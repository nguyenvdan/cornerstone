"""Probabilistic development projection (Phase 3 — the core).

Project a prospect as a *distribution* of outcomes, derived from how their
historical comparables actually developed. Nothing here is a point estimate.

Method
------
1. Find the prospect's similarity-weighted neighborhood (Phase 2 engine).
2. **Tier probabilities** — a similarity-weighted vote over the neighbors'
   realized outcome tiers (bust ... superstar).
3. **Career value distribution** — weighted percentiles of the neighbors'
   career VORP and peak BPM (the spread *is* the uncertainty).
4. **Development curve** — for each of the first N pro seasons, the weighted
   percentile band (p10/p25/p50/p75/p90) of the neighbors' season-by-season
   VORP. Non-played but *elapsed* seasons count as 0 (replacement value);
   seasons that haven't happened yet for a recent draftee are excluded.
5. **Swing factors** — local sensitivity: perturb each feature +/-1 SD and
   measure how the expected career VORP moves. The biggest movers are the
   attributes that most raise or lower the projection.

`project(prospect)` is the Phase 6 agent's `project` tool. The neighborhood
engine is injectable so the Phase 4 back-test can supply a leakage-safe,
draft-time-only universe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pipelines import config

from .comparables import ComparablesEngine
from .features import FEATURE_DISPLAY, SIMILARITY_FEATURES


def vorp_feature_weights(prospects: pd.DataFrame, floor: float = 0.1) -> dict[str, float]:
    """Weight each feature by its multivariate predictive contribution to VORP.

    Uses the absolute standardized Ridge coefficients of career VORP on the
    feature set (matured classes only). This turns the comparables metric from
    "pure profile similarity" into "similarity in the dimensions that relate to
    NBA value", and — unlike marginal correlations — it shares credit correctly
    across collinear stats (pts/usage/efg). A cross-validated bake-off picked
    Ridge weighting over correlation weighting and over gradient-boosted /
    engineered-feature variants (those overfit or added noise). A floor keeps
    every feature contributing something.
    """
    mat = prospects[
        prospects["has_college_stats"]
        & prospects["coll_pts_per_g"].notna()
        & (prospects["draft_year"] <= config.MATURE_DRAFT_CUTOFF)
        & prospects["career_vorp"].notna()
    ]
    X = mat[SIMILARITY_FEATURES].astype(float)
    X = X.fillna(X.median())
    vorp = mat["career_vorp"].astype(float).to_numpy()
    Xs = StandardScaler().fit_transform(X.values)
    coef = np.abs(Ridge(alpha=10.0).fit(Xs, vorp).coef_)
    coef = coef / coef.mean()
    return {f: float(max(coef[j], floor)) for j, f in enumerate(SIMILARITY_FEATURES)}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile (q in [0, 100]) via the standard CDF interpolation."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= weights.sum()
    return float(np.interp(q / 100.0, cum, values))


@dataclass
class SwingFactor:
    feature: str
    display: str
    effect_vorp: float        # change in expected career VORP per +1 SD of the feature
    direction: str            # "raises" / "lowers" projection when the feature increases


@dataclass
class Projection:
    prospect_name: str
    n_comparables: int
    tier_probabilities: dict[str, float]
    p_starter_plus: float          # P(starter or better) — cleaner than the modal bin
    p_star_plus: float             # P(all-star or better)
    modal_tier: str                # widest-probability bin (note: bins are uneven)
    expected_career_vorp: float
    career_vorp_band: dict[str, float]      # p10/p25/p50/p75/p90
    peak_bpm_band: dict[str, float]
    season_curve: list[dict] = field(default_factory=list)
    swing_factors: list[SwingFactor] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    top_comparables: list[dict] = field(default_factory=list)
    ceiling_comparable: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["swing_factors"] = [sf.__dict__ for sf in self.swing_factors]
        return d


POWER_CONFERENCES = {
    "Big 12", "SEC", "ACC", "Big Ten", "Big East", "Pac-12", "Pac-10", "Amer",
}


@dataclass
class ProjectionContext:
    """Optional, scouting-informed signals layered onto the profile projection.

    All default off, so the base ProjectionModel (used by the Phase 4 back-test)
    stays a pure, leakage-controlled profile model. Turned on, these add *real*
    pre-draft signal the profile-only model deliberately ignores.
    """
    draft_prior: bool = False          # weight comparables toward a similar draft slot
    draft_bandwidth: float = 8.0       # picks; smaller = tighter to the prospect's slot
    archetype_anchors: tuple[str, ...] = ()   # scout-named comps defining an archetype
    archetype_blend: float = 0.0       # 0..1 nudge of the query toward that archetype
    competition_match: bool = False    # compare to players who faced similar competition
    competition_penalty: float = 0.6   # weight multiplier for a competition-tier mismatch
    age_weight_boost: float = 1.0      # extra emphasis on (young) age at draft
    athleticism_match: bool = False    # boost comps with similar measured combine athleticism
    athleticism_boost: float = 0.6     # max extra weight for an athleticism match
    athleticism_bandwidth: float = 22.0  # percentile-points kernel width
    candidate_pool: int = 220          # pool to reweight before trimming to k


def _is_power(conf: object) -> bool:
    return isinstance(conf, str) and conf in POWER_CONFERENCES


class ProjectionModel:
    def __init__(
        self,
        prospects: pd.DataFrame,
        engine: ComparablesEngine | None = None,
        k: int = 75,
        bandwidth_rank: int = 10,
        mature_only: bool = True,
        context: ProjectionContext | None = None,
    ) -> None:
        self.context = context or ProjectionContext()
        universe = prospects.drop_duplicates("player_id")
        # Draw comparable outcomes only from matured draft classes so that recent
        # draftees with still-developing careers don't bias the projection down.
        # (A caller — e.g. the Phase 4 back-test — can inject its own engine and
        # set mature_only=False to control the universe explicitly.)
        if mature_only:
            universe = universe[universe["draft_year"] <= config.MATURE_DRAFT_CUTOFF]
        self.outcomes = universe.set_index("player_id")
        if engine is None:
            weights = vorp_feature_weights(prospects)
            if self.context.age_weight_boost != 1.0:
                weights = dict(weights)
                weights["age_at_draft"] *= self.context.age_weight_boost
            engine = ComparablesEngine().fit(universe, feature_weights=weights)
        self.engine = engine
        self.k = k
        self.bandwidth_rank = bandwidth_rank

        # Measured combine athleticism (partial coverage), used as a reweight.
        self._ath = {}
        if self.context.athleticism_match:
            ath_path = config.PROCESSED / "combine_athleticism.parquet"
            if ath_path.exists():
                adf = pd.read_parquet(ath_path)
                self._ath = dict(zip(adf["player_id"], adf["athleticism_pct"], strict=False))

        # Archetype centroid (raw feature space), averaged over the scout-named
        # anchors that exist in the matured universe.
        self._archetype = None
        if self.context.archetype_anchors and self.context.archetype_blend > 0:
            anchors = self.outcomes[
                self.outcomes["player_name"].isin(self.context.archetype_anchors)
            ]
            if not anchors.empty:
                feats = anchors[SIMILARITY_FEATURES].astype(float)
                self._archetype = feats.fillna(self.engine.medians_).mean()

    # -- helpers -----------------------------------------------------------
    def _effective_query(self, prospect: pd.Series) -> pd.Series:
        """Optionally nudge the query toward the scouting archetype centroid."""
        if self._archetype is None or self.context.archetype_blend <= 0:
            return prospect
        b = self.context.archetype_blend
        base = prospect.reindex(SIMILARITY_FEATURES).astype(float).fillna(self.engine.medians_)
        blended = (1 - b) * base + b * self._archetype
        q = prospect.copy()
        for f in SIMILARITY_FEATURES:
            q[f] = blended[f]
        return q

    def _neighbors(self, prospect: pd.Series):
        ctx = self.context
        query = self._effective_query(prospect)
        pid = prospect.get("player_id")
        active = ctx.draft_prior or ctx.competition_match
        pool = ctx.candidate_pool if active else self.k
        comps = self.engine.get_comparables(query, k=pool, exclude_player_id=pid)
        d = np.array([c.distance for c in comps])
        ids = [c.player_id for c in comps]
        h = max(float(d[min(self.bandwidth_rank - 1, len(d) - 1)]), 1e-6)
        weights = np.exp(-0.5 * (d / h) ** 2)

        if ctx.draft_prior and prospect.get("draft_pick") is not None \
                and not pd.isna(prospect.get("draft_pick")):
            picks = self.outcomes.loc[ids, "draft_pick"].to_numpy(float)
            pp = float(prospect["draft_pick"])
            dp = np.where(np.isnan(picks), 5.0, picks - pp)
            weights = weights * np.exp(-0.5 * (dp / ctx.draft_bandwidth) ** 2)
        if ctx.competition_match and isinstance(prospect.get("coll_conf"), str):
            same = np.array([_is_power(c) == _is_power(prospect["coll_conf"])
                             for c in self.outcomes.loc[ids, "coll_conf"]])
            weights = weights * np.where(same, 1.0, ctx.competition_penalty)
        if ctx.athleticism_match and self._ath.get(pid) is not None:
            pa = self._ath[pid]
            mult = np.ones(len(ids))
            for i, cid in enumerate(ids):
                ca = self._ath.get(cid)
                if ca is not None:  # boost athletically-similar comps; neutral if unmeasured
                    mult[i] = 1.0 + ctx.athleticism_boost * np.exp(
                        -0.5 * ((ca - pa) / ctx.athleticism_bandwidth) ** 2)
            weights = weights * mult

        if active and len(ids) > self.k:  # trim the reweighted pool back to k
            top = np.argsort(weights)[::-1][: self.k]
            comps = [comps[i] for i in top]
            ids = [ids[i] for i in top]
            weights = weights[top]
        return comps, ids, weights

    def _expected_career_vorp(self, prospect: pd.Series) -> float:
        _, ids, w = self._neighbors(prospect)
        v = self.outcomes.loc[ids, "career_vorp"].to_numpy(float)
        mask = ~np.isnan(v)
        return float(np.average(v[mask], weights=w[mask]))

    def _swing_factors(self, prospect: pd.Series, n: int = 6) -> list[SwingFactor]:
        # Keep the full row (draft_pick, conference, id) so any active context
        # still applies while we perturb individual features.
        base = prospect.copy()
        for feat in SIMILARITY_FEATURES:
            if pd.isna(base.get(feat)):
                base[feat] = self.engine.medians_[feat]
        factors: list[SwingFactor] = []
        for j, feat in enumerate(SIMILARITY_FEATURES):
            sd = float(self.engine.scaler_.scale_[j])
            up, dn = base.copy(), base.copy()
            up[feat] = float(up[feat]) + sd
            dn[feat] = float(dn[feat]) - sd
            effect = (self._expected_career_vorp(up) - self._expected_career_vorp(dn)) / 2.0
            factors.append(
                SwingFactor(
                    feature=feat,
                    display=FEATURE_DISPLAY[feat],
                    effect_vorp=round(effect, 2),
                    direction="raises" if effect >= 0 else "lowers",
                )
            )
        factors.sort(key=lambda f: abs(f.effect_vorp), reverse=True)
        return factors[:n]

    def _season_curve(self, ids: list[str], weights: np.ndarray) -> list[dict]:
        traj = self.outcomes.loc[ids, "trajectory_vorp"]
        draft_years = self.outcomes.loc[ids, "draft_year"].to_numpy(int)
        curve: list[dict] = []
        for s in range(config.OUTCOME_WINDOW):
            vals, wts = [], []
            for arr, dy, w in zip(traj, draft_years, weights, strict=False):
                season_elapsed = dy + s <= config.LAST_SEASON_START
                if not season_elapsed:
                    continue  # season hasn't happened yet -> genuinely unknown
                if arr is not None and s < len(arr) and arr[s] is not None:
                    vals.append(float(arr[s]))
                else:
                    vals.append(0.0)  # drafted but not in league that season -> replacement
                wts.append(w)
            if len(vals) < 10:
                continue
            vals_a, wts_a = np.array(vals), np.array(wts)
            curve.append(
                {
                    "season": s + 1,
                    "n_comps": len(vals),
                    "p10": round(weighted_percentile(vals_a, wts_a, 10), 2),
                    "p25": round(weighted_percentile(vals_a, wts_a, 25), 2),
                    "p50": round(weighted_percentile(vals_a, wts_a, 50), 2),
                    "p75": round(weighted_percentile(vals_a, wts_a, 75), 2),
                    "p90": round(weighted_percentile(vals_a, wts_a, 90), 2),
                }
            )
        return curve

    # -- public API --------------------------------------------------------
    def project(
        self,
        prospect: pd.Series,
        include_curve: bool = True,
        include_swing: bool = True,
    ) -> Projection:
        comps, ids, weights = self._neighbors(prospect)
        meta = self.outcomes.loc[ids]

        # 1) tier probabilities (similarity-weighted vote)
        tier_prob = {t: 0.0 for t in config.TIER_ORDER}
        for tier, w in zip(meta["outcome_tier"], weights, strict=False):
            tier_prob[tier] += w
        total = sum(tier_prob.values())
        tier_prob = {t: round(p / total, 3) for t, p in tier_prob.items()}
        modal_tier = max(tier_prob, key=tier_prob.get)
        p_star_plus = round(tier_prob["all_star"] + tier_prob["superstar"], 3)
        p_starter_plus = round(p_star_plus + tier_prob["starter"], 3)

        # 2) career value distribution
        cv = meta["career_vorp"].to_numpy(float)
        cmask = ~np.isnan(cv)
        cv_band = {f"p{q}": round(weighted_percentile(cv[cmask], weights[cmask], q), 1)
                   for q in (10, 25, 50, 75, 90, 95, 99)}
        expected_vorp = round(float(np.average(cv[cmask], weights=weights[cmask])), 1)

        # The single best realized outcome among his comparables = his data-grounded
        # ceiling ("if everything breaks right, this is who he becomes").
        ceil_i = int(np.argmax(np.where(cmask, cv, -np.inf)))
        ceiling_comparable = {
            "player_name": meta["player_name"].iloc[ceil_i],
            "career_vorp": round(float(cv[ceil_i]), 1),
            "outcome_tier": meta["outcome_tier"].iloc[ceil_i],
        }

        pb = meta["peak_bpm"].to_numpy(float)
        pmask = ~np.isnan(pb)
        peak_band = {f"p{q}": round(weighted_percentile(pb[pmask], weights[pmask], q), 1)
                     for q in (10, 50, 90)}

        # 3) development curve, 4) swing factors (skippable for fast back-testing)
        curve = self._season_curve(ids, weights) if include_curve else []
        swings = self._swing_factors(prospect) if include_swing else []

        # 5) honest uncertainty notes
        bust_p = tier_prob["bust"]
        notes = [
            f"80% career-VORP interval spans {cv_band['p10']} to {cv_band['p90']} "
            f"(median {cv_band['p50']}) — a wide range driven by how differently the "
            f"comparables developed.",
            f"P(starter or better) = {p_starter_plus:.0%}; P(all-star or better) = "
            f"{p_star_plus:.0%}; P(bust) = {bust_p:.0%}. The projection is a "
            f"distribution, not a single outcome.",
            "College box-score profiles are weak predictors of NBA stardom, so the "
            "uncertainty here is real, not hidden — comparables also don't capture "
            "team fit, health, or development environment.",
        ]

        top = [
            {"player_name": c.player_name, "draft_year": c.draft_year,
             "similarity": c.similarity, "outcome_tier": c.outcome_tier,
             "career_vorp": c.career_vorp}
            for c in comps[:8]
        ]

        return Projection(
            prospect_name=prospect.get("player_name", "prospect"),
            n_comparables=len(ids),
            tier_probabilities=tier_prob,
            p_starter_plus=p_starter_plus,
            p_star_plus=p_star_plus,
            modal_tier=modal_tier,
            expected_career_vorp=expected_vorp,
            career_vorp_band=cv_band,
            peak_bpm_band=peak_band,
            season_curve=curve,
            swing_factors=swings,
            key_uncertainties=notes,
            top_comparables=top,
            ceiling_comparable=ceiling_comparable,
        )


# --------------------------------------------------------------------------
# CLI: project Dybantsa, print the distribution, write JSON for the frontend.
# --------------------------------------------------------------------------
def _bar(p: float, width: int = 24) -> str:
    return "█" * int(round(p * width)) + "·" * (width - int(round(p * width)))


# Scout-named playstyle comps that exist within the 2003-2022 data window
# (Tracy McGrady is outside it — drafted 1997 from high school). Kevin Durant
# anchors the "KD ceiling" the front office sees in him.
DYBANTSA_ARCHETYPE = ("Kevin Durant", "Jayson Tatum", "Jaylen Brown",
                      "Shai Gilgeous-Alexander")


def dybantsa_context() -> ProjectionContext:
    """Scouting-informed context for AJ Dybantsa: layers in the real pre-draft
    signals the profile-only model ignores."""
    return ProjectionContext(
        draft_prior=True, draft_bandwidth=5.0,   # tight to his elite (top-3) slot
        archetype_anchors=DYBANTSA_ARCHETYPE, archetype_blend=0.3,
        competition_match=True, competition_penalty=0.6,
        age_weight_boost=2.0,
        athleticism_match=True,
    )


DYBANTSA_ADJUSTMENTS = [
    "Draft capital: conditioned on his #1-overall draft slot (Washington, 2026) — "
    "top picks historically bust far less than the average drafted player.",
    f"Scouting archetype: query nudged toward {', '.join(DYBANTSA_ARCHETYPE)} (his "
    "scout-cited comps), using the FULL outcome distribution of similar wings — "
    "cautionary cases (e.g. Wiggins, Barrett) included, not just the stars.",
    "Strength of competition: credited for producing in the Big 12 (a power "
    "conference), comparing him to players who faced similar competition.",
    "Age weighting: extra emphasis on how young he is for his production.",
    "Combine athleticism: his measured 2026 combine line (42\" max vertical — "
    "combine-best — 7'0.5\" wingspan, 8'10\" standing reach) upweights "
    "comparables of similar measured athleticism (where combine data exists).",
]


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]

    base = ProjectionModel(prospects).project(dyb, include_curve=False, include_swing=False)
    model = ProjectionModel(prospects, context=dybantsa_context())
    proj = model.project(dyb)

    print(f"AJ Dybantsa — SCOUTING-INFORMED projection (from {proj.n_comparables} comparables)")
    print(f"  [profile-only baseline: P(starter+) {base.p_starter_plus:.0%}, "
          f"P(all-star+) {base.p_star_plus:.0%}, P(bust) {base.tier_probabilities['bust']:.0%}]\n")
    print("Outcome tier probabilities:")
    for tier in reversed(config.TIER_ORDER):
        p = proj.tier_probabilities[tier]
        print(f"  {tier:10s} {p:5.1%}  {_bar(p)}")
    print(f"\nP(starter or better): {proj.p_starter_plus:.0%}   "
          f"P(all-star or better): {proj.p_star_plus:.0%}   "
          f"P(bust): {proj.tier_probabilities['bust']:.0%}")
    print(f"Expected career VORP: {proj.expected_career_vorp}  "
          f"(80% range {proj.career_vorp_band['p10']} to {proj.career_vorp_band['p90']}, "
          f"median {proj.career_vorp_band['p50']})")
    print(f"Peak BPM band: p10 {proj.peak_bpm_band['p10']} / "
          f"p50 {proj.peak_bpm_band['p50']} / p90 {proj.peak_bpm_band['p90']}")

    print("\nProjected VORP by season (p25–p75 band around median):")
    for row in proj.season_curve:
        print(f"  Year {row['season']}: median {row['p50']:5.1f}  "
              f"[{row['p25']:5.1f} – {row['p75']:5.1f}]   (n={row['n_comps']})")

    print("\nTop swing factors (effect on expected career VORP per +1 SD):")
    for sf in proj.swing_factors:
        arrow = "↑" if sf.direction == "raises" else "↓"
        print(f"  {arrow} {sf.display:24s} {sf.effect_vorp:+.2f}")

    print("\nAdjustments applied:")
    for note in DYBANTSA_ADJUSTMENTS:
        print(f"  • {note}")

    out = proj.to_dict()
    out["model_mode"] = "scouting-informed"
    out["adjustments"] = DYBANTSA_ADJUSTMENTS
    out["archetype_anchors"] = list(DYBANTSA_ARCHETYPE)

    # Surface AJ's measured 2026 combine line for the frontend.
    ath_path = config.PROCESSED / "combine_athleticism.parquet"
    if ath_path.exists():
        from pipelines.combine import DYBANTSA_COMBINE
        arow = pd.read_parquet(ath_path).query("player_id == 'dybanaj01'")
        out["combine"] = {
            "height_no_shoes_in": DYBANTSA_COMBINE["height_no_shoes"],
            "wingspan_in": DYBANTSA_COMBINE["wingspan"],
            "standing_reach_in": DYBANTSA_COMBINE["standing_reach"],
            "max_vertical_in": DYBANTSA_COMBINE["vertical_max"],
            "length_in": float(arow["length"].iloc[0]) if not arow.empty else None,
            "athleticism_pct": float(arow["athleticism_pct"].iloc[0]) if not arow.empty else None,
        }
    out["profile_only"] = {
        "p_starter_plus": base.p_starter_plus,
        "p_star_plus": base.p_star_plus,
        "p_bust": base.tier_probabilities["bust"],
        "expected_career_vorp": base.expected_career_vorp,
    }
    path = config.PROCESSED / "dybantsa_projection.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
