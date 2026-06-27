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

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["swing_factors"] = [sf.__dict__ for sf in self.swing_factors]
        return d


class ProjectionModel:
    def __init__(
        self,
        prospects: pd.DataFrame,
        engine: ComparablesEngine | None = None,
        k: int = 75,
        bandwidth_rank: int = 10,
        mature_only: bool = True,
    ) -> None:
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
            engine = ComparablesEngine().fit(universe, feature_weights=weights)
        self.engine = engine
        self.k = k
        # Adaptive-kernel bandwidth = distance to this-ranked nearest neighbor,
        # so the ~N closest analogs carry most of the weight and the long tail
        # tapers off (instead of dragging the projection toward the base rate).
        self.bandwidth_rank = bandwidth_rank

    # -- helpers -----------------------------------------------------------
    def _neighbors(self, prospect: pd.Series):
        pid = prospect.get("player_id")
        comps = self.engine.get_comparables(prospect, k=self.k, exclude_player_id=pid)
        d = np.array([c.distance for c in comps])
        h = max(float(d[min(self.bandwidth_rank - 1, len(d) - 1)]), 1e-6)
        weights = np.exp(-0.5 * (d / h) ** 2)
        ids = [c.player_id for c in comps]
        return comps, ids, weights

    def _expected_career_vorp(self, prospect: pd.Series) -> float:
        _, ids, w = self._neighbors(prospect)
        v = self.outcomes.loc[ids, "career_vorp"].to_numpy(float)
        mask = ~np.isnan(v)
        return float(np.average(v[mask], weights=w[mask]))

    def _swing_factors(self, prospect: pd.Series, n: int = 6) -> list[SwingFactor]:
        base = prospect.reindex(SIMILARITY_FEATURES).astype(float).fillna(self.engine.medians_)
        factors: list[SwingFactor] = []
        for j, feat in enumerate(SIMILARITY_FEATURES):
            sd = float(self.engine.scaler_.scale_[j])
            up, dn = base.copy(), base.copy()
            up[feat] += sd
            dn[feat] -= sd
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
    def project(self, prospect: pd.Series) -> Projection:
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
                   for q in (10, 25, 50, 75, 90)}
        expected_vorp = round(float(np.average(cv[cmask], weights=weights[cmask])), 1)

        pb = meta["peak_bpm"].to_numpy(float)
        pmask = ~np.isnan(pb)
        peak_band = {f"p{q}": round(weighted_percentile(pb[pmask], weights[pmask], q), 1)
                     for q in (10, 50, 90)}

        # 3) development curve, 4) swing factors
        curve = self._season_curve(ids, weights)
        swings = self._swing_factors(prospect)

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
        )


# --------------------------------------------------------------------------
# CLI: project Dybantsa, print the distribution, write JSON for the frontend.
# --------------------------------------------------------------------------
def _bar(p: float, width: int = 24) -> str:
    return "█" * int(round(p * width)) + "·" * (width - int(round(p * width)))


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    model = ProjectionModel(prospects)
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    proj = model.project(dyb)

    print(f"AJ Dybantsa — probabilistic projection (from {proj.n_comparables} comparables)\n")
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

    print("\nKey uncertainties:")
    for note in proj.key_uncertainties:
        print(f"  • {note}")

    path = config.PROCESSED / "dybantsa_projection.json"
    path.write_text(json.dumps(proj.to_dict(), indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
