"""Roster-fit engine (Phase 5 — the product layer).

Turns a cornerstone *projection* into team decision-support: what skills best
complement him, how the current roster covers those needs, where the gaps are,
and which archetypes (or what-if additions) would help most.

Approach (explainable, data-grounded heuristic):
1. Estimate the cornerstone's own skill *supply* from his profile (percentiles
   within the prospect universe) -> what he already provides.
2. Derive a *need* profile: importance of each skill around a high-usage
   perimeter cornerstone, scaled up for skills he lacks (you build around the
   star's weaknesses, not his strengths).
3. Score the roster's *supply* per skill (a blend of its best provider and its
   minutes-weighted depth), then gap = unmet need.
4. Rank complementary archetypes by how well they fill the gaps; support
   what-if additions.

`evaluate_fit(roster, cornerstone_supply)` is the Phase 6 agent's `evaluate_fit`
tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipelines import config
from pipelines.nba_skills import SEASON, SKILLS

# Importance of each skill *around a high-usage perimeter cornerstone* — you
# surround a ball-dominant scoring wing with spacing, rim protection and
# defense, not with more shot creation.
BASE_IMPORTANCE = {
    "spacing": 1.0,
    "rim_protection": 0.9,
    "perimeter_defense": 0.8,
    "playmaking": 0.7,
    "rebounding": 0.6,
    "shot_creation": 0.3,
}

# Complementary archetypes, described by the skills they supply (0-100).
ARCHETYPES: dict[str, dict[str, float]] = {
    "3&D Wing": {"spacing": 85, "perimeter_defense": 85},
    "Stretch Big": {"spacing": 85, "rebounding": 75, "rim_protection": 60},
    "Defensive Anchor Big": {"rim_protection": 92, "rebounding": 85, "perimeter_defense": 55},
    "Lob-Threat Center": {"rim_protection": 88, "rebounding": 88},
    "Floor General PG": {"playmaking": 90, "spacing": 65, "perimeter_defense": 55},
    "Movement Shooter": {"spacing": 92, "perimeter_defense": 45},
    "Two-Way Wing": {"perimeter_defense": 82, "spacing": 70, "shot_creation": 55},
}

# Map a prospect's college box stats to the six NBA skill dimensions.
_COLLEGE_SKILL_SOURCE = {
    "shot_creation": "coll_usg_pct",
    "playmaking": "coll_ast_per_g",
    "spacing": "coll_fg3_pct",
    "rim_protection": "coll_blk_per_g",
    "rebounding": "coll_trb_per_g",
    "perimeter_defense": "coll_stl_per_g",
}


@dataclass
class SkillRow:
    skill: str
    need: float            # 0-100 importance of this skill for this cornerstone
    roster_supply: float   # 0-100 how well the roster currently provides it
    gap: float             # 0-100 unmet need (high = priority)


@dataclass
class RosterFitReport:
    fit_score: float                       # 0-100 overall coverage of weighted needs
    skills: list[SkillRow]
    biggest_gaps: list[str]
    recommended_archetypes: list[dict]     # ranked, with score + the gaps they fill
    cornerstone_supply: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["skills"] = [s.__dict__ for s in self.skills]
        return d


def _pctile(college: pd.DataFrame, col: str, val: object) -> float | None:
    ref = college[col].dropna().to_numpy()
    if val is None or (isinstance(val, float) and np.isnan(val)) or len(ref) == 0:
        return None
    return 100.0 * float((ref < val).mean())


def cornerstone_skill_supply(prospect: pd.Series, prospects: pd.DataFrame) -> dict[str, float]:
    """Estimate the cornerstone's skill supply as percentiles vs the prospect
    universe (a projected *role tendency*, not NBA production)."""
    college = prospects[prospects["has_college_stats"]]
    supply: dict[str, float] = {}
    for skill, col in _COLLEGE_SKILL_SOURCE.items():
        p = _pctile(college, col, prospect.get(col))
        supply[skill] = round(p, 1) if p is not None else 50.0
    # Shooting is more than 3P% on a freshman sample: free-throw % is a strong,
    # well-established predictor of shooting development. Blend the two.
    f3 = _pctile(college, "coll_fg3_pct", prospect.get("coll_fg3_pct"))
    ft = _pctile(college, "coll_ft_pct", prospect.get("coll_ft_pct"))
    parts = [x for x in (f3, ft) if x is not None]
    if parts:
        supply["spacing"] = round(sum(parts) / len(parts), 1)
    return supply


def _need_profile(supply: dict[str, float]) -> dict[str, float]:
    # importance scaled up where the cornerstone is weak; normalized to 0-100.
    raw = {s: BASE_IMPORTANCE[s] * (1 - supply[s] / 100.0) for s in SKILLS}
    hi = max(raw.values()) or 1.0
    return {s: round(100.0 * raw[s] / hi, 1) for s in SKILLS}


# Trajectory: a cornerstone's competitive window is years out, so a young,
# ascending roster's *future* skill matters. Credit youth up to +25% (a 19-yo),
# fading to 0 by peak age — capped so nobody exceeds the 100 percentile ceiling.
PEAK_AGE = 27.0
YOUNG_AGE = 19.0
UPSIDE_MAX = 0.25


def _upside(age: float) -> float:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return 1.0
    frac = float(np.clip((PEAK_AGE - age) / (PEAK_AGE - YOUNG_AGE), 0.0, 1.0))
    return 1.0 + UPSIDE_MAX * frac


def _roster_supply(roster: pd.DataFrame, trajectory: bool = True) -> dict[str, float]:
    """Per skill: blend the roster's best provider with minutes-weighted depth,
    optionally crediting young/ascending players' upside."""
    w = roster["mp_per_g"].clip(lower=1).to_numpy()
    if trajectory and "age" in roster.columns:
        up = roster["age"].map(_upside).to_numpy(float)
    else:
        up = np.ones(len(roster))
    out: dict[str, float] = {}
    for skill in SKILLS:
        vals = np.minimum(roster[skill].to_numpy(float) * up, 100.0)
        best = float(np.nanmax(vals)) if len(vals) else 0.0
        depth = float(np.average(vals, weights=w)) if len(vals) else 0.0
        out[skill] = round(0.5 * best + 0.5 * depth, 1)
    return out


def _fit_from(need: dict[str, float], supply: dict[str, float]) -> tuple[float, list[SkillRow]]:
    rows, num, den = [], 0.0, 0.0
    for skill in SKILLS:
        gap = round(need[skill] * (100 - supply[skill]) / 100.0, 1)
        rows.append(SkillRow(skill, need[skill], supply[skill], gap))
        num += need[skill] * min(supply[skill], 100) / 100.0
        den += need[skill]
    fit = round(100.0 * num / den, 1) if den else 0.0
    rows.sort(key=lambda r: r.gap, reverse=True)
    return fit, rows


def _rank_archetypes(rows: list[SkillRow]) -> list[dict]:
    gap = {r.skill: r.gap for r in rows}
    ranked = []
    for name, vec in ARCHETYPES.items():
        score = sum(gap[s] * v / 100.0 for s, v in vec.items())
        fills = sorted(vec, key=lambda s: gap[s] * vec[s], reverse=True)[:2]
        ranked.append({"archetype": name, "score": round(score, 1),
                       "fills": fills, "provides": vec})
    ranked.sort(key=lambda a: a["score"], reverse=True)
    return ranked


def _team_cornerstone(team_players: pd.DataFrame) -> pd.Series:
    """The team's focal point: highest-usage real-minutes player."""
    pool = team_players[team_players["mp_per_g"] >= 25]
    if pool.empty:
        pool = team_players
    return pool.loc[pool["usg_pct"].idxmax()]


def compute_league_fit(skills: pd.DataFrame) -> pd.DataFrame:
    """Fit score for every team around its own cornerstone — the reference
    distribution that gives an absolute fit number context."""
    q = skills[skills["is_qualified"]]
    teams = sorted({t for ts in q["teams"] for t in ts})
    rows = []
    for team in teams:
        tp = q[q["teams"].apply(lambda ts, t=team: t in ts)]
        if len(tp) < 5:
            continue
        corner = _team_cornerstone(tp)
        supply = {s: float(corner[s]) for s in SKILLS}
        roster = tp[tp["player_id"] != corner["player_id"]]
        fit, _ = _fit_from(_need_profile(supply), _roster_supply(roster))
        rows.append({"team": team, "cornerstone": corner["player_name"], "fit_score": fit})
    return pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)


def calibrate(fit_score: float, league: pd.DataFrame) -> dict:
    """Place a fit score in the league distribution (rank + percentile)."""
    scores = league["fit_score"].to_numpy()
    better = int((scores > fit_score).sum())
    return {
        "n_teams": int(len(league)),
        "rank": better + 1,
        "percentile": round(100.0 * (scores < fit_score).mean(), 0),
        "league_median": round(float(np.median(scores)), 1),
        "league_best": {"team": league.iloc[0]["team"],
                        "fit_score": float(league.iloc[0]["fit_score"])},
    }


def evaluate_fit(roster: pd.DataFrame, cornerstone_supply: dict[str, float]) -> RosterFitReport:
    need = _need_profile(cornerstone_supply)
    supply = _roster_supply(roster)
    fit, rows = _fit_from(need, supply)
    archetypes = _rank_archetypes(rows)
    gaps = [r.skill for r in rows[:3]]
    notes = [
        f"Overall fit {fit:.0f}/100: how well the roster covers the skills this "
        f"cornerstone most needs around him.",
        f"Biggest unmet needs: {', '.join(g.replace('_', ' ') for g in gaps)}.",
        f"Top complementary archetype: {archetypes[0]['archetype']} "
        f"(fills {', '.join(f.replace('_', ' ') for f in archetypes[0]['fills'])}).",
        "Trajectory-adjusted: young, ascending players are credited for upside, "
        "since a cornerstone's window is years out. Calibrated against all 30 "
        "teams so the score has an anchor. Captures skills, not contracts or scheme.",
    ]
    return RosterFitReport(fit, rows, gaps, archetypes, cornerstone_supply, notes)


def what_if(roster: pd.DataFrame, cornerstone_supply: dict[str, float],
            archetype: str, mp_per_g: float = 28.0) -> dict:
    """Add a hypothetical archetype player and report the fit delta."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype: {archetype}")
    before = evaluate_fit(roster, cornerstone_supply)
    add = {"player_name": f"[hypothetical {archetype}]", "mp_per_g": mp_per_g}
    for skill in SKILLS:
        add[skill] = float(ARCHETYPES[archetype].get(skill, 40))
    after_roster = pd.concat([roster, pd.DataFrame([add])], ignore_index=True)
    after = evaluate_fit(after_roster, cornerstone_supply)
    return {"archetype": archetype, "fit_before": before.fit_score,
            "fit_after": after.fit_score, "fit_delta": round(after.fit_score - before.fit_score, 1),
            "new_biggest_gaps": after.biggest_gaps}


# --------------------------------------------------------------------------
def _load_wizards() -> pd.DataFrame:
    skills = pd.read_parquet(config.PROCESSED / f"nba_skills_{SEASON}.parquet")
    return skills[skills["teams"].apply(lambda t: "WAS" in t) & skills["is_qualified"]].copy()


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    skills = pd.read_parquet(config.PROCESSED / f"nba_skills_{SEASON}.parquet")
    roster = _load_wizards()

    supply = cornerstone_skill_supply(dyb, prospects)
    report = evaluate_fit(roster, supply)
    league = compute_league_fit(skills)
    cal = calibrate(report.fit_score, league)

    print(f"Wizards roster fit for AJ Dybantsa — {len(roster)} rotation players")
    print(f"  {report.fit_score}/100 — ranks {cal['rank']} of {cal['n_teams']} teams "
          f"({cal['percentile']:.0f}th pctile; league median {cal['league_median']}).\n")
    print("AJ Dybantsa projected skill supply (role tendency, percentile):")
    print("  " + "  ".join(f"{s.replace('_',' ')} {int(v)}" for s, v in supply.items()))
    print(f"\nOverall roster fit: {report.fit_score}/100\n")
    print(f"{'skill':20s} {'need':>6s} {'roster':>7s} {'gap':>6s}")
    print("-" * 42)
    for r in report.skills:
        print(f"{r.skill.replace('_',' '):20s} {r.need:6.0f} {r.roster_supply:7.0f} {r.gap:6.0f}")

    print("\nRecommended complementary archetypes:")
    for a in report.recommended_archetypes[:4]:
        print(f"  {a['archetype']:22s} score {a['score']:5.1f}  "
              f"(fills {', '.join(f.replace('_',' ') for f in a['fills'])})")

    print("\nWhat-if additions:")
    for arch in ["Defensive Anchor Big", "3&D Wing", "Movement Shooter"]:
        w = what_if(roster, supply, arch)
        print(f"  add {arch:22s} fit {w['fit_before']} -> {w['fit_after']}  "
              f"(Δ {w['fit_delta']:+.1f})")

    print(f"\nLeague calibration ({cal['n_teams']} teams): WAS ranks {cal['rank']} "
          f"(best: {cal['league_best']['team']} {cal['league_best']['fit_score']}).")

    roster_cols = ["player_name", "age", "mp_per_g", *SKILLS, "primary_skill"]
    out = {"cornerstone": "AJ Dybantsa", "report": report.to_dict(),
           "calibration": cal,
           "league_table": league.to_dict("records"),
           "roster": roster[roster_cols].to_dict("records")}
    path = config.PROCESSED / "wizards_fit.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
