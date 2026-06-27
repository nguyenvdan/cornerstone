"""The agent's toolbelt — the modeling functions exposed as callable tools.

Each tool returns a JSON-serializable dict (the same payload the LLM sees and
the scripted orchestrator consumes). Heavy resources (data, fitted models) load
once into a shared ``Context`` and are reused across calls.

Tools:
    lookup_prospect        pre-draft profile (+ realized outcome for historicals)
    find_comparables       explained historical analogs
    project_development    probabilistic projection (tiers, bands, swing factors)
    evaluate_roster_fit    team fit + gaps + recommended archetypes around a star
    team_skill_summary     a team's current rotation skill profile
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from models.comparables import ComparablesEngine
from models.projection import ProjectionModel
from models.roster_fit import cornerstone_skill_supply, evaluate_fit
from pipelines import config
from pipelines.nba_skills import SEASON, SKILLS


class Context:
    """Lazy, cached data + models shared by all tools."""

    def __init__(self) -> None:
        self._prospects = None
        self._dybantsa = None
        self._comp_engine = None
        self._proj_model = None
        self._skills = None

    @property
    def prospects(self) -> pd.DataFrame:
        if self._prospects is None:
            self._prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
        return self._prospects

    @property
    def dybantsa(self) -> pd.Series:
        if self._dybantsa is None:
            self._dybantsa = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
        return self._dybantsa

    @property
    def comp_engine(self) -> ComparablesEngine:
        if self._comp_engine is None:
            self._comp_engine = ComparablesEngine().fit(self.prospects)
        return self._comp_engine

    @property
    def proj_model(self) -> ProjectionModel:
        if self._proj_model is None:
            self._proj_model = ProjectionModel(self.prospects)
        return self._proj_model

    @property
    def skills(self) -> pd.DataFrame:
        if self._skills is None:
            self._skills = pd.read_parquet(config.PROCESSED / f"nba_skills_{SEASON}.parquet")
        return self._skills

    def resolve(self, name: str) -> pd.Series | None:
        if name and "dybantsa" in name.lower():
            return self.dybantsa
        hit = self.prospects[self.prospects["player_name"].str.lower() == (name or "").lower()]
        return hit.iloc[0] if not hit.empty else None

    def team_roster(self, team: str) -> pd.DataFrame:
        t = team.upper()
        return self.skills[self.skills["teams"].apply(lambda ts: t in ts)
                           & self.skills["is_qualified"]].copy()


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[..., dict]

    def anthropic_spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
def _tool_lookup_prospect(ctx: Context, name: str) -> dict:
    row = ctx.resolve(name)
    if row is None:
        return {"error": f"no prospect named '{name}' found"}
    fields = ["player_name", "draft_year", "draft_pick", "college_name", "age_at_draft",
              "height_in", "weight_lb", "position", "coll_pts_per_g", "coll_trb_per_g",
              "coll_ast_per_g", "coll_stl_per_g", "coll_blk_per_g", "coll_ts_pct",
              "coll_usg_pct", "coll_fg3_pct"]
    out = {f: (None if pd.isna(row.get(f)) else row.get(f)) for f in fields if f in row.index}
    if "outcome_tier" in row.index and not pd.isna(row.get("outcome_tier")):
        vorp = row.get("career_vorp")
        out["realized_outcome_tier"] = row["outcome_tier"]
        out["realized_career_vorp"] = None if pd.isna(vorp) else float(vorp)
    return out


def _tool_find_comparables(ctx: Context, name: str, k: int = 8) -> dict:
    row = ctx.resolve(name)
    if row is None:
        return {"error": f"no prospect named '{name}' found"}
    pid = row.get("player_id")
    comps = ctx.comp_engine.get_comparables(row, k=int(k), exclude_player_id=pid)
    return {
        "prospect": row.get("player_name"),
        "comparables": [
            {"player_name": c.player_name, "draft_year": c.draft_year,
             "similarity": c.similarity, "outcome_tier": c.outcome_tier,
             "career_vorp": c.career_vorp,
             "why_alike": [d.display for d in c.most_alike]}
            for c in comps
        ],
    }


def _tool_project_development(ctx: Context, name: str) -> dict:
    row = ctx.resolve(name)
    if row is None:
        return {"error": f"no prospect named '{name}' found"}
    p = ctx.proj_model.project(row, include_curve=True, include_swing=True)
    return {
        "prospect": p.prospect_name,
        "n_comparables": p.n_comparables,
        "tier_probabilities": p.tier_probabilities,
        "p_starter_plus": p.p_starter_plus,
        "p_star_plus": p.p_star_plus,
        "expected_career_vorp": p.expected_career_vorp,
        "career_vorp_band": p.career_vorp_band,
        "season_curve": p.season_curve,
        "swing_factors": [{"factor": s.display, "direction": s.direction,
                           "effect_vorp": s.effect_vorp} for s in p.swing_factors],
        "key_uncertainties": p.key_uncertainties,
    }


def _tool_evaluate_roster_fit(ctx: Context, team: str = "WAS",
                              cornerstone: str = "AJ Dybantsa") -> dict:
    roster = ctx.team_roster(team)
    if roster.empty:
        return {"error": f"no qualified roster found for team '{team}'"}
    star = ctx.resolve(cornerstone)
    if star is None:
        return {"error": f"no cornerstone named '{cornerstone}' found"}
    supply = cornerstone_skill_supply(star, ctx.prospects)
    rep = evaluate_fit(roster, supply)
    return {
        "team": team.upper(),
        "cornerstone": cornerstone,
        "fit_score": rep.fit_score,
        "cornerstone_supply": rep.cornerstone_supply,
        "skills": [s.__dict__ for s in rep.skills],
        "biggest_gaps": rep.biggest_gaps,
        "recommended_archetypes": [
            {"archetype": a["archetype"], "score": a["score"], "fills": a["fills"]}
            for a in rep.recommended_archetypes[:4]
        ],
    }


def _tool_team_skill_summary(ctx: Context, team: str = "WAS") -> dict:
    roster = ctx.team_roster(team)
    if roster.empty:
        return {"error": f"no qualified roster found for team '{team}'"}
    roster = roster.sort_values("mp_per_g", ascending=False)
    return {
        "team": team.upper(),
        "n_rotation": int(len(roster)),
        "players": [
            {"player_name": r["player_name"], "mp_per_g": round(float(r["mp_per_g"]), 1),
             "primary_skill": r["primary_skill"],
             "skills": {s: round(float(r[s]), 0) for s in SKILLS}}
            for _, r in roster.head(10).iterrows()
        ],
    }


def build_tools(ctx: Context) -> dict[str, Tool]:
    _name = {"type": "string", "description": "Player name, e.g. 'AJ Dybantsa'."}
    specs = [
        Tool("lookup_prospect",
             "Look up a prospect's pre-draft profile (and realized NBA outcome if "
             "they are a historical player).",
             {"type": "object", "properties": {"name": _name}, "required": ["name"]},
             lambda name: _tool_lookup_prospect(ctx, name)),
        Tool("find_comparables",
             "Find the most similar historical prospects by pre-draft profile, with "
             "an explanation of why each is alike.",
             {"type": "object", "properties": {
                 "name": _name,
                 "k": {"type": "integer", "description": "How many analogs (default 8)."}},
              "required": ["name"]},
             lambda name, k=8: _tool_find_comparables(ctx, name, k)),
        Tool("project_development",
             "Project a prospect's NBA development as a probability distribution: "
             "outcome-tier probabilities, expected career VORP with an 80% band, a "
             "season-by-season curve, and the swing factors that move the projection.",
             {"type": "object", "properties": {"name": _name}, "required": ["name"]},
             lambda name: _tool_project_development(ctx, name)),
        Tool("evaluate_roster_fit",
             "Given a cornerstone and an NBA team, score how well the team's current "
             "roster complements him, the biggest skill gaps, and recommended "
             "complementary archetypes.",
             {"type": "object", "properties": {
                 "team": {"type": "string", "description": "Team abbreviation, e.g. 'WAS'."},
                 "cornerstone": _name},
              "required": []},
             lambda team="WAS", cornerstone="AJ Dybantsa":
                 _tool_evaluate_roster_fit(ctx, team, cornerstone)),
        Tool("team_skill_summary",
             "Summarize a team's current rotation and each player's skill profile.",
             {"type": "object", "properties": {
                 "team": {"type": "string", "description": "Team abbreviation, e.g. 'WAS'."}},
              "required": []},
             lambda team="WAS": _tool_team_skill_summary(ctx, team)),
    ]
    return {t.name: t for t in specs}
