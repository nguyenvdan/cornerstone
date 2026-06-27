"""Feature representation for prospect similarity (Phase 2).

The similarity feature set is intentionally restricted to fields that are
(a) available at draft time and (b) well-populated across the 2003-2022 college
universe (>= ~90% non-null; the few gaps are median-imputed at fit time). The
sparse college advanced metrics ``coll_per`` / ``coll_bpm`` (~60-68% present)
are deliberately excluded so the embedding isn't dominated by imputed values.

Position is carried as *metadata* shown alongside each comparable, not as a
distance dimension: height, weight, assist rate and block rate already encode
role, and keeping the distance purely continuous makes per-feature attribution
clean and honest.
"""

from __future__ import annotations

# Ordered feature vector used for the embedding / nearest-neighbor search.
SIMILARITY_FEATURES: list[str] = [
    "age_at_draft",
    "height_in",
    "weight_lb",
    "coll_mp_per_g",
    "coll_pts_per_g",
    "coll_trb_per_g",
    "coll_ast_per_g",
    "coll_stl_per_g",
    "coll_blk_per_g",
    "coll_tov_per_g",
    "coll_fg_pct",
    "coll_fg3_pct",
    "coll_ft_pct",
    "coll_efg_pct",
    "coll_ts_pct",
    "coll_usg_pct",
]

# Human-readable labels for explanations.
FEATURE_DISPLAY: dict[str, str] = {
    "age_at_draft": "age at draft",
    "height_in": "height",
    "weight_lb": "weight",
    "coll_mp_per_g": "college minutes/game",
    "coll_pts_per_g": "college points/game",
    "coll_trb_per_g": "college rebounds/game",
    "coll_ast_per_g": "college assists/game",
    "coll_stl_per_g": "college steals/game",
    "coll_blk_per_g": "college blocks/game",
    "coll_tov_per_g": "college turnovers/game",
    "coll_fg_pct": "college FG%",
    "coll_fg3_pct": "college 3P%",
    "coll_ft_pct": "college FT%",
    "coll_efg_pct": "college eFG%",
    "coll_ts_pct": "college TS%",
    "coll_usg_pct": "college usage%",
    # candidate enriched features (Phase: missing predictors)
    "rsci_rank": "HS recruiting rank",
    "conference_strength": "conference strength",
    "age_minus_class": "age vs class",
    "athleticism_pct": "combine athleticism",
}

# Metadata columns surfaced with each comparable (not used in the distance).
META_COLUMNS: list[str] = [
    "player_name",
    "draft_year",
    "draft_pick",
    "position",
    "outcome_tier",
    "career_vorp",
]


def format_value(feature: str, value: float | None) -> str:
    """Pretty-print a raw feature value for human-facing explanations."""
    if value is None:
        return "n/a"
    if feature == "height_in":
        feet, inch = divmod(int(round(value)), 12)
        return f"{feet}'{inch}\""
    if feature == "weight_lb":
        return f"{int(round(value))} lb"
    if feature == "coll_usg_pct":
        # usage is already stored on a 0-100 scale
        return f"{value:.1f}%"
    if feature.endswith("_pct"):
        # shooting percentages are stored as fractions (e.g. 0.51) -> percent
        return f"{value * 100:.1f}%"
    if feature == "age_at_draft":
        return f"{value:.1f} yr"
    return f"{value:.1f}"
