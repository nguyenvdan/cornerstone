"""Assemble the pre-draft feature row for a prospect.

Every field here is knowable *at draft time* (no leakage): draft slot, body
measurements, age, and final-college-season production. These are the inputs to
the comparables and projection models in later phases.
"""

from __future__ import annotations

from datetime import date

from dateutil import parser as dateparser

from .sources.bbref import DraftPick, PlayerPage

# NBA drafts are held in late June; we approximate draft day for age math.
_DRAFT_DAY = (6, 26)


def age_at_draft(birth_date: str | None, draft_year: int) -> float | None:
    if not birth_date:
        return None
    try:
        bd = dateparser.parse(birth_date).date()
    except (ValueError, TypeError):
        return None
    draft_day = date(draft_year, *_DRAFT_DAY)
    return round((draft_day - bd).days / 365.25, 2)


def build_features(
    pick: DraftPick,
    player: PlayerPage | None,
    college: dict | None,
) -> dict:
    """Combine draft slot + bio + final college season into a feature row."""
    row: dict = {
        "player_id": pick.player_id,
        "player_name": pick.player_name,
        "draft_year": pick.draft_year,
        "draft_pick": pick.pick_overall,
        "draft_round": 1 if (pick.pick_overall or 99) <= 30 else 2,
        "draft_team": pick.team_id,
        "college_name": pick.college_name,
    }
    if player:
        row.update(
            {
                "birth_date": player.birth_date,
                "age_at_draft": age_at_draft(player.birth_date, pick.draft_year),
                "height_in": player.height_in,
                "weight_lb": player.weight_lb,
                "position": player.position,
                "has_college_stats": player.cbb_id is not None,
            }
        )
    else:
        row.update(
            {
                "birth_date": None,
                "age_at_draft": None,
                "height_in": None,
                "weight_lb": None,
                "position": None,
                "has_college_stats": False,
            }
        )
    if college:
        row.update({k: v for k, v in college.items() if k != "cbb_id"})
        row["has_college_stats"] = True
    return row
