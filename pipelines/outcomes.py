"""Derive realized NBA outcomes (continuous scores + tier label).

Outcomes combine the career summary from the draft page with the first
``OUTCOME_WINDOW`` seasons of the per-season ``advanced`` table to describe how
a prospect *actually* developed.
"""

from __future__ import annotations

from . import config
from .sources.bbref import DraftPick, PlayerPage


def _tier_from_vorp(career_vorp: float | None, career_g: int | None) -> str:
    if career_vorp is None:
        return "bust"
    if (career_g or 0) < config.MIN_GAMES_FOR_NON_BUST:
        return "bust"
    for tier, lower in config.VORP_TIER_BINS:
        if career_vorp >= lower:
            return tier
    return "bust"


def _safe(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def build_outcome(pick: DraftPick, player: PlayerPage | None) -> dict:
    """Return the realized-outcome record for one prospect."""
    window = (player.seasons[: config.OUTCOME_WINDOW] if player else [])
    vorp_w = _safe([s["vorp"] for s in window])
    bpm_w = _safe([s["bpm"] for s in window])
    ws_w = _safe([s["ws"] for s in window])
    per_w = _safe([s["per"] for s in window])

    all_bpm = _safe([s["bpm"] for s in (player.seasons if player else [])])

    return {
        "player_id": pick.player_id,
        # career summary (from draft page)
        "career_seasons": pick.career_seasons,
        "career_g": pick.career_g,
        "career_ws": pick.career_ws,
        "career_ws_per_48": pick.career_ws_per_48,
        "career_bpm": pick.career_bpm,
        "career_vorp": pick.career_vorp,
        "peak_bpm": max(all_bpm) if all_bpm else None,
        # first-window trajectory (from player page)
        "window_seasons": len(window),
        "early_vorp_sum": round(sum(vorp_w), 2) if vorp_w else None,
        "early_bpm_mean": round(sum(bpm_w) / len(bpm_w), 2) if bpm_w else None,
        "early_ws_sum": round(sum(ws_w), 2) if ws_w else None,
        "early_per_mean": round(sum(per_w) / len(per_w), 2) if per_w else None,
        # season-by-season vorp for trajectory plots downstream
        "trajectory_vorp": [s["vorp"] for s in window],
        "trajectory_bpm": [s["bpm"] for s in window],
        # label
        "outcome_tier": _tier_from_vorp(pick.career_vorp, pick.career_g),
    }
