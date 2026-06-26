"""Build AJ Dybantsa's pre-draft feature row (the project's subject).

His college production and body measurements are *scraped* from his Sports
Reference college page (authoritative). Two fields are documented external
inputs, not scraped, and are flagged as such:

* ``draft_pick = 1`` — consensus 2026 mock-draft projection (he had not been
  drafted at build time). Treated as his expected draft slot.
* ``birth_date``     — from public reporting (ESPN/247Sports). Used only for
  the age-at-draft calculation; a few months' error shifts age trivially.

Output: ``data/processed/dybantsa.parquet`` / ``.csv`` — identical schema to
``prospect_features`` so the comparables/projection models treat him like any
other prospect.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .features import build_features
from .sources import bbref
from .sources.bbref import DraftPick, PlayerPage

CBB_ID = "aj-dybantsa-1"

# --- Documented external inputs (NOT scraped) -----------------------------
DRAFT_YEAR = 2026
PROJECTED_DRAFT_PICK = 1          # consensus #1 projection
REPORTED_BIRTH_DATE = "2006-12-04"  # public reporting; verify
# Measurements fall back to these if the college page header can't be parsed.
FALLBACK_HEIGHT_IN = 81           # 6'9"
FALLBACK_WEIGHT_LB = 210


def build_row() -> dict:
    college = bbref.parse_college_final_season(CBB_ID)
    if college is None:
        raise RuntimeError(f"Could not parse Dybantsa college page ({CBB_ID})")

    # Re-read the header for measurements (height/weight live on the bio line).
    import re

    from .fetch import get

    html = get(bbref.college_url(CBB_ID))
    m = re.search(r"(\d)-(\d{1,2})[^0-9]{0,40}?(\d{2,3})\s*lb", html, re.S)
    if m:
        height_in = int(m.group(1)) * 12 + int(m.group(2))
        weight_lb = int(m.group(3))
    else:
        height_in, weight_lb = FALLBACK_HEIGHT_IN, FALLBACK_WEIGHT_LB

    pick = DraftPick(
        draft_year=DRAFT_YEAR,
        pick_overall=PROJECTED_DRAFT_PICK,
        team_id="WAS",  # the framing: Wizards' cornerstone
        player_id="dybanaj01",  # synthetic id (no BBRef NBA page yet)
        player_name="AJ Dybantsa",
        college_name="BYU",
        career_seasons=None, career_g=None, career_ws=None,
        career_ws_per_48=None, career_bpm=None, career_vorp=None,
    )
    player = PlayerPage(
        player_id="dybanaj01",
        birth_date=REPORTED_BIRTH_DATE,
        height_in=height_in,
        weight_lb=weight_lb,
        position=college.get("coll_pos"),
        cbb_id=CBB_ID,
        seasons=[],
    )
    row = build_features(pick, player, college)
    row["is_projection_subject"] = True
    row["draft_pick_is_projected"] = True
    return row


def main() -> int:
    row = build_row()
    df = pd.DataFrame([row])
    df.to_parquet(config.PROCESSED / "dybantsa.parquet", index=False)
    df.to_csv(config.PROCESSED / "dybantsa.csv", index=False)
    print("Built Dybantsa profile row:")
    for k, v in row.items():
        print(f"  {k:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
