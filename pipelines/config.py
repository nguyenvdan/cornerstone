"""Central configuration for the Cornerstone data pipeline."""

from __future__ import annotations

from pathlib import Path

# ---- Paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

for _p in (RAW, INTERIM, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ---- Universe -----------------------------------------------------------
# Draft classes to collect. 2003-2022 gives outcomes time to mature
# (>= ~3 seasons by the 2025-26 season) while spanning multiple eras.
DRAFT_START = 2003
DRAFT_END = 2022
DRAFT_YEARS = list(range(DRAFT_START, DRAFT_END + 1))

# Number of early NBA seasons used to summarize a "development trajectory".
OUTCOME_WINDOW = 5

# ---- Scraping etiquette -------------------------------------------------
# Basketball Reference asks for <= 20 requests/minute. We stay well under.
USER_AGENT = "cornerstone-research/0.1 (personal player-development project)"
REQUEST_DELAY_SEC = 3.2  # ~18 req/min
MAX_RETRIES = 4
TIMEOUT_SEC = 30

BBREF = "https://www.basketball-reference.com"
SREF_CBB = "https://www.sports-reference.com/cbb"

# ---- Outcome tiers ------------------------------------------------------
# Tier boundaries on *career* VORP (Value Over Replacement Player), the
# single best public summary of total career value available at draft-page
# scale. Documented + revisable in pipelines/data_dictionary.md.
# A player who never accrues meaningful minutes is labeled "bust".
TIER_ORDER = ["bust", "rotation", "starter", "all_star", "superstar"]
VORP_TIER_BINS = [
    ("superstar", 25.0),   # career VORP >= 25  (franchise cornerstone)
    ("all_star", 10.0),    # 10  <= VORP < 25
    ("starter", 3.0),      # 3   <= VORP < 10
    ("rotation", 0.0),     # 0   <= VORP < 3
    ("bust", float("-inf")),  # VORP < 0
]
# A player with fewer than this many career games is forced to "bust"
# regardless of rate stats (insufficient NBA footprint).
MIN_GAMES_FOR_NON_BUST = 50
