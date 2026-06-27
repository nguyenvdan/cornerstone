"""High-school recruiting rank (RSCI consensus) — a strong pre-draft signal.

The Recruiting Services Consensus Index aggregates the major recruiting
services into one ranking of each HS class's top ~100. Being the #1 recruit
(as AJ Dybantsa was in 2025) is real, predictive information the box score
misses. Basketball Reference hosts a page per class with player links, so we
join to the prospect universe by BBRef player id.

Output: data/processed/recruiting.parquet  (player_id, rsci_rank)
Unranked / international / JUCO prospects simply don't appear (handled as a
missing-but-flagged feature downstream).
"""

from __future__ import annotations

import re

import pandas as pd

from . import config
from .fetch import get
from .parse import make_soup, to_int

# Recruits ranked in years Y were drafted ~Y+1..Y+4, so this spans the
# 2003-2022 draft universe.
RSCI_YEARS = list(range(2000, 2022))
_PID = re.compile(r"/players/[a-z]/([a-z0-9]+)\.html")
# AJ Dybantsa: consensus #1 recruit, 2025 class.
DYBANTSA_RSCI = {"player_id": "dybanaj01", "rsci_rank": 1}


def _rsci_url(year: int) -> str:
    return f"{config.BBREF}/awards/recruit_rankings_{year}.html"


def parse_rsci(year: int) -> list[dict]:
    soup = make_soup(get(_rsci_url(year)))
    table = soup.find("table", id="rsci_rankings")
    if table is None:
        return []
    rows = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"]) if c.get("data-stat")}
        player_cell = cells.get("player")
        rank_cell = cells.get("rank")
        if player_cell is None or rank_cell is None:
            continue
        link = player_cell.find("a")
        if not (link and link.get("href")):
            continue
        m = _PID.search(link["href"])
        rank = to_int(rank_cell.get_text(strip=True))
        if m and rank is not None:
            rows.append({"player_id": m.group(1), "rsci_rank": rank, "rsci_year": year})
    return rows


def build_recruiting() -> pd.DataFrame:
    rows: list[dict] = []
    for year in RSCI_YEARS:
        try:
            rows.extend(parse_rsci(year))
        except (FileNotFoundError, RuntimeError):
            continue
    df = pd.DataFrame(rows)
    # A player can appear in one class; keep their best (lowest) rank if dupes.
    df = df.sort_values("rsci_rank").drop_duplicates("player_id", keep="first")
    df = pd.concat([df[["player_id", "rsci_rank"]],
                    pd.DataFrame([DYBANTSA_RSCI])], ignore_index=True)
    return df.drop_duplicates("player_id", keep="last").reset_index(drop=True)


def main() -> int:
    df = build_recruiting()
    path = config.PROCESSED / "recruiting.parquet"
    df.to_parquet(path, index=False)
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    matched = prospects["player_id"].isin(df["player_id"]).sum()
    print(f"RSCI ranks for {len(df)} players; {matched}/{len(prospects)} prospects matched.")
    print(f"Top-5 recruits in the data: "
          f"{df.nsmallest(5, 'rsci_rank')['player_id'].tolist()}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
