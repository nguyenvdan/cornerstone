"""Peak per-36-minute production for the comparable universe.

A scout-friendly way to express upside: at a player's *peak* season, what did
they produce per 36 minutes? We re-parse the (cached) player pages for the
per-36 table, pick each player's peak season (highest BPM among full seasons),
and store that per-36 line. The projection then reports a prospect's *projected*
peak per-36 as the similarity-weighted average of his comparables' peaks.

Output: data/processed/peak_per36.parquet
"""

from __future__ import annotations

import re

import pandas as pd
from tqdm import tqdm

from . import config
from .fetch import get, is_cached
from .parse import make_soup, table_records, to_float, to_int
from .sources import bbref

_SEASON = re.compile(r"\d{4}-\d{2}")
MIN_GAMES = 40            # a "full" season, so peak isn't a small-sample fluke
PER36 = {
    "pts_per_minute_36": "pts36", "trb_per_minute_36": "trb36",
    "ast_per_minute_36": "ast36", "stl_per_minute_36": "stl36",
    "blk_per_minute_36": "blk36", "fg3_per_minute_36": "fg3_36",
    "fg_pct": "fg_pct", "fg3_pct": "fg3_pct",
}


def _peak_for(player_id: str) -> dict | None:
    url = bbref.player_url(player_id)
    if not is_cached(url):
        return None
    soup = make_soup(get(url))
    per36 = {r.get("year_id"): r for r in table_records(soup, "per_minute_stats")
             if r.get("year_id") and _SEASON.match(r["year_id"])}
    adv = {r.get("year_id"): r for r in table_records(soup, "advanced")
           if r.get("year_id") and _SEASON.match(r["year_id"])}
    if not per36:
        return None
    # peak = season with highest BPM among full seasons that have a per-36 row
    best_year, best_bpm = None, -1e9
    for year, a in adv.items():
        if year not in per36 or (to_int(per36[year].get("games")) or 0) < MIN_GAMES:
            continue
        bpm = to_float(a.get("bpm"))
        if bpm is not None and bpm > best_bpm:
            best_bpm, best_year = bpm, year
    if best_year is None:
        return None
    row = per36[best_year]
    out = {"player_id": player_id, "peak_season": best_year, "peak_bpm": round(best_bpm, 1),
           "peak_ts_pct": to_float(adv[best_year].get("ts_pct"))}
    for src, dst in PER36.items():
        out[dst] = to_float(row.get(src))
    return out


def build_peak_per36(prospects: pd.DataFrame) -> pd.DataFrame:
    pool = prospects.drop_duplicates("player_id")
    rows = []
    for pid in tqdm(pool["player_id"].dropna(), desc="peak per-36", unit="player"):
        try:
            r = _peak_for(pid)
        except (FileNotFoundError, RuntimeError):
            r = None
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    df = build_peak_per36(prospects)
    path = config.PROCESSED / "peak_per36.parquet"
    df.to_parquet(path, index=False)
    print(f"\nPeak per-36 for {len(df)}/{prospects['player_id'].nunique()} players.")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
