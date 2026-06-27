"""Current-NBA skill profiles for roster-fit analysis (Phase 5 input).

Scrapes a season's league-wide per-game + advanced tables (two pages, cached),
joins them, and scores every rotation player on six skill dimensions as
league percentiles (0-100):

    spacing, playmaking, rim_protection, rebounding, shot_creation, perimeter_defense

Percentiles are taken over a *qualified* pool (rotation minutes) so role players
count but garbage-time lines don't distort the scale. Output:
``data/processed/nba_skills_{season}.parquet`` (whole league) — the Wizards are
just ``team == 'WAS'``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .fetch import get
from .parse import make_soup, to_float, to_int

SEASON = 2026  # 2025-26
LEAGUE_PER_GAME = f"{config.BBREF}/leagues/NBA_{SEASON}_per_game.html"
LEAGUE_ADVANCED = f"{config.BBREF}/leagues/NBA_{SEASON}_advanced.html"

SKILLS = ["spacing", "playmaking", "rim_protection", "rebounding",
          "shot_creation", "perimeter_defense"]

_PID = re.compile(r"/players/[a-z]/([a-z0-9]+)\.html")
_TEAM_TOKENS = {"TOT", "2TM", "3TM", "4TM"}


def _parse_league(html: str, table_id: str) -> pd.DataFrame:
    soup = make_soup(html)
    table = soup.find("table", id=table_id)
    rows = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"]) if c.get("data-stat")}
        if "name_display" not in cells:
            continue
        link = cells["name_display"].find("a")
        pid = _PID.search(link["href"]).group(1) if link and link.get("href") else None
        if not pid:
            continue
        team_cell = cells.get("team_name_abbr")
        rec = {"player_id": pid,
               "player_name": cells["name_display"].get_text(strip=True),
               "team": team_cell.get_text(strip=True) if team_cell is not None else None}
        for stat, cell in cells.items():
            if stat not in ("name_display", "team_name_abbr"):
                rec[stat] = cell.get_text(strip=True)
        rows.append(rec)
    return pd.DataFrame(rows)


def _percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to 0-100 percentiles of a reference distribution."""
    ref = np.sort(reference)
    idx = np.searchsorted(ref, values, side="right")
    return 100.0 * idx / len(ref)


def build_skills(min_mpg: float = 12.0, min_games: int = 15) -> pd.DataFrame:
    pg = _parse_league(get(LEAGUE_PER_GAME), "per_game_stats")
    adv = _parse_league(get(LEAGUE_ADVANCED), "advanced")

    # teams a player appeared for (exclude TOT/multi-team aggregate tokens)
    teams = (pg[~pg["team"].isin(_TEAM_TOKENS)]
             .groupby("player_id")["team"].apply(lambda s: sorted(set(s))))

    # one row per player: the line with the most games (season total / TOT)
    pg["games_i"] = pg["games"].map(to_int)
    pg = pg.sort_values("games_i").drop_duplicates("player_id", keep="last")
    adv["mp_i"] = adv["mp"].map(to_int)
    adv = adv.sort_values("mp_i").drop_duplicates("player_id", keep="last")

    df = pg.merge(adv[["player_id", "trb_pct", "ast_pct", "stl_pct", "blk_pct",
                       "usg_pct", "dbpm"]], on="player_id", how="left")
    for col in ["mp_per_g", "fg3_per_g", "fg3a_per_g", "fg3_pct", "trb_pct", "ast_pct",
                "stl_pct", "blk_pct", "usg_pct", "dbpm", "pts_per_g"]:
        df[col] = df[col].map(to_float)
    df["games_i"] = df["games"].map(to_int)
    df["teams"] = df["player_id"].map(teams).apply(lambda x: x if isinstance(x, list) else [])

    qual = df[(df["mp_per_g"] >= min_mpg) & (df["games_i"] >= min_games)].copy()

    # raw skill signals
    def raw(frame):
        space = frame["fg3a_per_g"].fillna(0) * frame["fg3_pct"].fillna(0)
        pdef = frame["stl_pct"].fillna(0) + frame["dbpm"].fillna(frame["dbpm"].median())
        return {
            "spacing": space.to_numpy(),
            "playmaking": frame["ast_pct"].fillna(0).to_numpy(),
            "rim_protection": frame["blk_pct"].fillna(0).to_numpy(),
            "rebounding": frame["trb_pct"].fillna(0).to_numpy(),
            "shot_creation": frame["usg_pct"].fillna(0).to_numpy(),
            "perimeter_defense": pdef.to_numpy(),
        }

    ref = raw(qual)
    cur = raw(df)
    for skill in SKILLS:
        df[skill] = _percentile(cur[skill], ref[skill]).round(1)

    df["age"] = df["age"].map(to_int)
    df["primary_skill"] = df[SKILLS].idxmax(axis=1)
    df["is_qualified"] = (df["mp_per_g"] >= min_mpg) & (df["games_i"] >= min_games)
    keep = (["player_id", "player_name", "team", "teams", "age", "mp_per_g", "games_i",
             "pts_per_g", "usg_pct"] + SKILLS + ["primary_skill", "is_qualified"])
    return df[keep].reset_index(drop=True)


def main() -> int:
    skills = build_skills()
    path = config.PROCESSED / f"nba_skills_{SEASON}.parquet"
    skills.to_parquet(path, index=False)
    wiz = skills[skills["teams"].apply(lambda t: "WAS" in t) & skills["is_qualified"]]
    print(f"Built skills for {len(skills)} players ({skills['is_qualified'].sum()} qualified).")
    print(f"Wizards qualified rotation ({len(wiz)}):")
    for _, r in wiz.sort_values("mp_per_g", ascending=False).iterrows():
        top = ", ".join(f"{s} {int(r[s])}" for s in SKILLS if r[s] >= 70) or "balanced"
        print(f"  {r['player_name']:22s} {r['mp_per_g']:4.1f} mpg  [{top}]")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
