"""Basketball Reference + Sports Reference (college) scrapers.

Three page types feed the pipeline:

* Draft index   ``/draft/NBA_{year}.html``  -> one row per pick, with the
  player's BBRef id, college, draft slot, and *career* outcome summary
  (WS / BPM / VORP). Cheap source of realized outcomes for everyone.
* NBA player    ``/players/{x}/{id}.html``  -> bio (birthdate, height,
  weight), the link to the player's college page, and a per-season
  ``advanced`` table (WS/BPM/VORP/PER/USG by year) for trajectory features.
* College       ``/cbb/players/{id}.html``  -> per-game + advanced college
  stats; the *final* college season is the pre-draft statistical profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import config
from ..fetch import get
from ..parse import make_soup, table_records, to_float, to_int

_PLAYER_HREF = re.compile(r"/players/[a-z]/([a-z0-9]+)\.html")
_CBB_HREF = re.compile(r"sports-reference\.com/cbb/players/([a-z0-9-]+)\.html")
_BIRTH = re.compile(r'data-birth="([0-9]{4}-[0-9]{2}-[0-9]{2})"')
_HEIGHT_WEIGHT = re.compile(r"(\d)-(\d{1,2})[^0-9]{0,40}?(\d{2,3})\s*lb", re.S)


def _cell_text(cells: dict, key: str) -> str | None:
    cell = cells.get(key)
    return (cell.get_text(strip=True) or None) if cell is not None else None


# --------------------------------------------------------------------------
# Draft index
# --------------------------------------------------------------------------
@dataclass
class DraftPick:
    draft_year: int
    pick_overall: int | None
    team_id: str | None
    player_id: str | None
    player_name: str | None
    college_name: str | None
    # career outcome summary straight off the draft page
    career_seasons: int | None
    career_g: int | None
    career_ws: float | None
    career_ws_per_48: float | None
    career_bpm: float | None
    career_vorp: float | None


def draft_url(year: int) -> str:
    return f"{config.BBREF}/draft/NBA_{year}.html"


def parse_draft(year: int) -> list[DraftPick]:
    """Return every pick of a draft class with career outcome columns."""
    soup = make_soup(get(draft_url(year)))
    table = soup.find("table", id="stats")
    picks: list[DraftPick] = []
    if table is None:
        return picks
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"]) if c.get("data-stat")}
        if "player" not in cells:
            continue
        player_cell = cells["player"]
        link = player_cell.find("a")
        player_id = None
        if link and link.get("href"):
            m = _PLAYER_HREF.search(link["href"])
            player_id = m.group(1) if m else None

        picks.append(
            DraftPick(
                draft_year=year,
                pick_overall=to_int(_cell_text(cells, "pick_overall")),
                team_id=_cell_text(cells, "team_id"),
                player_id=player_id,
                player_name=_cell_text(cells, "player"),
                college_name=_cell_text(cells, "college_name"),
                career_seasons=to_int(_cell_text(cells, "seasons")),
                career_g=to_int(_cell_text(cells, "g")),
                career_ws=to_float(_cell_text(cells, "ws")),
                career_ws_per_48=to_float(_cell_text(cells, "ws_per_48")),
                career_bpm=to_float(_cell_text(cells, "bpm")),
                career_vorp=to_float(_cell_text(cells, "vorp")),
            )
        )
    return picks


# --------------------------------------------------------------------------
# NBA player page
# --------------------------------------------------------------------------
@dataclass
class PlayerPage:
    player_id: str
    birth_date: str | None = None
    height_in: int | None = None
    weight_lb: int | None = None
    position: str | None = None
    cbb_id: str | None = None
    # per-season advanced rows (NBA), chronological
    seasons: list[dict] = field(default_factory=list)


def player_url(player_id: str) -> str:
    return f"{config.BBREF}/players/{player_id[0]}/{player_id}.html"


def parse_player(player_id: str) -> PlayerPage:
    html = get(player_url(player_id))
    soup = make_soup(html)
    page = PlayerPage(player_id=player_id)

    meta = soup.find("div", id="meta")
    meta_text = meta.get_text(" ", strip=True) if meta else html

    if (m := _BIRTH.search(html)):
        page.birth_date = m.group(1)
    if (m := _HEIGHT_WEIGHT.search(meta_text)):
        feet, inches, weight = int(m.group(1)), int(m.group(2)), int(m.group(3))
        page.height_in = feet * 12 + inches
        page.weight_lb = weight
    if (m := _CBB_HREF.search(html)):
        page.cbb_id = m.group(1)

    adv = table_records(soup, "advanced")
    seasons: list[dict] = []
    for row in adv:
        year = row.get("year_id")
        # skip career/aggregate footer rows (no concrete season)
        if not year or not re.match(r"\d{4}-\d{2}", year):
            continue
        seasons.append(
            {
                "season": year,
                "age": to_int(row.get("age")),
                "g": to_int(row.get("games")),
                "mp": to_int(row.get("mp")),
                "per": to_float(row.get("per")),
                "ts_pct": to_float(row.get("ts_pct")),
                "usg_pct": to_float(row.get("usg_pct")),
                "ws": to_float(row.get("ws")),
                "ws_per_48": to_float(row.get("ws_per_48")),
                "bpm": to_float(row.get("bpm")),
                "vorp": to_float(row.get("vorp")),
            }
        )
    page.seasons = seasons
    if seasons and not page.position:
        # position from most recent advanced row if present
        pos_rows = [r for r in adv if r.get("pos")]
        page.position = pos_rows[-1]["pos"] if pos_rows else None
    return page


# --------------------------------------------------------------------------
# College page (final season = pre-draft statistical profile)
# --------------------------------------------------------------------------
def college_url(cbb_id: str) -> str:
    return f"{config.SREF_CBB}/players/{cbb_id}.html"


def parse_college_final_season(cbb_id: str) -> dict | None:
    """Return the player's *last* college season (per-game + advanced)."""
    soup = make_soup(get(college_url(cbb_id)))
    pg = table_records(soup, "players_per_game")
    pg = [r for r in pg if r.get("year_id") and re.match(r"\d{4}-\d{2}", r["year_id"])]
    if not pg:
        return None
    last = pg[-1]
    adv = table_records(soup, "players_advanced")
    adv = [r for r in adv if r.get("year_id") and re.match(r"\d{4}-\d{2}", r["year_id"])]
    adv_last = adv[-1] if adv else {}

    return {
        "cbb_id": cbb_id,
        "coll_season": last.get("year_id"),
        "coll_class": last.get("class"),
        "coll_pos": last.get("pos"),
        "coll_conf": last.get("conf_abbr"),
        "coll_g": to_int(last.get("games")),
        "coll_mp_per_g": to_float(last.get("mp_per_g")),
        "coll_pts_per_g": to_float(last.get("pts_per_g")),
        "coll_trb_per_g": to_float(last.get("trb_per_g")),
        "coll_ast_per_g": to_float(last.get("ast_per_g")),
        "coll_stl_per_g": to_float(last.get("stl_per_g")),
        "coll_blk_per_g": to_float(last.get("blk_per_g")),
        "coll_tov_per_g": to_float(last.get("tov_per_g")),
        "coll_fg_pct": to_float(last.get("fg_pct")),
        "coll_fg3_pct": to_float(last.get("fg3_pct")),
        "coll_ft_pct": to_float(last.get("ft_pct")),
        "coll_efg_pct": to_float(last.get("efg_pct")),
        "coll_per": to_float(adv_last.get("per")),
        "coll_ts_pct": to_float(adv_last.get("ts_pct")),
        "coll_usg_pct": to_float(adv_last.get("usg_pct")),
        "coll_bpm": to_float(adv_last.get("bpm")),
    }
