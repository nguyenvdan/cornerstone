"""HTML parsing helpers shared across sources.

Basketball/Sports Reference defeat naive scrapers by wrapping many stat tables
in HTML comments. ``make_soup`` re-inflates those comments so every table is
queryable, and ``table_records`` reads a table into row dicts keyed by the
site's ``data-stat`` attributes (a stable, semantic schema).
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment


def make_soup(html: str) -> BeautifulSoup:
    """Parse HTML, lifting comment-wrapped tables into the live DOM."""
    soup = BeautifulSoup(html, "lxml")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        text = str(comment)
        if "<table" in text:
            comment.replace_with(BeautifulSoup(text, "lxml"))
    return soup


def _cell_value(cell) -> str | None:
    text = cell.get_text(strip=True)
    return text if text else None


def table_records(soup: BeautifulSoup, table_id: str) -> list[dict[str, str | None]]:
    """Return tbody rows of a table as dicts keyed by ``data-stat``.

    Header/spacer rows (class ``thead``) are skipped. Returns ``[]`` when the
    table is absent.
    """
    table = soup.find("table", id=table_id)
    if table is None:
        return []
    body = table.find("tbody") or table
    rows: list[dict[str, str | None]] = []
    for tr in body.find_all("tr"):
        classes = tr.get("class") or []
        if "thead" in classes:
            continue
        record: dict[str, str | None] = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                record[stat] = _cell_value(cell)
        if record:
            rows.append(record)
    return rows


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value.replace("%", ""))
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    f = to_float(value)
    return int(f) if f is not None else None
