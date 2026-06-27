"""NBA Draft Combine athleticism (Phase 5+ / projection input).

Adds *measured* pre-draft athleticism — the signal college box scores miss — by
joining public combine anthro/athletic data (wingspan, standing reach, max
vertical) to the prospect universe. Coverage is partial (the public set spans
2009-2017), so downstream this is used as a *reweight where measured, neutral
where not* — never imputed into the core distance.

Source: github.com/achou11/NBA_draft_combine_measurements (Draft Express).
Output: data/processed/combine_athleticism.parquet
    player_id, athleticism_pct (0-100 among measured), wingspan, vertical_max, length

AJ Dybantsa's own 2026 combine line is appended so his percentile is computed
against the same distribution.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from . import config
from .fetch import get

COMBINE_URL = ("https://raw.githubusercontent.com/achou11/"
               "NBA_draft_combine_measurements/master/nba_draft_combine_all_years.csv")

# AJ Dybantsa, 2026 NBA Draft Combine (public reporting): height 6'8.5" no shoes,
# 7'0.5" wingspan, 8'10" standing reach, 42.0" max vertical (combine-best).
DYBANTSA_COMBINE = {
    "player_id": "dybanaj01", "player_name": "AJ Dybantsa",
    "height_no_shoes": 80.5, "wingspan": 84.5, "standing_reach": 106.0,
    "vertical_max": 42.0,
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    toks = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(toks)


def _percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def _pos_group(pos: object) -> str | None:
    """Coarse position group for stable height baselines."""
    if not isinstance(pos, str):
        return None
    p = pos.upper()
    if p in ("PG", "SG", "G"):
        return "G"
    if p in ("SF", "F", "GF"):
        return "W"
    if p in ("PF", "C", "FC"):
        return "B"
    return None


def _banded_pct(heights: np.ndarray, values: np.ndarray, targets: np.ndarray,
                band: float = 2.0, min_n: int = 25) -> np.ndarray:
    """For each player, percentile of their metric among athletes within `band`
    inches of their height (widening the band until min_n peers are found)."""
    out = np.full(len(targets), np.nan)
    for i, (th, tv) in enumerate(zip(targets, values, strict=False)):
        if np.isnan(tv):
            continue
        b = band
        while True:
            mask = (np.abs(heights - th) <= b) & ~np.isnan(values)
            peers = values[mask]
            if len(peers) >= min_n or b > 8:
                break
            b += 1
        out[i] = 100.0 * np.mean(peers < tv) if len(peers) else np.nan
    return out


def _position_pct(groups: np.ndarray, heights: np.ndarray, min_n: int = 15) -> np.ndarray:
    """Percentile of each player's height within their position group."""
    out = np.full(len(heights), np.nan)
    valid = ~np.isnan(heights)
    for i, g in enumerate(groups):
        if np.isnan(heights[i]):
            continue
        mask = np.array([gg == g for gg in groups]) & valid if g else valid
        peers = heights[mask]
        if len(peers) < min_n:
            peers = heights[valid]
        out[i] = 100.0 * np.mean(peers < heights[i])
    return out


def build_combine(prospects: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(_local_csv())
    measured = raw.rename(columns={
        "Player": "player_name", "Wingspan": "wingspan",
        "Standing reach": "standing_reach", "Vertical (Max)": "vertical_max",
        "Height (No Shoes)": "height_no_shoes",
    })[["player_name", "wingspan", "standing_reach", "vertical_max", "height_no_shoes"]]
    measured = measured.dropna(subset=["wingspan", "standing_reach", "vertical_max"])

    # Append AJ so his percentile is on the same scale.
    aj = pd.DataFrame([{k: DYBANTSA_COMBINE[k] for k in
                        ("player_name", "wingspan", "standing_reach", "vertical_max",
                         "height_no_shoes")}])
    pool = pd.concat([measured, aj], ignore_index=True).reset_index(drop=True)
    pool["length"] = pool["wingspan"] - pool["height_no_shoes"]

    # Position (for the height comparison) joined from the prospect universe.
    tmp = (prospects.drop_duplicates("player_id")
           .assign(_n=lambda d: d["player_name"].map(normalize_name))
           .drop_duplicates("_n"))
    pmap = dict(zip(tmp["_n"], tmp["player_id"], strict=False))
    posmap = dict(zip(tmp["_n"], tmp["position"], strict=False))
    pool["position"] = pool["player_name"].map(
        lambda n: "F" if n == "AJ Dybantsa" else posmap.get(normalize_name(n)))
    pool["pos_group"] = pool["position"].map(_pos_group)

    # Athleticism is size-adjusted: explosiveness/length are scored vs athletes of
    # the SAME height (a 42" vertical on a 6'8" frame isn't diluted by small
    # guards), while height itself is scored vs the player's position.
    # Explosiveness (vertical) + length (wingspan) vs same-height peers; height
    # vs position. Standing reach is dropped (redundant with wingspan + height).
    h = pool["height_no_shoes"].to_numpy()
    vp = _banded_pct(h, pool["vertical_max"].to_numpy(), h)
    wp = _banded_pct(h, pool["wingspan"].to_numpy(), h)
    rp = _banded_pct(h, pool["standing_reach"].to_numpy(), h)
    hp = _position_pct(pool["pos_group"].to_numpy(), h)
    pool["vertical_pct"] = vp.round(1)
    pool["wingspan_pct"] = wp.round(1)
    pool["reach_pct"] = rp.round(1)
    pool["height_pct"] = hp.round(1)
    pool["athleticism_pct"] = np.nanmean(np.vstack([vp, wp, hp]), axis=0).round(1)

    out_rows = []
    for _, r in pool.iterrows():
        pid = "dybanaj01" if r["player_name"] == "AJ Dybantsa" \
            else pmap.get(normalize_name(r["player_name"]))
        if pid is None:
            continue
        out_rows.append({"player_id": pid, "athleticism_pct": float(r["athleticism_pct"]),
                         "vertical_pct": float(r["vertical_pct"]),
                         "wingspan_pct": float(r["wingspan_pct"]),
                         "reach_pct": float(r["reach_pct"]),
                         "height_pct": float(r["height_pct"]),
                         "wingspan": float(r["wingspan"]), "vertical_max": float(r["vertical_max"]),
                         "length": round(float(r["length"]), 1)})
    return pd.DataFrame(out_rows).drop_duplicates("player_id")


def _local_csv():
    """Cache the combine CSV under data/raw and return its path."""
    path = config.RAW / "nba_draft_combine_all_years.csv"
    if not path.exists():
        path.write_text(get(COMBINE_URL), encoding="utf-8")
    return path


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    df = build_combine(prospects)
    path = config.PROCESSED / "combine_athleticism.parquet"
    df.to_parquet(path, index=False)
    aj = df[df.player_id == "dybanaj01"].iloc[0]
    print(f"Combine athleticism: {len(df)} players matched to the universe "
          f"(+ AJ Dybantsa).")
    print(f"AJ Dybantsa: athleticism {aj['athleticism_pct']:.0f}th pct "
          f"(wingspan {aj['wingspan']}\", max vert {aj['vertical_max']}\", "
          f"length {aj['length']}\").")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
