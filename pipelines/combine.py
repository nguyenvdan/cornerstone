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
    pool = pd.concat([measured, aj], ignore_index=True)
    pool["length"] = pool["wingspan"] - pool["height_no_shoes"]

    # Athleticism = mean percentile of explosiveness (max vertical) + length
    # (wingspan, standing reach). Captures the "elite athlete" wing signal.
    ath = (_percentile(pool["vertical_max"]) + _percentile(pool["wingspan"])
           + _percentile(pool["standing_reach"])) / 3.0
    pool = pool.assign(athleticism_pct=ath.round(1))

    # Join historical rows to player_id by normalized name.
    tmp = (prospects.drop_duplicates("player_id")
           .assign(_n=lambda d: d["player_name"].map(normalize_name))
           .drop_duplicates("_n"))
    pmap = dict(zip(tmp["_n"], tmp["player_id"], strict=False))
    out_rows = []
    for _, r in pool.iterrows():
        if r["player_name"] == "AJ Dybantsa":
            pid = "dybanaj01"
        else:
            pid = pmap.get(normalize_name(r["player_name"]))
            if pid is None:
                continue
        out_rows.append({"player_id": pid, "athleticism_pct": float(r["athleticism_pct"]),
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
