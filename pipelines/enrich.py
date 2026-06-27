"""Augment the prospect dataset with the strongest missing pre-draft signals.

Adds four candidate features (their actual predictive value is then decided by
the back-test, not assumed):

    rsci_rank            HS recruiting consensus rank (lower = better)
    conference_strength  strength-of-schedule proxy from college conference
    age_minus_class      age relative to the typical age for the player's class
    athleticism_pct      height-adjusted combine athleticism (partial coverage)

Runs after build_dataset; re-joinable and idempotent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

POWER = {"ACC", "SEC", "Big 12", "Big Ten", "Big East", "Pac-12", "Pac-10"}
STRONG_MID = {"AAC", "MWC", "A-10", "WCC", "CUSA", "Amer"}
# Typical age-at-draft by college class (drafted after the season).
CLASS_AGE = {"FR": 19.3, "SO": 20.3, "JR": 21.3, "SR": 22.3}


def _conf_strength(conf: object) -> float | None:
    if not isinstance(conf, str):
        return None
    if conf in POWER:
        return 3.0
    if conf in STRONG_MID:
        return 2.0
    return 1.0


def _age_minus_class(row: pd.Series) -> float | None:
    cls, age = row.get("coll_class"), row.get("age_at_draft")
    if not isinstance(cls, str) or cls not in CLASS_AGE or age is None or np.isnan(age):
        return None
    return round(float(age) - CLASS_AGE[cls], 2)


def enrich(df: pd.DataFrame, recruiting: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conference_strength"] = df["coll_conf"].map(_conf_strength)
    df["age_minus_class"] = df.apply(_age_minus_class, axis=1)
    df = df.merge(recruiting[["player_id", "rsci_rank"]], on="player_id", how="left")
    df = df.merge(combine[["player_id", "athleticism_pct"]], on="player_id", how="left")
    return df


def main() -> int:
    recruiting = pd.read_parquet(config.PROCESSED / "recruiting.parquet")
    combine = pd.read_parquet(config.PROCESSED / "combine_athleticism.parquet")
    for name in ("prospects", "dybantsa"):
        path = config.PROCESSED / f"{name}.parquet"
        df = enrich(pd.read_parquet(path), recruiting, combine)
        df.to_parquet(path, index=False)
        cov = {c: f"{df[c].notna().mean():.0%}" for c in
               ("rsci_rank", "conference_strength", "age_minus_class", "athleticism_pct")}
        print(f"{name}: enriched ({len(df)} rows). coverage {cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
