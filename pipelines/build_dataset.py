"""Build the joined, versioned prospect dataset (Phase 1 entry point).

    uv run python -m pipelines.build_dataset            # full 2003-2022
    uv run python -m pipelines.build_dataset --years 2017 2018

Outputs (committed, reproducible) to ``data/processed/``:
    prospect_features.parquet / .csv   pre-draft features (no leakage)
    realized_outcomes.parquet / .csv   realized NBA outcomes + tier label
    prospects.parquet / .csv           the two joined on player_id
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from tqdm import tqdm

from . import config
from .features import build_features
from .outcomes import build_outcome
from .sources import bbref


def build(years: list[int], limit_per_year: int | None = None) -> dict[str, pd.DataFrame]:
    feature_rows: list[dict] = []
    outcome_rows: list[dict] = []
    skipped = 0

    for year in years:
        picks = bbref.parse_draft(year)
        if limit_per_year:
            picks = picks[:limit_per_year]
        for pick in tqdm(picks, desc=f"draft {year}", unit="pick"):
            if not pick.player_id:
                skipped += 1
                continue
            try:
                player = bbref.parse_player(pick.player_id)
            except (FileNotFoundError, RuntimeError):
                player = None
            college = None
            if player and player.cbb_id:
                try:
                    college = bbref.parse_college_final_season(player.cbb_id)
                except (FileNotFoundError, RuntimeError):
                    college = None
            feature_rows.append(build_features(pick, player, college))
            outcome_rows.append(build_outcome(pick, player))

    features = pd.DataFrame(feature_rows)
    outcomes = pd.DataFrame(outcome_rows)
    joined = features.merge(outcomes, on="player_id", how="inner", validate="one_to_one")
    print(f"\nBuilt {len(joined)} prospects across {len(years)} draft classes "
          f"({skipped} rows skipped for missing player id).")
    return {"prospect_features": features, "realized_outcomes": outcomes, "prospects": joined}


def _write(frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        df.to_parquet(config.PROCESSED / f"{name}.parquet", index=False)
        # list columns (trajectories) don't fit CSV cleanly; drop them there.
        csv_df = df.drop(columns=[c for c in df.columns if df[c].apply(type).eq(list).any()])
        csv_df.to_csv(config.PROCESSED / f"{name}.csv", index=False)
    print(f"Wrote {len(frames)} tables to {config.PROCESSED}")


def _summary(joined: pd.DataFrame) -> None:
    if joined.empty:
        return
    print("\nOutcome tier distribution:")
    counts = joined["outcome_tier"].value_counts()
    for tier in config.TIER_ORDER:
        print(f"  {tier:10s} {int(counts.get(tier, 0)):4d}")
    have_college = int(joined["has_college_stats"].sum())
    print(f"\nProspects with college stats: {have_college}/{len(joined)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Cornerstone prospect dataset.")
    ap.add_argument("--years", type=int, nargs="*", default=config.DRAFT_YEARS,
                    help="Draft years to build (default: full 2003-2022 universe).")
    ap.add_argument("--limit-per-year", type=int, default=None,
                    help="Cap picks per draft (smoke testing).")
    args = ap.parse_args(argv)

    frames = build(args.years, args.limit_per_year)
    _write(frames)
    _summary(frames["prospects"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
