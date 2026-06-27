"""Tune the scouting model's hyperparameters on the back-test (not on AJ).

Searches the generalizable scouting levers (draft-capital bandwidth, age
weighting, competition penalty, athleticism boost) by their leakage-free
back-test performance, optimizing a BALANCED score of discrimination
(star-detection AUC + VORP ranking) and calibration (ECE). To avoid tuning on
the data we report, the search runs on early cohorts and the winner is
validated on held-out late cohorts.

    uv run python -m eval.tune_context
"""

from __future__ import annotations

import itertools
import json
import warnings

import pandas as pd

from models.projection import scouting_context
from pipelines import config

from .backtest import run_backtest

warnings.filterwarnings("ignore")

TUNE_YEARS = range(2010, 2016)      # search the objective here
HOLDOUT_YEARS = range(2016, 2020)   # report the winner here

GRID = {
    "draft_bandwidth": [4.0, 8.0, 12.0],
    "age_weight_boost": [1.0, 2.0, 3.0],
    "competition_penalty": [0.6, 1.0],     # 1.0 = competition lever off
    "athleticism_boost": [0.0, 0.6],       # 0.0 = athleticism lever off
}


def balanced_score(m: dict) -> float:
    """0-1 blend of discrimination (AUC + VORP rank) and calibration (ECE)."""
    auc, ece, sp = m["star"]["auc"], m["star"]["ece"], m["vorp"]["spearman"]
    disc = 0.5 * max(0.0, (auc - 0.5) / 0.5) + 0.5 * max(0.0, min(1.0, sp / 0.6))
    cal = 1.0 - min(ece, 0.10) / 0.10
    return round(0.5 * disc + 0.5 * cal, 4)


def _metrics(m: dict) -> dict:
    return {"score": balanced_score(m), "auc": m["star"]["auc"],
            "ece": m["star"]["ece"], "spearman": m["vorp"]["spearman"],
            "within1": m.get("tier", {}).get("within_one")}


def search(df: pd.DataFrame) -> pd.DataFrame:
    keys = list(GRID)
    rows = []
    for i, combo in enumerate(itertools.product(*GRID.values()), 1):
        params = dict(zip(keys, combo, strict=False))
        _, s = run_backtest(df, test_years=TUNE_YEARS, context=scouting_context(**params))
        rows.append({**params, **_metrics(s["model"])})
        print(f"  [{i:2d}/{len(list(itertools.product(*GRID.values())))}] "
              f"{params} -> score {rows[-1]['score']}")
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def main() -> int:
    df = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    print(f"Searching {len(list(itertools.product(*GRID.values())))} configs on "
          f"tune cohorts {TUNE_YEARS.start}-{TUNE_YEARS.stop - 1}...\n")
    table = search(df)
    best = table.iloc[0]
    best_params = {k: best[k] for k in GRID}
    print(f"\nBest on tune split: {best_params}  (score {best['score']})")

    # Validate winner vs profile-only vs baseline on the held-out cohorts.
    _, prof = run_backtest(df, test_years=HOLDOUT_YEARS)
    _, scout = run_backtest(df, test_years=HOLDOUT_YEARS, context=scouting_context(**best_params))
    holdout = {
        "baseline": _metrics(prof["baseline"]),
        "profile_only": _metrics(prof["model"]),
        "scouting_tuned": _metrics(scout["model"]),
        "combined": _metrics(scout["combined"]),
    }
    print(f"\nHeld-out cohorts {HOLDOUT_YEARS.start}-{HOLDOUT_YEARS.stop - 1} "
          f"({scout['n_prospects']} prospects):")
    for name, m in holdout.items():
        print(f"  {name:16s} AUC {m['auc']:.3f}  ECE {m['ece']:.3f}  "
              f"Spearman {m['spearman']:.3f}  within1 {m['within1']:.3f}")

    out = {"tune_years": [TUNE_YEARS.start, TUNE_YEARS.stop - 1],
           "holdout_years": [HOLDOUT_YEARS.start, HOLDOUT_YEARS.stop - 1],
           "best_params": best_params, "tune_table": table.to_dict("records"),
           "holdout": holdout}
    path = config.ROOT / "eval" / "tuning_results.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
