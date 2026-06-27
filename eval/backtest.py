"""Leakage-aware temporal back-test (Phase 4 — the credibility anchor).

For each draft class Y in the test window, we fit the projection model **only on
players drafted before Y** (expanding window) and predict that class, then
compare to what actually happened. No target leakage, no same/future-cohort
leakage. The same protocol scores a draft-position baseline so we can show
whether the profile model adds value.

    uv run python -m eval.backtest

Outputs to ``eval/``:
    backtest_results.json   metrics + reliability curves + per-prospect rows
    calibration.png         reliability diagram (model vs baseline)

Honest-framing note: training uses earlier cohorts' *eventually realized*
careers (not their in-progress state on draft night). This is the standard
cohort back-test and is leakage-free with respect to the prediction target; it
slightly flatters both model and baseline equally, so the comparison is fair.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

from models.projection import ProjectionModel
from pipelines import config

from . import metrics
from .baseline import DraftPositionBaseline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_TEST_YEARS = range(2010, 2020)


def _college(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["has_college_stats"] & df["coll_pts_per_g"].notna()
              & df["career_vorp"].notna() & df["draft_pick"].notna()]


def run_backtest(prospects: pd.DataFrame, test_years=None, context=None
                 ) -> tuple[pd.DataFrame, dict]:
    """Leakage-free expanding-window back-test. ``context`` (a ProjectionContext)
    turns on the scouting levers (draft capital, competition, athleticism, age);
    None = the pure profile-only model."""
    test_years = test_years if test_years is not None else DEFAULT_TEST_YEARS
    pool = _college(prospects)
    rows: list[dict] = []

    for year in test_years:
        train = pool[pool["draft_year"] < year]
        test = pool[pool["draft_year"] == year]
        if len(train) < 50 or test.empty:
            continue
        model = ProjectionModel(train, mature_only=False, context=context)
        base = DraftPositionBaseline().fit(train)

        for _, p in test.iterrows():
            proj = model.project(p, include_curve=False, include_swing=False)
            pick = float(p["draft_pick"])
            base_tiers = base.tier_probabilities(pick)
            rows.append({
                "player_name": p["player_name"],
                "draft_year": int(p["draft_year"]),
                "draft_pick": int(pick),
                "actual_tier": p["outcome_tier"],
                "actual_vorp": float(p["career_vorp"]),
                "actual_star": int(p["outcome_tier"] in ("all_star", "superstar")),
                "model_p_star": proj.p_star_plus,
                "model_exp_vorp": proj.expected_career_vorp,
                "model_tier": proj.modal_tier,
                "base_p_star": base.p_star_plus(pick),
                "base_exp_vorp": base.expected_vorp(pick),
                "base_tier": max(base_tiers, key=base_tiers.get),
            })

    res = pd.DataFrame(rows)
    res["comb_p_star"] = 0.5 * (res["model_p_star"] + res["base_p_star"])
    res["comb_exp_vorp"] = 0.5 * (res["model_exp_vorp"] + res["base_exp_vorp"])
    return res, _score(res)


def _score(res: pd.DataFrame) -> dict:
    actual_star = res["actual_star"].to_numpy()
    actual_vorp = res["actual_vorp"].to_numpy()
    actual_tier = res["actual_tier"].tolist()

    def block(prefix: str, tier_col: str | None) -> dict:
        out = {
            "star": metrics.binary_scores(res[f"{prefix}_p_star"].to_numpy(), actual_star),
            "vorp": metrics.ranking_scores(res[f"{prefix}_exp_vorp"].to_numpy(), actual_vorp),
        }
        if tier_col:
            out["tier"] = metrics.tier_accuracy(res[tier_col].tolist(), actual_tier)
        return out

    return {
        "n_prospects": int(len(res)),
        "n_years": int(res["draft_year"].nunique()),
        "star_base_rate": round(float(actual_star.mean()), 3),
        "model": block("model", "model_tier"),
        "baseline": block("base", "base_tier"),
        "combined": block("comb", None),
    }


def _plot(res: pd.DataFrame, summary: dict, path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, key, color in [("Scouting model", "model", "#1f4e79"),
                             ("Draft-position baseline", "baseline", "#c0504d")]:
        rel = summary[key]["star"]["reliability"]
        ax1.plot(rel["confidence"], rel["observed"], "o-", color=color,
                 label=f"{name} (ECE {summary[key]['star']['ece']:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax1.set_xlabel("Predicted P(all-star or better)")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Calibration — P(all-star+)")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    labels = ["Exact tier", "Within-1 tier", "AUC P(star+)"]
    model_v = [summary["model"]["tier"]["exact"], summary["model"]["tier"]["within_one"],
               summary["model"]["star"]["auc"]]
    base_v = [summary["baseline"]["tier"]["exact"], summary["baseline"]["tier"]["within_one"],
              summary["baseline"]["star"]["auc"]]
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, model_v, 0.4, label="Scouting model", color="#1f4e79")
    ax2.bar(x + 0.2, base_v, 0.4, label="Draft-position baseline", color="#c0504d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1)
    ax2.set_title("Accuracy vs baseline")
    ax2.legend(fontsize=8)
    for i, (m, b) in enumerate(zip(model_v, base_v, strict=False)):
        ax2.text(i - 0.2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
        ax2.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Cornerstone back-test — {summary['n_prospects']} prospects, "
                 f"{summary['n_years']} draft classes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_summary(s: dict) -> None:
    m, b, c = s["model"], s["baseline"], s["combined"]
    print(f"\nBack-test: {s['n_prospects']} prospects across {s['n_years']} draft classes "
          f"(star base rate {s['star_base_rate']:.1%})\n")
    hdr = f"{'metric':22s} {'model':>10s} {'baseline':>10s} {'combined':>10s}"
    print(hdr)
    print("-" * len(hdr))

    def line(name, mv, bv, cv):
        cs = f"{cv:>10}" if cv is not None else f"{'—':>10}"
        print(f"{name:22s} {mv:>10} {bv:>10} {cs}")

    line("tier exact acc", m["tier"]["exact"], b["tier"]["exact"], None)
    line("tier within-1 acc", m["tier"]["within_one"], b["tier"]["within_one"], None)
    line("P(star+) AUC", m["star"]["auc"], b["star"]["auc"], c["star"]["auc"])
    line("P(star+) Brier", m["star"]["brier"], b["star"]["brier"], c["star"]["brier"])
    line("P(star+) ECE", m["star"]["ece"], b["star"]["ece"], c["star"]["ece"])
    line("VORP Spearman", m["vorp"]["spearman"], b["vorp"]["spearman"], c["vorp"]["spearman"])
    line("VORP MAE", m["vorp"]["mae"], b["vorp"]["mae"], c["vorp"]["mae"])


def _holdout(prospects: pd.DataFrame) -> dict:
    """Validate the tuned scouting model on the held-out 2016-2019 cohorts
    (settings were tuned only on 2010-2015), vs profile-only and the baseline."""
    from models.projection import scouting_context
    years = range(2016, 2020)
    _, prof = run_backtest(prospects, test_years=years)
    _, scout = run_backtest(prospects, test_years=years, context=scouting_context())

    def grab(m):
        return {"auc": m["star"]["auc"], "ece": m["star"]["ece"],
                "spearman": m["vorp"]["spearman"],
                "within_one": m.get("tier", {}).get("within_one")}
    return {"n_prospects": scout["n_prospects"], "years": [2016, 2019],
            "baseline": grab(prof["baseline"]), "profile_only": grab(prof["model"]),
            "scouting": grab(scout["model"]), "combined": grab(scout["combined"])}


def main() -> int:
    from models.projection import scouting_context
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    # Headline back-test uses the validated scouting model.
    res, summary = run_backtest(prospects, context=scouting_context())
    _print_summary(summary)
    summary["holdout"] = _holdout(prospects)

    out_dir = config.ROOT / "eval"
    plot_path = out_dir / "calibration.png"
    _plot(res, summary, plot_path)

    payload = {"summary": summary, "predictions": res.round(3).to_dict("records")}
    (out_dir / "backtest_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {plot_path}")
    print(f"Wrote {out_dir / 'backtest_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
