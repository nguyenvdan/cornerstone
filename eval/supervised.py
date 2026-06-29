"""Prototype: a *directly supervised* projection model (diagnostic for Phase 4+).

The kNN/comparables model predicts career value as a similarity-weighted **mean**
of a 75-neighbor cloud. The back-test exposes the structural cost of that: the
mean can't reach the tails, so the model *compresses the range* — it predicts
~11 career VORP for prospects who become 40-VORP superstars (residual +29). A
weighted average mathematically cannot extrapolate.

This module tests the fix: fit a model that predicts the outcome **directly**
from draft-time features (the 16-stat profile + draft capital + competition),
so it can output star-level values. Gradient-boosted trees with early stopping
and a monotonic draft-capital constraint, classification probabilities
calibrated (Platt). Evaluated in the *same* leakage-free expanding-window
protocol and scored with the *same* metrics as eval/backtest.py, plus a
residual-by-tier table so we can see directly whether the compression is gone.

    uv run python -m eval.supervised
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from models.features import SIMILARITY_FEATURES
from models.projection import POWER_CONFERENCES, scouting_context
from pipelines import config

from . import metrics
from .backtest import DEFAULT_TEST_YEARS, _college, run_backtest

warnings.filterwarnings("ignore")

# Profile (16) + draft capital + competition. log1p(pick) so the lottery is
# spread out and late picks compress, matching how draft value actually behaves.
_ENGINEERED = ["log_draft_pick", "is_power_conf"]
FEATURES = SIMILARITY_FEATURES + _ENGINEERED
# Outcome decreases as the pick *number* grows -> monotonic-decreasing constraint
# on draft capital; everything else unconstrained. Guards against the trees
# inventing a non-monotone draft-value curve from noise in small samples.
_MONOTONIC = [0] * len(SIMILARITY_FEATURES) + [-1, 0]


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SIMILARITY_FEATURES].astype(float).copy()
    X["log_draft_pick"] = np.log1p(df["draft_pick"].astype(float))
    X["is_power_conf"] = df["coll_conf"].isin(POWER_CONFERENCES).astype(float)
    return X[FEATURES]


def _tier_from_vorp(v: float) -> str:
    for tier, lower in config.VORP_TIER_BINS:
        if v >= lower:
            return tier
    return "bust"


class SupervisedModel:
    """Direct GBM: calibrated P(star+) classifier + career-VORP regressor."""

    def fit(self, train: pd.DataFrame) -> SupervisedModel:
        X = _design(train)
        self.medians_ = X.median()
        X = X.fillna(self.medians_)
        y_star = train["outcome_tier"].isin(("all_star", "superstar")).astype(int).to_numpy()
        y_vorp = train["career_vorp"].astype(float).to_numpy()

        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=1.0,
            min_samples_leaf=20, early_stopping=True, monotonic_cst=_MONOTONIC,
            random_state=0)
        # Platt-scale on internal CV for honest probabilities; sigmoid is robust
        # to the small positive counts in early cohorts where isotonic would be jumpy.
        n_pos = int(y_star.sum())
        if n_pos >= 12:
            self.clf_ = CalibratedClassifierCV(clf, method="sigmoid",
                                               cv=min(3, n_pos // 4)).fit(X, y_star)
        else:
            self.clf_ = clf.fit(X, y_star)

        self.reg_ = HistGradientBoostingRegressor(
            max_depth=3, max_iter=300, learning_rate=0.05, l2_regularization=1.0,
            min_samples_leaf=20, early_stopping=True, monotonic_cst=_MONOTONIC,
            random_state=0).fit(X, y_vorp)
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        X = _design(test).fillna(self.medians_)
        p_star = self.clf_.predict_proba(X)[:, 1]
        exp_vorp = self.reg_.predict(X)
        return pd.DataFrame({"sup_p_star": p_star, "sup_exp_vorp": exp_vorp,
                             "sup_tier": [_tier_from_vorp(v) for v in exp_vorp]})


def run_supervised_backtest(prospects: pd.DataFrame, test_years=None) -> pd.DataFrame:
    """Same expanding-window protocol as eval/backtest, supervised predictor."""
    test_years = test_years if test_years is not None else DEFAULT_TEST_YEARS
    pool = _college(prospects)
    rows = []
    for year in test_years:
        train = pool[pool["draft_year"] < year]
        test = pool[pool["draft_year"] == year]
        if len(train) < 50 or test.empty:
            continue
        preds = SupervisedModel().fit(train).predict(test).reset_index(drop=True)
        t = test.reset_index(drop=True)
        for i in range(len(t)):
            rows.append({
                "player_name": t.loc[i, "player_name"],
                "draft_year": int(year),
                "actual_tier": t.loc[i, "outcome_tier"],
                "actual_vorp": float(t.loc[i, "career_vorp"]),
                "actual_star": int(t.loc[i, "outcome_tier"] in ("all_star", "superstar")),
                "sup_p_star": float(preds.loc[i, "sup_p_star"]),
                "sup_exp_vorp": float(preds.loc[i, "sup_exp_vorp"]),
                "sup_tier": preds.loc[i, "sup_tier"],
            })
    return pd.DataFrame(rows)


def _score(res: pd.DataFrame) -> dict:
    star = metrics.binary_scores(res["sup_p_star"].to_numpy(), res["actual_star"].to_numpy())
    vorp = metrics.ranking_scores(res["sup_exp_vorp"].to_numpy(), res["actual_vorp"].to_numpy())
    tier = metrics.tier_accuracy(res["sup_tier"].tolist(), res["actual_tier"].tolist())
    return {"star": star, "vorp": vorp, "tier": tier}


def _compression_table(res: pd.DataFrame, pred_col: str) -> list[dict]:
    out = []
    for t in config.TIER_ORDER:
        s = res[res["actual_tier"] == t]
        if len(s):
            out.append({"tier": t, "n": int(len(s)),
                        "mean_actual_vorp": round(float(s["actual_vorp"].mean()), 1),
                        "mean_pred_vorp": round(float(s[pred_col].mean()), 1),
                        "resid": round(float((s["actual_vorp"] - s[pred_col]).mean()), 1)})
    return out


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    sup = run_supervised_backtest(prospects)
    s_sup = _score(sup)
    # kNN scouting model + baseline on the identical window, for a fair table.
    knn_res, s_knn = run_backtest(prospects, context=scouting_context())

    print(f"Supervised prototype vs kNN model vs baseline — "
          f"{len(sup)} prospects, {sup['draft_year'].nunique()} classes\n")
    hdr = f"{'metric':22s} {'supervised':>11s} {'kNN model':>11s} {'baseline':>11s}"
    print(hdr)
    print("-" * len(hdr))
    km, kb = s_knn["model"], s_knn["baseline"]
    rows = [
        ("P(star+) AUC", s_sup["star"]["auc"], km["star"]["auc"], kb["star"]["auc"]),
        ("P(star+) Brier", s_sup["star"]["brier"], km["star"]["brier"], kb["star"]["brier"]),
        ("P(star+) ECE", s_sup["star"]["ece"], km["star"]["ece"], kb["star"]["ece"]),
        ("VORP Spearman", s_sup["vorp"]["spearman"], km["vorp"]["spearman"],
         kb["vorp"]["spearman"]),
        ("VORP MAE", s_sup["vorp"]["mae"], km["vorp"]["mae"], kb["vorp"]["mae"]),
        ("tier within-1", s_sup["tier"]["within_one"], km["tier"]["within_one"],
         kb["tier"]["within_one"]),
    ]
    for name, a, b, c in rows:
        print(f"{name:22s} {a:>11} {b:>11} {c:>11}")

    # Significance: does the supervised model beat the baseline (and the kNN)?
    merged = sup.merge(knn_res[["player_name", "draft_year", "base_p_star", "model_p_star",
                                "base_exp_vorp", "model_exp_vorp"]],
                       on=["player_name", "draft_year"], how="inner")
    d_base = metrics.bootstrap_delta(merged["sup_p_star"].to_numpy(),
                                     merged["base_p_star"].to_numpy(),
                                     merged["actual_star"].to_numpy(), "auc")
    d_knn = metrics.bootstrap_delta(merged["sup_p_star"].to_numpy(),
                                    merged["model_p_star"].to_numpy(),
                                    merged["actual_star"].to_numpy(), "auc")
    print(f"\nAUC  supervised − baseline : Δ {d_base['delta']:+.4f}  "
          f"[{d_base['ci_low']:+.4f}, {d_base['ci_high']:+.4f}]"
          f"{'  *sig*' if d_base['significant'] else '  (n.s.)'}")
    print(f"AUC  supervised − kNN model: Δ {d_knn['delta']:+.4f}  "
          f"[{d_knn['ci_low']:+.4f}, {d_knn['ci_high']:+.4f}]"
          f"{'  *sig*' if d_knn['significant'] else '  (n.s.)'}")

    print("\nRange compression — VORP residual (actual − predicted) by realized tier:")
    print(f"  {'tier':10s} {'n':>4s} {'actual':>8s} {'supervised':>11s} {'kNN model':>11s}")
    knn_comp = {r["tier"]: r for r in _compression_table(
        knn_res.rename(columns={"model_exp_vorp": "pred"}), "pred")}
    for r in _compression_table(sup, "sup_exp_vorp"):
        k = knn_comp.get(r["tier"], {})
        print(f"  {r['tier']:10s} {r['n']:>4d} {r['mean_actual_vorp']:>8.1f} "
              f"{r['mean_pred_vorp']:>11.1f} {k.get('mean_pred_vorp', float('nan')):>11.1f}")
    print("  (closer 'predicted' to 'actual' for all_star/superstar = less compression)")

    out = {"n_prospects": int(len(sup)), "scores": s_sup,
           "significance": {"vs_baseline": d_base, "vs_knn": d_knn},
           "compression": _compression_table(sup, "sup_exp_vorp")}
    path = config.ROOT / "eval" / "supervised_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
