"""Scoring utilities for the back-test: calibration, accuracy, ranking."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pipelines import config


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts (lower is better)."""
    return float(np.mean((np.asarray(prob) - np.asarray(outcome)) ** 2))


def reliability_curve(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10):
    """Return (bin_conf, bin_obs, bin_count) for a reliability diagram + ECE."""
    prob = np.asarray(prob, float)
    outcome = np.asarray(outcome, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, obs, count = [], [], []
    ece = 0.0
    n = len(prob)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        c, o, k = prob[mask].mean(), outcome[mask].mean(), int(mask.sum())
        conf.append(round(float(c), 4))
        obs.append(round(float(o), 4))
        count.append(k)
        ece += (k / n) * abs(o - c)
    return {"confidence": conf, "observed": obs, "count": count, "ece": round(float(ece), 4)}


def binary_scores(prob: np.ndarray, outcome: np.ndarray) -> dict:
    prob = np.asarray(prob, float)
    outcome = np.asarray(outcome, int)
    auc = float(roc_auc_score(outcome, prob)) if 0 < outcome.sum() < len(outcome) else float("nan")
    rel = reliability_curve(prob, outcome)
    return {"auc": round(auc, 3), "brier": round(brier(prob, outcome), 4),
            "ece": rel["ece"], "reliability": rel}


def tier_index(tier: str) -> int:
    return config.TIER_ORDER.index(tier)


def tier_accuracy(pred_tiers: list[str], actual_tiers: list[str]) -> dict:
    exact = np.mean([p == a for p, a in zip(pred_tiers, actual_tiers, strict=False)])
    within1 = np.mean([abs(tier_index(p) - tier_index(a)) <= 1
                       for p, a in zip(pred_tiers, actual_tiers, strict=False)])
    return {"exact": round(float(exact), 3), "within_one": round(float(within1), 3)}


def ranking_scores(pred: np.ndarray, actual: np.ndarray) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return {"spearman": round(float(spearmanr(pred, actual).correlation), 3),
            "mae": round(float(np.mean(np.abs(pred - actual))), 2)}


# --------------------------------------------------------------------------
# Significance & segmented discrimination
# --------------------------------------------------------------------------
def _auc(prob: np.ndarray, outcome: np.ndarray) -> float:
    return roc_auc_score(outcome, prob) if 0 < outcome.sum() < len(outcome) else float("nan")


def bootstrap_delta(
    pred_a: np.ndarray, pred_b: np.ndarray, actual: np.ndarray,
    kind: str = "auc", n: int = 2000, seed: int = 0,
) -> dict:
    """Paired bootstrap 95% CI of a metric *difference* (model A − model B).

    Resamples prospects with replacement, recomputes the metric for both
    predictors on each resample, and reports the distribution of the gap. The
    interval crossing 0 means the edge isn't distinguishable from noise — the
    honest test of "does the model actually beat the baseline?". ``kind`` is
    ``"auc"`` (vs a binary ``actual``) or ``"spearman"`` (vs a continuous one).
    """
    a, b, y = np.asarray(pred_a, float), np.asarray(pred_b, float), np.asarray(actual, float)
    rng = np.random.default_rng(seed)
    if kind == "auc":
        score = lambda p, o: _auc(p, o.astype(int))   # noqa: E731
    elif kind == "spearman":
        score = lambda p, o: spearmanr(p, o).correlation   # noqa: E731
    else:
        raise ValueError(f"unknown kind {kind!r}")
    base = score(a, y) - score(b, y)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        da, db = score(a[idx], y[idx]), score(b[idx], y[idx])
        if not (np.isnan(da) or np.isnan(db)):
            deltas.append(da - db)
    if not deltas:
        return {"delta": None, "ci_low": None, "ci_high": None, "significant": False}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float(base), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "significant": bool(lo > 0 or hi < 0)}


# Draft-pick segments. Within a narrow pick band the draft-position baseline is
# nearly flat, so any discrimination there is signal the *profile* adds — the
# one place the model can beat "just look at the draft slot".
PICK_BUCKETS = [(1, 5), (6, 14), (15, 30), (31, 60)]


def auc_within_buckets(
    model_p: np.ndarray, base_p: np.ndarray, actual_star: np.ndarray, picks: np.ndarray,
) -> list[dict]:
    """Per-pick-bucket star-detection AUC for model vs baseline."""
    model_p = np.asarray(model_p, float)
    base_p = np.asarray(base_p, float)
    y = np.asarray(actual_star, int)
    picks = np.asarray(picks, float)
    out = []
    for lo, hi in PICK_BUCKETS:
        m = (picks >= lo) & (picks <= hi)
        ys = y[m]
        row = {"bucket": f"{lo}-{hi}", "n": int(m.sum()), "stars": int(ys.sum())}
        if 0 < ys.sum() < len(ys):
            row["model_auc"] = round(float(_auc(model_p[m], ys)), 3)
            row["base_auc"] = round(float(_auc(base_p[m], ys)), 3)
            row["delta"] = round(row["model_auc"] - row["base_auc"], 3)
        else:
            row["model_auc"] = row["base_auc"] = row["delta"] = None
        out.append(row)
    return out
