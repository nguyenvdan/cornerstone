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
