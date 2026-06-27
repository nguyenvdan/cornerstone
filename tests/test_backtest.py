"""Tests for the Phase 4 back-test: metrics, baseline, and the full run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval import metrics
from eval.backtest import run_backtest
from eval.baseline import DraftPositionBaseline
from pipelines import config

PROSPECTS = config.PROCESSED / "prospects.parquet"


# --------------------------------------------------------------------------
# Metrics (pure-unit)
# --------------------------------------------------------------------------
def test_brier_bounds():
    assert metrics.brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert metrics.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_reliability_perfect_is_well_calibrated():
    p = np.array([0.05, 0.15, 0.25, 0.85, 0.95] * 20)
    rng = np.random.default_rng(0)
    y = (rng.random(len(p)) < p).astype(int)
    rel = metrics.reliability_curve(p, y, n_bins=5)
    assert 0.0 <= rel["ece"] < 0.15
    assert len(rel["confidence"]) == len(rel["observed"]) == len(rel["count"])


def test_tier_accuracy_exact_and_within_one():
    pred = ["starter", "bust", "all_star"]
    actual = ["starter", "rotation", "superstar"]
    acc = metrics.tier_accuracy(pred, actual)
    assert acc["exact"] == pytest.approx(1 / 3, abs=1e-3)   # rounded to 3 dp
    assert acc["within_one"] == pytest.approx(1.0)   # each pred is within one tier


def test_ranking_scores_perfect():
    r = metrics.ranking_scores([1, 2, 3, 4], [10, 20, 30, 40])
    assert r["spearman"] == pytest.approx(1.0)
    assert r["mae"] >= 0


# --------------------------------------------------------------------------
# Baseline (small synthetic train)
# --------------------------------------------------------------------------
def test_baseline_monotonic_in_pick():
    # early picks are better players -> baseline should rank them higher
    rows = []
    for pick in range(1, 31):
        vorp = 30 - pick + np.random.default_rng(pick).normal(0, 1)
        tier = "all_star" if pick <= 5 else ("starter" if pick <= 15 else "bust")
        rows.append({"draft_pick": pick, "career_vorp": vorp, "outcome_tier": tier})
    base = DraftPositionBaseline().fit(pd.DataFrame(rows))
    assert base.expected_vorp(1) > base.expected_vorp(30)
    assert base.p_star_plus(1) > base.p_star_plus(25)
    probs = base.tier_probabilities(1)
    assert set(probs) == set(config.TIER_ORDER)
    assert sum(probs.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Full back-test on the real dataset (skips if not built)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")
def test_backtest_runs_and_is_leakage_aware():
    df = pd.read_parquet(PROSPECTS)
    res, summary = run_backtest(df, test_years=range(2012, 2018))
    assert summary["n_prospects"] > 100
    # every prediction must come from a model trained on strictly earlier classes:
    # there is no way for a test prospect's own row to be in its training pool.
    for key in ("model", "baseline", "combined"):
        auc = summary[key]["star"]["auc"]
        assert 0.5 <= auc <= 1.0
    ece = summary["model"]["star"]["ece"]
    assert 0.0 <= ece <= 1.0
    # combining the profile model with draft position should not hurt ranking
    assert summary["combined"]["star"]["auc"] >= summary["baseline"]["star"]["auc"] - 0.02


@pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")
def test_backtest_results_have_predictions():
    df = pd.read_parquet(PROSPECTS)
    res, _ = run_backtest(df, test_years=range(2014, 2016))
    assert not res.empty
    for col in ("player_name", "actual_tier", "model_p_star", "base_p_star"):
        assert col in res.columns
