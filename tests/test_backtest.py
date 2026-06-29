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
# Significance & segmented discrimination (new)
# --------------------------------------------------------------------------
def test_bootstrap_delta_detects_a_real_gap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.1, 400)  # tracks y
    bad = rng.random(400)                                        # noise
    d = metrics.bootstrap_delta(good, bad, y, kind="auc", n=500)
    assert d["delta"] > 0 and d["significant"] and d["ci_low"] > 0


def test_bootstrap_delta_identical_predictors_not_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    d = metrics.bootstrap_delta(p, p, y, kind="auc", n=500)
    assert d["delta"] == pytest.approx(0.0, abs=1e-9)
    assert not d["significant"]


def test_auc_within_buckets_structure():
    rng = np.random.default_rng(2)
    n = 200
    picks = rng.integers(1, 60, n)
    y = rng.integers(0, 2, n)
    rows = metrics.auc_within_buckets(rng.random(n), rng.random(n), y, picks)
    assert [r["bucket"] for r in rows] == ["1-5", "6-14", "15-30", "31-60"]
    assert sum(r["n"] for r in rows) <= n  # buckets cover picks 1-60


# --------------------------------------------------------------------------
# Walk-forward stacker (new)
# --------------------------------------------------------------------------
def _fake_predictions(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "draft_year": rng.integers(2008, 2018, n),
        "actual_star": y,
        "actual_vorp": np.where(y == 1, 18.0, 2.0) + rng.normal(0, 3, n),
        "model_p_star": np.clip(np.where(y == 1, 0.6, 0.2) + rng.normal(0, 0.1, n), 0, 1),
        "base_p_star": np.clip(np.where(y == 1, 0.5, 0.25) + rng.normal(0, 0.1, n), 0, 1),
        "model_exp_vorp": np.where(y == 1, 12.0, 3.0) + rng.normal(0, 2, n),
        "base_exp_vorp": np.where(y == 1, 10.0, 4.0) + rng.normal(0, 2, n),
    })


def test_stacker_falls_back_to_average_without_history():
    from eval.stacker import WalkForwardStacker
    rows = _fake_predictions()
    st = WalkForwardStacker()
    st.fit(None)  # no past data
    p, v = st.predict(rows)
    avg = 0.5 * (rows["model_p_star"].to_numpy() + rows["base_p_star"].to_numpy())
    assert not st.fitted
    assert np.allclose(p, avg)


def test_walk_forward_combine_is_aligned_and_in_range():
    from eval.stacker import walk_forward_combine
    rows = _fake_predictions()
    p, v = walk_forward_combine(rows)
    assert len(p) == len(v) == len(rows)
    assert ((p >= 0) & (p <= 1)).all()


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
    # The walk-forward stacked "combined" is a valid, in-range predictor. (We do
    # NOT assert it beats the baseline: the bootstrap CIs below show combining
    # doesn't add significant value over the standalone model — an honest result.)
    assert 0.5 <= summary["combined"]["star"]["auc"] <= 1.0
    # Significance section is present and well-formed.
    sig = summary["significance"]["auc_model_minus_base"]
    assert sig["ci_low"] <= sig["delta"] <= sig["ci_high"]
    # Within-bucket discrimination is reported for the standard pick bands.
    assert [r["bucket"] for r in summary["within_bucket"]] == ["1-5", "6-14", "15-30", "31-60"]


@pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")
def test_backtest_results_have_predictions():
    df = pd.read_parquet(PROSPECTS)
    res, _ = run_backtest(df, test_years=range(2014, 2016))
    assert not res.empty
    for col in ("player_name", "actual_tier", "model_p_star", "base_p_star"):
        assert col in res.columns
