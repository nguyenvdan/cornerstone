"""Tests for the Phase 3 probabilistic projection model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.features import SIMILARITY_FEATURES
from models.projection import (
    ProjectionModel,
    dybantsa_context,
    vorp_feature_weights,
    weighted_percentile,
)
from pipelines import config

PROSPECTS = config.PROCESSED / "prospects.parquet"


# --------------------------------------------------------------------------
# Pure-unit (no data dependency)
# --------------------------------------------------------------------------
def test_weighted_percentile_matches_unweighted():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones_like(v)
    assert weighted_percentile(v, w, 50) == pytest.approx(3.0, abs=1e-9)
    assert weighted_percentile(v, w, 0) == pytest.approx(1.0)
    assert weighted_percentile(v, w, 100) == pytest.approx(5.0)


def test_weighted_percentile_respects_weights():
    v = np.array([0.0, 10.0])
    # nearly all weight on the high value -> median pulled toward 10
    assert weighted_percentile(v, np.array([0.01, 0.99]), 50) > 9.0


# --------------------------------------------------------------------------
# Model behavior on the real committed dataset (skips if not built)
# --------------------------------------------------------------------------
pytestmark_data = pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")


@pytest.fixture(scope="module")
def prospects() -> pd.DataFrame:
    return pd.read_parquet(PROSPECTS)


@pytest.fixture(scope="module")
def model(prospects) -> ProjectionModel:
    return ProjectionModel(prospects)


@pytestmark_data
def test_feature_weights_cover_all_features(prospects):
    w = vorp_feature_weights(prospects)
    assert set(w) == set(SIMILARITY_FEATURES)
    assert all(v >= 0.1 - 1e-9 for v in w.values())     # respects the floor
    # age should carry more weight than a near-noise feature like 3P%
    assert w["age_at_draft"] > w["coll_fg3_pct"]


@pytestmark_data
def test_tier_probabilities_form_distribution(model, prospects):
    row = prospects[prospects.player_name == "Anthony Davis"].iloc[0]
    proj = model.project(row)
    probs = proj.tier_probabilities
    assert set(probs) == set(config.TIER_ORDER)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    # cumulative helpers are internally consistent
    assert proj.p_starter_plus >= proj.p_star_plus
    assert proj.p_star_plus == pytest.approx(probs["all_star"] + probs["superstar"], abs=1e-6)


@pytestmark_data
def test_value_band_is_monotonic(model, prospects):
    row = prospects[prospects.player_name == "Anthony Davis"].iloc[0]
    b = model.project(row).career_vorp_band
    assert b["p10"] <= b["p25"] <= b["p50"] <= b["p75"] <= b["p90"]


@pytestmark_data
def test_season_curve_bands_ordered(model, prospects):
    row = prospects[prospects.player_name == "Anthony Davis"].iloc[0]
    curve = model.project(row).season_curve
    assert len(curve) >= 1
    for s in curve:
        assert s["p10"] <= s["p25"] <= s["p50"] <= s["p75"] <= s["p90"]
        assert 1 <= s["season"] <= config.OUTCOME_WINDOW


@pytestmark_data
def test_swing_factors_present_and_ranked(model, prospects):
    row = prospects[prospects.player_name == "Anthony Davis"].iloc[0]
    sf = model.project(row).swing_factors
    assert 1 <= len(sf) <= 6
    mags = [abs(s.effect_vorp) for s in sf]
    assert mags == sorted(mags, reverse=True)          # ranked by |effect|


@pytestmark_data
def test_discriminates_star_from_bust(model, prospects):
    """The headline validity check: a real star out-projects a real bust."""
    durant = model.project(prospects[prospects.player_name == "Kevin Durant"].iloc[0])
    morrison = model.project(prospects[prospects.player_name == "Adam Morrison"].iloc[0])
    assert durant.p_star_plus > morrison.p_star_plus
    assert durant.expected_career_vorp > morrison.expected_career_vorp


@pytestmark_data
def test_scouting_context_raises_floor_but_keeps_risk(prospects):
    """Scouting-informed signals (draft capital, archetype, competition, age)
    should lower bust and raise star vs profile-only — without zeroing risk."""
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    base = ProjectionModel(prospects).project(dyb, include_curve=False, include_swing=False)
    prod = ProjectionModel(prospects, context=dybantsa_context()).project(
        dyb, include_curve=False, include_swing=False)
    assert prod.tier_probabilities["bust"] < base.tier_probabilities["bust"]
    assert prod.p_star_plus > base.p_star_plus
    assert prod.expected_career_vorp > base.expected_career_vorp
    assert prod.tier_probabilities["bust"] > 0.0          # not false-precision 0%
    assert abs(sum(prod.tier_probabilities.values()) - 1.0) < 1e-6


@pytestmark_data
def test_context_defaults_off_leave_model_unchanged(prospects):
    # The Phase 4 back-test relies on this: no context => pure profile model.
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    a = ProjectionModel(prospects).project(dyb, include_curve=False, include_swing=False)
    b = ProjectionModel(prospects).project(dyb, include_curve=False, include_swing=False)
    assert a.tier_probabilities == b.tier_probabilities


@pytestmark_data
def test_dybantsa_projection_sensible():
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    proj = ProjectionModel(pd.read_parquet(PROSPECTS)).project(dyb)
    assert proj.n_comparables > 0
    assert sum(proj.tier_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # a top prospect should carry meaningful upside but honest uncertainty
    assert proj.p_star_plus > 0.05
    assert proj.tier_probabilities["bust"] > 0.0
    assert proj.career_vorp_band["p90"] > proj.career_vorp_band["p10"]
    assert len(proj.key_uncertainties) >= 3
