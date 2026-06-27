"""Tests for the Phase 2 comparables engine.

Mechanics are tested on a small synthetic universe (no data dependency); the
archetype sanity checks run on the real committed dataset and skip if absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.comparables import ComparablesEngine
from models.features import SIMILARITY_FEATURES
from pipelines import config


# --------------------------------------------------------------------------
# Synthetic universe: 3 guards + 3 bigs with clearly separated profiles.
# --------------------------------------------------------------------------
def _synthetic() -> pd.DataFrame:
    guards = {
        "age_at_draft": 19.5, "height_in": 75, "weight_lb": 185, "coll_mp_per_g": 32,
        "coll_pts_per_g": 18, "coll_trb_per_g": 3.5, "coll_ast_per_g": 5.0,
        "coll_stl_per_g": 1.6, "coll_blk_per_g": 0.2, "coll_tov_per_g": 2.5,
        "coll_fg_pct": 0.44, "coll_fg3_pct": 0.38, "coll_ft_pct": 0.80,
        "coll_efg_pct": 0.52, "coll_ts_pct": 0.57, "coll_usg_pct": 24.0,
    }
    bigs = {
        "age_at_draft": 19.8, "height_in": 83, "weight_lb": 245, "coll_mp_per_g": 30,
        "coll_pts_per_g": 14, "coll_trb_per_g": 9.5, "coll_ast_per_g": 1.0,
        "coll_stl_per_g": 0.7, "coll_blk_per_g": 2.6, "coll_tov_per_g": 1.8,
        "coll_fg_pct": 0.58, "coll_fg3_pct": 0.20, "coll_ft_pct": 0.62,
        "coll_efg_pct": 0.59, "coll_ts_pct": 0.61, "coll_usg_pct": 20.0,
    }
    rows = []
    for i in range(3):
        g = dict(guards, coll_pts_per_g=18 + i)        # small perturbations
        b = dict(bigs, coll_pts_per_g=14 + i)
        rows.append({"player_id": f"g{i}", "player_name": f"Guard {i}", "draft_year": 2010 + i,
                     "draft_pick": 5 + i, "position": "PG", "outcome_tier": "starter",
                     "career_vorp": 5.0, "has_college_stats": True, **g})
        rows.append({"player_id": f"b{i}", "player_name": f"Big {i}", "draft_year": 2010 + i,
                     "draft_pick": 8 + i, "position": "C", "outcome_tier": "starter",
                     "career_vorp": 6.0, "has_college_stats": True, **b})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def engine() -> ComparablesEngine:
    return ComparablesEngine().fit(_synthetic())


def test_fit_universe_size(engine):
    assert len(engine.ids_) == 6
    assert engine.Z_.shape == (6, len(SIMILARITY_FEATURES))


def test_results_sorted_and_scored(engine):
    df = _synthetic()
    comps = engine.get_comparables(df.iloc[0], k=5)
    assert len(comps) == 5
    sims = [c.similarity for c in comps]
    assert sims == sorted(sims, reverse=True)         # descending
    assert all(0 < c.similarity <= 100 for c in comps)


def test_self_query_is_perfect_match(engine):
    df = _synthetic()
    row = df[df.player_id == "g0"].iloc[0]
    top = engine.get_comparables(row, k=1)[0]
    assert top.player_id == "g0"
    assert top.distance == pytest.approx(0.0, abs=1e-6)
    assert top.similarity == pytest.approx(100.0, abs=0.1)


def test_exclude_player_id(engine):
    df = _synthetic()
    row = df[df.player_id == "g0"].iloc[0]
    top = engine.get_comparables(row, k=1, exclude_player_id="g0")[0]
    assert top.player_id != "g0"
    assert top.player_id.startswith("g")             # nearest is another guard


def test_archetype_separation(engine):
    df = _synthetic()
    big = df[df.player_id == "b0"].iloc[0]
    nearest = engine.get_comparables(big, k=1, exclude_player_id="b0")[0]
    assert nearest.player_id.startswith("b")         # a big matches a big


def test_explanation_structure(engine):
    df = _synthetic()
    comps = engine.get_comparables(df.iloc[0], k=2, exclude_player_id="g0")
    c = comps[0]
    assert len(c.most_alike) == 3 and len(c.biggest_gaps) == 2
    # "most alike" features have smaller z-gaps than the "biggest gap" features
    assert max(d.z_gap for d in c.most_alike) <= min(d.z_gap for d in c.biggest_gaps)


# --------------------------------------------------------------------------
# Real-data archetype sanity (skips without the built dataset).
# --------------------------------------------------------------------------
PROSPECTS = config.PROCESSED / "prospects.parquet"


@pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")
def test_real_data_bigman_archetype():
    df = pd.read_parquet(PROSPECTS)
    eng = ComparablesEngine().fit(df)
    davis = df[df.player_name == "Anthony Davis"].iloc[0]
    comps = eng.get_comparables(davis, k=5, exclude_player_id=davis["player_id"])
    # Davis was a shot-blocking big; comps should skew tall + rim-protecting.
    avg_height = df["height_in"].mean()
    comp_heights = df.set_index("player_id").loc[[c.player_id for c in comps], "height_in"]
    assert comp_heights.mean() > avg_height + 3        # clearly taller than average
    assert all(meta in c.to_dict() for c in comps for meta in ("similarity", "most_alike"))


@pytest.mark.skipif(not PROSPECTS.exists(), reason="dataset not built")
def test_dybantsa_comparables_present():
    df = pd.read_parquet(PROSPECTS)
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    eng = ComparablesEngine().fit(df)
    comps = eng.get_comparables(dyb, k=10)
    assert len(comps) == 10
    assert all(np.isfinite(c.distance) for c in comps)
    # Dybantsa is a perimeter scorer: most comps should be guards/wings, not centers.
    centers = sum(1 for c in comps if c.position == "C")
    assert centers <= 3
