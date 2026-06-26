"""Integrity checks on the built dataset.

These run only when ``make data`` has produced ``data/processed/``; otherwise
they skip, so the unit suite still passes on a fresh checkout / in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines import config

PROSPECTS = config.PROCESSED / "prospects.parquet"
DYBANTSA = config.PROCESSED / "dybantsa.parquet"

pytestmark = pytest.mark.skipif(
    not PROSPECTS.exists(), reason="dataset not built (run `make data`)"
)


@pytest.fixture(scope="module")
def prospects() -> pd.DataFrame:
    return pd.read_parquet(PROSPECTS)


def test_non_trivial_size(prospects):
    # Spec acceptance: "at least a few hundred historical prospects".
    assert len(prospects) >= 300


def test_unique_player_id(prospects):
    assert prospects["player_id"].is_unique


def test_full_year_span(prospects):
    assert prospects["draft_year"].min() <= config.DRAFT_START
    assert prospects["draft_year"].max() >= config.DRAFT_END


def test_no_null_keys(prospects):
    for col in ("player_id", "player_name", "draft_year", "outcome_tier"):
        assert prospects[col].isna().sum() == 0, col


def test_tiers_are_valid_and_spread(prospects):
    tiers = set(prospects["outcome_tier"].unique())
    assert tiers.issubset(set(config.TIER_ORDER))
    # A healthy universe spans most tiers, not one bucket.
    assert len(tiers) >= 4


def test_no_leakage_columns_in_features():
    feats = pd.read_parquet(config.PROCESSED / "prospect_features.parquet")
    leaky = [c for c in feats.columns if c.startswith(("career_", "early_", "peak_"))]
    leaky += [c for c in ("outcome_tier", "trajectory_vorp") if c in feats.columns]
    assert leaky == [], f"outcome columns leaked into features: {leaky}"


def test_known_superstars_classified():
    df = pd.read_parquet(PROSPECTS)
    for name in ("LeBron James", "Kevin Durant", "Stephen Curry"):
        row = df[df["player_name"] == name]
        assert not row.empty, f"{name} missing"
        assert row["outcome_tier"].iloc[0] == "superstar", name


@pytest.mark.skipif(not DYBANTSA.exists(), reason="dybantsa row not built")
def test_dybantsa_row_complete():
    d = pd.read_parquet(DYBANTSA)
    assert len(d) == 1
    row = d.iloc[0]
    assert row["player_name"] == "AJ Dybantsa"
    for col in ("coll_pts_per_g", "coll_usg_pct", "age_at_draft", "height_in"):
        assert pd.notna(row[col]), col
    # schema is a superset of the historical feature schema
    feats = set(pd.read_parquet(config.PROCESSED / "prospect_features.parquet").columns)
    assert feats.issubset(set(d.columns))
