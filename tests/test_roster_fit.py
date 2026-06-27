"""Tests for the Phase 5 roster-fit engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.roster_fit import (
    ARCHETYPES,
    cornerstone_skill_supply,
    evaluate_fit,
    what_if,
)
from pipelines import config
from pipelines.nba_skills import SEASON, SKILLS, _percentile

SKILLS_PARQUET = config.PROCESSED / f"nba_skills_{SEASON}.parquet"
PROSPECTS = config.PROCESSED / "prospects.parquet"


def _roster(weak_skill: str | None = None) -> pd.DataFrame:
    """Five players, all strong (70) everywhere except one optionally-weak skill."""
    rows = []
    for i in range(5):
        r = {"player_name": f"P{i}", "mp_per_g": 28.0}
        for s in SKILLS:
            r[s] = 30.0 if s == weak_skill else 70.0
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def test_percentile_monotonic():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    out = _percentile(np.array([0.0, 2.5, 5.0]), ref)
    assert out[0] == 0.0 and out[2] == 100.0
    assert out[0] < out[1] < out[2]


def test_evaluate_fit_structure():
    rep = evaluate_fit(_roster(), {s: 50.0 for s in SKILLS})
    assert 0.0 <= rep.fit_score <= 100.0
    assert [s.skill for s in rep.skills] == [s.skill for s in
            sorted(rep.skills, key=lambda r: r.gap, reverse=True)]
    assert len(rep.biggest_gaps) == 3
    scores = [a["score"] for a in rep.recommended_archetypes]
    assert scores == sorted(scores, reverse=True)


def test_weak_roster_skill_becomes_top_gap_and_recommendation():
    # cornerstone lacks rim protection; roster also weak there -> top gap + a
    # rim-protecting archetype should be recommended first.
    supply = {s: 50.0 for s in SKILLS}
    supply["rim_protection"] = 10.0       # he can't protect the rim
    rep = evaluate_fit(_roster(weak_skill="rim_protection"), supply)
    assert rep.biggest_gaps[0] == "rim_protection"
    assert "rim_protection" in rep.recommended_archetypes[0]["provides"]


def test_high_supply_skill_has_low_need():
    supply = {s: 50.0 for s in SKILLS}
    supply["shot_creation"] = 99.0
    rep = evaluate_fit(_roster(), supply)
    need = {s.skill: s.need for s in rep.skills}
    assert need["shot_creation"] == min(need.values())


def test_what_if_relevant_addition_helps_most():
    supply = {s: 50.0 for s in SKILLS}
    supply["rim_protection"] = 10.0
    roster = _roster(weak_skill="rim_protection")
    relevant = what_if(roster, supply, "Defensive Anchor Big")
    irrelevant = what_if(roster, supply, "Floor General PG")
    assert relevant["fit_delta"] >= irrelevant["fit_delta"]
    assert relevant["fit_after"] >= relevant["fit_before"]


def test_what_if_unknown_archetype_raises():
    with pytest.raises(ValueError):
        what_if(_roster(), {s: 50.0 for s in SKILLS}, "Point Forward Unicorn")


# --------------------------------------------------------------------------
@pytest.mark.skipif(not (SKILLS_PARQUET.exists() and PROSPECTS.exists()),
                    reason="nba skills / prospects not built")
def test_real_wizards_fit_is_sensible():
    skills = pd.read_parquet(SKILLS_PARQUET)
    wiz = skills[skills["teams"].apply(lambda t: "WAS" in t) & skills["is_qualified"]]
    assert len(wiz) >= 8
    prospects = pd.read_parquet(PROSPECTS)
    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    supply = cornerstone_skill_supply(dyb, prospects)
    # Dybantsa: a high-usage scorer who doesn't protect the rim.
    assert supply["shot_creation"] > 80
    assert supply["rim_protection"] < 45
    rep = evaluate_fit(wiz, supply)
    assert 0.0 <= rep.fit_score <= 100.0
    assert set(s.skill for s in rep.skills) == set(SKILLS)
    assert rep.recommended_archetypes[0]["archetype"] in ARCHETYPES
