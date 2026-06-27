"""Tests for enriched predictors + the tuned scouting context."""

from __future__ import annotations

import pandas as pd

from models.projection import ProjectionContext, dybantsa_context, scouting_context
from pipelines.enrich import _age_minus_class, _conf_strength


def test_conf_strength_tiers():
    assert _conf_strength("Big 12") == 3.0          # power
    assert _conf_strength("MWC") == 2.0             # strong mid-major
    assert _conf_strength("Southland") == 1.0       # low
    assert _conf_strength(None) is None


def test_age_minus_class():
    young_fr = pd.Series({"coll_class": "FR", "age_at_draft": 18.8})
    old_fr = pd.Series({"coll_class": "FR", "age_at_draft": 19.9})
    assert _age_minus_class(young_fr) < 0           # young for a freshman
    assert _age_minus_class(old_fr) > 0
    assert _age_minus_class(pd.Series({"coll_class": None, "age_at_draft": 20})) is None


def test_scouting_context_is_tuned_and_neutralizes_dead_levers():
    ctx = scouting_context()
    assert isinstance(ctx, ProjectionContext)
    assert ctx.draft_prior and ctx.competition_match
    # the back-test found these didn't help out-of-sample -> off
    assert ctx.athleticism_match is False
    assert ctx.age_weight_boost == 1.0
    # no per-player archetype in the generalizable context
    assert ctx.archetype_blend == 0.0


def test_dybantsa_context_adds_archetype_overlay():
    ctx = dybantsa_context()
    assert ctx.archetype_blend > 0
    assert "Kevin Durant" in ctx.archetype_anchors
    # but shares the same tuned generalizable settings
    assert ctx.draft_bandwidth == scouting_context().draft_bandwidth
