"""Tests for the Phase 6 agent (scripted mode — no API key needed)."""

from __future__ import annotations

import json

import pytest

from agent.runner import run
from agent.tools import Context, build_tools
from pipelines import config

PROSPECTS = config.PROCESSED / "prospects.parquet"
pytestmark = pytest.mark.skipif(
    not PROSPECTS.exists(), reason="dataset/models not built")


@pytest.fixture(scope="module")
def ctx() -> Context:
    return Context()


@pytest.fixture(scope="module")
def tools(ctx):
    return build_tools(ctx)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def test_lookup_resolves_dybantsa_and_historicals(tools):
    dyb = tools["lookup_prospect"].func(name="AJ Dybantsa")
    assert dyb["player_name"] == "AJ Dybantsa"
    assert dyb["coll_pts_per_g"] == 25.5
    lebron = tools["lookup_prospect"].func(name="LeBron James")
    assert "realized_outcome_tier" in lebron      # historical -> outcome present
    assert tools["lookup_prospect"].func(name="Nobody XYZ")["error"]


def test_find_comparables_tool(tools):
    out = tools["find_comparables"].func(name="AJ Dybantsa", k=6)
    assert len(out["comparables"]) == 6
    assert out["comparables"][0]["why_alike"]


def test_project_tool_is_probabilistic(tools):
    out = tools["project_development"].func(name="AJ Dybantsa")
    assert abs(sum(out["tier_probabilities"].values()) - 1.0) < 1e-6
    assert 0 <= out["p_star_plus"] <= 1
    assert out["career_vorp_band"]["p90"] > out["career_vorp_band"]["p10"]


def test_roster_fit_tool(tools):
    out = tools["evaluate_roster_fit"].func(team="WAS", cornerstone="AJ Dybantsa")
    assert 0 <= out["fit_score"] <= 100
    assert len(out["biggest_gaps"]) == 3
    assert out["recommended_archetypes"]


def test_all_tools_have_anthropic_specs(tools):
    for t in tools.values():
        spec = t.anthropic_spec()
        assert set(spec) == {"name", "description", "input_schema"}
        assert spec["input_schema"]["type"] == "object"


# --------------------------------------------------------------------------
# Agent loop (scripted)
# --------------------------------------------------------------------------
def test_scripted_run_produces_grounded_report(ctx):
    res = run("Analyze the Wizards' path with Dybantsa as cornerstone.",
              ctx=ctx, force_scripted=True)
    assert res.mode == "scripted"
    assert len(res.steps) >= 4
    md = res.report_markdown
    assert "# Scouting & strategy report" in md
    assert "Development projection" in md and "Roster fit" in md
    # grounded: the fit score in the prose equals the tool's returned value
    assert str(res.tool_results["evaluate_roster_fit"]["fit_score"]) in md
    # states uncertainty, not a single outcome
    assert "P(bust)" in md or "80% range" in md
    json.dumps(res.to_dict(), default=str)          # must be serializable


def test_intent_routing_skips_roster_for_development_query(ctx):
    res = run("How is AJ Dybantsa likely to develop?", ctx=ctx, force_scripted=True)
    assert "project_development" in res.tool_results
    assert "evaluate_roster_fit" not in res.tool_results     # no team intent


def test_detects_non_default_cornerstone(ctx):
    res = run("Project Kevin Durant's development.", ctx=ctx, force_scripted=True)
    assert res.tool_results["project_development"]["prospect"] == "Kevin Durant"
