"""Unit tests for pipeline logic that don't require the network."""

from __future__ import annotations

from pipelines import config
from pipelines.features import age_at_draft, build_features
from pipelines.outcomes import _tier_from_vorp, build_outcome
from pipelines.parse import make_soup, table_records, to_float, to_int
from pipelines.sources.bbref import DraftPick, PlayerPage


def test_to_float_and_int():
    assert to_float("12.5") == 12.5
    assert to_float("65%") == 65.0
    assert to_float("") is None
    assert to_float(None) is None
    assert to_int("3") == 3
    assert to_int(None) is None


def test_age_at_draft():
    # born 2006-12-04, 2026 draft (~late June) -> ~19.6
    age = age_at_draft("2006-12-04", 2026)
    assert age is not None and 19.0 < age < 20.0
    assert age_at_draft(None, 2026) is None


def test_tier_thresholds():
    assert _tier_from_vorp(40.0, 500) == "superstar"
    assert _tier_from_vorp(15.0, 400) == "all_star"
    assert _tier_from_vorp(5.0, 300) == "starter"
    assert _tier_from_vorp(1.0, 200) == "rotation"
    assert _tier_from_vorp(-2.0, 200) == "bust"
    # too few games forces bust regardless of rate value
    assert _tier_from_vorp(5.0, 10) == "bust"
    assert _tier_from_vorp(None, 100) == "bust"


def test_make_soup_lifts_comment_tables():
    html = """
    <div><!--
      <table id="advanced"><tbody>
        <tr><td data-stat="year_id">2018-19</td><td data-stat="vorp">4.1</td></tr>
      </tbody></table>
    --></div>
    """
    soup = make_soup(html)
    rows = table_records(soup, "advanced")
    assert rows == [{"year_id": "2018-19", "vorp": "4.1"}]


def test_build_outcome_window_aggregation():
    pick = DraftPick(
        draft_year=2015, pick_overall=4, team_id="NYK", player_id="x01",
        player_name="Test", college_name="Duke", career_seasons=8, career_g=500,
        career_ws=60.0, career_ws_per_48=0.15, career_bpm=3.0, career_vorp=18.0,
    )
    seasons = [{"season": f"201{i}-1{i+1}", "age": 20 + i, "g": 70, "mp": 2000,
                "per": 18.0, "ts_pct": 0.55, "usg_pct": 24.0, "ws": 5.0,
                "ws_per_48": 0.12, "bpm": 2.0 + i, "vorp": 1.5 + i} for i in range(7)]
    player = PlayerPage(player_id="x01", seasons=seasons)
    out = build_outcome(pick, player)
    assert out["outcome_tier"] == "all_star"
    assert out["window_seasons"] == config.OUTCOME_WINDOW
    # early_vorp_sum uses only the first OUTCOME_WINDOW seasons
    assert out["early_vorp_sum"] == round(sum(1.5 + i for i in range(config.OUTCOME_WINDOW)), 2)
    assert out["peak_bpm"] == max(2.0 + i for i in range(7))


def test_build_features_schema_consistency():
    pick = DraftPick(
        draft_year=2018, pick_overall=1, team_id="PHO", player_id="y01",
        player_name="Test C", college_name="Arizona", career_seasons=5, career_g=300,
        career_ws=30.0, career_ws_per_48=0.13, career_bpm=2.0, career_vorp=9.0,
    )
    player = PlayerPage(player_id="y01", birth_date="1998-07-23", height_in=84,
                        weight_lb=250, position="C", cbb_id="aaa-1")
    college = {"cbb_id": "aaa-1", "coll_pts_per_g": 20.0, "coll_trb_per_g": 11.0}
    row = build_features(pick, player, college)
    assert row["draft_round"] == 1
    assert row["height_in"] == 84
    assert row["has_college_stats"] is True
    assert row["coll_pts_per_g"] == 20.0
    assert "cbb_id" not in row  # internal id is not leaked into the feature row
