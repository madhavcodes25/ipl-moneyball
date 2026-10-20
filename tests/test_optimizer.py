"""
Unit tests for the IPL Moneyball optimizer.

Run with:
    pytest tests/test_optimizer.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Ensure the project root is on the path when running tests from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimizer import calculate_fantasy, optimize_team, get_efficient_frontier


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def minimal_player_pool():
    """A minimal 15-player pool that satisfies all IPL composition rules."""
    players = [
        # Batsmen (5)
        {"Player Name": "Batter A", "Role": "Batsman", "Subrole": "WK",      "Cost_Cr": 10.0, "Is_Foreign": 0,
         "Batting Runs": 600, "Batting Average": 45.0, "Batting Strike Rate": 160.0,
         "Wickets": 0, "Economy Rate": 0, "Bowling Average": 0},
        {"Player Name": "Batter B", "Role": "Batsman", "Subrole": "Non-WK",  "Cost_Cr": 8.0, "Is_Foreign": 0,
         "Batting Runs": 500, "Batting Average": 40.0, "Batting Strike Rate": 150.0,
         "Wickets": 0, "Economy Rate": 0, "Bowling Average": 0},
        {"Player Name": "Batter C", "Role": "Batsman", "Subrole": "Non-WK",  "Cost_Cr": 7.0, "Is_Foreign": 0,
         "Batting Runs": 450, "Batting Average": 38.0, "Batting Strike Rate": 145.0,
         "Wickets": 0, "Economy Rate": 0, "Bowling Average": 0},
        {"Player Name": "Batter D", "Role": "Batsman", "Subrole": "Non-WK",  "Cost_Cr": 6.0, "Is_Foreign": 1,
         "Batting Runs": 420, "Batting Average": 36.0, "Batting Strike Rate": 155.0,
         "Wickets": 0, "Economy Rate": 0, "Bowling Average": 0},
        {"Player Name": "Batter E", "Role": "Batsman", "Subrole": "Non-WK",  "Cost_Cr": 5.0, "Is_Foreign": 0,
         "Batting Runs": 380, "Batting Average": 32.0, "Batting Strike Rate": 140.0,
         "Wickets": 0, "Economy Rate": 0, "Bowling Average": 0},
        # All-Rounders (4)
        {"Player Name": "AllR A",   "Role": "All-Rounder", "Subrole": "Pace", "Cost_Cr": 12.0, "Is_Foreign": 0,
         "Batting Runs": 300, "Batting Average": 28.0, "Batting Strike Rate": 150.0,
         "Wickets": 18, "Economy Rate": 7.5, "Bowling Average": 25.0},
        {"Player Name": "AllR B",   "Role": "All-Rounder", "Subrole": "Spin", "Cost_Cr": 9.0, "Is_Foreign": 1,
         "Batting Runs": 250, "Batting Average": 25.0, "Batting Strike Rate": 140.0,
         "Wickets": 15, "Economy Rate": 8.0, "Bowling Average": 28.0},
        {"Player Name": "AllR C",   "Role": "All-Rounder", "Subrole": "Pace", "Cost_Cr": 7.5, "Is_Foreign": 0,
         "Batting Runs": 200, "Batting Average": 22.0, "Batting Strike Rate": 135.0,
         "Wickets": 12, "Economy Rate": 8.5, "Bowling Average": 30.0},
        {"Player Name": "AllR D",   "Role": "All-Rounder", "Subrole": "Spin", "Cost_Cr": 6.5, "Is_Foreign": 0,
         "Batting Runs": 180, "Batting Average": 20.0, "Batting Strike Rate": 130.0,
         "Wickets": 10, "Economy Rate": 9.0, "Bowling Average": 32.0},
        # Bowlers (6)
        {"Player Name": "Bowler A", "Role": "Bowler", "Subrole": "Fast",     "Cost_Cr": 14.0, "Is_Foreign": 1,
         "Batting Runs": 10, "Batting Average": 5.0, "Batting Strike Rate": 80.0,
         "Wickets": 25, "Economy Rate": 7.0, "Bowling Average": 20.0},
        {"Player Name": "Bowler B", "Role": "Bowler", "Subrole": "Medium",   "Cost_Cr": 11.0, "Is_Foreign": 0,
         "Batting Runs": 8, "Batting Average": 4.0, "Batting Strike Rate": 75.0,
         "Wickets": 22, "Economy Rate": 7.5, "Bowling Average": 22.0},
        {"Player Name": "Bowler C", "Role": "Bowler", "Subrole": "Pace",     "Cost_Cr": 9.0, "Is_Foreign": 0,
         "Batting Runs": 5, "Batting Average": 3.0, "Batting Strike Rate": 70.0,
         "Wickets": 20, "Economy Rate": 8.0, "Bowling Average": 24.0},
        {"Player Name": "Bowler D", "Role": "Bowler", "Subrole": "Spin",     "Cost_Cr": 8.0, "Is_Foreign": 0,
         "Batting Runs": 3, "Batting Average": 2.0, "Batting Strike Rate": 65.0,
         "Wickets": 18, "Economy Rate": 8.5, "Bowling Average": 26.0},
        {"Player Name": "Bowler E", "Role": "Bowler", "Subrole": "Spin",     "Cost_Cr": 6.0, "Is_Foreign": 0,
         "Batting Runs": 2, "Batting Average": 1.5, "Batting Strike Rate": 60.0,
         "Wickets": 15, "Economy Rate": 9.0, "Bowling Average": 28.0},
        {"Player Name": "Bowler F", "Role": "Bowler", "Subrole": "Pace",     "Cost_Cr": 5.0, "Is_Foreign": 0,
         "Batting Runs": 1, "Batting Average": 1.0, "Batting Strike Rate": 55.0,
         "Wickets": 12, "Economy Rate": 9.5, "Bowling Average": 30.0},
    ]
    return pd.DataFrame(players)


# ─────────────────────────────────────────────
# calculate_fantasy tests
# ─────────────────────────────────────────────

class TestCalculateFantasy:
    def test_returns_dataframe(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert isinstance(result, pd.DataFrame)

    def test_fantasy_score_column_created(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert 'Fantasy_Score' in result.columns

    def test_value_score_column_created(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert 'Value_Score' in result.columns

    def test_fantasy_scores_are_non_negative(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert (result['Fantasy_Score'] >= 0).all(), "Fantasy scores must be non-negative"

    def test_value_scores_are_non_negative(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert (result['Value_Score'] >= 0).all(), "Value scores must be non-negative"

    def test_value_score_equals_fantasy_over_cost(self, minimal_player_pool):
        """Value Score should equal Fantasy Score / Cost_Cr for players with Cost > 0."""
        result = calculate_fantasy(minimal_player_pool.copy())
        for _, row in result[result['Cost_Cr'] > 0].iterrows():
            expected = row['Fantasy_Score'] / row['Cost_Cr']
            assert abs(row['Value_Score'] - expected) < 1e-9, (
                f"Value Score mismatch for {row['Player Name']}: "
                f"expected {expected:.6f}, got {row['Value_Score']:.6f}"
            )

    def test_batting_heavy_boosts_batters(self, minimal_player_pool):
        """Pure batters should score higher under Batting Heavy than Bowling Heavy."""
        df_bat = calculate_fantasy(minimal_player_pool.copy(), strategy="Batting Heavy")
        df_bowl = calculate_fantasy(minimal_player_pool.copy(), strategy="Bowling Heavy")
        top_batter = "Batter A"
        assert df_bat[df_bat['Player Name'] == top_batter]['Fantasy_Score'].iloc[0] > \
               df_bowl[df_bowl['Player Name'] == top_batter]['Fantasy_Score'].iloc[0]

    def test_bowling_heavy_boosts_bowlers(self, minimal_player_pool):
        """Pure bowlers should score higher under Bowling Heavy than Batting Heavy."""
        df_bat = calculate_fantasy(minimal_player_pool.copy(), strategy="Batting Heavy")
        df_bowl = calculate_fantasy(minimal_player_pool.copy(), strategy="Bowling Heavy")
        top_bowler = "Bowler A"
        assert df_bowl[df_bowl['Player Name'] == top_bowler]['Fantasy_Score'].iloc[0] > \
               df_bat[df_bat['Player Name'] == top_bowler]['Fantasy_Score'].iloc[0]

    def test_no_nan_in_fantasy_scores(self, minimal_player_pool):
        result = calculate_fantasy(minimal_player_pool.copy())
        assert not result['Fantasy_Score'].isna().any(), "Fantasy Score must not contain NaN"

    def test_normalised_metrics_bounded_0_1(self, minimal_player_pool):
        """All norm_ columns must stay within [0, 1]."""
        result = calculate_fantasy(minimal_player_pool.copy())
        norm_cols = [c for c in result.columns if c.startswith('norm_')]
        for col in norm_cols:
            assert result[col].between(0, 1).all(), f"Column {col} has values outside [0, 1]"


# ─────────────────────────────────────────────
# optimize_team tests
# ─────────────────────────────────────────────

class TestOptimizeTeam:
    def test_returns_11_players(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, score = optimize_team(scored, budget=120.0)
        assert team is not None
        assert len(team) == 11

    def test_total_score_is_positive(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        _, score = optimize_team(scored, budget=120.0)
        assert score > 0

    def test_budget_constraint_respected(self, minimal_player_pool):
        budget = 80.0
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, _ = optimize_team(scored, budget=budget)
        assert team is not None
        assert team['Cost_Cr'].sum() <= budget + 1e-6, "Team exceeds budget"

    def test_foreigner_limit_respected(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, _ = optimize_team(scored, budget=120.0, max_foreigners=4)
        assert team is not None
        assert team['Is_Foreign'].sum() <= 4

    def test_must_include_honoured(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, _ = optimize_team(scored, budget=120.0, must_include=["Batter A"])
        assert team is not None
        assert "Batter A" in team['Player Name'].values

    def test_must_exclude_honoured(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, _ = optimize_team(scored, budget=120.0, must_exclude=["Bowler A"])
        assert team is not None
        assert "Bowler A" not in team['Player Name'].values

    def test_infeasible_budget_returns_none(self, minimal_player_pool):
        """A budget of 1 Cr is too small to pick 11 players — should return None gracefully."""
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, score = optimize_team(scored, budget=1.0)
        assert team is None
        assert score == 0

    def test_value_score_in_output(self, minimal_player_pool):
        """Output team DataFrame must include the Value_Score column."""
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, _ = optimize_team(scored, budget=120.0)
        assert team is not None
        assert 'Value_Score' in team.columns

    def test_strict_mode_optimization(self, minimal_player_pool):
        """Strict constraint mode should correctly respect exact counts and pace/spin constraints."""
        scored = calculate_fantasy(minimal_player_pool.copy())
        team, score = optimize_team(
            scored,
            budget=120.0,
            constraint_mode="Strict (Custom Roles)",
            num_batters=5,
            num_bowlers=4,
            num_all_rounders=2,
            num_pacers=3,
            num_spinners=2
        )
        assert team is not None
        assert len(team) == 11
        assert (team['Role'] == 'Batsman').sum() == 5
        assert (team['Role'] == 'Bowler').sum() == 4
        assert (team['Role'] == 'All-Rounder').sum() == 2


# ─────────────────────────────────────────────
# get_efficient_frontier tests
# ─────────────────────────────────────────────

class TestGetEfficientFrontier:
    def test_returns_dataframe(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        frontier = get_efficient_frontier(scored)
        assert isinstance(frontier, pd.DataFrame)

    def test_frontier_is_non_empty(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        frontier = get_efficient_frontier(scored)
        assert len(frontier) > 0

    def test_frontier_subset_of_original(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        frontier = get_efficient_frontier(scored)
        assert set(frontier['Player Name']).issubset(set(scored['Player Name']))

    def test_frontier_sorted_by_cost(self, minimal_player_pool):
        scored = calculate_fantasy(minimal_player_pool.copy())
        frontier = get_efficient_frontier(scored)
        assert list(frontier['Cost_Cr']) == sorted(frontier['Cost_Cr'].tolist())

    def test_no_dominated_player_in_frontier(self, minimal_player_pool):
        """
        For every pair (A, B) in the frontier, no player B should have both
        higher/equal Fantasy Score AND lower/equal Cost vs player A (i.e., no player dominates another).
        """
        scored = calculate_fantasy(minimal_player_pool.copy())
        frontier = get_efficient_frontier(scored)
        rows = frontier.to_dict('records')
        for i, a in enumerate(rows):
            for j, b in enumerate(rows):
                if i == j:
                    continue
                if b['Fantasy_Score'] >= a['Fantasy_Score'] and b['Cost_Cr'] <= a['Cost_Cr']:
                    pytest.fail(
                        f"Player '{b['Player Name']}' dominates '{a['Player Name']}' "
                        f"but both are on the frontier."
                    )
