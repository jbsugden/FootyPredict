"""Tests for the Elo rating calculator.

Uses known mathematical outcomes to verify correctness of the core functions.
"""

from __future__ import annotations

import pytest

from predictor.engine.elo import (
    DEFAULT_INITIAL_RATING,
    HOME_ADVANTAGE_ELO,
    K_NON_LEAGUE,
    K_PREMIER_LEAGUE,
    EloCalculator,
)


class TestExpectedScore:
    """Tests for EloCalculator.expected_score."""

    def test_equal_ratings_no_home_advantage_is_fifty_fifty(self) -> None:
        """Two teams with identical ratings should each have 0.5 expected score."""
        calc = EloCalculator(home_advantage=0.0)
        result = calc.expected_score(1500, 1500, a_is_home=False)
        assert abs(result - 0.5) < 1e-9

    def test_equal_ratings_with_home_advantage_favours_home(self) -> None:
        """Home team should have > 0.5 expected score when ratings are equal."""
        calc = EloCalculator(home_advantage=HOME_ADVANTAGE_ELO)
        p_home = calc.expected_score(1500, 1500, a_is_home=True)
        assert p_home > 0.5

    def test_higher_rated_team_has_higher_expected_score(self) -> None:
        """A team rated 200 points higher should have a significantly higher E."""
        calc = EloCalculator(home_advantage=0.0)
        p_strong = calc.expected_score(1700, 1500)
        p_weak = calc.expected_score(1500, 1700)
        assert p_strong > p_weak
        assert p_strong + p_weak == pytest.approx(1.0, abs=1e-9)

    def test_known_value_400_point_gap(self) -> None:
        """A 400 Elo gap corresponds to expected score of 10/11 ≈ 0.909."""
        calc = EloCalculator(home_advantage=0.0)
        result = calc.expected_score(1900, 1500)
        assert result == pytest.approx(10 / 11, rel=1e-6)


class TestActualScore:
    """Tests for EloCalculator.actual_score."""

    def test_home_win_returns_one_zero(self) -> None:
        calc = EloCalculator()
        assert calc.actual_score(3, 1) == (1.0, 0.0)

    def test_away_win_returns_zero_one(self) -> None:
        calc = EloCalculator()
        assert calc.actual_score(0, 2) == (0.0, 1.0)

    def test_draw_returns_half_half(self) -> None:
        calc = EloCalculator()
        assert calc.actual_score(1, 1) == (0.5, 0.5)

    def test_zero_zero_draw(self) -> None:
        calc = EloCalculator()
        assert calc.actual_score(0, 0) == (0.5, 0.5)


class TestUpdateRatings:
    """Tests for EloCalculator.update_ratings."""

    def test_home_win_increases_home_rating(self) -> None:
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        new_home, new_away = calc.update_ratings(1500, 1500, 2, 0)
        assert new_home > 1500
        assert new_away < 1500

    def test_away_win_increases_away_rating(self) -> None:
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        new_home, new_away = calc.update_ratings(1500, 1500, 0, 1)
        assert new_home < 1500
        assert new_away > 1500

    def test_draw_between_equal_teams_preserves_ratings(self) -> None:
        """A draw between equal teams (no home advantage) should not change ratings."""
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        new_home, new_away = calc.update_ratings(1500, 1500, 1, 1)
        assert new_home == pytest.approx(1500, abs=1e-6)
        assert new_away == pytest.approx(1500, abs=1e-6)

    def test_ratings_sum_is_conserved(self) -> None:
        """Total Elo should be conserved (zero-sum property)."""
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        for home_g, away_g in [(3, 0), (0, 2), (1, 1)]:
            new_h, new_a = calc.update_ratings(1500, 1600, home_g, away_g)
            assert new_h + new_a == pytest.approx(1500 + 1600, abs=1e-6)

    def test_upset_win_gives_more_points_than_expected_win(self) -> None:
        """Beating a much stronger side should yield more Elo gain than beating a weaker one."""
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        # Upset: weak (1200) beats strong (1800) — weak team is home, wins 1-0
        upset_new_home, _ = calc.update_ratings(1200, 1800, 1, 0)
        upset_gain = upset_new_home - 1200  # gain for the weak team

        # Expected: strong (1800) beats weak (1200) — strong team is home, wins 1-0
        expected_new_home, _ = calc.update_ratings(1800, 1200, 1, 0)
        expected_gain = expected_new_home - 1800  # gain for the strong team

        # The weaker team gains more Elo from an upset than the strong team gains
        # from an expected result
        assert upset_gain > expected_gain

    def test_k_factor_scales_update_magnitude(self) -> None:
        """Higher K-factor should produce larger rating changes."""
        calc_low = EloCalculator(k_factor=10, home_advantage=0.0)
        calc_high = EloCalculator(k_factor=40, home_advantage=0.0)
        new_h_low, _ = calc_low.update_ratings(1500, 1500, 1, 0)
        new_h_high, _ = calc_high.update_ratings(1500, 1500, 1, 0)
        assert (new_h_high - 1500) > (new_h_low - 1500)


class TestBulkUpdate:
    """Tests for EloCalculator.bulk_update."""

    def test_bulk_update_processes_all_results(self) -> None:
        calc = EloCalculator(k_factor=32, home_advantage=0.0)
        ratings: dict[str, float] = {}
        results = [
            ("team_a", "team_b", 2, 1),
            ("team_a", "team_c", 0, 0),
            ("team_b", "team_c", 3, 0),
        ]
        updated = calc.bulk_update(ratings, results)
        # All three teams should have entries
        assert "team_a" in updated
        assert "team_b" in updated
        assert "team_c" in updated

    def test_missing_teams_are_initialised_to_default(self) -> None:
        calc = EloCalculator()
        updated = calc.bulk_update({}, [("x", "y", 1, 0)])
        assert "x" in updated
        assert "y" in updated

    def test_original_dict_is_not_mutated(self) -> None:
        calc = EloCalculator()
        original: dict[str, float] = {"a": 1500.0, "b": 1500.0}
        calc.bulk_update(original, [("a", "b", 2, 0)])
        assert original["a"] == 1500.0
        assert original["b"] == 1500.0
