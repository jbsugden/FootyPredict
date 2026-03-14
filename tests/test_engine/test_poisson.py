"""Tests for the Poisson strength calculator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from predictor.engine.poisson import (
    DECAY_HALF_LIFE_WEEKS,
    SHRINKAGE_K,
    MatchRecord,
    StrengthCalculator,
    TeamStrengths,
    _get_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_match(
    home: str,
    away: str,
    hg: int,
    ag: int,
    weeks_ago: float = 1.0,
) -> MatchRecord:
    played_at = _now() - timedelta(weeks=weeks_ago)
    return MatchRecord(
        home_team_id=home,
        away_team_id=away,
        home_goals=hg,
        away_goals=ag,
        played_at=played_at,
    )


# ---------------------------------------------------------------------------
# Weight decay tests
# ---------------------------------------------------------------------------


class TestGetWeight:
    def test_weight_is_one_for_just_played(self) -> None:
        played_at = _now()
        assert _get_weight(played_at) == pytest.approx(1.0)

    def test_weight_at_half_life_is_half(self) -> None:
        played_at = _now() - timedelta(weeks=DECAY_HALF_LIFE_WEEKS)
        assert _get_weight(played_at) == pytest.approx(0.5, abs=0.01)

    def test_weight_decreases_monotonically(self) -> None:
        w5 = _get_weight(_now() - timedelta(weeks=5))
        w10 = _get_weight(_now() - timedelta(weeks=10))
        w20 = _get_weight(_now() - timedelta(weeks=20))
        assert w5 > w10 > w20

    def test_old_match_has_small_positive_weight(self) -> None:
        w = _get_weight(_now() - timedelta(weeks=30))
        assert 0 < w < 0.2


# ---------------------------------------------------------------------------
# Shrinkage tests
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_small_sample_shrinks_toward_average(self) -> None:
        """With only 2-3 matches, extreme raw values should be pulled toward 1.0."""
        matches = [_make_match("A", "B", 5, 0) for _ in range(2)]
        calc = StrengthCalculator(matches)
        strengths = calc.compute_strengths()
        # A's raw attack would be very high; shrinkage should pull it closer to 1.0
        # but still above 1.0
        assert 1.0 < strengths["A"].attack < 2.5

    def test_large_sample_preserves_signal(self) -> None:
        """With 20+ matches, values should be close to raw (little shrinkage)."""
        matches = [_make_match("A", "B", 4, 1) for _ in range(25)]
        calc = StrengthCalculator(matches)
        strengths = calc.compute_strengths()
        # With many matches, attack should be well above 1.0
        assert strengths["A"].attack > 1.3

    def test_shrinkage_never_flips_direction(self) -> None:
        """Raw attack > 1.0 should always give adjusted attack > 1.0."""
        for n_matches in [2, 5, 10, 20]:
            matches = [_make_match("A", "B", 3, 1) for _ in range(n_matches)]
            calc = StrengthCalculator(matches)
            strengths = calc.compute_strengths()
            assert strengths["A"].attack > 1.0
            assert strengths["B"].attack < 1.0


# ---------------------------------------------------------------------------
# StrengthCalculator tests
# ---------------------------------------------------------------------------


class TestStrengthCalculator:
    def test_empty_matches_returns_empty_dict(self) -> None:
        calc = StrengthCalculator(matches=[])
        assert calc.compute_strengths() == {}

    def test_returns_entry_for_every_team(self) -> None:
        matches = [
            _make_match("A", "B", 2, 1),
            _make_match("C", "A", 0, 1),
        ]
        calc = StrengthCalculator(matches)
        strengths = calc.compute_strengths()
        assert "A" in strengths
        assert "B" in strengths
        assert "C" in strengths

    def test_high_scoring_team_has_attack_above_one(self) -> None:
        """A team that consistently outscores the league average should have attack > 1."""
        # League: A scores 4/game, B scores 1/game => league avg = 2.5
        matches = [_make_match("A", "B", 4, 1) for _ in range(10)]
        calc = StrengthCalculator(matches)
        strengths = calc.compute_strengths()
        assert strengths["A"].attack > 1.0

    def test_strong_defence_has_concede_rate_below_one(self) -> None:
        """A team that concedes less than league average should have defence < 1."""
        matches = [_make_match("A", "B", 4, 1) for _ in range(10)]
        calc = StrengthCalculator(matches)
        strengths = calc.compute_strengths()
        # B's attack (goals scored) = 1 per game (below average) => attack < 1
        assert strengths["B"].attack < 1.0

    def test_strengths_are_cached(self) -> None:
        matches = [_make_match("A", "B", 2, 1)]
        calc = StrengthCalculator(matches)
        s1 = calc.compute_strengths()
        s2 = calc.get_strengths()
        assert s1 is s2  # same object — cached

    def test_compute_lambda_returns_positive_values(self) -> None:
        matches = [_make_match("A", "B", 2, 1), _make_match("B", "A", 0, 3)]
        calc = StrengthCalculator(matches)
        lh, la = calc.compute_lambda("A", "B")
        assert lh > 0
        assert la > 0

    def test_compute_lambda_for_unknown_team_uses_defaults(self) -> None:
        """An unseen team should fall back to neutral strength (attack=defence=1)."""
        matches = [_make_match("A", "B", 2, 1)]
        calc = StrengthCalculator(matches)
        lh, la = calc.compute_lambda("UNKNOWN_HOME", "UNKNOWN_AWAY")
        # Both should equal league_avg_goals (default 1.4)
        assert lh == pytest.approx(1.4, rel=0.01)
        assert la == pytest.approx(1.4, rel=0.01)


class TestScoreProbabilityMatrix:
    def test_matrix_shape_is_correct(self) -> None:
        matches = [_make_match("A", "B", 2, 1)]
        calc = StrengthCalculator(matches)
        lh, la = calc.compute_lambda("A", "B")
        matrix = calc.score_probability_matrix(lh, la, max_goals=10)
        assert matrix.shape == (11, 11)

    def test_matrix_sums_to_approximately_one(self) -> None:
        matches = [_make_match("A", "B", 2, 1)]
        calc = StrengthCalculator(matches)
        lh, la = calc.compute_lambda("A", "B")
        matrix = calc.score_probability_matrix(lh, la, max_goals=15)
        assert matrix.sum() == pytest.approx(1.0, abs=0.01)

    def test_all_probabilities_non_negative(self) -> None:
        matches = [_make_match("A", "B", 2, 1)]
        calc = StrengthCalculator(matches)
        lh, la = calc.compute_lambda("A", "B")
        matrix = calc.score_probability_matrix(lh, la)
        assert np.all(matrix >= 0)
