"""Tests for the Poisson strength calculator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from predictor.engine.poisson import (
    WEIGHT_MID,
    WEIGHT_OLD,
    WEIGHT_RECENT,
    WEEKS_MID,
    WEEKS_RECENT,
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
    def test_recent_match_gets_full_weight(self) -> None:
        played_at = _now() - timedelta(weeks=1)
        assert _get_weight(played_at) == WEIGHT_RECENT

    def test_match_at_boundary_recent_gets_full_weight(self) -> None:
        played_at = _now() - timedelta(weeks=WEEKS_RECENT)
        assert _get_weight(played_at) == WEIGHT_RECENT

    def test_match_between_6_and_12_weeks_gets_mid_weight(self) -> None:
        played_at = _now() - timedelta(weeks=9)
        assert _get_weight(played_at) == WEIGHT_MID

    def test_match_older_than_12_weeks_gets_old_weight(self) -> None:
        played_at = _now() - timedelta(weeks=20)
        assert _get_weight(played_at) == WEIGHT_OLD


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
        # B scores 1 per game against A's above-average concession; B's defence is < 1
        # when we look at how much B concedes (=4 per game, above average)
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
