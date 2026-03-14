"""Tests for the per-fixture prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from predictor.engine.fixture_prediction import predict_fixtures


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class _FakeMatch:
    """Minimal match-like object for testing (mirrors Match ORM fields used)."""

    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    played_at: datetime
    status: str = "FINISHED"


@dataclass
class _FakeTeam:
    """Minimal team-like object."""

    id: str
    name: str
    short_name: str | None = None


class TestPredictFixtures:
    """Tests for predict_fixtures()."""

    def _make_data(self):
        """Build a small set of finished and scheduled matches."""
        team_map = {
            "A": _FakeTeam(id="A", name="Team Alpha"),
            "B": _FakeTeam(id="B", name="Team Bravo"),
            "C": _FakeTeam(id="C", name="Team Charlie"),
        }

        finished = [
            _FakeMatch("A", "B", 2, 1, _now()),
            _FakeMatch("B", "C", 0, 0, _now()),
            _FakeMatch("C", "A", 1, 3, _now()),
            _FakeMatch("A", "C", 1, 0, _now()),
            _FakeMatch("B", "A", 1, 2, _now()),
            _FakeMatch("C", "B", 2, 1, _now()),
        ]

        scheduled = [
            _FakeMatch("A", "C", 0, 0, _now(), status="SCHEDULED"),
            _FakeMatch("B", "A", 0, 0, _now(), status="SCHEDULED"),
        ]

        return finished, scheduled, team_map

    def test_lambda_values_positive(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        for f in result.fixtures:
            assert f.lambda_team > 0, "Lambda team should be positive"
            assert f.lambda_opponent > 0, "Lambda opponent should be positive"

    def test_probabilities_sum_to_one(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        for f in result.fixtures:
            total = f.p_win + f.p_draw + f.p_loss
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_score_matrix_is_6x6(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        for f in result.fixtures:
            assert len(f.score_matrix) == 6
            for row in f.score_matrix:
                assert len(row) == 6
                for val in row:
                    assert val >= 0, "Score matrix values should be non-negative"

    def test_top_scores_sorted_descending(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        for f in result.fixtures:
            assert len(f.top_scores) > 0
            probs = [p for _, _, p in f.top_scores]
            for i in range(len(probs) - 1):
                assert probs[i] >= probs[i + 1], "Top scores should be sorted by probability"

    def test_expected_remaining_points_in_range(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        n_fixtures = len(result.fixtures)
        assert result.expected_remaining_points >= 0
        assert result.expected_remaining_points <= 3 * n_fixtures

    def test_empty_scheduled_returns_empty_fixtures(self):
        finished, _, team_map = self._make_data()
        result = predict_fixtures(finished, [], "A", team_map)

        assert result.fixtures == []
        assert result.expected_remaining_points == 0.0

    def test_team_not_in_scheduled(self):
        """Team exists in finished but has no scheduled matches."""
        finished, _, team_map = self._make_data()
        # Pass scheduled matches that don't involve team C
        scheduled_no_c = [
            _FakeMatch("A", "B", 0, 0, _now(), status="SCHEDULED"),
        ]
        result = predict_fixtures(finished, scheduled_no_c, "C", team_map)

        assert result.fixtures == []
        assert result.expected_remaining_points == 0.0

    def test_home_away_correctness(self):
        """Verify is_home flag and opponent assignment."""
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        # First scheduled: A vs C (A is home)
        assert result.fixtures[0].is_home is True
        assert result.fixtures[0].opponent_id == "C"

        # Second scheduled: B vs A (A is away)
        assert result.fixtures[1].is_home is False
        assert result.fixtures[1].opponent_id == "B"

    def test_strength_profile_populated(self):
        finished, scheduled, team_map = self._make_data()
        result = predict_fixtures(finished, scheduled, "A", team_map)

        assert result.team_strength.attack > 0
        assert result.team_strength.defence > 0
        assert result.league_avg_goals > 0
