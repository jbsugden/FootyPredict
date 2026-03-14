"""Tests for head-to-head lambda adjustment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from predictor.engine.head_to_head import (
    H2H_K,
    H2H_MAX_WEIGHT,
    adjust_lambdas_h2h,
)
from predictor.engine.poisson import MatchRecord


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
# Tests
# ---------------------------------------------------------------------------


class TestNoAdjustment:
    """Cases where base lambdas should be returned unchanged."""

    def test_no_h2h_matches_returns_base_lambdas(self) -> None:
        """Unrelated matches → no change."""
        matches = [
            _make_match("X", "Y", 2, 1),
            _make_match("Y", "X", 0, 0),
        ]
        result = adjust_lambdas_h2h(1.5, 1.2, matches, "A", "B")
        assert result == (1.5, 1.2)

    def test_few_h2h_matches_returns_base_lambdas(self) -> None:
        """1 match → below minimum, no change."""
        matches = [_make_match("A", "B", 3, 0)]
        result = adjust_lambdas_h2h(1.5, 1.2, matches, "A", "B")
        assert result == (1.5, 1.2)

    def test_empty_matches_returns_base_lambdas(self) -> None:
        result = adjust_lambdas_h2h(1.5, 1.2, [], "A", "B")
        assert result == (1.5, 1.2)


class TestAdjustment:
    """Cases where h2h data should nudge the lambdas."""

    def test_h2h_adjustment_nudges_toward_history(self) -> None:
        """If h2h avg is higher than base lambda, adjusted lambda should increase."""
        # A scores 4 goals per game against B at home
        matches = [
            _make_match("A", "B", 4, 0, weeks_ago=2),
            _make_match("A", "B", 4, 0, weeks_ago=4),
        ]
        base_home, base_away = 1.5, 1.2
        adj_home, adj_away = adjust_lambdas_h2h(
            base_home, base_away, matches, "A", "B"
        )
        # Home lambda should increase (h2h avg ~4 > base 1.5)
        assert adj_home > base_home
        # Away lambda should decrease (h2h avg ~0 < base 1.2)
        assert adj_away < base_away

    def test_h2h_weight_increases_with_more_matches(self) -> None:
        """More h2h data → larger adjustment."""
        few = [
            _make_match("A", "B", 4, 0, weeks_ago=2),
            _make_match("A", "B", 4, 0, weeks_ago=4),
        ]
        many = few + [
            _make_match("A", "B", 4, 0, weeks_ago=6),
            _make_match("A", "B", 4, 0, weeks_ago=8),
            _make_match("A", "B", 4, 0, weeks_ago=10),
            _make_match("A", "B", 4, 0, weeks_ago=12),
        ]
        base = 1.5
        adj_few, _ = adjust_lambdas_h2h(base, 1.2, few, "A", "B")
        adj_many, _ = adjust_lambdas_h2h(base, 1.2, many, "A", "B")
        # Both should be above base, but many-match version further
        assert adj_few > base
        assert adj_many > adj_few

    def test_h2h_respects_home_away_perspective(self) -> None:
        """Goals attributed correctly when home/away is reversed."""
        # In the historical match B was home and scored 5, A scored 0.
        # For the upcoming fixture A is home, B is away.
        # So from A's perspective: A scored 0 (away in that match) → h2h home avg = 0
        # B scored 5 (home in that match) → h2h away avg = 5
        matches = [
            _make_match("B", "A", 5, 0, weeks_ago=2),
            _make_match("B", "A", 5, 0, weeks_ago=4),
        ]
        base_home, base_away = 1.5, 1.2
        adj_home, adj_away = adjust_lambdas_h2h(
            base_home, base_away, matches, "A", "B"
        )
        # A's home lambda should decrease (h2h home avg ~0 < 1.5)
        assert adj_home < base_home
        # B's away lambda should increase (h2h away avg ~5 > 1.2)
        assert adj_away > base_away

    def test_adjusted_lambdas_always_positive(self) -> None:
        """Floor at 0.01 even when h2h history is all zeros."""
        matches = [
            _make_match("A", "B", 0, 0, weeks_ago=2),
            _make_match("A", "B", 0, 0, weeks_ago=4),
        ]
        adj_home, adj_away = adjust_lambdas_h2h(0.05, 0.05, matches, "A", "B")
        assert adj_home >= 0.01
        assert adj_away >= 0.01


class TestBlendingWeight:
    """Verify the shrinkage formula produces expected weights."""

    @pytest.mark.parametrize(
        "n, expected_w",
        [
            (2, H2H_MAX_WEIGHT * 2 / (2 + H2H_K)),
            (4, H2H_MAX_WEIGHT * 4 / (4 + H2H_K)),
            (6, H2H_MAX_WEIGHT * 6 / (6 + H2H_K)),
            (10, H2H_MAX_WEIGHT * 10 / (10 + H2H_K)),
        ],
    )
    def test_weight_formula(self, n: int, expected_w: float) -> None:
        """Verify weight matches the documented formula."""
        # With n identical h2h matches and same time-decay, the blend weight
        # should match the formula from the plan.
        matches = [_make_match("A", "B", 3, 1, weeks_ago=1) for _ in range(n)]
        # h2h avg should be ~3 for home, ~1 for away
        base_home = 1.5
        adj_home, _ = adjust_lambdas_h2h(base_home, 1.2, matches, "A", "B")

        # adj = (1-w)*base + w*h2h_avg → w = (adj - base) / (h2h_avg - base)
        h2h_avg_home = 3.0  # all matches are 3-1
        actual_w = (adj_home - base_home) / (h2h_avg_home - base_home)
        assert abs(actual_w - expected_w) < 0.001
