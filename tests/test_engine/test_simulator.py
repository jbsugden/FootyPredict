"""Tests for the Monte Carlo simulator."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from predictor.engine.poisson import MatchRecord, StrengthCalculator
from predictor.engine.simulator import (
    Fixture,
    MonteCarloSimulator,
    SimulationInput,
    TeamPrediction,
)
from predictor.engine.standings import initialise_standings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_sim_input(
    teams: list[str],
    fixtures: list[tuple[str, str]],
    historical: list[tuple[str, str, int, int]] | None = None,
    n_games_played: int = 5,
) -> SimulationInput:
    """Build a minimal SimulationInput for testing.

    Args:
        teams: Team IDs.
        fixtures: Remaining fixtures as (home_id, away_id).
        historical: Historical results as (home, away, hg, ag). Defaults to
            equal-strength round-robin results.
        n_games_played: How many dummy historical matches to generate if
            ``historical`` is None.

    Returns:
        :class:`SimulationInput` ready for the simulator.
    """
    if historical is None:
        # Generate balanced historical data so all teams have equal strength
        historical = []
        for i in range(n_games_played):
            for j, home in enumerate(teams):
                away = teams[(j + 1) % len(teams)]
                historical.append((home, away, 1, 1))

    records = [
        MatchRecord(
            home_team_id=h,
            away_team_id=a,
            home_goals=hg,
            away_goals=ag,
            played_at=_now() - timedelta(days=14),
        )
        for h, a, hg, ag in historical
    ]
    strength_calc = StrengthCalculator(records)

    standings = initialise_standings(teams)
    # Give teams some existing points so it's not a blank slate
    for i, tid in enumerate(teams):
        standings[tid].points = (len(teams) - i) * 3
        standings[tid].played = 5
        standings[tid].won = len(teams) - i
        standings[tid].goals_for = (len(teams) - i) * 2
        standings[tid].goals_against = i + 1

    return SimulationInput(
        team_ids=teams,
        current_standings=standings,
        remaining_fixtures=[Fixture(h, a) for h, a in fixtures],
        strength_calculator=strength_calc,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMonteCarloSimulator:
    TEAMS = ["A", "B", "C", "D"]
    FIXTURES = [("A", "B"), ("C", "D"), ("B", "C"), ("D", "A")]

    def test_returns_entry_for_every_team(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        assert set(results.keys()) == set(self.TEAMS)

    def test_mean_pos_is_in_valid_range(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        n = len(self.TEAMS)
        for pred in results.values():
            assert 1.0 <= pred.mean_pos <= n

    def test_pos_dist_length_equals_team_count(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        n = len(self.TEAMS)
        for pred in results.values():
            assert len(pred.pos_dist) == n

    def test_pos_dist_sums_to_approximately_one(self) -> None:
        sim = MonteCarloSimulator(n_simulations=500, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        for pred in results.values():
            assert sum(pred.pos_dist) == pytest.approx(1.0, abs=0.01)

    def test_pos_dist_probabilities_are_non_negative(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        for pred in results.values():
            assert all(p >= 0 for p in pred.pos_dist)

    def test_mean_points_is_positive(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        results = sim.run(inp)
        for pred in results.values():
            assert pred.mean_points >= 0

    def test_no_remaining_fixtures_preserves_current_standings(self) -> None:
        """With no fixtures left, positions should reflect current standings deterministically."""
        sim = MonteCarloSimulator(n_simulations=200, rng_seed=0)
        inp = _make_sim_input(self.TEAMS, fixtures=[])
        results = sim.run(inp)
        # Team A has most points (set up in _make_sim_input), so mean_pos should be ~1
        assert results["A"].mean_pos == pytest.approx(1.0, abs=0.5)

    def test_results_to_dict_produces_correct_structure(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        inp = _make_sim_input(self.TEAMS, self.FIXTURES)
        predictions = sim.run(inp)
        serialised = sim.results_to_dict(predictions)
        for team_id, data in serialised.items():
            assert "mean_pos" in data
            assert "mean_points" in data
            assert "pos_dist" in data
            assert isinstance(data["pos_dist"], list)

    def test_seed_produces_reproducible_results(self) -> None:
        inp1 = _make_sim_input(self.TEAMS, self.FIXTURES)
        inp2 = _make_sim_input(self.TEAMS, self.FIXTURES)
        sim1 = MonteCarloSimulator(n_simulations=200, rng_seed=99)
        sim2 = MonteCarloSimulator(n_simulations=200, rng_seed=99)
        r1 = sim1.run(inp1)
        r2 = sim2.run(inp2)
        for tid in self.TEAMS:
            assert r1[tid].mean_pos == r2[tid].mean_pos
