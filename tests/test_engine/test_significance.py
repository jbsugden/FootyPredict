"""Tests for match significance scoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from predictor.engine.poisson import MatchRecord, StrengthCalculator
from predictor.engine.significance import (
    FixtureSignificance,
    compute_fixture_significance,
    rank_fixtures_by_significance,
    _outcome_probabilities,
)
from predictor.engine.simulator import (
    Fixture,
    MonteCarloSimulator,
    SimulationInput,
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
) -> SimulationInput:
    """Build a minimal SimulationInput for significance tests."""
    historical = []
    for i in range(5):
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

TEAMS = ["A", "B", "C", "D"]
FIXTURES = [("A", "B"), ("C", "D"), ("B", "C"), ("D", "A")]
TEAM_NAMES = {"A": "Team A", "B": "Team B", "C": "Team C", "D": "Team D"}


class TestOutcomeProbabilities:
    def test_probabilities_sum_to_approximately_one(self) -> None:
        p_hw, p_d, p_aw = _outcome_probabilities(1.5, 1.2)
        assert p_hw + p_d + p_aw == pytest.approx(1.0, abs=0.01)

    def test_equal_lambdas_give_symmetric_win_probs(self) -> None:
        p_hw, p_d, p_aw = _outcome_probabilities(1.5, 1.5)
        # Home/away win probs should be equal with equal lambdas
        assert p_hw == pytest.approx(p_aw, abs=0.01)

    def test_higher_home_lambda_gives_higher_home_win(self) -> None:
        p_hw, _, p_aw = _outcome_probabilities(2.5, 0.8)
        assert p_hw > p_aw


class TestComputeFixtureSignificance:
    def test_returns_correct_structure(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        sig = compute_fixture_significance(
            inp, 0, baseline, TEAM_NAMES, n_simulations=50, rng_seed=42,
        )
        assert isinstance(sig, FixtureSignificance)
        assert sig.home_id == "A"
        assert sig.away_id == "B"
        assert sig.home_name == "Team A"
        assert sig.away_name == "Team B"

    def test_significance_score_non_negative(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        sig = compute_fixture_significance(
            inp, 0, baseline, TEAM_NAMES, n_simulations=50, rng_seed=42,
        )
        assert sig.significance_score >= 0
        assert sig.shift_home_win >= 0
        assert sig.shift_draw >= 0
        assert sig.shift_away_win >= 0

    def test_shifts_are_finite(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        sig = compute_fixture_significance(
            inp, 0, baseline, TEAM_NAMES, n_simulations=50, rng_seed=42,
        )
        import math
        assert math.isfinite(sig.significance_score)
        assert math.isfinite(sig.shift_home_win)
        assert math.isfinite(sig.shift_draw)
        assert math.isfinite(sig.shift_away_win)


class TestRankFixturesBySignificance:
    def test_returns_sorted_descending(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        ranked = rank_fixtures_by_significance(
            inp, baseline, TEAM_NAMES,
            n_simulations=50, max_fixtures=4, rng_seed=42,
        )
        scores = [r.significance_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_max_fixtures_caps_output(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        ranked = rank_fixtures_by_significance(
            inp, baseline, TEAM_NAMES,
            n_simulations=50, max_fixtures=2, rng_seed=42,
        )
        assert len(ranked) == 2

    def test_empty_fixtures_returns_empty(self) -> None:
        inp = _make_sim_input(TEAMS, [])
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        ranked = rank_fixtures_by_significance(
            inp, baseline, TEAM_NAMES,
            n_simulations=50, rng_seed=42,
        )
        assert ranked == []

    def test_all_results_have_team_names(self) -> None:
        inp = _make_sim_input(TEAMS, FIXTURES)
        sim = MonteCarloSimulator(n_simulations=100, rng_seed=42)
        baseline = sim.run(inp)

        ranked = rank_fixtures_by_significance(
            inp, baseline, TEAM_NAMES,
            n_simulations=50, max_fixtures=4, rng_seed=42,
        )
        for sig in ranked:
            assert sig.home_name.startswith("Team")
            assert sig.away_name.startswith("Team")
