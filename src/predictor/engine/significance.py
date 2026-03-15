"""Match significance scoring for upcoming fixtures.

Ranks remaining fixtures by how much their outcome would shift the
predicted final league table.  For each fixture, three scenarios are
simulated (home win, draw, away win) and the resulting mean-position
shifts are weighted by the probability of each outcome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from scipy.stats import poisson as poisson_dist  # type: ignore[import-untyped]

from predictor.engine.simulator import (
    MonteCarloSimulator,
    SimulationInput,
    TeamPrediction,
)
from predictor.engine.standings import apply_result


@dataclass
class FixtureSignificance:
    """Significance scoring result for a single fixture."""

    home_id: str
    away_id: str
    home_name: str
    away_name: str
    significance_score: float
    shift_home_win: float
    shift_draw: float
    shift_away_win: float


# Canonical score lines for each outcome
_OUTCOMES: list[tuple[int, int]] = [
    (2, 0),  # home win
    (1, 1),  # draw
    (0, 2),  # away win
]


def _outcome_probabilities(
    lambda_home: float, lambda_away: float,
) -> tuple[float, float, float]:
    """Estimate P(home win), P(draw), P(away win) from Poisson lambdas."""
    max_goals = 8
    p_hw = 0.0
    p_draw = 0.0
    p_aw = 0.0
    for h in range(max_goals + 1):
        ph = poisson_dist.pmf(h, lambda_home)
        for a in range(max_goals + 1):
            pa = poisson_dist.pmf(a, lambda_away)
            joint = ph * pa
            if h > a:
                p_hw += joint
            elif h == a:
                p_draw += joint
            else:
                p_aw += joint
    return p_hw, p_draw, p_aw


def _compute_shift(
    baseline: dict[str, TeamPrediction],
    scenario: dict[str, TeamPrediction],
) -> float:
    """Sum of absolute mean-position shifts across all teams."""
    total = 0.0
    for tid in baseline:
        total += abs(scenario[tid].mean_pos - baseline[tid].mean_pos)
    return total


def compute_fixture_significance(
    sim_input: SimulationInput,
    fixture_index: int,
    baseline_results: dict[str, TeamPrediction],
    team_names: dict[str, str],
    n_simulations: int = 1_000,
    rng_seed: int | None = None,
) -> FixtureSignificance:
    """Compute the significance score for a single remaining fixture.

    Args:
        sim_input: The full simulation input (will be deep-copied per scenario).
        fixture_index: Index into ``sim_input.remaining_fixtures``.
        baseline_results: Baseline prediction results for comparison.
        team_names: Mapping of team_id -> team name.
        n_simulations: Simulation runs per scenario.
        rng_seed: Optional seed for reproducibility.

    Returns:
        A :class:`FixtureSignificance` with the computed score and shifts.
    """
    fixture = sim_input.remaining_fixtures[fixture_index]

    # Get outcome probabilities from the strength model
    lh, la = sim_input.strength_calculator.compute_lambda(
        fixture.home_id, fixture.away_id, sim_input.league_avg_goals,
    )
    p_hw, p_draw, p_aw = _outcome_probabilities(lh, la)
    probs = [p_hw, p_draw, p_aw]

    shifts = []
    for hg, ag in _OUTCOMES:
        # Deep-copy standings and apply the fixed result
        scenario_standings = copy.deepcopy(sim_input.current_standings)
        apply_result(scenario_standings, fixture.home_id, fixture.away_id, hg, ag)

        # Remove this fixture from remaining
        scenario_fixtures = [
            f for i, f in enumerate(sim_input.remaining_fixtures)
            if i != fixture_index
        ]

        scenario_input = SimulationInput(
            team_ids=sim_input.team_ids,
            current_standings=scenario_standings,
            remaining_fixtures=scenario_fixtures,
            strength_calculator=sim_input.strength_calculator,
            league_avg_goals=sim_input.league_avg_goals,
        )

        seed = rng_seed + fixture_index * 10 + len(shifts) if rng_seed is not None else None
        simulator = MonteCarloSimulator(n_simulations=n_simulations, rng_seed=seed)
        scenario_results = simulator.run(scenario_input)
        shifts.append(_compute_shift(baseline_results, scenario_results))

    # Probability-weighted significance score
    significance = sum(p * s for p, s in zip(probs, shifts))

    return FixtureSignificance(
        home_id=fixture.home_id,
        away_id=fixture.away_id,
        home_name=team_names.get(fixture.home_id, fixture.home_id),
        away_name=team_names.get(fixture.away_id, fixture.away_id),
        significance_score=significance,
        shift_home_win=shifts[0],
        shift_draw=shifts[1],
        shift_away_win=shifts[2],
    )


def rank_fixtures_by_significance(
    sim_input: SimulationInput,
    baseline_results: dict[str, TeamPrediction],
    team_names: dict[str, str],
    n_simulations: int = 1_000,
    max_fixtures: int = 10,
    rng_seed: int | None = None,
) -> list[FixtureSignificance]:
    """Rank remaining fixtures by predicted table impact.

    Args:
        sim_input: The full simulation input.
        baseline_results: Baseline prediction results.
        team_names: Mapping of team_id -> team name.
        n_simulations: Simulation runs per scenario per fixture.
        max_fixtures: Maximum number of fixtures to evaluate.
        rng_seed: Optional seed for reproducibility.

    Returns:
        List of :class:`FixtureSignificance` sorted by score descending.
    """
    n = min(max_fixtures, len(sim_input.remaining_fixtures))
    results = []
    for i in range(n):
        sig = compute_fixture_significance(
            sim_input, i, baseline_results, team_names,
            n_simulations=n_simulations, rng_seed=rng_seed,
        )
        results.append(sig)

    results.sort(key=lambda s: s.significance_score, reverse=True)
    return results
