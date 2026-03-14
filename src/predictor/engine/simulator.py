"""Monte Carlo simulator for predicting final league table positions.

Simulates the remaining fixtures N times, sampling scores from a
Dixon-Coles corrected Poisson distribution.  The result is a distribution
over final league positions for each team.

All computationally intensive work uses vectorised NumPy operations where
possible to keep simulation times reasonable (target: 10,000 runs < 10s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson  # type: ignore[import-untyped]

from predictor.engine.dixon_coles import DixonColesCorrection
from predictor.engine.head_to_head import adjust_lambdas_h2h
from predictor.engine.poisson import StrengthCalculator
from predictor.engine.standings import (
    StandingsDict,
    apply_result,
    initialise_standings,
    rank_standings,
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    """A remaining fixture to be simulated."""

    home_id: str
    away_id: str


@dataclass
class SimulationInput:
    """Everything the simulator needs for one league run."""

    team_ids: list[str]
    """All teams in the league."""

    current_standings: StandingsDict
    """Points/GD/GF as they stand today (deepcopied inside the simulator)."""

    remaining_fixtures: list[Fixture]
    """Fixtures not yet played."""

    strength_calculator: StrengthCalculator
    """Pre-fitted :class:`StrengthCalculator`."""

    league_avg_goals: float = 1.4
    """League average goals per team per game."""

    rho: float = -0.13
    """Dixon-Coles correlation parameter."""


@dataclass
class TeamPrediction:
    """Aggregated Monte Carlo result for a single team."""

    team_id: str
    mean_pos: float
    """Average finishing position across all simulations (1 = best)."""

    mean_points: float
    """Average final points total."""

    pos_dist: list[float] = field(default_factory=list)
    """Probability of finishing in each position.

    ``pos_dist[0]`` = P(finish 1st), ``pos_dist[1]`` = P(finish 2nd), etc.
    Length equals the number of teams in the league.
    """


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MonteCarloSimulator:
    """Simulate remaining fixtures N times and aggregate final-table predictions.

    Args:
        n_simulations: Number of Monte Carlo iterations.
        correction: Pre-configured :class:`DixonColesCorrection`. A default
            instance is created if not provided.
        rng_seed: Optional random seed for reproducible tests.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        correction: DixonColesCorrection | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.correction = correction or DixonColesCorrection()
        self._rng = np.random.default_rng(rng_seed)

    def run(self, sim_input: SimulationInput) -> dict[str, TeamPrediction]:
        """Execute the Monte Carlo simulation.

        Args:
            sim_input: All data required for the simulation run.

        Returns:
            Dict mapping ``team_id -> TeamPrediction`` with mean position,
            mean points, and a position probability distribution.
        """
        n_teams = len(sim_input.team_ids)
        n_fixtures = len(sim_input.remaining_fixtures)

        # Pre-compute lambdas for every remaining fixture (vectorisable)
        lambdas_home = np.zeros(n_fixtures)
        lambdas_away = np.zeros(n_fixtures)
        for i, fixture in enumerate(sim_input.remaining_fixtures):
            lh, la = sim_input.strength_calculator.compute_lambda(
                fixture.home_id,
                fixture.away_id,
                sim_input.league_avg_goals,
            )
            lh, la = adjust_lambdas_h2h(
                lh, la, sim_input.strength_calculator.matches,
                fixture.home_id, fixture.away_id,
            )
            lambdas_home[i] = lh
            lambdas_away[i] = la

        # Accumulators: position_counts[team_idx, pos_idx] = count of times
        # team finished in that position across all simulations
        team_id_to_idx = {tid: i for i, tid in enumerate(sim_input.team_ids)}
        position_counts = np.zeros((n_teams, n_teams), dtype=np.int64)
        total_points = np.zeros(n_teams, dtype=np.float64)

        for _ in range(self.n_simulations):
            # Sample goals for all fixtures simultaneously
            sampled_home = self._rng.poisson(lambdas_home)
            sampled_away = self._rng.poisson(lambdas_away)

            # Apply Dixon-Coles correction by re-sampling low-score outcomes
            # (full matrix correction is expensive at scale; we approximate by
            # adjusting individual low-score draws via rejection sampling)
            # TODO: Implement proper vectorised DC correction for performance
            for i, fixture in enumerate(sim_input.remaining_fixtures):
                hg = int(sampled_home[i])
                ag = int(sampled_away[i])
                if hg <= 1 and ag <= 1:
                    hg, ag = self._dc_corrected_sample(
                        lambdas_home[i], lambdas_away[i]
                    )
                sampled_home[i] = hg
                sampled_away[i] = ag

            # Build standings for this simulation
            import copy

            sim_standings = copy.deepcopy(sim_input.current_standings)
            # Ensure all teams are present
            for tid in sim_input.team_ids:
                if tid not in sim_standings:
                    from predictor.engine.standings import TeamStanding

                    sim_standings[tid] = TeamStanding(team_id=tid)

            for i, fixture in enumerate(sim_input.remaining_fixtures):
                apply_result(
                    sim_standings,
                    fixture.home_id,
                    fixture.away_id,
                    int(sampled_home[i]),
                    int(sampled_away[i]),
                )

            ranked = rank_standings(sim_standings)
            for pos_idx, standing in enumerate(ranked):
                if standing.team_id in team_id_to_idx:
                    team_idx = team_id_to_idx[standing.team_id]
                    position_counts[team_idx, pos_idx] += 1
                    total_points[team_idx] += standing.points

        # Aggregate results
        results: dict[str, TeamPrediction] = {}
        for team_id, team_idx in team_id_to_idx.items():
            counts = position_counts[team_idx]
            pos_dist = (counts / self.n_simulations).tolist()
            mean_pos = float(
                np.dot(np.arange(1, n_teams + 1), counts) / self.n_simulations
            )
            mean_pts = float(total_points[team_idx] / self.n_simulations)
            results[team_id] = TeamPrediction(
                team_id=team_id,
                mean_pos=round(mean_pos, 2),
                mean_points=round(mean_pts, 2),
                pos_dist=[round(p, 4) for p in pos_dist],
            )

        return results

    def _dc_corrected_sample(
        self,
        lambda_home: float,
        lambda_away: float,
        max_attempts: int = 20,
    ) -> tuple[int, int]:
        """Sample a score using Dixon-Coles rejection sampling for low scores.

        For low-scoring outcomes (hg+ag <= 2) we use rejection sampling with
        the DC correction factor as an acceptance probability.

        Args:
            lambda_home: Expected goals for the home side.
            lambda_away: Expected goals for the away side.
            max_attempts: Maximum rejection attempts before returning a raw
                Poisson sample.

        Returns:
            Tuple ``(home_goals, away_goals)``.
        """
        for _ in range(max_attempts):
            hg = int(self._rng.poisson(lambda_home))
            ag = int(self._rng.poisson(lambda_away))
            tau = self.correction.tau(hg, ag, lambda_home, lambda_away)
            # Accept with probability tau (tau <= 1 for negative rho)
            if tau >= 1.0 or self._rng.random() < tau:
                return hg, ag
        # Fallback
        return int(self._rng.poisson(lambda_home)), int(self._rng.poisson(lambda_away))

    def results_to_dict(
        self, predictions: dict[str, TeamPrediction]
    ) -> dict[str, dict]:
        """Serialise predictions to a plain dict suitable for JSON storage.

        Args:
            predictions: Output from :meth:`run`.

        Returns:
            Dict of ``team_id -> {mean_pos, mean_points, pos_dist}``.
        """
        return {
            team_id: {
                "mean_pos": pred.mean_pos,
                "mean_points": pred.mean_points,
                "pos_dist": pred.pos_dist,
            }
            for team_id, pred in predictions.items()
        }
