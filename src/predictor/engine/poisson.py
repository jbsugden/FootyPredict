"""Poisson-based attack/defence strength model for match prediction.

Each team is assigned:
- an *attack strength* (goals scored relative to league average)
- a *defence strength* (goals conceded relative to league average)

These are combined to estimate the expected goal lambda for any matchup.
Recent matches are weighted more heavily than older ones.

References:
  - Dixon & Coles (1997) "Modelling Association Football Scores and
    Inefficiencies in the Football Betting Market"
  - Maher (1982) "Modelling Association Football Scores"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Time-decay weights for form-weighting historical results
# ---------------------------------------------------------------------------

# Matches played within the last 6 weeks get full weight
WEIGHT_RECENT = 1.0
# Matches from 6–12 weeks ago get reduced weight
WEIGHT_MID = 0.7
# Matches older than 12 weeks get lowest weight
WEIGHT_OLD = 0.4

WEEKS_RECENT = 6
WEEKS_MID = 12


def _get_weight(played_at: datetime, now: datetime | None = None) -> float:
    """Return the time-decay weight for a single match.

    Args:
        played_at: When the match was played (timezone-aware).
        now: Reference datetime (defaults to UTC now).

    Returns:
        Weight scalar in ``{WEIGHT_OLD, WEIGHT_MID, WEIGHT_RECENT}``.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    weeks_ago = (now - played_at).days / 7.0
    if weeks_ago <= WEEKS_RECENT:
        return WEIGHT_RECENT
    if weeks_ago <= WEEKS_MID:
        return WEIGHT_MID
    return WEIGHT_OLD


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MatchRecord:
    """A single historical match used for strength calculation."""

    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    played_at: datetime


@dataclass
class TeamStrengths:
    """Attack and defence strength parameters for one team."""

    team_id: str
    attack: float = 1.0
    defence: float = 1.0


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------


class StrengthCalculator:
    """Compute Poisson attack/defence strengths from historical match data.

    Args:
        matches: List of historical :class:`MatchRecord` objects.
        now: Reference time for weight decay (defaults to UTC now).
    """

    def __init__(
        self,
        matches: list[MatchRecord],
        now: datetime | None = None,
    ) -> None:
        self._matches = matches
        self._now = now or datetime.now(tz=timezone.utc)
        self._strengths: dict[str, TeamStrengths] | None = None

    def compute_strengths(self) -> dict[str, TeamStrengths]:
        """Compute attack and defence strengths for all teams.

        Uses a single-pass weighted average approach:
          attack_strength = weighted_goals_scored / weighted_league_avg_goals
          defence_strength = weighted_goals_conceded / weighted_league_avg_goals

        Returns:
            Dict mapping ``team_id -> TeamStrengths``.
        """
        if not self._matches:
            return {}

        # Accumulate weighted goal totals per team
        # Format: {team_id: [weighted_scored, weighted_conceded, total_weight]}
        accum: dict[str, list[float]] = {}

        total_weighted_goals = 0.0
        total_weight = 0.0

        for m in self._matches:
            w = _get_weight(m.played_at, self._now)

            for team_id, scored, conceded in (
                (m.home_team_id, m.home_goals, m.away_goals),
                (m.away_team_id, m.away_goals, m.home_goals),
            ):
                if team_id not in accum:
                    accum[team_id] = [0.0, 0.0, 0.0]
                accum[team_id][0] += scored * w
                accum[team_id][1] += conceded * w
                accum[team_id][2] += w

            total_weighted_goals += (m.home_goals + m.away_goals) * w
            total_weight += 2 * w  # two teams per match

        # League average goals per weighted game
        league_avg = total_weighted_goals / total_weight if total_weight > 0 else 1.0

        strengths: dict[str, TeamStrengths] = {}
        for team_id, (scored, conceded, weight) in accum.items():
            if weight == 0:
                strengths[team_id] = TeamStrengths(team_id=team_id)
                continue
            avg_scored = scored / weight
            avg_conceded = conceded / weight
            strengths[team_id] = TeamStrengths(
                team_id=team_id,
                attack=avg_scored / league_avg if league_avg > 0 else 1.0,
                defence=avg_conceded / league_avg if league_avg > 0 else 1.0,
            )

        self._strengths = strengths
        return strengths

    def get_strengths(self) -> dict[str, TeamStrengths]:
        """Return cached strengths, computing them if necessary."""
        if self._strengths is None:
            return self.compute_strengths()
        return self._strengths

    def compute_lambda(
        self,
        home_id: str,
        away_id: str,
        league_avg_goals: float = 1.4,
    ) -> tuple[float, float]:
        """Compute the expected goal lambdas for a fixture.

        ``lambda_home = league_avg * home_attack * away_defence``
        ``lambda_away = league_avg * away_attack * home_defence``

        Args:
            home_id: Team ID of the home side.
            away_id: Team ID of the away side.
            league_avg_goals: Overall league average goals per team per game.

        Returns:
            Tuple ``(lambda_home, lambda_away)``.
        """
        strengths = self.get_strengths()
        home_s = strengths.get(home_id, TeamStrengths(team_id=home_id))
        away_s = strengths.get(away_id, TeamStrengths(team_id=away_id))

        lambda_home = league_avg_goals * home_s.attack * away_s.defence
        lambda_away = league_avg_goals * away_s.attack * home_s.defence

        # Guard against non-positive lambdas which break Poisson
        lambda_home = max(lambda_home, 0.01)
        lambda_away = max(lambda_away, 0.01)

        return lambda_home, lambda_away

    def score_probability_matrix(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
    ) -> np.ndarray:
        """Return an (N+1) x (N+1) matrix of score probabilities.

        ``matrix[i, j]`` = P(home scores i, away scores j).

        Args:
            lambda_home: Expected goals for the home team.
            lambda_away: Expected goals for the away team.
            max_goals: Maximum goals per side to model (scores above this are
                ignored — probability mass is negligible for reasonable lambdas).

        Returns:
            2-D float64 numpy array of shape ``(max_goals+1, max_goals+1)``.
        """
        from scipy.stats import poisson  # type: ignore[import-untyped]

        home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
        away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
        return np.outer(home_probs, away_probs)
