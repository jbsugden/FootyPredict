"""League standings helper functions.

Provides a pure function :func:`apply_result` that updates an in-memory
standings dictionary with the outcome of a single match. Used during Monte
Carlo simulation to accumulate a final table without database I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TeamStanding:
    """Mutable standings record for one team in a simulation pass."""

    team_id: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def sort_key(self) -> tuple[int, int, int]:
        """Primary sort key for standard league table ordering.

        Returns:
            Tuple ``(-points, -goal_difference, -goals_for)`` so that
            sorting ascending gives the correct league order.
        """
        return (-self.points, -self.goal_difference, -self.goals_for)


# Type alias for a standings dict
StandingsDict: TypeAlias = dict[str, TeamStanding]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def apply_result(
    standings: StandingsDict,
    home_id: str,
    away_id: str,
    home_goals: int,
    away_goals: int,
) -> StandingsDict:
    """Update standings in-place with the result of a single match.

    If ``home_id`` or ``away_id`` are not already in ``standings``, new
    :class:`TeamStanding` entries are created automatically.

    Args:
        standings: Current standings dictionary (mutated in-place AND returned).
        home_id: Team ID of the home side.
        away_id: Team ID of the away side.
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.

    Returns:
        The same ``standings`` dict, updated in-place.

    Example::

        s: StandingsDict = {}
        apply_result(s, "team_a", "team_b", 2, 1)
        # s["team_a"].points == 3, s["team_b"].points == 0
    """
    if home_id not in standings:
        standings[home_id] = TeamStanding(team_id=home_id)
    if away_id not in standings:
        standings[away_id] = TeamStanding(team_id=away_id)

    home = standings[home_id]
    away = standings[away_id]

    home.played += 1
    away.played += 1
    home.goals_for += home_goals
    home.goals_against += away_goals
    away.goals_for += away_goals
    away.goals_against += home_goals

    if home_goals > away_goals:
        home.won += 1
        home.points += 3
        away.lost += 1
    elif home_goals < away_goals:
        away.won += 1
        away.points += 3
        home.lost += 1
    else:
        home.drawn += 1
        away.drawn += 1
        home.points += 1
        away.points += 1

    return standings


def rank_standings(standings: StandingsDict) -> list[TeamStanding]:
    """Return teams sorted in standard league table order.

    Ordering: points (desc), goal difference (desc), goals for (desc).

    Args:
        standings: Standings dict to rank.

    Returns:
        List of :class:`TeamStanding` from 1st to last place.
    """
    return sorted(standings.values(), key=TeamStanding.sort_key)


def initialise_standings(team_ids: list[str]) -> StandingsDict:
    """Create a fresh standings dict for a list of teams.

    Args:
        team_ids: List of team ID strings.

    Returns:
        Dict mapping each ``team_id`` to a zeroed :class:`TeamStanding`.
    """
    return {tid: TeamStanding(team_id=tid) for tid in team_ids}
