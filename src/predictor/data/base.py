"""Abstract base types for league data sources.

Define the :class:`AbstractLeagueSource` protocol and the shared data-transfer
objects (:class:`LeagueData`, :class:`MatchData`) that all source adapters
must produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from predictor.db.models import MatchStatus


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamData:
    """Minimal team information returned by a data source."""

    name: str
    short_name: str | None = None
    external_id: str | None = None


@dataclass(frozen=True)
class MatchData:
    """A single fixture as returned by a data source.

    Fields map directly to the :class:`~predictor.db.models.Match` model.
    ``home_goals`` / ``away_goals`` are ``None`` for scheduled matches.
    """

    season: str
    home_team: TeamData
    away_team: TeamData
    played_at: datetime
    status: MatchStatus
    home_goals: int | None = None
    away_goals: int | None = None
    matchday: int | None = None
    external_id: str | None = None
    """Optional upstream fixture identifier for deduplication."""


@dataclass
class StandingRow:
    """One row in a live league standings table."""

    team: TeamData
    position: int
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    points: int
    form: str = ""


@dataclass
class LeagueData:
    """Complete data payload returned for one fetch cycle."""

    league_code: str
    season: str
    standings: list[StandingRow] = field(default_factory=list)
    finished_matches: list[MatchData] = field(default_factory=list)
    scheduled_fixtures: list[MatchData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractLeagueSource(Protocol):
    """Protocol that every data-source adapter must satisfy.

    Implement this protocol (either by subclassing or duck typing) in each
    concrete adapter (e.g. :class:`~predictor.data.football_data_org.FootballDataOrgSource`).

    All methods are async and must be awaited.
    """

    async def fetch_standings(
        self, league_code: str, season: str
    ) -> list[StandingRow]:
        """Fetch the current standings table for a league/season.

        Args:
            league_code: Short code identifying the league (e.g. ``'PL'``).
            season: Season identifier (e.g. ``'2024'`` or ``'2024-25'``).

        Returns:
            List of :class:`StandingRow` ordered by position ascending.
        """
        ...

    async def fetch_finished_matches(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch all completed results for a league/season.

        Args:
            league_code: Short code identifying the league.
            season: Season identifier.

        Returns:
            List of :class:`MatchData` with status ``FINISHED``.
        """
        ...

    async def fetch_scheduled_fixtures(
        self, league_code: str, season: str
    ) -> list[MatchData]:
        """Fetch upcoming fixtures that have not yet been played.

        Args:
            league_code: Short code identifying the league.
            season: Season identifier.

        Returns:
            List of :class:`MatchData` with status ``SCHEDULED``.
        """
        ...
