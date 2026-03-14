"""Repository for Match database operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import Match, MatchStatus


class MatchRepository:
    """Data-access layer for :class:`~predictor.db.models.Match` records.

    All methods are coroutines and must be awaited.

    Args:
        session: An active :class:`AsyncSession`.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_by_league_season(
        self, league_id: str, season: str
    ) -> list[Match]:
        """Return all matches for a given league and season.

        Args:
            league_id: UUID string of the league.
            season: Season string, e.g. ``'2024-25'``.

        Returns:
            List of :class:`Match` instances ordered by ``played_at``.
        """
        stmt = (
            select(Match)
            .where(Match.league_id == league_id, Match.season == season)
            .order_by(Match.played_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_finished(
        self, league_id: str, season: str
    ) -> list[Match]:
        """Return all finished matches for a league/season.

        Args:
            league_id: UUID string of the league.
            season: Season string.

        Returns:
            List of finished :class:`Match` instances ordered by ``played_at``.
        """
        stmt = (
            select(Match)
            .where(
                Match.league_id == league_id,
                Match.season == season,
                Match.status == MatchStatus.FINISHED,
            )
            .order_by(Match.played_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_scheduled(
        self, league_id: str, season: str
    ) -> list[Match]:
        """Return all scheduled (upcoming) matches for a league/season.

        Args:
            league_id: UUID string of the league.
            season: Season string.

        Returns:
            List of scheduled :class:`Match` instances ordered by ``played_at``.
        """
        stmt = (
            select(Match)
            .where(
                Match.league_id == league_id,
                Match.season == season,
                Match.status == MatchStatus.SCHEDULED,
            )
            .order_by(Match.played_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def upsert(self, match_data: dict) -> Match:
        """Insert or update a match record.

        Matches are identified by ``(league_id, season, home_team_id,
        away_team_id, played_at)`` to avoid creating duplicates on
        re-import.

        Args:
            match_data: Dictionary with keys corresponding to :class:`Match`
                column names. Must include at minimum ``league_id``,
                ``season``, ``home_team_id``, ``away_team_id``, and
                ``played_at``.

        Returns:
            The persisted :class:`Match` instance (not yet committed).
        """
        # Try to find an existing record first
        stmt = select(Match).where(
            Match.league_id == match_data["league_id"],
            Match.season == match_data["season"],
            Match.home_team_id == match_data["home_team_id"],
            Match.away_team_id == match_data["away_team_id"],
            Match.played_at == match_data["played_at"],
        )
        result = await self._session.execute(stmt)
        existing = result.scalars().first()

        if existing is not None:
            # Update mutable fields
            for field in (
                "home_goals",
                "away_goals",
                "status",
                "matchday",
                "updated_at",
            ):
                if field in match_data:
                    setattr(existing, field, match_data[field])
            existing.updated_at = datetime.now(tz=timezone.utc)
            return existing

        new_match = Match(**match_data)
        self._session.add(new_match)
        return new_match
