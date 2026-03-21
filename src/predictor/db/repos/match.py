"""Repository for Match database operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import Match, MatchStatus


def previous_season(current: str) -> str:
    """Derive the previous season string from the current one.

    Example: ``"2024-25"`` → ``"2023-24"``.
    """
    start_year = int(current.split("-")[0])
    end_suffix = f"{start_year % 100:02d}"
    return f"{start_year - 1}-{end_suffix}"


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

    async def get_finished_multi_season(
        self, league_id: str, seasons: list[str]
    ) -> list[Match]:
        """Return finished matches across multiple seasons.

        Args:
            league_id: UUID string of the league.
            seasons: List of season strings, e.g. ``['2023-24', '2024-25']``.

        Returns:
            List of finished :class:`Match` instances ordered by ``played_at``.
        """
        stmt = (
            select(Match)
            .where(
                Match.league_id == league_id,
                Match.season.in_(seasons),
                Match.status == MatchStatus.FINISHED,
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

        When ``external_id`` is provided the match is looked up by
        ``(league_id, external_id)`` first, which correctly handles
        rescheduled fixtures whose ``played_at`` date has changed.
        Falls back to ``(league_id, season, home_team_id, away_team_id,
        played_at)`` for sources that do not supply an external ID.

        Args:
            match_data: Dictionary with keys corresponding to :class:`Match`
                column names. Must include at minimum ``league_id``,
                ``season``, ``home_team_id``, ``away_team_id``, and
                ``played_at``.

        Returns:
            The persisted :class:`Match` instance (not yet committed).
        """
        existing: Match | None = None

        # Prefer external_id lookup (handles rescheduled fixtures)
        ext_id = match_data.get("external_id")
        if ext_id:
            stmt = select(Match).where(
                Match.league_id == match_data["league_id"],
                Match.external_id == ext_id,
            )
            result = await self._session.execute(stmt)
            existing = result.scalars().first()

        # Fallback: match by teams + date
        if existing is None:
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
            # Update mutable fields (including played_at for rescheduled matches)
            for field in (
                "home_goals",
                "away_goals",
                "status",
                "matchday",
                "played_at",
                "external_id",
            ):
                if field in match_data and match_data[field] is not None:
                    setattr(existing, field, match_data[field])
            existing.updated_at = datetime.now(tz=timezone.utc)
            return existing

        new_match = Match(**match_data)
        self._session.add(new_match)
        return new_match
