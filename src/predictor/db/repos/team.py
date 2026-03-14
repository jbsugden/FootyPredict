"""Repository for Team database operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import Team


class TeamRepository:
    """Data-access layer for :class:`~predictor.db.models.Team` records.

    Args:
        session: An active :class:`AsyncSession`.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_by_league(self, league_id: str) -> list[Team]:
        """Return all teams belonging to a league.

        Args:
            league_id: UUID string of the league.

        Returns:
            List of :class:`Team` instances ordered alphabetically by name.
        """
        stmt = (
            select(Team)
            .where(Team.league_id == league_id)
            .order_by(Team.name)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, team_id: str) -> Team | None:
        """Return a single team by its UUID.

        Args:
            team_id: UUID string.

        Returns:
            :class:`Team` or ``None`` if not found.
        """
        result = await self._session.get(Team, team_id)
        return result

    async def get_by_external_id(
        self, league_id: str, external_id: str
    ) -> Team | None:
        """Look up a team by its upstream data-source identifier.

        Args:
            league_id: UUID of the league (scopes the search to one league).
            external_id: The external ID from the data source.

        Returns:
            :class:`Team` or ``None`` if not found.
        """
        stmt = select(Team).where(
            Team.league_id == league_id,
            Team.external_id == external_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def get_or_create(
        self,
        league_id: str,
        name: str,
        short_name: str | None = None,
        external_id: str | None = None,
    ) -> tuple[Team, bool]:
        """Fetch an existing team or create a new one.

        Lookup priority:
        1. ``external_id`` match (if provided)
        2. Exact ``name`` match within the league

        Args:
            league_id: UUID of the parent league.
            name: Full club name.
            short_name: Optional abbreviated name (e.g. ``'Man Utd'``).
            external_id: Optional upstream source identifier.

        Returns:
            A tuple of ``(team, created)`` where ``created`` is ``True`` if a
            new record was inserted.
        """
        # 1. Try external_id first (most reliable)
        if external_id is not None:
            existing = await self.get_by_external_id(league_id, external_id)
            if existing is not None:
                return existing, False

        # 2. Fall back to name match
        stmt = select(Team).where(
            Team.league_id == league_id,
            Team.name == name,
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            return existing, False

        # 3. Create new team
        team = Team(
            league_id=league_id,
            name=name,
            short_name=short_name,
            external_id=external_id,
        )
        self._session.add(team)
        await self._session.flush()  # populate team.id without full commit
        return team, True
