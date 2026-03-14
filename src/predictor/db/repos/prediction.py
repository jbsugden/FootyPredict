"""Repository for Prediction database operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import Prediction


class PredictionRepository:
    """Data-access layer for :class:`~predictor.db.models.Prediction` records.

    Args:
        session: An active :class:`AsyncSession`.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_latest(
        self, league_id: str, season: str
    ) -> Prediction | None:
        """Return the most recently generated prediction for a league/season.

        Args:
            league_id: UUID of the league.
            season: Season string, e.g. ``'2024-25'``.

        Returns:
            Most recent :class:`Prediction` or ``None`` if none exist yet.
        """
        stmt = (
            select(Prediction)
            .where(
                Prediction.league_id == league_id,
                Prediction.season == season,
            )
            .order_by(Prediction.generated_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_history(
        self, league_id: str, season: str, limit: int = 10
    ) -> list[Prediction]:
        """Return recent predictions for a league/season, newest first.

        Args:
            league_id: UUID of the league.
            season: Season string.
            limit: Maximum number of records to return.

        Returns:
            List of :class:`Prediction` instances.
        """
        stmt = (
            select(Prediction)
            .where(
                Prediction.league_id == league_id,
                Prediction.season == season,
            )
            .order_by(Prediction.generated_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def save(
        self,
        league_id: str,
        season: str,
        simulation_runs: int,
        results: dict,
    ) -> Prediction:
        """Persist a new prediction record.

        A new row is always inserted (predictions are immutable snapshots).
        Use :meth:`get_latest` to retrieve the most relevant one.

        Args:
            league_id: UUID of the league.
            season: Season string, e.g. ``'2024-25'``.
            simulation_runs: Number of Monte Carlo iterations used.
            results: Serialisable dict of simulation output keyed by team_id.

        Returns:
            The newly created :class:`Prediction` instance (not yet committed).
        """
        prediction = Prediction(
            league_id=league_id,
            season=season,
            simulation_runs=simulation_runs,
            results=results,
        )
        self._session.add(prediction)
        await self._session.flush()  # populate prediction.id
        return prediction
