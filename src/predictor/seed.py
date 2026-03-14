"""Database seed script.

Creates the initial League records for FootyPredict:
  - Premier League (football-data.org API)
  - Northern Premier League West (FA Full Time scraper)

Run once after the database is initialised:

    uv run python -m predictor.seed

Safe to run multiple times — skips leagues that already exist.
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select

from predictor.db.models import DataSource, League
from predictor.db.session import create_all_tables, get_session_factory

logger = structlog.get_logger(__name__)

LEAGUES = [
    {
        "name": "Premier League",
        "code": "PL",
        "tier": 1,
        "data_source": DataSource.API_FOOTBALL_DATA,
        "current_season": "2025",
    },
    {
        "name": "Northern Premier League West",
        "code": "NPL_W",
        "tier": 6,
        "data_source": DataSource.SCRAPE_FA_FULLTIME,
        "current_season": "2025-26",
    },
]


async def seed() -> None:
    """Create initial league records if they don't already exist."""
    await create_all_tables()

    factory = get_session_factory()
    async with factory() as session:
        for league_data in LEAGUES:
            result = await session.execute(
                select(League).where(League.code == league_data["code"])
            )
            existing = result.scalar_one_or_none()

            if existing is not None:
                logger.info("league_already_exists", code=league_data["code"])
                continue

            league = League(**league_data)
            session.add(league)
            logger.info("league_created", code=league_data["code"], name=league_data["name"])

        await session.commit()
        logger.info("seed_completed")


if __name__ == "__main__":
    asyncio.run(seed())
