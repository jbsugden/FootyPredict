"""Shared pytest fixtures for FootyPredict tests.

Provides:
- In-memory SQLite async engine and session
- FastAPI test client
- Sample League, Team, and Match factory fixtures
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from predictor.db.models import Base, DataSource, League, Match, MatchStatus, Team, TeamSeason
from predictor.db.session import get_session_factory

# ---------------------------------------------------------------------------
# In-memory SQLite engine
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create an async in-memory SQLite engine for the test session."""
    eng = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield a fresh database session for each test.

    Each test runs inside a transaction that is rolled back after completion
    so tests are isolated from one another.
    """
    async with engine.connect() as conn:
        await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as session:
            yield session
        await conn.rollback()


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Yield an async HTTPX client wired to the FastAPI app.

    The database dependency is overridden to use the test session so that
    all requests share the same in-memory DB as the test.
    """
    from predictor.api.app import create_app
    from predictor.api.deps import get_db

    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest_asyncio.fixture
async def sample_league(db_session: AsyncSession) -> League:
    """Insert and return a sample Premier League record."""
    league = League(
        id=_uuid(),
        name="Test Premier League",
        code="TEST_PL",
        tier=1,
        data_source=DataSource.API_FOOTBALL_DATA,
        current_season="2024-25",
    )
    db_session.add(league)
    await db_session.flush()
    return league


@pytest_asyncio.fixture
async def sample_teams(db_session: AsyncSession, sample_league: League) -> list[Team]:
    """Insert and return a list of 4 sample teams for the sample league."""
    names = ["Arsenal FC", "Chelsea FC", "Liverpool FC", "Manchester City"]
    teams: list[Team] = []
    for name in names:
        t = Team(
            id=_uuid(),
            league_id=sample_league.id,
            name=name,
            short_name=name.split()[0],
            external_id=str(abs(hash(name)))[:6],
        )
        db_session.add(t)
        teams.append(t)
    await db_session.flush()
    return teams


@pytest_asyncio.fixture
async def sample_matches(
    db_session: AsyncSession,
    sample_league: League,
    sample_teams: list[Team],
) -> list[Match]:
    """Insert and return a list of sample finished matches."""
    team_a, team_b, team_c, team_d = sample_teams
    fixtures = [
        (team_a, team_b, 2, 1),
        (team_c, team_d, 0, 0),
        (team_b, team_c, 1, 3),
        (team_d, team_a, 1, 2),
    ]
    matches: list[Match] = []
    for i, (home, away, hg, ag) in enumerate(fixtures):
        m = Match(
            id=_uuid(),
            league_id=sample_league.id,
            season="2024-25",
            matchday=i + 1,
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=hg,
            away_goals=ag,
            status=MatchStatus.FINISHED,
            played_at=_now(),
        )
        db_session.add(m)
        matches.append(m)
    await db_session.flush()
    return matches


@pytest_asyncio.fixture
async def sample_team_seasons(
    db_session: AsyncSession,
    sample_league: League,
    sample_teams: list[Team],
) -> list[TeamSeason]:
    """Insert stub TeamSeason records for each sample team."""
    team_seasons: list[TeamSeason] = []
    for i, team in enumerate(sample_teams):
        ts = TeamSeason(
            id=_uuid(),
            team_id=team.id,
            league_id=sample_league.id,
            season="2024-25",
            played=10,
            won=5 - i,
            drawn=2,
            lost=3 + i,
            goals_for=15 - i * 2,
            goals_against=10 + i,
            points=(5 - i) * 3 + 2,
            form="WWDLW",
            elo_rating=1500.0 + (i * 20),
        )
        db_session.add(ts)
        team_seasons.append(ts)
    await db_session.flush()
    return team_seasons
