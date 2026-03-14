"""Tests for the TeamSeason rebuild utility."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.db.models import (
    DataSource,
    League,
    Match,
    MatchStatus,
    Team,
    TeamSeason,
)
from predictor.engine.team_season import rebuild_team_seasons


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest_asyncio.fixture
async def rebuild_league(db_session: AsyncSession) -> League:
    """A dedicated league for rebuild tests (avoids clashing with sample_league)."""
    league = League(
        id=_uuid(),
        name="Rebuild Test League",
        code="REBUILD_TEST",
        tier=2,
        data_source=DataSource.SCRAPE_FA_FULLTIME,
        current_season="2024-25",
    )
    db_session.add(league)
    await db_session.flush()
    return league


@pytest_asyncio.fixture
async def rebuild_teams(
    db_session: AsyncSession, rebuild_league: League
) -> list[Team]:
    teams = []
    for name in ["Team Alpha", "Team Beta", "Team Gamma"]:
        t = Team(
            id=_uuid(),
            league_id=rebuild_league.id,
            name=name,
            short_name=name.split()[-1],
        )
        db_session.add(t)
        teams.append(t)
    await db_session.flush()
    return teams


@pytest_asyncio.fixture
async def rebuild_matches(
    db_session: AsyncSession,
    rebuild_league: League,
    rebuild_teams: list[Team],
) -> list[Match]:
    """Insert matches: Alpha beats Beta 2-1, Beta draws Gamma 0-0, Alpha beats Gamma 3-0."""
    a, b, c = rebuild_teams
    fixtures = [
        (a, b, 2, 1, 1),
        (b, c, 0, 0, 2),
        (a, c, 3, 0, 3),
    ]
    matches = []
    for home, away, hg, ag, day_offset in fixtures:
        m = Match(
            id=_uuid(),
            league_id=rebuild_league.id,
            season=rebuild_league.current_season,
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=hg,
            away_goals=ag,
            status=MatchStatus.FINISHED,
            played_at=_now() - timedelta(days=day_offset),
            matchday=day_offset,
        )
        db_session.add(m)
        matches.append(m)
    await db_session.flush()
    return matches


class TestRebuildTeamSeasons:
    @pytest.mark.asyncio
    async def test_returns_correct_upsert_count(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        count = await rebuild_team_seasons(db_session, rebuild_league)
        assert count == 3  # one per team

    @pytest.mark.asyncio
    async def test_points_calculated_correctly(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        await rebuild_team_seasons(db_session, rebuild_league)
        from sqlalchemy import select

        result = await db_session.execute(
            select(TeamSeason).where(
                TeamSeason.league_id == rebuild_league.id
            )
        )
        seasons = {ts.team_id: ts for ts in result.scalars().all()}

        a, b, c = rebuild_teams
        # Alpha: 2 wins = 6 pts
        assert seasons[a.id].points == 6
        # Beta: 1 loss + 1 draw = 1 pt
        assert seasons[b.id].points == 1
        # Gamma: 1 draw + 1 loss = 1 pt
        assert seasons[c.id].points == 1

    @pytest.mark.asyncio
    async def test_goals_calculated_correctly(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        await rebuild_team_seasons(db_session, rebuild_league)
        from sqlalchemy import select

        result = await db_session.execute(
            select(TeamSeason).where(
                TeamSeason.league_id == rebuild_league.id
            )
        )
        seasons = {ts.team_id: ts for ts in result.scalars().all()}

        a, b, c = rebuild_teams
        # Alpha: scored 2+3=5, conceded 1+0=1
        assert seasons[a.id].goals_for == 5
        assert seasons[a.id].goals_against == 1

    @pytest.mark.asyncio
    async def test_form_string(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        await rebuild_team_seasons(db_session, rebuild_league)
        from sqlalchemy import select

        result = await db_session.execute(
            select(TeamSeason).where(
                TeamSeason.league_id == rebuild_league.id
            )
        )
        seasons = {ts.team_id: ts for ts in result.scalars().all()}

        a, b, c = rebuild_teams
        # Matches ordered by played_at ascending (earliest first):
        # match 3 (3 days ago): A beats C  -> Alpha W, Gamma L
        # match 2 (2 days ago): B draws C  -> Beta D, Gamma D
        # match 1 (1 day ago):  A beats B  -> Alpha W, Beta L
        assert seasons[a.id].form == "WW"
        # Beta: D (from B-C draw, 2 days ago) then L (from A-B, 1 day ago)
        assert seasons[b.id].form == "DL"

    @pytest.mark.asyncio
    async def test_elo_ratings_computed(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        await rebuild_team_seasons(db_session, rebuild_league)
        from sqlalchemy import select

        result = await db_session.execute(
            select(TeamSeason).where(
                TeamSeason.league_id == rebuild_league.id
            )
        )
        seasons = {ts.team_id: ts for ts in result.scalars().all()}

        a, b, c = rebuild_teams
        # Alpha won both games, should have higher ELO than default 1500
        assert seasons[a.id].elo_rating > 1500.0
        # Beta lost one, drew one — should be below 1500
        assert seasons[b.id].elo_rating < 1500.0

    @pytest.mark.asyncio
    async def test_no_matches_returns_zero(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
    ) -> None:
        count = await rebuild_team_seasons(db_session, rebuild_league)
        assert count == 0

    @pytest.mark.asyncio
    async def test_idempotent_rebuild(
        self,
        db_session: AsyncSession,
        rebuild_league: League,
        rebuild_teams: list[Team],
        rebuild_matches: list[Match],
    ) -> None:
        """Running rebuild twice should produce the same results."""
        await rebuild_team_seasons(db_session, rebuild_league)
        await rebuild_team_seasons(db_session, rebuild_league)
        from sqlalchemy import select

        result = await db_session.execute(
            select(TeamSeason).where(
                TeamSeason.league_id == rebuild_league.id
            )
        )
        seasons = list(result.scalars().all())
        # Should still be exactly 3 records, not duplicated
        assert len(seasons) == 3
