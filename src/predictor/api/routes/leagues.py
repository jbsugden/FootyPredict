"""API routes for league and standings data."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from predictor.api.deps import DbSession
from predictor.db.models import League, TeamSeason
from predictor.db.repos.team import TeamRepository

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class LeagueResponse(BaseModel):
    """Summary representation of a league."""

    id: str
    name: str
    code: str
    tier: int
    current_season: str
    data_source: str


class StandingRowResponse(BaseModel):
    """One row in a league table response."""

    position: int
    team_id: str
    team_name: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    form: str
    elo_rating: float


class LeagueTableResponse(BaseModel):
    """Full league table for a season."""

    league_id: str
    league_name: str
    season: str
    standings: list[StandingRowResponse]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/leagues", response_model=list[LeagueResponse])
async def list_leagues(db: DbSession) -> list[LeagueResponse]:
    """Return all leagues tracked by FootyPredict.

    Returns:
        List of :class:`LeagueResponse` objects.
    """
    result = await db.execute(select(League).order_by(League.tier, League.name))
    leagues = result.scalars().all()
    return [
        LeagueResponse(
            id=lg.id,
            name=lg.name,
            code=lg.code,
            tier=lg.tier,
            current_season=lg.current_season,
            data_source=lg.data_source.value,
        )
        for lg in leagues
    ]


@router.get("/leagues/{league_id}/table", response_model=LeagueTableResponse)
async def get_league_table(league_id: str, db: DbSession) -> LeagueTableResponse:
    """Return the current standings table for a specific league.

    Args:
        league_id: UUID of the league.

    Returns:
        :class:`LeagueTableResponse` with all teams ranked by points.

    Raises:
        HTTPException 404: If the league does not exist.
    """
    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    # Fetch team seasons for the current season
    result = await db.execute(
        select(TeamSeason)
        .where(
            TeamSeason.league_id == league_id,
            TeamSeason.season == league.current_season,
        )
        .order_by(
            TeamSeason.points.desc(),
            # SQLAlchemy expression for goal difference
        )
    )
    team_seasons = result.scalars().all()

    # Resolve team names
    team_repo = TeamRepository(db)
    teams = await team_repo.get_by_league(league_id)
    team_name_map = {t.id: t.name for t in teams}

    # Sort by points desc, then goal difference desc, then goals for desc
    sorted_seasons = sorted(
        team_seasons,
        key=lambda ts: (-ts.points, -(ts.goals_for - ts.goals_against), -ts.goals_for),
    )

    standings = [
        StandingRowResponse(
            position=idx + 1,
            team_id=ts.team_id,
            team_name=team_name_map.get(ts.team_id, "Unknown"),
            played=ts.played,
            won=ts.won,
            drawn=ts.drawn,
            lost=ts.lost,
            goals_for=ts.goals_for,
            goals_against=ts.goals_against,
            goal_difference=ts.goals_for - ts.goals_against,
            points=ts.points,
            form=ts.form,
            elo_rating=ts.elo_rating,
        )
        for idx, ts in enumerate(sorted_seasons)
    ]

    return LeagueTableResponse(
        league_id=league_id,
        league_name=league.name,
        season=league.current_season,
        standings=standings,
    )
