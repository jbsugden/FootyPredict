"""Jinja2 HTML page routes for FootyPredict.

These routes return full HTML pages rendered from Jinja2 templates and are
separate from the ``/api/`` JSON routes.
"""

from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from predictor.api.deps import DbSession
from predictor.db.models import League, TeamSeason
from predictor.db.repos.prediction import PredictionRepository
from predictor.db.repos.team import TeamRepository

logger = structlog.get_logger(__name__)

router = APIRouter()

# Locate templates directory relative to this file
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def _top_positions(pos_dist: list[float], n: int = 3) -> list[tuple[int, float]]:
    """Return top-n (index, probability) pairs sorted by probability descending."""
    indexed = list(enumerate(pos_dist))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:n]


templates.env.filters["top_positions"] = _top_positions


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: DbSession) -> HTMLResponse:
    """Homepage — display cards linking to each tracked league.

    Returns:
        Rendered ``index.html`` template.
    """
    result = await db.execute(select(League).order_by(League.tier, League.name))
    leagues = result.scalars().all()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"leagues": leagues, "nav_leagues": leagues},
    )


@router.get("/league/{league_id}", response_class=HTMLResponse)
async def league_detail(
    request: Request,
    league_id: str,
    db: DbSession,
) -> HTMLResponse:
    """League detail page — current standings and predicted final table.

    Args:
        league_id: UUID of the league to display.

    Returns:
        Rendered ``league.html`` template.

    Raises:
        HTTPException 404: If the league does not exist.
    """
    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    # Current standings
    ts_result = await db.execute(
        select(TeamSeason).where(
            TeamSeason.league_id == league_id,
            TeamSeason.season == league.current_season,
        )
    )
    team_seasons = ts_result.scalars().all()

    team_repo = TeamRepository(db)
    teams = await team_repo.get_by_league(league_id)
    team_map = {t.id: t for t in teams}

    # Sort standings
    sorted_seasons = sorted(
        team_seasons,
        key=lambda ts: (-ts.points, -(ts.goals_for - ts.goals_against), -ts.goals_for),
    )

    # Latest prediction
    pred_repo = PredictionRepository(db)
    prediction = await pred_repo.get_latest(league_id, league.current_season)

    predicted_table = []
    if prediction:
        items = [
            {
                "team_id": tid,
                "team_name": team_map.get(tid, None) and team_map[tid].name or tid,
                "mean_pos": data.get("mean_pos", 0.0),
                "mean_points": data.get("mean_points", 0.0),
                "pos_dist": data.get("pos_dist", []),
            }
            for tid, data in prediction.results.items()
        ]
        predicted_table = sorted(items, key=lambda x: x["mean_pos"])

    # Load leagues for nav
    nav_result = await db.execute(select(League).order_by(League.tier, League.name))
    nav_leagues = nav_result.scalars().all()

    return templates.TemplateResponse(
        request=request,
        name="league.html",
        context={
            "league": league,
            "standings": sorted_seasons,
            "team_map": team_map,
            "predicted_table": predicted_table,
            "prediction": prediction,
            "n_teams": len(teams),
            "nav_leagues": nav_leagues,
        },
    )


@router.get("/admin/import", response_class=HTMLResponse)
async def admin_import_page(request: Request, db: DbSession) -> HTMLResponse:
    """Admin CSV import form page.

    Returns:
        Rendered ``admin/import.html`` template.
    """
    nav_result = await db.execute(select(League).order_by(League.tier, League.name))
    nav_leagues = nav_result.scalars().all()

    return templates.TemplateResponse(
        request=request,
        name="admin/import.html",
        context={"nav_leagues": nav_leagues},
    )
