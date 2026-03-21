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
from predictor.db.models import League, Match, MatchStatus, TeamSeason
from predictor.db.repos.match import MatchRepository
from predictor.db.repos.prediction import PredictionRepository
from predictor.db.repos.team import TeamRepository
from predictor.engine.fixture_prediction import predict_fixtures
from predictor.engine.zones import build_zone_class_map, compute_zone_probabilities

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

    # Prediction history for timeline chart
    prediction_history = await pred_repo.get_history(league_id, league.current_season, limit=50)

    timeline_data = None
    if len(prediction_history) > 1:
        chronological = list(reversed(prediction_history))
        timeline_data = {
            "dates": [p.generated_at.strftime("%d %b") for p in chronological],
            "teams": {},
            "crests": {},
        }
        for pred in chronological:
            for tid, data in pred.results.items():
                if tid.startswith("__"):
                    continue
                team = team_map.get(tid)
                name = team.name if team else tid
                timeline_data["teams"].setdefault(name, []).append(
                    data.get("mean_pos")
                )
                if team and team.crest_url and name not in timeline_data["crests"]:
                    timeline_data["crests"][name] = team.crest_url

    predicted_table = []
    n_teams = len(teams)
    if prediction:
        items = []
        for tid, data in prediction.results.items():
            if tid.startswith("__"):
                continue
            pos_dist = data.get("pos_dist", [])
            items.append({
                "team_id": tid,
                "team_name": team_map.get(tid, None) and team_map[tid].name or tid,
                "mean_pos": data.get("mean_pos", 0.0),
                "mean_points": data.get("mean_points", 0.0),
                "pos_dist": pos_dist,
                "zones": compute_zone_probabilities(pos_dist, league.code, n_teams),
            })
        predicted_table = sorted(items, key=lambda x: x["mean_pos"])

    zone_classes = build_zone_class_map(league.code, n_teams)

    # Key matches from significance analysis
    key_matches = []
    fixture_completeness = {}
    if prediction and "__meta__" in prediction.results:
        key_matches = prediction.results["__meta__"].get("key_matches", [])
        fixture_completeness = prediction.results["__meta__"].get("fixture_completeness", {})

    # Scheduled matches for scenario explorer
    match_repo = MatchRepository(db)
    scheduled_matches = await match_repo.get_scheduled(league_id, league.current_season)
    import json as _json
    scenario_fixtures = _json.dumps([
        {
            "home_id": m.home_team_id,
            "away_id": m.away_team_id,
            "home_name": team_map.get(m.home_team_id, None) and team_map[m.home_team_id].name or m.home_team_id,
            "away_name": team_map.get(m.away_team_id, None) and team_map[m.away_team_id].name or m.away_team_id,
            "home_goals": 0,
            "away_goals": 0,
            "locked": False,
        }
        for m in scheduled_matches[:30]
    ])

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
            "n_teams": n_teams,
            "zone_classes": zone_classes,
            "timeline_data": timeline_data,
            "key_matches": key_matches,
            "fixture_completeness": fixture_completeness,
            "scenario_fixtures": scenario_fixtures,
            "nav_leagues": nav_leagues,
        },
    )


@router.get("/league/{league_id}/team/{team_id}", response_class=HTMLResponse)
async def team_detail(
    request: Request,
    league_id: str,
    team_id: str,
    db: DbSession,
) -> HTMLResponse:
    """Team deep-dive page — per-fixture prediction breakdowns.

    Args:
        league_id: UUID of the league.
        team_id: UUID of the team.

    Returns:
        Rendered ``team.html`` template.

    Raises:
        HTTPException 404: If the league or team does not exist.
    """
    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    team_repo = TeamRepository(db)
    team = await team_repo.get_by_id(team_id)
    if team is None or team.league_id != league_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team {team_id!r} not found in league {league_id!r}.",
        )

    # Current season record
    ts_result = await db.execute(
        select(TeamSeason).where(
            TeamSeason.team_id == team_id,
            TeamSeason.league_id == league_id,
            TeamSeason.season == league.current_season,
        )
    )
    team_season = ts_result.scalars().first()

    # All teams for name lookups
    teams = await team_repo.get_by_league(league_id)
    team_map = {t.id: t for t in teams}

    # Matches
    match_repo = MatchRepository(db)
    finished_matches = await match_repo.get_finished(league_id, league.current_season)
    scheduled_matches = await match_repo.get_scheduled(league_id, league.current_season)

    # Include previous-season matches for cross-season strength prior
    from predictor.db.repos.match import previous_season
    prev = previous_season(league.current_season)
    prev_finished = await match_repo.get_finished_multi_season(league_id, [prev])

    # Per-fixture predictions (previous season matches provide historical anchor)
    fp_result = predict_fixtures(
        prev_finished + finished_matches, scheduled_matches, team_id, team_map
    )

    # Latest prediction for position distribution
    pred_repo = PredictionRepository(db)
    prediction = await pred_repo.get_latest(league_id, league.current_season)

    # Team prediction timeline
    team_timeline = None
    if prediction:
        history = await pred_repo.get_history(league_id, league.current_season, limit=50)
        if len(history) > 1:
            chronological = list(reversed(history))
            team_timeline = {
                "dates": [p.generated_at.strftime("%d %b") for p in chronological],
                "positions": [
                    p.results.get(team_id, {}).get("mean_pos")
                    for p in chronological
                ],
            }

    team_prediction = None
    team_zones = []
    if prediction and team_id in prediction.results:
        team_prediction = prediction.results[team_id]
        n_league_teams = len(teams)
        team_zones = compute_zone_probabilities(
            team_prediction.get("pos_dist", []), league.code, n_league_teams
        )

    # Last 5 finished matches involving this team (most recent first)
    team_finished = [
        m
        for m in finished_matches
        if m.home_team_id == team_id or m.away_team_id == team_id
    ]
    recent_results = team_finished[-5:][::-1]

    # Aggregate outlook: toughest and easiest fixture
    toughest = None
    easiest = None
    if fp_result.fixtures:
        toughest = min(fp_result.fixtures, key=lambda f: f.p_win)
        easiest = max(fp_result.fixtures, key=lambda f: f.p_win)

    # Nav leagues
    nav_result = await db.execute(select(League).order_by(League.tier, League.name))
    nav_leagues = nav_result.scalars().all()

    return templates.TemplateResponse(
        request=request,
        name="team.html",
        context={
            "league": league,
            "team": team,
            "team_season": team_season,
            "team_map": team_map,
            "fp_result": fp_result,
            "team_prediction": team_prediction,
            "prediction": prediction,
            "team_zones": team_zones,
            "team_timeline": team_timeline,
            "recent_results": recent_results,
            "toughest": toughest,
            "easiest": easiest,
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
