"""API routes for Monte Carlo predictions."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from predictor.api.deps import AdminKeyVerified, DbSession
from predictor.db.models import League
from predictor.db.repos.prediction import PredictionRepository

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TeamPredictionResponse(BaseModel):
    """Prediction data for a single team."""

    team_id: str
    mean_pos: float
    mean_points: float
    pos_dist: list[float]


class PredictionResponse(BaseModel):
    """Full prediction payload for a league."""

    prediction_id: str
    league_id: str
    season: str
    generated_at: str
    simulation_runs: int
    teams: list[TeamPredictionResponse]


class RunPredictionResponse(BaseModel):
    """Acknowledgement returned when a prediction job is queued."""

    message: str
    league_id: str


class ScenarioFixture(BaseModel):
    """A single locked-in fixture result for a scenario simulation."""

    home_team_id: str
    away_team_id: str
    home_goals: int = Field(ge=0, le=15)
    away_goals: int = Field(ge=0, le=15)


class ScenarioRequest(BaseModel):
    """Request body for a scenario (what-if) simulation."""

    locked_results: list[ScenarioFixture] = Field(min_length=1, max_length=20)


class ScenarioTeamResult(BaseModel):
    """Per-team result comparing baseline vs scenario predictions."""

    team_id: str
    team_name: str
    mean_pos: float
    mean_points: float
    baseline_mean_pos: float
    pos_change: float


class ScenarioResponse(BaseModel):
    """Response from a scenario simulation."""

    teams: list[ScenarioTeamResult]
    simulation_runs: int
    locked_count: int


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/predictions/{league_id}", response_model=PredictionResponse)
async def get_latest_prediction(
    league_id: str,
    db: DbSession,
) -> PredictionResponse:
    """Return the most recent Monte Carlo prediction for a league.

    Args:
        league_id: UUID of the league.

    Returns:
        :class:`PredictionResponse` with per-team position distributions.

    Raises:
        HTTPException 404: If no prediction exists yet for this league.
    """
    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    pred_repo = PredictionRepository(db)
    prediction = await pred_repo.get_latest(league_id, league.current_season)

    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No prediction found for league {league_id!r} "
                f"season {league.current_season!r}. "
                "Run POST /api/predictions/{league_id}/run to generate one."
            ),
        )

    teams = [
        TeamPredictionResponse(
            team_id=team_id,
            mean_pos=data.get("mean_pos", 0.0),
            mean_points=data.get("mean_points", 0.0),
            pos_dist=data.get("pos_dist", []),
        )
        for team_id, data in prediction.results.items()
        if not team_id.startswith("__")
    ]
    # Sort by mean position ascending
    teams.sort(key=lambda t: t.mean_pos)

    return PredictionResponse(
        prediction_id=prediction.id,
        league_id=league_id,
        season=prediction.season,
        generated_at=prediction.generated_at.isoformat(),
        simulation_runs=prediction.simulation_runs,
        teams=teams,
    )


@router.post(
    "/predictions/{league_id}/run",
    response_model=RunPredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_prediction(
    league_id: str,
    db: DbSession,
    background_tasks: BackgroundTasks,
    _: AdminKeyVerified,
) -> RunPredictionResponse:
    """Queue a Monte Carlo simulation run for a league (admin only).

    The simulation is executed asynchronously in the background.  Poll
    ``GET /api/predictions/{league_id}`` to retrieve results once complete.

    Args:
        league_id: UUID of the league to simulate.

    Returns:
        202 Accepted with a confirmation message.

    Raises:
        HTTPException 404: If the league does not exist.

    TODO: Replace background task with a proper task queue (e.g. ARQ/Celery)
          to support progress tracking and retries.
    """
    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    logger.info("prediction_run_queued", league_id=league_id)
    background_tasks.add_task(_run_prediction_task, league_id)

    return RunPredictionResponse(
        message="Prediction job queued. Results will be available shortly.",
        league_id=league_id,
    )


@router.post(
    "/predictions/{league_id}/scenario",
    response_model=ScenarioResponse,
)
async def run_scenario(
    league_id: str,
    scenario: ScenarioRequest,
    db: DbSession,
) -> ScenarioResponse:
    """Run a what-if scenario simulation with locked-in fixture results.

    No admin key required — this is a read-only hypothetical computation.
    Nothing is persisted.

    Args:
        league_id: UUID of the league.
        scenario: Locked-in fixture results to apply before simulating.

    Returns:
        Per-team comparison of baseline vs scenario predicted positions.

    Raises:
        HTTPException 404: If league not found or no baseline prediction exists.
    """
    import copy

    from predictor.engine.pipeline import build_simulation_input, build_team_name_map
    from predictor.engine.simulator import MonteCarloSimulator
    from predictor.engine.standings import apply_result

    league = await db.get(League, league_id)
    if league is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"League {league_id!r} not found.",
        )

    # Get baseline prediction for comparison
    pred_repo = PredictionRepository(db)
    baseline = await pred_repo.get_latest(league_id, league.current_season)
    if baseline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No baseline prediction exists. Run a prediction first.",
        )

    # Build simulation input
    sim_input = await build_simulation_input(db, league_id, league.current_season)
    if sim_input is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No finished matches available for simulation.",
        )

    # Apply locked-in results: remove from remaining fixtures, apply to standings
    locked_set = {
        (lr.home_team_id, lr.away_team_id) for lr in scenario.locked_results
    }
    sim_input.remaining_fixtures = [
        f for f in sim_input.remaining_fixtures
        if (f.home_id, f.away_id) not in locked_set
    ]
    sim_input.current_standings = copy.deepcopy(sim_input.current_standings)
    for lr in scenario.locked_results:
        apply_result(
            sim_input.current_standings,
            lr.home_team_id,
            lr.away_team_id,
            lr.home_goals,
            lr.away_goals,
        )

    # Run reduced simulation
    n_sims = 2_000
    simulator = MonteCarloSimulator(n_simulations=n_sims)
    predictions = simulator.run(sim_input)
    results_dict = simulator.results_to_dict(predictions)

    # Build comparison
    team_names = await build_team_name_map(db, league_id)
    teams = []
    for team_id, data in results_dict.items():
        if team_id.startswith("__"):
            continue
        baseline_pos = baseline.results.get(team_id, {}).get("mean_pos", 0.0)
        scenario_pos = data.get("mean_pos", 0.0)
        teams.append(ScenarioTeamResult(
            team_id=team_id,
            team_name=team_names.get(team_id, team_id),
            mean_pos=round(scenario_pos, 2),
            mean_points=round(data.get("mean_points", 0.0), 2),
            baseline_mean_pos=round(baseline_pos, 2),
            pos_change=round(scenario_pos - baseline_pos, 2),
        ))
    teams.sort(key=lambda t: t.mean_pos)

    return ScenarioResponse(
        teams=teams,
        simulation_runs=n_sims,
        locked_count=len(scenario.locked_results),
    )


async def _run_prediction_task(league_id: str) -> None:
    """Background task: run Monte Carlo simulation and persist results.

    Uses the shared pipeline to build a :class:`SimulationInput`, runs
    10,000 Monte Carlo simulations, and saves the result.

    Args:
        league_id: UUID of the league.
    """
    from predictor.db.session import get_session_factory
    from predictor.db.models import League
    from predictor.db.repos.prediction import PredictionRepository
    from predictor.engine.pipeline import build_simulation_input, build_team_name_map
    from predictor.engine.significance import rank_fixtures_by_significance
    from predictor.engine.simulator import MonteCarloSimulator

    logger.info("prediction_task_started", league_id=league_id)
    try:
        factory = get_session_factory()
        async with factory() as session:
            league = await session.get(League, league_id)
            if league is None:
                logger.error("prediction_task_league_not_found", league_id=league_id)
                return

            sim_input = await build_simulation_input(
                session, league_id, league.current_season
            )
            if sim_input is None:
                logger.warning("prediction_task_no_finished_matches", league_id=league_id)
                return

            simulator = MonteCarloSimulator(n_simulations=10_000)
            predictions = simulator.run(sim_input)
            results_dict = simulator.results_to_dict(predictions)

            # Compute match significance index
            try:
                team_names = await build_team_name_map(session, league_id)
                key_matches = rank_fixtures_by_significance(
                    sim_input, predictions, team_names,
                    n_simulations=1_000, max_fixtures=10, rng_seed=42,
                )
                results_dict["__meta__"] = {
                    "key_matches": [
                        {
                            "home_id": km.home_id,
                            "away_id": km.away_id,
                            "home_name": km.home_name,
                            "away_name": km.away_name,
                            "significance_score": round(km.significance_score, 3),
                            "shift_home_win": round(km.shift_home_win, 3),
                            "shift_draw": round(km.shift_draw, 3),
                            "shift_away_win": round(km.shift_away_win, 3),
                        }
                        for km in key_matches
                    ]
                }
            except Exception as sig_exc:
                logger.warning(
                    "significance_computation_failed",
                    league_id=league_id, error=str(sig_exc),
                )

            pred_repo = PredictionRepository(session)
            await pred_repo.save(
                league_id=league_id,
                season=league.current_season,
                simulation_runs=10_000,
                results=results_dict,
            )
            await session.commit()

        logger.info("prediction_task_completed", league_id=league_id)
    except Exception as exc:
        logger.error("prediction_task_failed", league_id=league_id, error=str(exc))
        raise
