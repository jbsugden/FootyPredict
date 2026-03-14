"""API routes for Monte Carlo predictions."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

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


async def _run_prediction_task(league_id: str) -> None:
    """Background task: run Monte Carlo simulation and persist results.

    Pipeline:
      1. Open a fresh DB session (independent of the triggering request).
      2. Load finished matches and scheduled fixtures.
      3. Build StrengthCalculator from finished matches.
      4. Build current standings from finished matches.
      5. Run MonteCarloSimulator.
      6. Save via PredictionRepository.save().

    Args:
        league_id: UUID of the league.
    """
    from predictor.db.session import get_session_factory
    from predictor.db.models import League
    from predictor.db.repos.match import MatchRepository
    from predictor.db.repos.prediction import PredictionRepository
    from predictor.engine.poisson import MatchRecord, StrengthCalculator
    from predictor.engine.simulator import Fixture, MonteCarloSimulator, SimulationInput
    from predictor.engine.standings import apply_result, initialise_standings

    logger.info("prediction_task_started", league_id=league_id)
    try:
        factory = get_session_factory()
        async with factory() as session:
            league = await session.get(League, league_id)
            if league is None:
                logger.error("prediction_task_league_not_found", league_id=league_id)
                return

            match_repo = MatchRepository(session)
            finished = await match_repo.get_finished(league_id, league.current_season)
            scheduled = await match_repo.get_scheduled(league_id, league.current_season)

            if not finished:
                logger.warning("prediction_task_no_finished_matches", league_id=league_id)
                return

            # Include previous-season matches for cross-season prior
            from predictor.db.repos.match import previous_season
            prev = previous_season(league.current_season)
            prev_finished = await match_repo.get_finished_multi_season(
                league_id, [prev]
            )

            # Build StrengthCalculator from current + previous season
            match_records = [
                MatchRecord(
                    home_team_id=m.home_team_id,
                    away_team_id=m.away_team_id,
                    home_goals=m.home_goals or 0,
                    away_goals=m.away_goals or 0,
                    played_at=m.played_at,
                )
                for m in prev_finished + finished
            ]
            strength_calc = StrengthCalculator(match_records)
            strength_calc.compute_strengths()

            # Compute league average goals
            total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in finished)
            league_avg = total_goals / (len(finished) * 2) if finished else 1.4

            # Build current standings from finished matches
            all_team_ids = list({m.home_team_id for m in finished} | {m.away_team_id for m in finished})
            current_standings = initialise_standings(all_team_ids)
            for m in finished:
                apply_result(
                    current_standings,
                    m.home_team_id,
                    m.away_team_id,
                    m.home_goals or 0,
                    m.away_goals or 0,
                )

            # Build remaining fixtures
            remaining_fixtures = [
                Fixture(home_id=m.home_team_id, away_id=m.away_team_id)
                for m in scheduled
            ]

            sim_input = SimulationInput(
                team_ids=all_team_ids,
                current_standings=current_standings,
                remaining_fixtures=remaining_fixtures,
                strength_calculator=strength_calc,
                league_avg_goals=league_avg,
            )

            simulator = MonteCarloSimulator(n_simulations=10_000)
            predictions = simulator.run(sim_input)
            results_dict = simulator.results_to_dict(predictions)

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
