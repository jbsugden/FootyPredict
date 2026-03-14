"""APScheduler configuration for FootyPredict background jobs.

Two nightly jobs are configured:
- ``nightly_sync``  — 02:00 UTC: pull fresh data from all data sources.
- ``daily_predict`` — 03:00 UTC: run Monte Carlo simulations for all leagues.

The scheduler is started during the FastAPI lifespan and shut down cleanly
on application exit.
"""

from __future__ import annotations

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = structlog.get_logger(__name__)

# Module-level scheduler instance
_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Return (or lazily create) the global :class:`AsyncIOScheduler`."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
        _register_jobs(_scheduler)
    return _scheduler


def _register_jobs(scheduler: AsyncIOScheduler) -> None:
    """Add all scheduled jobs to the scheduler instance."""
    scheduler.add_job(
        nightly_sync,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="nightly_sync",
        name="Nightly data sync from all sources",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        daily_predict,
        trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="daily_predict",
        name="Daily Monte Carlo prediction run",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    logger.info("scheduler_jobs_registered", job_count=2)


async def nightly_sync() -> None:
    """Pull fresh data from all configured league data sources (02:00 UTC)."""
    from sqlalchemy import select

    from predictor.config import get_settings
    from predictor.data.fa_fulltime_scraper import FAFullTimeScraper
    from predictor.data.football_data_org import FootballDataOrgSource
    from predictor.data.importer import DataImporter
    from predictor.db.models import DataSource, League
    from predictor.db.repos.match import MatchRepository
    from predictor.db.repos.team import TeamRepository
    from predictor.db.session import get_session_factory
    from predictor.engine.team_season import rebuild_team_seasons

    logger.info("nightly_sync_started")
    settings = get_settings()

    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(select(League).order_by(League.tier))
            leagues = list(result.scalars().all())

            for league in leagues:
                try:
                    if league.data_source == DataSource.API_FOOTBALL_DATA:
                        source = FootballDataOrgSource(api_key=settings.FOOTBALL_DATA_API_KEY)
                    else:
                        source = FAFullTimeScraper()

                    importer = DataImporter(
                        source=source,
                        team_repo=TeamRepository(session),
                        match_repo=MatchRepository(session),
                    )
                    stats = await importer.sync_league(league)
                    await session.flush()
                    await rebuild_team_seasons(session, league)
                    await session.commit()
                    await source.aclose()
                    logger.info("nightly_sync_league_done", league=league.code, **stats)

                except Exception as exc:
                    logger.error("nightly_sync_league_failed", league=league.code, error=str(exc))
                    await session.rollback()

        logger.info("nightly_sync_completed")
    except Exception as exc:
        logger.error("nightly_sync_failed", error=str(exc), exc_info=True)
        raise


async def daily_predict() -> None:
    """Run Monte Carlo simulations for all active leagues (03:00 UTC)."""
    from sqlalchemy import select

    from predictor.db.models import League, MatchStatus
    from predictor.db.repos.match import MatchRepository
    from predictor.db.repos.prediction import PredictionRepository
    from predictor.db.session import get_session_factory
    from predictor.engine.poisson import MatchRecord, StrengthCalculator
    from predictor.engine.simulator import Fixture, MonteCarloSimulator, SimulationInput
    from predictor.engine.standings import apply_result, initialise_standings

    logger.info("daily_predict_started")
    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(select(League))
            leagues = list(result.scalars().all())

            for league in leagues:
                try:
                    match_repo = MatchRepository(session)
                    finished = await match_repo.get_finished(league.id, league.current_season)
                    scheduled = await match_repo.get_scheduled(league.id, league.current_season)

                    if not finished:
                        logger.info("daily_predict_no_data", league=league.code)
                        continue

                    # Include previous-season matches for cross-season prior
                    from predictor.db.repos.match import previous_season
                    prev = previous_season(league.current_season)
                    prev_finished = await match_repo.get_finished_multi_season(
                        league.id, [prev]
                    )

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

                    total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in finished)
                    league_avg = total_goals / (len(finished) * 2) if finished else 1.4

                    all_team_ids = list(
                        {m.home_team_id for m in finished} | {m.away_team_id for m in finished}
                    )
                    current_standings = initialise_standings(all_team_ids)
                    for m in finished:
                        apply_result(
                            current_standings,
                            m.home_team_id,
                            m.away_team_id,
                            m.home_goals or 0,
                            m.away_goals or 0,
                        )

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
                        league_id=league.id,
                        season=league.current_season,
                        simulation_runs=10_000,
                        results=results_dict,
                    )
                    await session.commit()
                    logger.info("daily_predict_league_done", league=league.code)

                except Exception as exc:
                    logger.error("daily_predict_league_failed", league=league.code, error=str(exc))
                    await session.rollback()

        logger.info("daily_predict_completed")
    except Exception as exc:
        logger.error("daily_predict_failed", error=str(exc), exc_info=True)
        raise


async def start_scheduler() -> None:
    """Start the scheduler (call from FastAPI lifespan startup)."""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler_started")


async def stop_scheduler() -> None:
    """Gracefully shut down the scheduler (call from FastAPI lifespan shutdown)."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")
