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

    from predictor.db.models import League
    from predictor.db.repos.prediction import PredictionRepository
    from predictor.db.session import get_session_factory
    from predictor.db.repos.match import MatchRepository
    from predictor.engine.completeness import check_fixture_completeness
    from predictor.engine.pipeline import build_simulation_input, build_team_name_map
    from predictor.engine.significance import rank_fixtures_by_significance
    from predictor.engine.simulator import MonteCarloSimulator

    logger.info("daily_predict_started")
    try:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(select(League))
            leagues = list(result.scalars().all())

            for league in leagues:
                try:
                    sim_input = await build_simulation_input(
                        session, league.id, league.current_season
                    )
                    if sim_input is None:
                        logger.info("daily_predict_no_data", league=league.code)
                        continue

                    simulator = MonteCarloSimulator(n_simulations=10_000)
                    predictions = simulator.run(sim_input)
                    results_dict = simulator.results_to_dict(predictions)

                    # Compute match significance index
                    try:
                        team_names = await build_team_name_map(session, league.id)
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
                            league=league.code, error=str(sig_exc),
                        )

                    # Fixture completeness check
                    try:
                        match_repo = MatchRepository(session)
                        all_matches = await match_repo.get_by_league_season(
                            league.id, league.current_season
                        )
                        try:
                            team_names  # noqa: B018 — reuse from significance block
                        except NameError:
                            team_names = await build_team_name_map(session, league.id)
                        completeness = check_fixture_completeness(
                            sim_input.team_ids, all_matches, team_names,
                        )
                        results_dict.setdefault("__meta__", {})["fixture_completeness"] = completeness
                    except Exception as comp_exc:
                        logger.warning(
                            "completeness_check_failed",
                            league=league.code, error=str(comp_exc),
                        )

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
