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
    """Add all scheduled jobs to the scheduler instance.

    Args:
        scheduler: The scheduler to configure.
    """
    scheduler.add_job(
        nightly_sync,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="nightly_sync",
        name="Nightly data sync from all sources",
        replace_existing=True,
        misfire_grace_time=3600,  # allow up to 1 hour late start
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
    """Pull fresh data from all configured league data sources.

    This job runs at 02:00 UTC every day.

    TODO: Load all active leagues from the DB.
    TODO: Instantiate the correct data source per league (API vs scraper).
    TODO: Call DataImporter.sync_league() for each league.
    TODO: Update TeamSeason ELO ratings after sync.
    TODO: Add error handling and alerting for failed syncs.
    """
    logger.info("nightly_sync_started")
    try:
        # TODO: import and call the real sync logic
        # from predictor.db.session import get_session_factory
        # from predictor.data.importer import DataImporter
        # ...
        logger.info("nightly_sync_completed")
    except Exception as exc:
        logger.error("nightly_sync_failed", error=str(exc), exc_info=True)
        raise


async def daily_predict() -> None:
    """Run Monte Carlo simulations for all active leagues.

    This job runs at 03:00 UTC every day (after nightly_sync has completed).

    TODO: Load all active leagues from the DB.
    TODO: Fetch finished matches + scheduled fixtures per league.
    TODO: Run StrengthCalculator and MonteCarloSimulator.
    TODO: Persist the Prediction record via PredictionRepository.save().
    TODO: Implement retry logic for transient DB failures.
    """
    logger.info("daily_predict_started")
    try:
        # TODO: import and call the real prediction pipeline
        # from predictor.engine.simulator import MonteCarloSimulator
        # ...
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
