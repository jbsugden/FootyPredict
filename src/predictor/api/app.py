"""FastAPI application factory for FootyPredict.

Call :func:`create_app` to create a configured :class:`FastAPI` instance.
This module is also the Uvicorn entry point (``predictor.api.app:app``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from predictor.api.routes import admin, leagues, predictions
from predictor.db.session import create_all_tables, dispose_engine
from predictor.scheduler import start_scheduler, stop_scheduler
from predictor.web import routes as web_routes

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown side-effects.

    Startup:
    - Create DB tables (dev/test only; use Alembic for production).
    - Start the APScheduler background scheduler.

    Shutdown:
    - Stop the scheduler gracefully.
    - Dispose the DB connection pool.
    """
    # --- Startup ---
    logger.info("application_startup")
    await create_all_tables()
    await start_scheduler()
    yield
    # --- Shutdown ---
    await stop_scheduler()
    await dispose_engine()
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured :class:`FastAPI` instance.
    """
    from predictor.config import get_settings

    settings = get_settings()

    application = FastAPI(
        title="FootyPredict",
        description=(
            "Football league table predictor — "
            "Monte Carlo simulation powered by Dixon-Coles Poisson model."
        ),
        version="0.1.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    import os

    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    static_dir = os.path.abspath(static_dir)
    if os.path.isdir(static_dir):
        application.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ------------------------------------------------------------------
    # API routers
    # ------------------------------------------------------------------
    application.include_router(leagues.router, prefix="/api", tags=["leagues"])
    application.include_router(
        predictions.router, prefix="/api", tags=["predictions"]
    )
    application.include_router(admin.router, prefix="/admin", tags=["admin"])

    # ------------------------------------------------------------------
    # Web (Jinja2 page) router
    # ------------------------------------------------------------------
    application.include_router(web_routes.router, tags=["web"])

    logger.info("application_created", debug=settings.DEBUG)
    return application


# Module-level app instance consumed by Uvicorn
app = create_app()


def main() -> None:
    """Entry point for ``footypredict`` CLI command."""
    import uvicorn

    from predictor.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "predictor.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )


if __name__ == "__main__":
    main()
