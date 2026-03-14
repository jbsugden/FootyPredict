"""Database session management for FootyPredict.

Provides:
- async engine created from application settings
- async session factory
- ``get_db`` FastAPI dependency that yields a session per request
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from predictor.config import Settings, get_settings

logger = structlog.get_logger(__name__)

# Module-level singletons — initialised lazily on first use via get_engine().
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return (or create) the module-level async engine.

    Args:
        settings: Application settings. Defaults to ``get_settings()``.

    Returns:
        The shared :class:`AsyncEngine` instance.
    """
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        connect_args: dict = {}
        if "sqlite" in cfg.DATABASE_URL:
            # SQLite requires check_same_thread=False when used across coroutines
            connect_args["check_same_thread"] = False
        _engine = create_async_engine(
            cfg.DATABASE_URL,
            connect_args=connect_args,
            echo=cfg.DEBUG,
            # Use a smaller pool for SQLite (which doesn't support concurrent writes)
            pool_pre_ping=True,
        )
        logger.info("database_engine_created", url=cfg.DATABASE_URL)
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return (or create) the module-level async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(settings)
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a database session for the current request.

    The session is committed on clean exit and rolled back on exception.

    Usage::

        @router.get("/example")
        async def example(db: Annotated[AsyncSession, Depends(get_db)]):
            ...
    """
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """Create all tables defined in the ORM models (for development / testing).

    In production use Alembic migrations instead.
    """
    from predictor.db.models import Base  # local import to avoid circular deps

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def dispose_engine() -> None:
    """Dispose the engine and release all connections (call on app shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_engine_disposed")
