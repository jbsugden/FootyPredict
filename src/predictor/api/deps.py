"""FastAPI dependency providers for FootyPredict.

Centralises common dependencies so they can be injected consistently across
all route handlers using ``Annotated[..., Depends(...)]``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from predictor.config import Settings, get_settings
from predictor.db.session import get_session_factory

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Database session dependency
# ---------------------------------------------------------------------------


async def get_db(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for the duration of a request.

    The session is committed on clean exit and rolled back on exception.

    Usage::

        @router.get("/example")
        async def example(db: DbSession) -> ...:
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


# ---------------------------------------------------------------------------
# Settings dependency
# ---------------------------------------------------------------------------


def get_app_settings() -> Settings:
    """Return the cached application settings.

    Thin wrapper around :func:`~predictor.config.get_settings` so it can be
    used directly as a FastAPI dependency.
    """
    return get_settings()


# ---------------------------------------------------------------------------
# Admin API key dependency
# ---------------------------------------------------------------------------


async def verify_admin_key(
    x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None,
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Verify the ``X-Admin-Key`` header for protected admin endpoints.

    Raises:
        HTTPException 401: If the header is missing.
        HTTPException 403: If the key does not match the configured value.

    Usage::

        @router.post("/admin/sync")
        async def sync(_: Annotated[None, Depends(verify_admin_key)]):
            ...
    """
    if x_admin_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Admin-Key header is required for admin endpoints.",
        )
    if x_admin_key != settings.ADMIN_API_KEY:
        logger.warning("invalid_admin_key_attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key.",
        )


# ---------------------------------------------------------------------------
# Convenience type aliases for route annotations
# ---------------------------------------------------------------------------

DbSession = Annotated[AsyncSession, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]
AdminKeyVerified = Annotated[None, Depends(verify_admin_key)]
