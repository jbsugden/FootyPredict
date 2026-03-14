"""Application configuration.

Settings are read from environment variables / a .env file using Pydantic Settings.
Copy .env.example to .env and fill in the values before running the app.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for FootyPredict.

    All fields can be overridden by environment variables (case-insensitive)
    or by values in a .env file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # External data source credentials
    # ------------------------------------------------------------------
    FOOTBALL_DATA_API_KEY: str = ""
    """API key for https://www.football-data.org/ (free tier available)."""

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = "sqlite+aiosqlite:///./footypredict.db"
    """Async SQLAlchemy database URL.

    Examples:
        sqlite+aiosqlite:///./footypredict.db   (default, development)
        postgresql+asyncpg://user:pw@localhost/footypredict  (production)
    """

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    ADMIN_API_KEY: str = "change_me"
    """Bearer token required for /admin/* endpoints.

    Change this before deploying. Generate with:
        python -c "import secrets; print(secrets.token_hex(32))"
    """

    # ------------------------------------------------------------------
    # Application behaviour
    # ------------------------------------------------------------------
    DEBUG: bool = False
    """Enable debug mode: verbose logging and Uvicorn auto-reload."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Using lru_cache means the .env file is parsed only once per process.
    In tests, call ``get_settings.cache_clear()`` after monkeypatching env vars.
    """
    return Settings()
