"""Alembic environment configuration for FootyPredict.

Uses SQLAlchemy async engine so it is compatible with aiosqlite / asyncpg.
Migrations run synchronously via run_sync() on the async connection.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Alembic Config object (access to values in alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging (if present)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import application models so Alembic can auto-generate migrations
# ---------------------------------------------------------------------------
# The src layout requires the package to be installed (e.g. `uv pip install -e .`)
# or for the PYTHONPATH to include src/.
from predictor.config import get_settings  # noqa: E402
from predictor.db.models import Base  # noqa: E402

target_metadata = Base.metadata

# Override sqlalchemy.url from application settings so we use the same DB
# as the running application, respecting .env overrides.
config.set_main_option("sqlalchemy.url", get_settings().DATABASE_URL)


# ---------------------------------------------------------------------------
# Helper: run migrations inside an async connection using run_sync
# ---------------------------------------------------------------------------


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations synchronously on the provided connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via run_sync."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    In this mode we don't need an actual DB connection — Alembic just
    generates the SQL statements as a script.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
