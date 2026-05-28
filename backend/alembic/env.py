from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool


# Allow `backend/` to import `app.*`
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.config import Settings  # noqa: E402
from app.models import Base as LegacyBase  # noqa: E402
from app.dbv2 import Base as BaseV2  # noqa: E402
from app.dbv2 import models as _models  # noqa: F401,E402  (ensure models are imported)


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Manage both:
# - legacy tables in `public` schema (`app/models.py`)
# - v2 tables in `core` schema (`app/dbv2/*`)
target_metadata = [LegacyBase.metadata, BaseV2.metadata]


def _get_url() -> str:
    # Prefer explicit env var if present, otherwise fall back to Settings.
    env_url = os.environ.get("TF_BACKEND_DB_DSN")
    if env_url:
        return env_url
    return Settings().db_dsn


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


