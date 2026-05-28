from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import Settings, validate_security_at_startup
from .db import engine, session_scope
from .dbv2 import Base as BaseV2
from .routers.analysis import router as analysis_router
from .routers.auth import router as auth_router
from .routers.channels import router as channels_router
from .routers.health import router as health_router
from .routers.runs import router as runs_router
from .routers.videos import router as videos_router
from .routers.webhooks import router as webhooks_router
from .routers.workspaces import router as workspaces_router
from .services.profiles import seed_public_profiles
from .services.storage import ensure_dirs

settings = Settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    validate_security_at_startup(settings)
    ensure_dirs()
    if settings.db_auto_create:
        # Ensure schema for core tables exists
        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS core"))
        BaseV2.metadata.create_all(bind=engine)
    # Seed публичных профилей из DataProcessor/profiles/*.yaml (см. PROFILES.md)
    try:
        profiles_dir = settings.resolve_paths().dataproc_root / "profiles"
        with session_scope() as db:
            seed_public_profiles(db, profiles_dir)
    except Exception as exc:
        logger.warning(
            "Startup: seed_public_profiles skipped (%s). "
            "Check TF_BACKEND_DATAPROC_ROOT and DB migrations.",
            exc,
            exc_info=bool(settings.debug),
        )


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(workspaces_router)
app.include_router(channels_router)
app.include_router(videos_router)
app.include_router(analysis_router)
app.include_router(runs_router)
app.include_router(webhooks_router, prefix="/api")

