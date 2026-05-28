from __future__ import annotations

import logging
from typing import Any

import redis
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from ..config import Settings
from ..db import engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
settings = Settings()


@router.get("/health/live")
def health_live() -> dict[str, str]:
    """Liveness: процесс отвечает (orchestrators / LB)."""
    return {"status": "live"}


@router.get("/health")
def health_root() -> dict[str, str]:
    """Alias для окружений, где ожидают именно GET /health."""
    return {"status": "live"}


@router.get("/health/ready")
def health_ready() -> dict[str, Any]:
    """Readiness: PostgreSQL и Redis доступны."""
    failed: list[str] = []
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("health_ready: database unreachable: %s", exc)
        failed.append("database")
    try:
        client = redis.from_url(settings.redis_url)
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:
        logger.warning("health_ready: redis unreachable: %s", exc)
        failed.append("redis")
    if failed:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks_failed": failed},
        )
    return {"status": "ready", "database": "ok", "redis": "ok"}
