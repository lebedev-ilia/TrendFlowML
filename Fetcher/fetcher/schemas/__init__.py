"""Pydantic схемы для Fetcher API (пакет: health, stats, api)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response для health check endpoint."""

    status: str = Field(..., description="Общий статус: healthy, degraded, unhealthy")
    api: str = Field(..., description="Статус API сервера")
    version: str = Field(..., description="Версия Fetcher")
    uptime_seconds: float = Field(..., description="Время работы сервиса в секундах")
    dependencies: Dict[str, Any] = Field(
        default_factory=dict,
        description="Статусы зависимостей (database, redis, storage)",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Метрики сервиса (активные runs, etc.)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Время проверки",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
        }


class StatsResponse(BaseModel):
    """Упрощённая схема статистики для stats_aggregator.

    Полная версия также определена в `fetcher/schemas/api.py` для REST API.
    """

    period: str
    runs: Dict[str, int]
    throughput: Dict[str, float]
    cache: Dict[str, Any]
    platforms: Dict[str, int]
    errors: Dict[str, int]


__all__ = ["HealthResponse", "StatsResponse"]
