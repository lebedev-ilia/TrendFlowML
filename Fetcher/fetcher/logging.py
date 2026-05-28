"""
Structured logging для Fetcher.

Поддерживает JSON-формат с обязательными полями:
- run_id
- stage
- level
- timestamp
- platform / platform_video_id (опционально)

Соответствует требованиям из `Fetcher/docs/checklist.md` (Phase 2 — Logging).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .config import settings
from .db import session_scope
from .models import FetchLog

# Импортируем handlers для централизованного логирования (опционально)
try:
    from .logging_handlers import setup_centralized_logging
except ImportError:
    setup_centralized_logging = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter для structured logging в Fetcher."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует log record в JSON-строку."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Добавляем структурированные поля из extra, если они есть
        if hasattr(record, "run_id"):
            log_data["run_id"] = str(record.run_id)
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "platform"):
            log_data["platform"] = record.platform
        if hasattr(record, "platform_video_id"):
            log_data["platform_video_id"] = record.platform_video_id

        # Добавляем exception info, если есть
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Настроить логирование для Fetcher.

    Поддерживает:
    - Логирование в stdout (всегда)
    - Централизованное логирование (Loki, Elasticsearch, CloudWatch) - опционально

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR).
        log_format: Формат логов ("json" или "text").
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Удаляем существующие handlers
    root_logger.handlers.clear()

    # 1. Стандартный handler для stdout (всегда)
    handler = logging.StreamHandler()
    handler.setLevel(level)

    if log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # 2. Централизованное логирование (опционально)
    if setup_centralized_logging and hasattr(settings, "logging_backend") and settings.logging_backend:
        backend = settings.logging_backend.lower()
        kwargs: dict[str, Any] = {}

        if backend == "loki":
            if hasattr(settings, "logging_loki_url") and settings.logging_loki_url:
                kwargs["loki_url"] = settings.logging_loki_url
                kwargs["labels"] = {"job": "fetcher", "component": "fetcher"}
        elif backend == "elasticsearch":
            if hasattr(settings, "logging_elasticsearch_url") and settings.logging_elasticsearch_url:
                kwargs["es_url"] = settings.logging_elasticsearch_url
                kwargs["index"] = getattr(settings, "logging_elasticsearch_index", "fetcher-logs")
        elif backend == "cloudwatch":
            kwargs["log_group"] = getattr(settings, "logging_cloudwatch_log_group", "/aws/fetcher")
            kwargs["region_name"] = getattr(settings, "logging_cloudwatch_region", None)

        if kwargs:
            central_handler = setup_centralized_logging(backend=backend, **kwargs)
            if central_handler:
                central_handler.setLevel(level)
                # Используем тот же formatter для централизованного логирования
                if log_format == "json":
                    central_handler.setFormatter(StructuredFormatter())
                else:
                    central_handler.setFormatter(formatter)
                root_logger.addHandler(central_handler)


def get_logger(name: str) -> logging.Logger:
    """Получить logger с именем модуля."""
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Логировать сообщение с контекстом Fetcher.

    Args:
        logger: Logger instance.
        level: Уровень логирования (logging.INFO, logging.ERROR, etc.).
        message: Текст сообщения.
        run_id: UUID run'а (опционально).
        stage: Стадия pipeline (опционально).
        platform: Платформа (youtube, tiktok, etc.) (опционально).
        platform_video_id: ID видео на платформе (опционально).
        **kwargs: Дополнительные поля для логирования.
    """
    extra: dict[str, Any] = {}
    if run_id:
        extra["run_id"] = run_id
    if stage:
        extra["stage"] = stage
    if platform:
        extra["platform"] = platform
    if platform_video_id:
        extra["platform_video_id"] = platform_video_id
    extra.update(kwargs)

    # Лог в stdout / агрегатор логов
    logger.log(level, message, extra=extra)

    # Дублирование в таблицу fetch_logs (best-effort, ошибки игнорируем)
    try:
        with session_scope() as db:
            db_log = FetchLog(
                run_id=run_id,
                stage=stage,
                level=logging.getLevelName(level).lower(),
                message=message,
            )
            db.add(db_log)
            db.flush()
    except Exception:
        # Не допускаем, чтобы проблемы с БД-логами ломали основной поток.
        return


__all__ = ["setup_logging", "get_logger", "log_with_context", "StructuredFormatter"]

