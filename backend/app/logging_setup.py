"""Структурные JSON-логи + correlation_id для backend (task 5 / P9).

Зависимостей нет (stdlib). Даёт:
  * configure_logging()        — JSON-формат в stdout, уровень из настроек;
  * CorrelationIdMiddleware    — берёт/генерит X-Request-ID на каждый HTTP-запрос;
  * set_correlation_id()       — для Celery-задач (correlation_id = run_id);
  * во все записи лога добавляется поле correlation_id (если задано).

Включение (в app/main.py):
    from .logging_setup import configure_logging, CorrelationIdMiddleware
    configure_logging(level=..., json_format=...)
    app.add_middleware(CorrelationIdMiddleware)
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(value: Optional[str]) -> None:
    _correlation_id.set(value)


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


class _CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Минимальный JSON-форматтер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _dt.datetime.fromtimestamp(record.created, _dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": "backend",
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # дополнительные поля (logger.info(..., extra={...}))
        for k, v in record.__dict__.items():
            if k not in payload and k not in _RESERVED:
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"correlation_id", "message", "asctime", "taskName"}


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper() if isinstance(level, str) else level)
    handler = logging.StreamHandler()
    handler.addFilter(_CorrelationIdFilter())
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s")
        )
    root.handlers = [handler]


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get(self.HEADER) or uuid.uuid4().hex
        token = _correlation_id.set(cid)
        try:
            response = await call_next(request)
            response.headers[self.HEADER] = cid
            return response
        finally:
            _correlation_id.reset(token)
