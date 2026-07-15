"""Prometheus metrics for the Backend API (task 5 / P4).

Exposes RED metrics for HTTP (rate/errors/duration) plus Celery broker queue
depth, on ``GET /metrics``. Self-contained: only depends on ``prometheus_client``
(+ ``redis`` for queue depth, already a backend dependency).

Wire-up (in app/main.py):
    from .metrics import PrometheusMiddleware, register_celery_queue_collector, metrics_router
    app.add_middleware(PrometheusMiddleware)
    app.include_router(metrics_router)
    register_celery_queue_collector(broker_url, queues)   # on startup
"""
from __future__ import annotations

import os
import time
from typing import Iterable, List

from fastapi import APIRouter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS = Counter(
    "backend_http_requests_total",
    "Total HTTP requests handled by the backend.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "backend_http_request_duration_seconds",
    "HTTP request latency (seconds).",
    ["method", "path"],
)


def _route_template(request: Request) -> str:
    """Low-cardinality label: matched route template, not the raw URL."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        method = request.method
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            path = _route_template(request)
            # skip the scrape endpoint itself to avoid self-noise
            if path != "/metrics":
                HTTP_REQUESTS.labels(method, path, str(status)).inc()
                HTTP_LATENCY.labels(method, path).observe(time.perf_counter() - start)


class _CeleryQueueCollector:
    """Reads list lengths from the Celery (Redis) broker on each scrape."""

    def __init__(self, broker_url: str, queues: List[str]):
        self.broker_url = broker_url
        self.queues = queues

    def collect(self) -> Iterable[GaugeMetricFamily]:
        g = GaugeMetricFamily(
            "backend_celery_queue_length",
            "Number of messages waiting in a Celery (Redis) broker queue.",
            labels=["queue"],
        )
        try:
            import redis  # local import: optional at metric time

            client = redis.from_url(self.broker_url, socket_connect_timeout=2, socket_timeout=2)
            for q in self.queues:
                try:
                    g.add_metric([q], float(client.llen(q)))
                except Exception:
                    continue
        except Exception:
            # broker unreachable -> emit nothing rather than crash the scrape
            return
        yield g


_collector_registered = False


def register_celery_queue_collector(broker_url: str | None, queues: Iterable[str] | None = None) -> None:
    global _collector_registered
    if _collector_registered or not broker_url:
        return
    qs = list(queues) if queues else [
        q.strip() for q in os.environ.get("BACKEND_CELERY_QUEUES", "celery").split(",") if q.strip()
    ]
    try:
        REGISTRY.register(_CeleryQueueCollector(broker_url, qs))
        _collector_registered = True
    except Exception:
        # already registered / registry issue -> ignore
        pass


metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
