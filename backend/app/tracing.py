"""Опциональный OpenTelemetry-трейсинг для backend (task 5 / P9-tracing).

No-op, если OTel-библиотеки не установлены ИЛИ не задан endpoint. Включается, когда:
  * установлены пакеты opentelemetry-* (см. requirements, секция optional);
  * задан OTEL_EXPORTER_OTLP_ENDPOINT (напр. http://jaeger:4317 или collector).

Паттерн повторяет DataProcessor/api/main.py, чтобы трейсы были сквозными
(Backend → DataProcessor) при общем коллекторе.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(app, service_name: str = "backend") -> bool:
    """Инструментирует FastAPI-приложение OTel-трейсингом. Возвращает True, если включено."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.info("OTel libs not installed -> tracing disabled (set up opentelemetry-* to enable)")
        return False

    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OTel tracing enabled -> %s", endpoint)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("OTel tracing setup failed (%s) -> disabled", exc)
        return False
