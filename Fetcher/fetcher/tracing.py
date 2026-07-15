"""Опциональный OpenTelemetry-трейсинг для Fetcher (task 5 / observability).

No-op, если OTel-библиотеки не установлены ИЛИ не задан endpoint. Включается, когда
установлены opentelemetry-* и задан OTEL_EXPORTER_OTLP_ENDPOINT. Тот же паттерн,
что в backend/app/tracing.py и DataProcessor/api/main.py — для сквозных трейсов
Backend → Fetcher → DataProcessor при общем коллекторе.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing(app, service_name: str = "fetcher") -> bool:
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
        logger.info("OTel libs not installed -> tracing disabled")
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
