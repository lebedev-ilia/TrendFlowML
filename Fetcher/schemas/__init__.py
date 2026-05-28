"""
Pydantic-схемы Fetcher для backend-контрактов.

Здесь живут модели:

- manifest.json (контракт Fetcher → DataProcessor),
- события ingestion-пайплайна Fetcher (contracts для Backend/WebSocket),
- общие enum'ы и типы для run/status/platform.

Схемы должны соответствовать описанию в `Fetcher/docs/BACKEND_CONTRACTS.md`.
"""

from .manifest import FetcherManifest
from .events import (
    FetcherEventBase,
    FetcherRunStatusChangedPayload,
    FetcherJobStartedPayload,
    FetcherJobFinishedPayload,
    FetcherJobFailedPayload,
    FetcherLogLinePayload,
    FetcherEvent,
)

__all__ = [
    "FetcherManifest",
    "FetcherEventBase",
    "FetcherRunStatusChangedPayload",
    "FetcherJobStartedPayload",
    "FetcherJobFinishedPayload",
    "FetcherJobFailedPayload",
    "FetcherLogLinePayload",
    "FetcherEvent",
]


