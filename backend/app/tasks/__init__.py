"""
Celery tasks: точка входа для worker (`include=["app.tasks"]`).

Реализация разнесена: `analysis` (AnalysisJob), `ingestion` (Fetcher + sync),
`events` (Redis pub/sub), `manifest` (manifest.json / артефакты).
"""

from __future__ import annotations

from .analysis import process_analysis_job
from .ingestion import process_ingestion_run, sync_ingestion_run_status
from .manifest import (
    _register_artifact,
    _scan_and_register_artifacts,
    _sync_from_manifest_v2,
)

__all__ = [
    "process_analysis_job",
    "process_ingestion_run",
    "sync_ingestion_run_status",
    "_register_artifact",
    "_scan_and_register_artifacts",
    "_sync_from_manifest_v2",
]
