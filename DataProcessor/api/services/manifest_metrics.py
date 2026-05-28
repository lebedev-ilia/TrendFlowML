"""
Запись метрик Prometheus на основе manifest.json (длительности по компонентам).

Вызывается из ProcessorService после завершения subprocess main.py, чтобы
`dataprocessor_component_stage_seconds` отражал реальные тайминги, а не только
сквозной `dataprocessor_processing_seconds` с pipeline/main_py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _worker_id() -> str:
    try:
        from api.worker import get_worker_id

        return get_worker_id()
    except Exception:
        return "unknown"


def record_from_manifest(manifest_path: str) -> int:
    """
    Прочитать manifest и для каждого компонента с duration_ms вызвать observe.

    Returns:
        Число компонентов, для которых учтена длительность
    """
    if not os.path.isfile(manifest_path):
        return 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f) or {}
    except Exception as e:
        logger.debug("manifest_metrics: read failed %s: %s", manifest_path, e)
        return 0

    comps: Optional[List] = data.get("components")
    if not isinstance(comps, list) or not comps:
        return 0

    try:
        from api.services.metrics import component_stage_seconds
    except ImportError:
        return 0

    wid = _worker_id()
    n = 0
    for c in comps:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str) or not name:
            continue
        kind = str(c.get("kind") or "other")
        status = str(c.get("status") or "error")
        duration_ms = c.get("duration_ms")
        if duration_ms is None:
            continue
        try:
            sec = float(duration_ms) / 1000.0
        except (TypeError, ValueError):
            continue
        if sec < 0 or sec > 24 * 3600:
            continue
        try:
            component_stage_seconds.labels(
                worker_id=wid,
                component=name,
                component_kind=kind,
                status=status,
            ).observe(sec)
            n += 1
        except Exception as e:
            logger.debug("component_stage observe failed: %s", e)
    return n
