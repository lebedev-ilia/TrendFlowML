"""
Каркас телеметрии оркестратора AudioProcessor: снимки ресурсов вокруг каждого экстрактора.

Внешние оркестраторы читают события из scheduler_runtime_report.json (ключ orchestrator_telemetry).
См. docs/ORCHESTRATOR_TELEMETRY.md.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def telemetry_enabled_from_env() -> bool:
    v = os.environ.get("AP_ORCHESTRATOR_TELEMETRY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resource_snapshot(*, include_children: bool) -> Dict[str, Any]:
    """Best-effort снимок процесса (и опционально дерева дочерних)."""
    snap: Dict[str, Any] = {"ts_monotonic": time.perf_counter()}
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        mi = p.memory_info()
        snap["rss_mb"] = float(mi.rss) / (1024.0 * 1024.0)
        snap["vms_mb"] = float(mi.vms) / (1024.0 * 1024.0)
        try:
            snap["num_threads"] = int(p.num_threads())
        except Exception:
            snap["num_threads"] = None
        if include_children:
            ch_rss = 0
            try:
                for c in p.children(recursive=True):
                    try:
                        ch_rss += int(c.memory_info().rss)
                    except Exception:
                        pass
            except Exception:
                pass
            snap["children_rss_mb"] = float(ch_rss) / (1024.0 * 1024.0)
            snap["rss_tree_mb"] = float(snap["rss_mb"]) + float(snap["children_rss_mb"])
    except Exception:
        snap["rss_mb"] = None
        snap["vms_mb"] = None
        snap["num_threads"] = None

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            snap["gpu_allocated_mb"] = float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
            snap["gpu_reserved_mb"] = float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
        else:
            snap["gpu_allocated_mb"] = None
            snap["gpu_reserved_mb"] = None
    except Exception:
        snap["gpu_allocated_mb"] = None
        snap["gpu_reserved_mb"] = None

    return snap


def _num_delta(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


class OrchestratorTelemetryCollector:
    """
    Собирает события «экстрактор начался / закончился» с парой снимков ресурсов.

    При enabled=False все методы no-op (дешёвая проверка булева флага).
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        include_children: Optional[bool] = None,
        log_each_event: Optional[bool] = None,
    ) -> None:
        if enabled is None:
            enabled = telemetry_enabled_from_env()
        self.enabled = bool(enabled)
        self._include_children = (
            _truthy_env("AP_ORCHESTRATOR_TELEMETRY_CHILDREN")
            if include_children is None
            else bool(include_children)
        )
        self._log_each = (
            _truthy_env("AP_ORCHESTRATOR_TELEMETRY_LOG") if log_each_event is None else bool(log_each_event)
        )
        self.events: List[Dict[str, Any]] = []
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def mark_extractor_start(self, extractor_key: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._pending[str(extractor_key)] = {
                "snap_before": _resource_snapshot(include_children=self._include_children),
            }

    def mark_extractor_end(
        self,
        extractor_key: str,
        wall_ms: float,
        success: bool,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        key = str(extractor_key)
        with self._lock:
            pend = self._pending.pop(key, None)
            snap_after = _resource_snapshot(include_children=self._include_children)
            evt: Dict[str, Any] = {
                "extractor_key": key,
                "wall_ms": float(wall_ms),
                "success": bool(success),
                "snap_before": (pend or {}).get("snap_before"),
                "snap_after": snap_after,
            }
            if context:
                evt["context"] = dict(context)
            b = evt["snap_before"] or {}
            a = snap_after
            evt["delta"] = {
                "rss_mb": _num_delta(b.get("rss_mb"), a.get("rss_mb")),
                "rss_tree_mb": _num_delta(b.get("rss_tree_mb"), a.get("rss_tree_mb")),
                "gpu_allocated_mb": _num_delta(b.get("gpu_allocated_mb"), a.get("gpu_allocated_mb")),
                "gpu_reserved_mb": _num_delta(b.get("gpu_reserved_mb"), a.get("gpu_reserved_mb")),
            }
            self.events.append(evt)
        if self._log_each:
            try:
                logger.info("orchestrator_telemetry %s", json.dumps(evt, ensure_ascii=False, default=str))
            except Exception:
                logger.info("orchestrator_telemetry extractor=%s wall_ms=%s", key, wall_ms)

    def cancel_pending(self, extractor_key: str) -> None:
        """Если экстрактор не дошёл до run — снять висящий start (редко)."""
        with self._lock:
            self._pending.pop(str(extractor_key), None)

    def to_report_section(self) -> Dict[str, Any]:
        with self._lock:
            ev = list(self.events)
        return {
            "schema_version": "orchestrator_telemetry_v1",
            "host": socket.gethostname(),
            "include_children": self._include_children,
            "events": ev,
        }
