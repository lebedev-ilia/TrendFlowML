"""
Prometheus Metrics Service

Этот модуль предоставляет метрики Prometheus для мониторинга DataProcessor API.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2167-2217)
"""

import logging
import os
import threading
import time as time_module

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

_HOST_LOCK = threading.Lock()
_HOST_LAST_REFRESH = 0.0
_HOST_REFRESH_SEC = 3.0

# ============================================================================
# Queue метрики
# ============================================================================

queue_length = Gauge(
    "dataprocessor_queue_length",
    "Current queue length",
    ["priority"]  # high, normal, low
)

queue_wait_time = Histogram(
    "dataprocessor_queue_wait_seconds",
    "Time spent waiting in queue",
    buckets=[10, 30, 60, 300, 600]
)

# ============================================================================
# Processing метрики
# ============================================================================

processing_time = Histogram(
    "dataprocessor_processing_seconds",
    "Processing time per run",
    ["processor", "component"],
    buckets=[60, 300, 600, 1800, 3600]
)

failure_rate = Counter(
    "dataprocessor_failures_total",
    "Total failures",
    ["processor", "component", "error_type"]
)

# ============================================================================
# Resource метрики
# ============================================================================

memory_usage = Gauge(
    "dataprocessor_memory_bytes",
    "Memory usage per run",
    ["run_id"]
)

active_runs = Gauge(
    "dataprocessor_active_runs",
    "Current number of active runs"
)

crashed_runs = Counter(
    "dataprocessor_crashed_runs_total",
    "Total crashed runs (no heartbeat)"
)

# Per-component wall time (из manifest.json после прогона main.py)
component_stage_seconds = Histogram(
    "dataprocessor_component_stage_seconds",
    "Wall time of a single component from run manifest (seconds)",
    ["worker_id", "component", "component_kind", "status"],
    buckets=[
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
        1800.0,
        3600.0,
    ],
)

# Снимок хоста (обновляется не чаще чем раз в _HOST_REFRESH_SEC при scrape /metrics)
host_cpu_ratio = Gauge(
    "dataprocessor_host_cpu_usage_ratio",
    "Host CPU usage ratio 0..1 (non-idle, short sample via psutil)",
)
host_load1 = Gauge(
    "dataprocessor_host_load1",
    "Host load average 1m (Unix getloadavg), or 0 if unavailable",
)
host_memory_available_bytes = Gauge(
    "dataprocessor_host_memory_available_bytes",
    "Host available RAM bytes (psutil.virtual_memory().available)",
)
host_disk_free_bytes = Gauge(
    "dataprocessor_host_disk_free_bytes",
    "Free space on a mount (bytes)",
    ["mountpoint"],
)
host_num_cpus = Gauge(
    "dataprocessor_host_num_cpus",
    "Logical CPU count on host (psutil.cpu_count)",
)

# ============================================================================
# Вспомогательные функции
# ============================================================================

def _refresh_host_resource_gauges() -> None:
    """Периодически обновить gauges по хосту (psutil, лёгкий throttle)."""
    global _HOST_LAST_REFRESH
    now = time_module.time()
    with _HOST_LOCK:
        if now - _HOST_LAST_REFRESH < _HOST_REFRESH_SEC:
            return
        _HOST_LAST_REFRESH = now
    try:
        import psutil
    except ImportError:
        return
    try:
        p = float(psutil.cpu_percent(interval=None)) / 100.0
        host_cpu_ratio.set(min(1.0, max(0.0, p)))
    except Exception as e:
        logger.debug("host_cpu_ratio update failed: %s", e)
    try:
        host_num_cpus.set(int(psutil.cpu_count(logical=True) or 0))
    except Exception:
        pass
    try:
        host_memory_available_bytes.set(int(getattr(psutil.virtual_memory(), "available", 0)))
    except Exception as e:
        logger.debug("host_memory update failed: %s", e)
    try:
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            if la:
                host_load1.set(float(la[0]))
    except (OSError, AttributeError) as e:
        logger.debug("host load update skipped: %s", e)
    for mp in ("/", os.environ.get("DATAPROCESSOR_HOST_DISK_PATH", "")):
        if not (isinstance(mp, str) and mp.strip()):
            continue
        m = mp.strip()
        try:
            host_disk_free_bytes.labels(mountpoint=m).set(
                int(getattr(psutil.disk_usage(m), "free", 0))
            )
        except Exception as e:
            logger.debug("host disk %s: %s", m, e)


def get_metrics() -> bytes:
    """
    Получить метрики в формате Prometheus.
    
    Returns:
        Байты с метриками в формате Prometheus text format
    """
    _refresh_host_resource_gauges()
    return generate_latest()


def get_metrics_content_type() -> str:
    """
    Получить Content-Type для метрик.
    
    Returns:
        Content-Type для Prometheus метрик
    """
    return CONTENT_TYPE_LATEST

