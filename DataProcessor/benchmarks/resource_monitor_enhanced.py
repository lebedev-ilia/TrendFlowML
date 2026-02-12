"""
Enhanced resource monitor with time series data collection.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, List


def _read_proc_stat() -> Optional[Dict[str, int]]:
    """Returns CPU jiffies from /proc/stat for aggregate "cpu" line."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = line.split()
                    vals = [int(x) for x in parts[1:]]
                    keys = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
                    out = {}
                    for i, k in enumerate(keys):
                        if i < len(vals):
                            out[k] = int(vals[i])
                    return out
    except Exception:
        return None
    return None


def _cpu_util_percent(prev: Dict[str, int], cur: Dict[str, int]) -> Optional[float]:
    try:
        prev_total = float(sum(prev.values()))
        cur_total = float(sum(cur.values()))
        dt = cur_total - prev_total
        if dt <= 0:
            return None
        prev_idle = float(prev.get("idle", 0) + prev.get("iowait", 0))
        cur_idle = float(cur.get("idle", 0) + cur.get("iowait", 0))
        didle = cur_idle - prev_idle
        util = max(0.0, min(100.0, 100.0 * (1.0 - (didle / dt))))
        return util
    except Exception:
        return None


def _gpu_query() -> Optional[Dict[str, float]]:
    """Best-effort GPU 0 stats from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,nounits,noheader",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    line = (p.stdout or "").strip().splitlines()[0] if (p.stdout or "").strip() else ""
    if not line:
        return None
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "gpu_util": float(parts[0]),
            "mem_used_mb": float(parts[1]),
            "mem_total_mb": float(parts[2]),
        }
    except Exception:
        return None


def _mem_used_mb() -> Optional[float]:
    """Best-effort RAM used MB from /proc/meminfo."""
    try:
        mem_total_kb = None
        mem_avail_kb = None
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
        if mem_total_kb is None or mem_avail_kb is None:
            return None
        used_kb = max(0, int(mem_total_kb) - int(mem_avail_kb))
        return float(used_kb) / 1024.0
    except Exception:
        return None


@dataclass
class ResourceTimePoint:
    """Single time point in resource monitoring."""
    timestamp_iso: str
    elapsed_sec: float
    cpu_util_pct: Optional[float] = None
    cpu_mem_used_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResourcePeaks:
    cpu_util_peak_pct: Optional[float] = None
    ram_used_peak_mb: Optional[float] = None
    gpu_util_peak_pct: Optional[float] = None
    vram_used_peak_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class EnhancedResourceMonitor:
    """
    Enhanced resource monitor with time series collection.
    """

    def __init__(self, interval_sec: float = 0.25):
        self.interval_sec = float(max(0.05, interval_sec))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peaks = ResourcePeaks()
        self._time_series: List[ResourceTimePoint] = []
        self._start_time: Optional[float] = None
        self._peak_timestamps: Dict[str, str] = {}
        self._th = threading.Thread(target=self._loop, name="enhanced_resource_monitor", daemon=True)

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._th.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop.set()
        try:
            self._th.join(timeout=float(timeout_sec))
        except Exception:
            pass

    @property
    def peaks(self) -> ResourcePeaks:
        with self._lock:
            return ResourcePeaks(**self._peaks.to_dict())

    @property
    def time_series(self) -> List[ResourceTimePoint]:
        with self._lock:
            return [ResourceTimePoint(**tp.to_dict()) for tp in self._time_series]

    @property
    def peak_timestamps(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._peak_timestamps)

    def _loop(self) -> None:
        prev_cpu = _read_proc_stat()
        while not self._stop.wait(self.interval_sec):
            cur_time = time.perf_counter()
            elapsed = cur_time - (self._start_time or cur_time)
            timestamp_iso = datetime.utcnow().isoformat()

            cur_cpu = _read_proc_stat()
            cpu_util = _cpu_util_percent(prev_cpu, cur_cpu) if prev_cpu and cur_cpu else None
            if cur_cpu:
                prev_cpu = cur_cpu

            ram_used = _mem_used_mb()
            gpu = _gpu_query()

            gpu_util = gpu.get("gpu_util") if gpu else None
            gpu_mem = gpu.get("mem_used_mb") if gpu else None

            # Create time point
            time_point = ResourceTimePoint(
                timestamp_iso=timestamp_iso,
                elapsed_sec=elapsed,
                cpu_util_pct=cpu_util,
                cpu_mem_used_mb=ram_used,
                gpu_util_pct=gpu_util,
                gpu_mem_used_mb=gpu_mem,
            )

            with self._lock:
                # Add to time series
                self._time_series.append(time_point)

                # Update peaks and record timestamps
                if cpu_util is not None:
                    if self._peaks.cpu_util_peak_pct is None or cpu_util > self._peaks.cpu_util_peak_pct:
                        self._peaks.cpu_util_peak_pct = cpu_util
                        self._peak_timestamps["cpu_util_peak"] = timestamp_iso
                        self._peak_timestamps["cpu_util_peak_elapsed_sec"] = f"{elapsed:.3f}"

                if ram_used is not None:
                    if self._peaks.ram_used_peak_mb is None or ram_used > self._peaks.ram_used_peak_mb:
                        self._peaks.ram_used_peak_mb = ram_used
                        self._peak_timestamps["ram_peak"] = timestamp_iso
                        self._peak_timestamps["ram_peak_elapsed_sec"] = f"{elapsed:.3f}"

                if gpu_util is not None:
                    if self._peaks.gpu_util_peak_pct is None or gpu_util > self._peaks.gpu_util_peak_pct:
                        self._peaks.gpu_util_peak_pct = gpu_util
                        self._peak_timestamps["gpu_util_peak"] = timestamp_iso
                        self._peak_timestamps["gpu_util_peak_elapsed_sec"] = f"{elapsed:.3f}"

                if gpu_mem is not None:
                    if self._peaks.vram_used_peak_mb is None or gpu_mem > self._peaks.vram_used_peak_mb:
                        self._peaks.vram_used_peak_mb = gpu_mem
                        self._peak_timestamps["vram_peak"] = timestamp_iso
                        self._peak_timestamps["vram_peak_elapsed_sec"] = f"{elapsed:.3f}"

