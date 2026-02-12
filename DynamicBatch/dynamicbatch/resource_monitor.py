from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict


def _read_proc_stat() -> Optional[Dict[str, int]]:
    """
    Returns CPU jiffies from /proc/stat for aggregate "cpu" line.
    """
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
    """
    Best-effort GPU 0 stats from nvidia-smi.
    Returns dict with keys: gpu_util, mem_used_mb, mem_total_mb.
    """
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
    """
    Best-effort RAM used MB from /proc/meminfo.
    Uses MemTotal - MemAvailable.
    """
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
class ResourcePeaks:
    cpu_util_peak_pct: Optional[float] = None
    ram_used_peak_mb: Optional[float] = None
    gpu_util_peak_pct: Optional[float] = None
    vram_used_peak_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ResourceMonitor:
    """
    Lightweight sampling monitor (no psutil dependency):
    - CPU utilization from /proc/stat deltas
    - RAM used from /proc/meminfo
    - GPU util + VRAM used from nvidia-smi
    """

    def __init__(self, interval_sec: float = 0.25):
        self.interval_sec = float(max(0.05, interval_sec))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peaks = ResourcePeaks()
        self._th = threading.Thread(target=self._loop, name="dynamicbatch_resource_monitor", daemon=True)

    def start(self) -> None:
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

    def _loop(self) -> None:
        prev_cpu = _read_proc_stat()
        while not self._stop.wait(self.interval_sec):
            cur_cpu = _read_proc_stat()
            cpu_util = _cpu_util_percent(prev_cpu, cur_cpu) if prev_cpu and cur_cpu else None
            if cur_cpu:
                prev_cpu = cur_cpu

            ram_used = _mem_used_mb()
            gpu = _gpu_query()

            with self._lock:
                if cpu_util is not None:
                    self._peaks.cpu_util_peak_pct = (
                        cpu_util
                        if self._peaks.cpu_util_peak_pct is None
                        else max(float(self._peaks.cpu_util_peak_pct), float(cpu_util))
                    )
                if ram_used is not None:
                    self._peaks.ram_used_peak_mb = (
                        ram_used
                        if self._peaks.ram_used_peak_mb is None
                        else max(float(self._peaks.ram_used_peak_mb), float(ram_used))
                    )
                if gpu is not None:
                    gu = gpu.get("gpu_util")
                    mu = gpu.get("mem_used_mb")
                    if gu is not None:
                        self._peaks.gpu_util_peak_pct = (
                            float(gu)
                            if self._peaks.gpu_util_peak_pct is None
                            else max(float(self._peaks.gpu_util_peak_pct), float(gu))
                        )
                    if mu is not None:
                        self._peaks.vram_used_peak_mb = (
                            float(mu)
                            if self._peaks.vram_used_peak_mb is None
                            else max(float(self._peaks.vram_used_peak_mb), float(mu))
                        )


