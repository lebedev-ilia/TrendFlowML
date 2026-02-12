from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MemInfo:
    total_mb: int
    used_mb: int
    free_mb: int


def _mb(x: float) -> int:
    return int(round(float(x)))


def probe_cpu_mem_mb() -> Optional[MemInfo]:
    """
    Best-effort Linux RAM probe (no external deps).
    Uses /proc/meminfo.
    """
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None

    mem_total_kb = None
    mem_avail_kb = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
    except Exception:
        return None

    if mem_total_kb is None or mem_avail_kb is None:
        return None

    total_mb = _mb(mem_total_kb / 1024.0)
    free_mb = _mb(mem_avail_kb / 1024.0)
    used_mb = max(0, total_mb - free_mb)
    return MemInfo(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)


def _run_nvidia_smi() -> Optional[Tuple[int, int]]:
    """
    Returns (total_mb, used_mb) for GPU 0.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used",
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
    if len(parts) < 2:
        return None
    try:
        total_mb = int(float(parts[0]))
        used_mb = int(float(parts[1]))
    except Exception:
        return None
    return total_mb, used_mb


def probe_gpu_mem_mb() -> Optional[MemInfo]:
    """
    Best-effort VRAM probe for GPU 0.
    """
    r = _run_nvidia_smi()
    if r is None:
        return None
    total_mb, used_mb = r
    free_mb = max(0, int(total_mb) - int(used_mb))
    return MemInfo(total_mb=int(total_mb), used_mb=int(used_mb), free_mb=int(free_mb))


