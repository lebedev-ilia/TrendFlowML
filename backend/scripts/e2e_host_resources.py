"""Снимок ОЗУ / loadavg / GPU для диагностики E2E (Linux, опционально nvidia-smi)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List


def host_resource_snapshot_line() -> str:
    """Одна строка для лога: RAM, load1, GPU (если есть nvidia-smi).

    GPU: значение из nvidia-smi — суммарная память по карте (все процессы: Triton,
    воркеры DataProcessor и т.д.), не «память только текущего Python-процесса».
    """
    parts: list[str] = []
    try:
        mi: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            tok = rest.split()
            if tok and tok[0].isdigit():
                mi[key.strip()] = int(tok[0])
        tot = mi.get("MemTotal", 0)
        avail = mi.get("MemAvailable", mi.get("MemFree", 0))
        if tot > 0:
            used_pct = 100.0 * (tot - avail) / tot
            parts.append(f"RAM {used_pct:.0f}% used ({avail // 1024} MiB avail)")
        else:
            parts.append("RAM ?")
    except OSError:
        parts.append("RAM ?")
    try:
        la = Path("/proc/loadavg").read_text(encoding="utf-8").split()
        if la:
            parts.append(f"load1 {la[0]}")
    except OSError:
        parts.append("load1 ?")
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if r.returncode == 0 and (r.stdout or "").strip():
            gpus: list[str] = []
            for ln in r.stdout.strip().splitlines():
                bits = [x.strip() for x in ln.split(",")]
                if len(bits) >= 3:
                    used, total, util = bits[0], bits[1], bits[2]
                    gpus.append(f"{used}/{total} MiB util {util}%")
            if gpus:
                parts.append("GPU " + " | ".join(gpus))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return " · ".join(parts) if parts else "host ?"


def host_resource_snapshot_dict() -> Dict[str, Any]:
    """
    Развёрнутый снимок для JSONL (без печати огромных таблиц в stdout).
    """
    out: Dict[str, Any] = {}
    meminfo: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            tok = rest.split()
            if tok and tok[0].isdigit():
                meminfo[key.strip()] = int(tok[0])
        tot = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        out["mem_kib"] = {
            "MemTotal": meminfo.get("MemTotal"),
            "MemAvailable": meminfo.get("MemAvailable"),
            "MemFree": meminfo.get("MemFree"),
            "SwapTotal": meminfo.get("SwapTotal"),
            "SwapFree": meminfo.get("SwapFree"),
        }
        if tot > 0:
            out["mem_used_pct"] = round(100.0 * (tot - avail) / tot, 2)
    except OSError:
        out["mem_kib"] = {}

    try:
        la = Path("/proc/loadavg").read_text(encoding="utf-8").split()
        if la:
            out["loadavg"] = {"1m": la[0], "5m": la[1], "15m": la[2]}
    except OSError:
        pass

    gpus: List[Dict[str, Any]] = []
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if r.returncode == 0 and (r.stdout or "").strip():
            for ln in r.stdout.strip().splitlines():
                bits = [x.strip() for x in ln.split(",")]
                if len(bits) >= 5:
                    gpus.append(
                        {
                            "index": bits[0],
                            "name": bits[1],
                            "memory_used_mib": bits[2],
                            "memory_total_mib": bits[3],
                            "utilization_gpu_pct": bits[4],
                        }
                    )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if gpus:
        out["gpus"] = gpus

    return out


def parent_process_gpu_gc() -> None:
    """Лучшее-effort: освободить CUDA-кэш только в текущем процессе (оркестратор E2E).

    Не освобождает VRAM других процессов на GPU; строка из host_resource_snapshot_line()
    после этого может не измениться.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc):
                ipc()
    except Exception:
        pass
