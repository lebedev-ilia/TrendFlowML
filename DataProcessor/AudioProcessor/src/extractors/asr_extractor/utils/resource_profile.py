"""
Снимки RSS/VRAM для ASR (best-effort, без обязательного psutil).

Включение тяжёлых снимков в payload/meta и расширенных полей профиля:
env ``AP_ASR_RESOURCE_PROFILE=1`` (иначе в meta попадают только тайминги, без RSS/GPU снимков).
"""
from __future__ import annotations

import os
from typing import Any, Dict


def resource_profile_enabled() -> bool:
    v = os.environ.get("AP_ASR_RESOURCE_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def lang_detect_once_enabled() -> bool:
    """
    При ``language=auto`` — после первого успешного detect (с непустым lang_code)
    на файл / на run_segments повторно не вызывать detect_language.

    Риск: смена языка внутри файла перестаёт учитываться. Включать осознанно.
    """
    v = os.environ.get("AP_ASR_LANG_DETECT_ONCE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def snapshot_process_resources() -> Dict[str, Any]:
    """Один снимок текущего процесса (+ GPU через torch, если есть)."""
    snap: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        mi = p.memory_info()
        snap["rss_mb"] = float(mi.rss) / (1024.0 * 1024.0)
        snap["vms_mb"] = float(mi.vms) / (1024.0 * 1024.0)
    except Exception:
        snap["rss_mb"] = None
        snap["vms_mb"] = None

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


def prefix_snapshot(prefix: str, snap: Dict[str, Any]) -> Dict[str, Any]:
    """Плоский dict с префиксом ключей для meta (например rss_mb -> rss_mb_at_start)."""
    out: Dict[str, Any] = {}
    for k, v in snap.items():
        out[f"{k}_{prefix}"] = v
    return out
