"""
Best-effort RSS snapshots for onset_extractor (Audit v4.2).

Enable via env:
  AP_ONSET_RESOURCE_PROFILE=1
"""
from __future__ import annotations

import os
from typing import Any, Dict


def resource_profile_enabled() -> bool:
    v = os.environ.get("AP_ONSET_RESOURCE_PROFILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def snapshot_process_resources() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore
        import os as _os

        p = psutil.Process(_os.getpid())
        mi = p.memory_info()
        snap["rss_mb"] = float(mi.rss) / (1024.0 * 1024.0)
        snap["vms_mb"] = float(mi.vms) / (1024.0 * 1024.0)
    except Exception:
        snap["rss_mb"] = None
        snap["vms_mb"] = None
    return snap


def prefix_snapshot(prefix: str, snap: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{k}_{prefix}": v for k, v in snap.items()}

