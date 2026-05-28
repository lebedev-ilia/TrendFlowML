from __future__ import annotations

import os
from typing import Any, Dict, Optional


def is_spectral_resource_profile_enabled() -> bool:
    v = os.getenv("AP_SPECTRAL_RESOURCE_PROFILE", "")
    return v.strip() in ("1", "true", "True", "yes", "on", "ON")


def capture_spectral_resource_profile(*, stage: str) -> Optional[Dict[str, Any]]:
    if not is_spectral_resource_profile_enabled():
        return None

    snap: Dict[str, Any] = {"stage": stage}

    try:
        import psutil

        p = psutil.Process()
        mem = p.memory_info()
        snap["rss_bytes"] = int(getattr(mem, "rss", 0))
        snap["vms_bytes"] = int(getattr(mem, "vms", 0))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            snap["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:
        pass

    return snap

