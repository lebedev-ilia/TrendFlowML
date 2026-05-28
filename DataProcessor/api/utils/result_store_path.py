"""
Единообразное вычисление per-run result_store, совместимо с DataProcessor/main.py.
"""

from __future__ import annotations

import os
from typing import Optional


def get_dataprocessor_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[2])


def default_rs_base() -> str:
    return os.path.join(get_dataprocessor_root(), "VisualProcessor", "result_store")


def resolve_run_result_store_path(
    *,
    platform_id: str,
    video_id: str,
    run_id: str,
    rs_base: Optional[str] = None,
) -> str:
    base = os.path.abspath(rs_base or default_rs_base())
    return os.path.join(base, platform_id, video_id, run_id)
