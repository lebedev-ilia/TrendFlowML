from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class UnitCost:
    component: str
    unit: str
    model_branch: str
    latency_ms_per_unit: Optional[float]
    cpu_rss_peak_mb: Optional[float]
    vram_triton_peak_mb: Optional[float]
    vram_triton_delta_run_mb: Optional[float]


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_resource_costs_json(path: str) -> List[UnitCost]:
    """
    Loads a single resource_costs_*.json and normalizes to a list of UnitCost.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    costs = data.get("costs") or []
    out: List[UnitCost] = []
    if not isinstance(costs, list):
        return out
    for row in costs:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        out.append(
            UnitCost(
                component=str(row.get("component") or ""),
                unit=str(row.get("unit") or ""),
                model_branch=str(row.get("model_branch") or ""),
                latency_ms_per_unit=_as_float(metrics.get("latency_ms_mean_stable_per_unit")),
                cpu_rss_peak_mb=_as_float(metrics.get("cpu_rss_peak_mb")),
                vram_triton_peak_mb=_as_float(metrics.get("vram_triton_peak_mb")),
                vram_triton_delta_run_mb=_as_float(metrics.get("vram_triton_delta_run_mb")),
            )
        )
    return out


def load_resource_costs_dir(resource_costs_dir: str) -> List[UnitCost]:
    out: List[UnitCost] = []
    if not os.path.isdir(resource_costs_dir):
        return out
    for name in sorted(os.listdir(resource_costs_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(resource_costs_dir, name)
        try:
            out.extend(load_resource_costs_json(path))
        except Exception:
            # best-effort; scheduler must be robust to partial/malformed cost files
            continue
    return out


def find_best_cost(
    costs: List[UnitCost],
    *,
    component: str,
    prefer_branch: Optional[str] = None,
) -> Optional[UnitCost]:
    """
    Selects a cost row for a component.

    MVP policy:
    - filter by exact component name (e.g., "core_clip.clip_image")
    - if prefer_branch provided and present, pick it
    - else pick the first row deterministically (sorted by model_branch)
    """
    rows = []
    for c in costs:
        if c.component == component:
            rows.append(c)
    if not rows:
        return None
    rows.sort(key=lambda r: (str(r.model_branch), str(r.unit)))
    if prefer_branch is not None:
        for r in rows:
            if str(r.model_branch) == str(prefer_branch):
                return r
    return rows[0]


def gpu_mem_per_task_mb(cost: Optional[UnitCost], default_mb: int = 64) -> int:
    """
    For scheduling batch_size we prefer per-request delta VRAM (activations):
    - use vram_triton_delta_run_mb if present and > 0
    - else use default_mb (MVP conservative fallback)
    """
    if cost is None:
        return int(max(1, default_mb))
    d = cost.vram_triton_delta_run_mb
    if d is not None and d > 0:
        return int(max(1, round(d)))
    return int(max(1, default_mb))


