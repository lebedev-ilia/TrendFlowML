from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml  # type: ignore


@dataclass(frozen=True)
class VisualBatchOverrides:
    # Core providers (baseline)
    core_clip_batch_size: Optional[int] = None
    core_depth_midas_batch_size: Optional[int] = None
    core_optical_flow_batch_size: Optional[int] = None
    core_object_detections_batch_size: Optional[int] = None


def _ensure_dict(d: Any) -> Dict[str, Any]:
    return d if isinstance(d, dict) else {}


def build_visual_runtime_cfg(
    *,
    base_cfg_path: str,
    overrides: VisualBatchOverrides,
) -> Dict[str, Any]:
    """
    Loads VisualProcessor config YAML and applies scheduler overrides.
    """
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _ensure_dict(cfg)

    # IMPORTANT:
    # - `cfg["core_providers"][name]` is a boolean enable flag (or any truthy value).
    # - Actual per-component runtime config lives at top-level `cfg[name]`.
    # Therefore, scheduler must patch `cfg[name]["batch_size"]` without touching enable flags.
    core_flags = _ensure_dict(cfg.get("core_providers"))

    def _patch_component_batch(name: str, bs: Optional[int]) -> None:
        if bs is None:
            return
        if not bool(core_flags.get(name)):
            # Do not accidentally enable disabled components.
            return
        node = _ensure_dict(cfg.get(name))
        node["batch_size"] = int(bs)
        cfg[name] = node

    _patch_component_batch("core_clip", overrides.core_clip_batch_size)
    _patch_component_batch("core_depth_midas", overrides.core_depth_midas_batch_size)
    _patch_component_batch("core_optical_flow", overrides.core_optical_flow_batch_size)
    _patch_component_batch("core_object_detections", overrides.core_object_detections_batch_size)
    return cfg


def write_temp_yaml(cfg: Dict[str, Any], prefix: str = "vp_sched_", suffix: str = ".yaml") -> str:
    fd, p = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return p


