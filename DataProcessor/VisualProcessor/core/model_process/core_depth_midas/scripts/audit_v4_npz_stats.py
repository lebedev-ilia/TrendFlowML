#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_npz(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    try:
        return {k: npz[k] for k in npz.files}
    finally:
        try:
            npz.close()
        except Exception:
            pass


def _unbox_meta(meta_arr: Any) -> Dict[str, Any]:
    if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object and meta_arr.shape == ():
        try:
            meta_arr = meta_arr.item()
        except Exception:
            pass
    if isinstance(meta_arr, dict):
        return dict(meta_arr)
    return {}


def _finite_min_max(x: np.ndarray) -> Tuple[Optional[float], Optional[float], int, int, int]:
    a = np.asarray(x, dtype=np.float64)
    n_total = int(a.size)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None, None, n_total, n_nan, n_inf
    return float(finite.min()), float(finite.max()), n_total, n_nan, n_inf


def _is_strictly_increasing(x: np.ndarray) -> bool:
    a = np.asarray(x).reshape(-1)
    if a.size <= 1:
        return True
    return bool(np.all(a[1:] > a[:-1]))


@dataclass
class RunStats:
    npz_path: str
    platform_id: Optional[str]
    video_id: Optional[str]
    run_id: Optional[str]
    config_hash: Optional[str]
    sampling_policy_version: Optional[str]
    schema_version: Optional[str]
    producer_version: Optional[str]

    N: int
    H: int
    W: int
    K: int

    frame_indices_strict_inc: bool
    preview_subset_ok: bool

    depth_maps_norm_min: Optional[float]
    depth_maps_norm_max: Optional[float]
    depth_maps_norm_total: int
    depth_maps_norm_nan: int
    depth_maps_norm_inf: int

    depth_maps_min: Optional[float]
    depth_maps_max: Optional[float]
    depth_maps_total: int
    depth_maps_nan: int
    depth_maps_inf: int


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    depth_maps_norm = np.asarray(d.get("depth_maps_norm", np.asarray([], dtype=np.float32)), dtype=np.float32)
    depth_maps = np.asarray(d.get("depth_maps", np.asarray([], dtype=np.float32)), dtype=np.float32)
    preview_frame_indices = np.asarray(d.get("preview_frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)

    if depth_maps_norm.ndim == 3:
        N, H, W = map(int, depth_maps_norm.shape)
    else:
        N, H, W = 0, 0, 0
    K = int(preview_frame_indices.shape[0])

    norm_min, norm_max, norm_total, norm_nan, norm_inf = _finite_min_max(depth_maps_norm)
    raw_min, raw_max, raw_total, raw_nan, raw_inf = _finite_min_max(depth_maps)

    # preview subset check
    fi_set = set(map(int, frame_indices.tolist()))
    prev_ok = all(int(x) in fi_set for x in preview_frame_indices.tolist())

    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        N=N,
        H=H,
        W=W,
        K=K,
        frame_indices_strict_inc=_is_strictly_increasing(frame_indices),
        preview_subset_ok=bool(prev_ok),
        depth_maps_norm_min=norm_min,
        depth_maps_norm_max=norm_max,
        depth_maps_norm_total=norm_total,
        depth_maps_norm_nan=norm_nan,
        depth_maps_norm_inf=norm_inf,
        depth_maps_min=raw_min,
        depth_maps_max=raw_max,
        depth_maps_total=raw_total,
        depth_maps_nan=raw_nan,
        depth_maps_inf=raw_inf,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for core_depth_midas (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to core_depth_midas/depth.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc = {
        "component": "core_depth_midas",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "N_set": sorted({int(r.N) for r in runs}),
            "H_set": sorted({int(r.H) for r in runs}),
            "W_set": sorted({int(r.W) for r in runs}),
            "K_set": sorted({int(r.K) for r in runs}),
            "frame_indices_strict_inc_all": bool(all(r.frame_indices_strict_inc for r in runs)),
            "preview_subset_ok_all": bool(all(r.preview_subset_ok for r in runs)),
            "depth_maps_norm_min_min": min([r.depth_maps_norm_min for r in runs if r.depth_maps_norm_min is not None], default=None),
            "depth_maps_norm_max_max": max([r.depth_maps_norm_max for r in runs if r.depth_maps_norm_max is not None], default=None),
            "depth_maps_norm_nan_total": int(sum(r.depth_maps_norm_nan for r in runs)),
            "depth_maps_norm_inf_total": int(sum(r.depth_maps_norm_inf for r in runs)),
            "depth_maps_nan_total": int(sum(r.depth_maps_nan for r in runs)),
            "depth_maps_inf_total": int(sum(r.depth_maps_inf for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

