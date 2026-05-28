#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _is_strictly_increasing_int(x: np.ndarray) -> bool:
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size <= 1:
        return True
    return bool(np.all(np.diff(a) > 0))


def _is_monotonic_non_decreasing(x: np.ndarray, tol: float = 0.0) -> bool:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size <= 1:
        return True
    return bool(np.all(np.diff(a) >= -float(tol)))


def _finite_stats_1d(x: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "finite": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }


def _nan_at_0_only(x: np.ndarray) -> bool:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return True
    m = np.isnan(a)
    if not bool(m[0]):
        return False
    return bool(np.sum(m[1:]) == 0)


def _finite_from_1(x: np.ndarray) -> bool:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size <= 1:
        return True
    return bool(np.all(np.isfinite(a[1:])))


def _preview_stats(m: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(m, dtype=np.float64)
    finite = a[np.isfinite(a)]
    in_01 = None
    if finite.size:
        in_01 = bool((finite.min() >= -1e-6) and (finite.max() <= 1.0 + 1e-6))
    return {
        "shape": list(a.shape),
        "finite": int(finite.size),
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "in_01": in_01,
    }


FLOW_DEP_KEYS: Tuple[str, ...] = (
    "dt_seconds",
    "flow_mag_std_per_sec_norm",
    "flow_mag_p95_per_sec_norm",
    "flow_dx_mean_per_sec_norm",
    "flow_dy_mean_per_sec_norm",
    "flow_dir_sin_mean",
    "flow_dir_cos_mean",
    "flow_dir_dispersion",
    "flow_div_abs_mean",
    "flow_consistency",
    "cam_affine_scale",
    "cam_affine_rotation",
    "cam_tx_per_sec_norm",
    "cam_ty_per_sec_norm",
    "cam_shake_std_norm",
    "bg_ratio",
)


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
    backend_proxy_version: Optional[str]

    N: int
    K: int
    preview_map_hw: List[int]

    frame_indices_strict_inc: bool
    times_s_monotonic: bool

    motion0_is_zero: Optional[bool]
    motion_finite_all: bool

    flow_dep_nan_at_0_only_all: bool
    flow_dep_finite_from_1_all: bool
    flow_dep_nan_total: int

    preview_flow_mag_map_norm: Dict[str, Any]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    times_s = np.asarray(d.get("times_s", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    motion = np.asarray(d.get("motion_norm_per_sec_mean", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    preview_pair_pos = np.asarray(d.get("preview_pair_pos", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    preview_map = np.asarray(d.get("preview_flow_mag_map_norm", np.asarray([], dtype=np.float32)), dtype=np.float32)

    N = int(frame_indices.size)
    K = int(preview_pair_pos.size)
    preview_hw = list(preview_map.shape[1:3]) if preview_map.ndim == 3 else []

    frame_inc = _is_strictly_increasing_int(frame_indices)
    times_mon = _is_monotonic_non_decreasing(times_s, tol=1e-3)

    motion_finite_all = bool(np.all(np.isfinite(motion))) if motion.size else True
    motion0_is_zero = None
    if motion.size:
        motion0_is_zero = bool(abs(float(motion[0])) <= 1e-8)

    nan_total = 0
    nan_at0_all = True
    finite_from1_all = True
    for k in FLOW_DEP_KEYS:
        arr = np.asarray(d.get(k, np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        nan_total += int(np.isnan(arr).sum())
        nan_at0_all = bool(nan_at0_all and _nan_at_0_only(arr))
        finite_from1_all = bool(finite_from1_all and _finite_from_1(arr))

    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        backend_proxy_version=str(meta.get("backend_proxy_version")) if meta.get("backend_proxy_version") is not None else None,
        N=N,
        K=K,
        preview_map_hw=[int(x) for x in preview_hw],
        frame_indices_strict_inc=frame_inc,
        times_s_monotonic=times_mon,
        motion0_is_zero=motion0_is_zero,
        motion_finite_all=motion_finite_all,
        flow_dep_nan_at_0_only_all=nan_at0_all,
        flow_dep_finite_from_1_all=finite_from1_all,
        flow_dep_nan_total=int(nan_total),
        preview_flow_mag_map_norm=_preview_stats(preview_map),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for core_optical_flow (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to core_optical_flow/flow.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc: Dict[str, Any] = {
        "component": "core_optical_flow",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "N_set": sorted({int(r.N) for r in runs}),
            "K_set": sorted({int(r.K) for r in runs}),
            "preview_map_hw_set": sorted({tuple(r.preview_map_hw) for r in runs}),
            "frame_indices_strict_inc_all": bool(all(r.frame_indices_strict_inc for r in runs)),
            "times_s_monotonic_all": bool(all(r.times_s_monotonic for r in runs)),
            "motion0_is_zero_all": bool(all((r.motion0_is_zero is True) for r in runs if r.motion0_is_zero is not None)),
            "motion_finite_all": bool(all(r.motion_finite_all for r in runs)),
            "flow_dep_nan_at_0_only_all": bool(all(r.flow_dep_nan_at_0_only_all for r in runs)),
            "flow_dep_finite_from_1_all": bool(all(r.flow_dep_finite_from_1_all for r in runs)),
            "flow_dep_nan_total": int(sum(r.flow_dep_nan_total for r in runs)),
            "preview_nan_total": int(sum(int(r.preview_flow_mag_map_norm.get("nan") or 0) for r in runs)),
            "preview_in_01_all": bool(all(bool(r.preview_flow_mag_map_norm.get("in_01")) for r in runs)),
            "preview_finite_all": bool(all(int(r.preview_flow_mag_map_norm.get("finite") or 0) > 0 for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

