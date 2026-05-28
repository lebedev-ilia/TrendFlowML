#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf
    except Exception:
        return None


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
    if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object:
        if meta_arr.shape == ():
            v = meta_arr.item()
        else:
            v = meta_arr.flat[0].item() if hasattr(meta_arr.flat[0], "item") else meta_arr.flat[0]
        if isinstance(v, dict):
            return dict(v)
    if isinstance(meta_arr, dict):
        return dict(meta_arr)
    return {}


def _as_obj_str_list(arr: Any) -> List[str]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [str(x) for x in arr.reshape(-1).tolist()]
    if isinstance(arr, (list, tuple)):
        return [str(x) for x in arr]
    return []


def _numeric_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out: Dict[str, Any] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "size": int(arr.size)}
    if arr.size == 0:
        out.update({"nan": 0, "inf": 0, "n_valid": 0, "min": None, "max": None, "mean": None, "std": None})
        return out

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        a = arr.astype(np.int64, copy=False)
        out.update(
            {
                "nan": 0,
                "inf": 0,
                "n_valid": int(a.size),
                "min": int(a.min()) if a.size else None,
                "max": int(a.max()) if a.size else None,
                "mean": float(a.mean()) if a.size else None,
                "std": float(a.std()) if a.size else None,
            }
        )
        return out

    a = arr.astype(np.float64, copy=False)
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    finite = a[np.isfinite(a)]
    out.update(
        {
            "nan": nan,
            "inf": inf,
            "n_valid": int(finite.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
        }
    )
    return out


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

    keys: List[str]
    N: int
    D: int
    F: int

    missing_ratio_curve: Optional[float]
    missing_ratio_matrix: Optional[float]
    missing_ratio_max_abs_diff: Optional[float]

    motion_curve_summary_finite: Dict[str, Any]
    feature_values_summary: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    curve = np.asarray(d.get("motion_norm_per_sec_mean", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    mat = np.asarray(d.get("frame_feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32)
    ff_names = _as_obj_str_list(d.get("frame_feature_names"))
    fv_names = _as_obj_str_list(d.get("feature_names"))
    fv = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    N = int(curve.size)
    D = int(len(ff_names))
    F = int(len(fv_names))

    miss_curve = float(np.isnan(curve.astype(np.float64, copy=False)).mean()) if curve.size else None
    miss_mat = float(np.isnan(mat.astype(np.float64, copy=False)).mean()) if (mat.ndim == 2 and mat.size) else None
    miss_diff = None
    if miss_curve is not None and miss_mat is not None:
        miss_diff = float(abs(miss_curve - miss_mat))

    curve_finite = curve[np.isfinite(curve.astype(np.float64, copy=False))]
    motion_summary = _numeric_summary(curve_finite.astype(np.float32, copy=False))
    feat_summary = _numeric_summary(fv.astype(np.float32, copy=False))

    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        keys=keys,
        N=N,
        D=D,
        F=F,
        missing_ratio_curve=miss_curve,
        missing_ratio_matrix=miss_mat,
        missing_ratio_max_abs_diff=miss_diff,
        motion_curve_summary_finite=motion_summary,
        feature_values_summary=feat_summary,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for optical_flow (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to optical_flow.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    doc = {
        "component": "optical_flow",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "D_set": sorted({int(r.D) for r in runs}),
            "F_set": sorted({int(r.F) for r in runs}),
            "missing_ratio_curve_mean": float(np.mean([r.missing_ratio_curve for r in runs if r.missing_ratio_curve is not None])) if runs else 0.0,
            "missing_ratio_matrix_mean": float(np.mean([r.missing_ratio_matrix for r in runs if r.missing_ratio_matrix is not None])) if runs else 0.0,
            "missing_ratio_max_abs_diff_max": float(max([r.missing_ratio_max_abs_diff for r in runs if r.missing_ratio_max_abs_diff is not None], default=0.0)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

