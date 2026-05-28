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


def _percentiles(xs: np.ndarray, ps: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return {f"p{int(p):02d}": None for p in ps}
    out = np.percentile(xs, ps).astype(np.float64)
    return {f"p{int(p):02d}": float(v) for p, v in zip(ps, out)}


def _numeric_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out: Dict[str, Any] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "size": int(arr.size)}
    if arr.size == 0:
        out.update({"nan": 0, "inf": 0, "n_valid": 0, "min": None, "max": None, "mean": None, "std": None})
        out.update(_percentiles(np.asarray([], dtype=np.float64), [1, 5, 50, 95, 99]))
        return out

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        a64 = arr.astype(np.int64, copy=False)
        out.update(
            {
                "nan": 0,
                "inf": 0,
                "n_valid": int(a64.size),
                "min": int(a64.min()) if a64.size else None,
                "max": int(a64.max()) if a64.size else None,
                "mean": float(a64.mean()) if a64.size else None,
                "std": float(a64.std()) if a64.size else None,
            }
        )
        out.update(_percentiles(a64.astype(np.float64), [1, 5, 50, 95, 99]))
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
    out.update(_percentiles(finite, [1, 5, 50, 95, 99]))
    return out


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


def _unbox_dict_scalar_object(arr: Any) -> Dict[str, Any]:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            v = arr.item()
        else:
            v = arr.flat[0].item() if hasattr(arr.flat[0], "item") else arr.flat[0]
        if isinstance(v, dict):
            return dict(v)
    if isinstance(arr, dict):
        return dict(arr)
    return {}


def _flatten_numeric_leaves(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_numeric_leaves(v, key))
            continue
        fv = _as_float(v)
        if fv is not None:
            out[key] = float(fv)
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

    face_present_true: int
    processed_mask_true: int
    primary_valid_true: int
    face_present_ratio: float
    processed_ratio: float
    primary_valid_ratio: float

    face_count: Dict[str, Any]
    tracking_id_unique: int
    tracking_id_neg1_ratio: float

    compact: Dict[str, Any]
    compact_zero_row_ratio: Optional[float]
    compact_l2_all: Dict[str, Any]
    compact_l2_when_primary_valid: Dict[str, Any]

    aggregated_numeric: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]

    optional_curves_present: List[str]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    face_present = np.asarray(d.get("face_present", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    processed_mask = np.asarray(d.get("processed_mask", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    primary_valid = np.asarray(d.get("primary_valid", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    face_count = np.asarray(d.get("face_count", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    tracking_id = np.asarray(d.get("primary_tracking_id", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    compact = np.asarray(d.get("primary_compact_features", np.asarray([], dtype=np.float32)), dtype=np.float32)

    N = int(face_present.size)
    fp_true = int(face_present.sum())
    pm_true = int(processed_mask.sum())
    pv_true = int(primary_valid.sum())
    fp_ratio = float(fp_true / max(N, 1))
    pm_ratio = float(pm_true / max(N, 1))
    pv_ratio = float(pv_true / max(N, 1))

    # tracking id stats
    unique_ids = int(len(set(int(x) for x in tracking_id.tolist()))) if tracking_id.size else 0
    neg1_ratio = float((tracking_id == -1).mean()) if tracking_id.size else 1.0

    # compact: zero rows ratio (exactly all zeros)
    compact_zero_row_ratio = None
    compact_l2_all = {"note": "missing_or_bad_shape"}
    compact_l2_pv = {"note": "missing_or_bad_shape"}
    if compact.ndim == 2 and compact.shape[0] == N and compact.shape[1] > 0:
        zero_row = np.all(compact == 0.0, axis=1)
        compact_zero_row_ratio = float(zero_row.mean()) if zero_row.size else None
        l2 = np.linalg.norm(compact.astype(np.float64), axis=1)
        compact_l2_all = _numeric_summary(l2)
        if primary_valid.size == l2.size:
            compact_l2_pv = _numeric_summary(l2[primary_valid])

    # aggregated numeric leaves
    aggregated = _unbox_dict_scalar_object(d.get("aggregated"))
    aggregated_numeric_flat = _flatten_numeric_leaves(aggregated)
    aggregated_numeric = {
        "n_total_keys_top": int(len(aggregated.keys())) if isinstance(aggregated, dict) else 0,
        "n_numeric_leaves": int(len(aggregated_numeric_flat)),
        "numeric_leaves": {k: aggregated_numeric_flat[k] for k in sorted(aggregated_numeric_flat.keys())},
    }

    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    # Optional curves presence (NPZ keys)
    optional_curves = [
        "primary_gaze_at_camera_prob",
        "primary_blink_rate",
        "primary_attention_score",
        "primary_quality_proxy_score",
        "primary_face_sharpness",
        "primary_occlusion_proxy",
        "primary_speech_activity_prob",
    ]
    optional_present = [k for k in optional_curves if k in d]

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
        face_present_true=fp_true,
        processed_mask_true=pm_true,
        primary_valid_true=pv_true,
        face_present_ratio=fp_ratio,
        processed_ratio=pm_ratio,
        primary_valid_ratio=pv_ratio,
        face_count=_numeric_summary(face_count),
        tracking_id_unique=unique_ids,
        tracking_id_neg1_ratio=neg1_ratio,
        compact=_numeric_summary(compact),
        compact_zero_row_ratio=compact_zero_row_ratio,
        compact_l2_all=compact_l2_all,
        compact_l2_when_primary_valid=compact_l2_pv,
        aggregated_numeric=aggregated_numeric,
        stage_timings_ms=stage_timings_ms,
        optional_curves_present=optional_present,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for detalize_face (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to detalize_face.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc = {
        "component": "detalize_face",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "face_present_true_total": int(sum(r.face_present_true for r in runs)),
            "processed_mask_true_total": int(sum(r.processed_mask_true for r in runs)),
            "primary_valid_true_total": int(sum(r.primary_valid_true for r in runs)),
            "face_present_ratio_weighted": float(sum(r.face_present_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "processed_ratio_weighted": float(sum(r.processed_mask_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "primary_valid_ratio_weighted": float(sum(r.primary_valid_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "compact_zero_row_ratio_mean": float(
                np.mean([r.compact_zero_row_ratio for r in runs if r.compact_zero_row_ratio is not None])
            )
            if any(r.compact_zero_row_ratio is not None for r in runs)
            else None,
            "optional_curves_any_present": sorted({k for r in runs for k in (r.optional_curves_present or [])}),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

