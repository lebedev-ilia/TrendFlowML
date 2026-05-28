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


def _resolve_model_facing_path(features_npz_path: str, model_facing_path_value: Any) -> Optional[str]:
    # In current artifacts this is stored as an absolute path string (0-d ndarray).
    try:
        if isinstance(model_facing_path_value, np.ndarray) and model_facing_path_value.shape == ():
            mp = model_facing_path_value.item()
        else:
            mp = model_facing_path_value
        if isinstance(mp, bytes):
            mp = mp.decode("utf-8", errors="ignore")
        mp = str(mp)
        if not mp:
            return None
        if os.path.isabs(mp):
            return mp if os.path.exists(mp) else None
        cand = os.path.join(os.path.dirname(features_npz_path), mp)
        return cand if os.path.exists(cand) else None
    except Exception:
        return None


@dataclass
class RunStats:
    features_npz_path: str
    model_facing_npz_path: Optional[str]

    platform_id: Optional[str]
    video_id: Optional[str]
    run_id: Optional[str]
    config_hash: Optional[str]
    sampling_policy_version: Optional[str]

    features_schema_version: Optional[str]
    model_facing_schema_version: Optional[str]
    producer_version: Optional[str]

    # Axes / coverage
    N: int
    pairs: int
    E: int

    # Masks coverage
    deep_valid_ratio: Optional[float]
    ssim_valid_ratio: Optional[float]
    flow_valid_ratio: Optional[float]

    # Model-facing numeric summaries
    hist_diff_l1: Dict[str, Any]
    flow_mag: Dict[str, Any]
    hard_score: Dict[str, Any]
    pair_dt_s: Dict[str, Any]
    deep_cosine_dist: Dict[str, Any]
    ssim_drop: Dict[str, Any]

    # Events
    event_strength: Dict[str, Any]
    event_type_id_counts: Dict[str, int]

    # Analytics NPZ (features/detections)
    analytics: Dict[str, Any]

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(features_npz_path: str) -> RunStats:
    d = _load_npz(features_npz_path)
    keys = set(d.keys())

    meta = _unbox_meta(d.get("meta"))
    platform_id = meta.get("platform_id")
    video_id = meta.get("video_id")
    run_id = meta.get("run_id")
    config_hash = meta.get("config_hash")
    sampling_policy_version = meta.get("sampling_policy_version")
    producer_version = meta.get("producer_version")
    features_schema_version = meta.get("schema_version")

    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    mf_path = None
    if "model_facing_npz_path" in keys:
        mf_path = _resolve_model_facing_path(features_npz_path, d.get("model_facing_npz_path"))

    N = int(np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32))).size)

    # Analytics dicts
    features_dict = _unbox_dict_scalar_object(d.get("features"))
    detections_dict = _unbox_dict_scalar_object(d.get("detections"))
    # Count numeric NaN in top-level features dict
    nan_keys = []
    numeric_count = 0
    for k, v in (features_dict or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_count += 1
            if float(v) != float(v):
                nan_keys.append(str(k))
    analytics = {
        "npz_keys": sorted(list(keys)),
        "features_n_keys_top": int(len(features_dict.keys())) if isinstance(features_dict, dict) else 0,
        "features_n_numeric_top": int(numeric_count),
        "features_nan_numeric_top_keys": sorted(nan_keys),
        "detections_n_keys_top": int(len(detections_dict.keys())) if isinstance(detections_dict, dict) else 0,
    }

    # Load model-facing NPZ and summarize main arrays
    model_facing_schema_version = None
    pairs = max(0, N - 1)
    E = 0
    deep_valid_ratio = None
    ssim_valid_ratio = None
    flow_valid_ratio = None
    hist_diff_l1 = {"note": "no_model_facing"}
    flow_mag = {"note": "no_model_facing"}
    hard_score = {"note": "no_model_facing"}
    pair_dt_s = {"note": "no_model_facing"}
    deep_cosine_dist = {"note": "no_model_facing"}
    ssim_drop = {"note": "no_model_facing"}
    event_strength = {"note": "no_model_facing"}
    event_type_id_counts: Dict[str, int] = {}

    if mf_path:
        mfd = _load_npz(mf_path)
        mf_meta = _unbox_meta(mfd.get("meta"))
        model_facing_schema_version = mf_meta.get("schema_version")

        # Axes
        N_mf = int(np.asarray(mfd.get("frame_indices", np.asarray([], dtype=np.int32))).size)
        pairs = max(0, N_mf - 1)
        E = int(np.asarray(mfd.get("event_type_id", np.asarray([], dtype=np.int16))).size)

        # Masks
        deep_valid = np.asarray(mfd.get("deep_valid_mask", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
        ssim_valid = np.asarray(mfd.get("ssim_valid_mask", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
        flow_valid = np.asarray(mfd.get("flow_valid_mask", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
        if deep_valid.size:
            deep_valid_ratio = float(deep_valid.mean())
        if ssim_valid.size:
            ssim_valid_ratio = float(ssim_valid.mean())
        if flow_valid.size:
            flow_valid_ratio = float(flow_valid.mean())

        # Numeric series (mask-aware summaries kept simple: overall numeric stats)
        hist_diff_l1 = _numeric_summary(np.asarray(mfd.get("hist_diff_l1", np.asarray([], dtype=np.float32))))
        flow_mag = _numeric_summary(np.asarray(mfd.get("flow_mag", np.asarray([], dtype=np.float32))))
        hard_score = _numeric_summary(np.asarray(mfd.get("hard_score", np.asarray([], dtype=np.float32))))
        pair_dt_s = _numeric_summary(np.asarray(mfd.get("pair_dt_s", np.asarray([], dtype=np.float32))))
        deep_cosine_dist = _numeric_summary(np.asarray(mfd.get("deep_cosine_dist", np.asarray([], dtype=np.float32))))
        ssim_drop = _numeric_summary(np.asarray(mfd.get("ssim_drop", np.asarray([], dtype=np.float32))))

        event_strength = _numeric_summary(np.asarray(mfd.get("event_strength", np.asarray([], dtype=np.float32))))
        eti = np.asarray(mfd.get("event_type_id", np.asarray([], dtype=np.int16))).reshape(-1)
        if eti.size:
            for x in eti.astype(np.int64).tolist():
                k = str(int(x))
                event_type_id_counts[k] = int(event_type_id_counts.get(k, 0) + 1)

    return RunStats(
        features_npz_path=features_npz_path,
        model_facing_npz_path=mf_path,
        platform_id=str(platform_id) if platform_id is not None else None,
        video_id=str(video_id) if video_id is not None else None,
        run_id=str(run_id) if run_id is not None else None,
        config_hash=str(config_hash) if config_hash is not None else None,
        sampling_policy_version=str(sampling_policy_version) if sampling_policy_version is not None else None,
        features_schema_version=str(features_schema_version) if features_schema_version is not None else None,
        model_facing_schema_version=str(model_facing_schema_version) if model_facing_schema_version is not None else None,
        producer_version=str(producer_version) if producer_version is not None else None,
        N=int(N),
        pairs=int(pairs),
        E=int(E),
        deep_valid_ratio=deep_valid_ratio,
        ssim_valid_ratio=ssim_valid_ratio,
        flow_valid_ratio=flow_valid_ratio,
        hist_diff_l1=hist_diff_l1,
        flow_mag=flow_mag,
        hard_score=hard_score,
        pair_dt_s=pair_dt_s,
        deep_cosine_dist=deep_cosine_dist,
        ssim_drop=ssim_drop,
        event_strength=event_strength,
        event_type_id_counts=event_type_id_counts,
        analytics=analytics,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for cut_detection (VisualProcessor)")
    ap.add_argument("--features-npz", action="append", required=True, help="Path to cut_detection_features_*.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.features_npz or [])]

    doc = {
        "component": "cut_detection",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "pairs_total": int(sum(r.pairs for r in runs)),
            "E_total": int(sum(r.E for r in runs)),
            "deep_valid_ratio_mean": float(np.mean([r.deep_valid_ratio for r in runs if r.deep_valid_ratio is not None])) if any(r.deep_valid_ratio is not None for r in runs) else None,
            "ssim_valid_ratio_mean": float(np.mean([r.ssim_valid_ratio for r in runs if r.ssim_valid_ratio is not None])) if any(r.ssim_valid_ratio is not None for r in runs) else None,
            "flow_valid_ratio_mean": float(np.mean([r.flow_valid_ratio for r in runs if r.flow_valid_ratio is not None])) if any(r.flow_valid_ratio is not None for r in runs) else None,
            "event_type_id_counts_total": {
                k: int(sum(int(r.event_type_id_counts.get(k, 0)) for r in runs)) for k in sorted({k for r in runs for k in r.event_type_id_counts.keys()})
            },
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

