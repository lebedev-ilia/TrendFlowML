#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    landmarks_present_true: int
    landmarks_present_ratio: float

    seq_summaries: Dict[str, Any]
    gesture_prob_sum: Dict[str, Any]
    aggregated_numeric: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    lp = np.asarray(d.get("landmarks_present", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    N = int(lp.size)
    lp_true = int(lp.sum())
    lp_ratio = float(lp_true / max(N, 1))

    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    # Sequence fields
    seq_summaries: Dict[str, Any] = {}
    for k in keys:
        if not k.startswith("seq_"):
            continue
        arr = np.asarray(d[k])
        if arr.ndim != 1:
            seq_summaries[k] = {"note": "non-1d", **_numeric_summary(arr.astype(np.float64, copy=False))}
            continue
        all_summary = _numeric_summary(arr)
        present_summary = _numeric_summary(arr[lp]) if lp.size == arr.size else {"note": "mask/shape mismatch"}
        absent_summary = _numeric_summary(arr[~lp]) if lp.size == arr.size else {"note": "mask/shape mismatch"}
        seq_summaries[k] = {
            "all": all_summary,
            "when_landmarks_present": present_summary,
            "when_landmarks_absent": absent_summary,
        }

    # Gesture prob sum sanity (only meaningful where at least one prob is finite)
    prob_keys = [k for k in keys if k.startswith("seq_gesture_prob_")]
    prob_sum = None
    if prob_keys:
        probs = []
        for k in sorted(prob_keys):
            probs.append(np.asarray(d[k], dtype=np.float64).reshape(-1))
        if probs and all(p.size == probs[0].size for p in probs):
            prob_sum = np.sum(np.stack(probs, axis=0), axis=0)
    gesture_prob_sum = _numeric_summary(prob_sum) if isinstance(prob_sum, np.ndarray) else {"note": "no gesture prob keys"}

    aggregated = _unbox_dict_scalar_object(d.get("aggregated"))
    aggregated_numeric_flat = _flatten_numeric_leaves(aggregated)
    aggregated_numeric: Dict[str, Any] = {
        "n_total_keys_top": int(len(aggregated.keys())) if isinstance(aggregated, dict) else 0,
        "n_numeric_leaves": int(len(aggregated_numeric_flat)),
        "numeric_leaves": {},
    }
    for ak, av in sorted(aggregated_numeric_flat.items()):
        aggregated_numeric["numeric_leaves"][ak] = av

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
        landmarks_present_true=lp_true,
        landmarks_present_ratio=lp_ratio,
        seq_summaries=seq_summaries,
        gesture_prob_sum=gesture_prob_sum,
        aggregated_numeric=aggregated_numeric,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for behavioral (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to behavioral_features.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc = {
        "component": "behavioral",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "landmarks_present_true_total": int(sum(r.landmarks_present_true for r in runs)),
            "landmarks_present_ratio_weighted": float(
                (sum(r.landmarks_present_true for r in runs) / max(sum(r.N for r in runs), 1))
            ),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

