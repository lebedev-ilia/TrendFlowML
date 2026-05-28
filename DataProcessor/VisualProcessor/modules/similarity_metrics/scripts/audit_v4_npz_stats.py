#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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


def _as_bool0(x: Any) -> Optional[bool]:
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
    except Exception:
        return None
    return None


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


def _summary_finite_1d(x: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(x).astype(np.float64, copy=False).reshape(-1)
    finite = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "n_finite": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "nan": int(np.isnan(a).sum()),
        "inf": int(np.isinf(a).sum()),
    }


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
    F: int
    reference_present: Optional[bool]
    feature_finite: int
    feature_total: int

    centroid_sims: Dict[str, Any]
    temporal_sim_next: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    centroid = np.asarray(d.get("centroid_sims", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    temporal = np.asarray(d.get("temporal_sim_next", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    feature_names = np.asarray(d.get("feature_names", np.asarray([], dtype=object)))
    feature_values = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    N = int(frame_indices.size)
    F = int(feature_names.size)

    ref_present = _as_bool0(d.get("reference_present"))

    finite = np.isfinite(feature_values.astype(np.float64, copy=False))
    feature_finite = int(finite.sum())
    feature_total = int(feature_values.size)

    st_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    st: Dict[str, Optional[float]] = {}
    if isinstance(st_raw, dict):
        for k, v in st_raw.items():
            if isinstance(k, str) and k:
                st[k] = _as_float(v)

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
        F=F,
        reference_present=ref_present,
        feature_finite=feature_finite,
        feature_total=feature_total,
        centroid_sims=_summary_finite_1d(centroid),
        temporal_sim_next=_summary_finite_1d(temporal),
        stage_timings_ms=st,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for similarity_metrics (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to results.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    doc = {
        "component": "similarity_metrics",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "F_set": sorted({int(r.F) for r in runs}),
            "reference_present_true_count": int(sum(1 for r in runs if r.reference_present is True)),
            "reference_present_false_count": int(sum(1 for r in runs if r.reference_present is False)),
            "feature_finite_total": int(sum(r.feature_finite for r in runs)),
            "feature_total_total": int(sum(r.feature_total for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

