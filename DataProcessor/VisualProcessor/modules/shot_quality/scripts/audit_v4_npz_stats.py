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


def _unbox0(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    if isinstance(x, np.ndarray) and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _unbox_meta(meta_arr: Any) -> Dict[str, Any]:
    v = _unbox0(meta_arr)
    if isinstance(v, dict):
        return dict(v)
    return {}


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


def _nan_inf_counts(arr: np.ndarray) -> Dict[str, int]:
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.floating):
        return {"nan": 0, "inf": 0}
    af = a.astype(np.float64, copy=False)
    return {"nan": int(np.isnan(af).sum()), "inf": int(np.isinf(af).sum())}


def _finite_ratio(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.size == 0:
        return np.asarray([], dtype=np.float32)
    if not np.issubdtype(a.dtype, np.floating):
        return np.ones((a.shape[-1],), dtype=np.float32) if a.ndim == 2 else np.asarray([], dtype=np.float32)
    af = a.astype(np.float64, copy=False)
    if af.ndim != 2:
        return np.asarray([], dtype=np.float32)
    finite = np.isfinite(af)
    return (finite.sum(axis=0) / float(af.shape[0])).astype(np.float32)


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
    S: int
    F: int
    P: int
    K: int

    quality_row_sum: Dict[str, Any]
    shot_topk_sum: Dict[str, Any]

    frame_features_nan: int
    frame_features_inf: int
    n_features_any_nan: int
    n_features_all_nan: int
    all_nan_feature_names: List[str]

    present_ratio_max_abs_diff_vs_computed: Optional[float]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_features = np.asarray(d.get("frame_features", np.asarray([], dtype=np.float32)), dtype=np.float32)
    feature_names = np.asarray(d.get("feature_names", np.asarray([], dtype=object)))
    quality_probs = np.asarray(d.get("quality_probs", np.asarray([], dtype=np.float16)))
    shot_topk_probs = np.asarray(d.get("shot_quality_topk_probs", np.asarray([], dtype=np.float32)), dtype=np.float32)
    shot_topk_ids = np.asarray(d.get("shot_quality_topk_ids", np.asarray([], dtype=np.int32)))
    frame_present_ratio = np.asarray(d.get("frame_feature_present_ratio", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    N = int(frame_features.shape[0]) if frame_features.ndim == 2 else 0
    F = int(frame_features.shape[1]) if frame_features.ndim == 2 else 0
    S = int(np.asarray(d.get("shot_start_frame", np.asarray([], dtype=np.int32))).size)
    P = int(quality_probs.shape[1]) if quality_probs.ndim == 2 else 0
    K = int(shot_topk_probs.shape[1]) if shot_topk_probs.ndim == 2 else (int(shot_topk_ids.shape[1]) if shot_topk_ids.ndim == 2 else 0)

    # quality_probs row sum (should be ~1)
    if quality_probs.ndim == 2 and quality_probs.shape[0] > 0:
        qsum = quality_probs.astype(np.float32).sum(axis=1)
    else:
        qsum = np.asarray([], dtype=np.float32)

    # shot topk probs sum (not expected to be 1)
    if shot_topk_probs.ndim == 2 and shot_topk_probs.shape[0] > 0:
        stsum = shot_topk_probs.astype(np.float64).sum(axis=1).astype(np.float32)
    else:
        stsum = np.asarray([], dtype=np.float32)

    # NaN per feature
    if frame_features.ndim == 2 and N > 0:
        finite = np.isfinite(frame_features.astype(np.float64, copy=False))
        ratio = (finite.sum(axis=0) / float(N)).astype(np.float32)
        any_nan = int((ratio < 1.0).sum())
        all_nan = int((ratio == 0.0).sum())
        all_nan_idx = np.where(ratio == 0.0)[0].tolist()
    else:
        ratio = np.asarray([], dtype=np.float32)
        any_nan = 0
        all_nan = 0
        all_nan_idx = []

    all_nan_names: List[str] = []
    try:
        if feature_names is not None and len(all_nan_idx) > 0:
            for i in all_nan_idx:
                if 0 <= int(i) < int(feature_names.size):
                    all_nan_names.append(str(feature_names[int(i)]))
    except Exception:
        all_nan_names = []

    # present ratio diff
    diff = None
    try:
        if frame_present_ratio.size == ratio.size and ratio.size > 0:
            diff = float(np.max(np.abs(frame_present_ratio.astype(np.float64) - ratio.astype(np.float64))))
    except Exception:
        diff = None

    # stage timings
    st_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    st: Dict[str, Optional[float]] = {}
    if isinstance(st_raw, dict):
        for k, v in st_raw.items():
            if isinstance(k, str) and k:
                st[k] = _as_float(v)

    counts = _nan_inf_counts(frame_features)

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
        S=S,
        F=F,
        P=P,
        K=K,
        quality_row_sum=_summary_finite_1d(qsum),
        shot_topk_sum=_summary_finite_1d(stsum),
        frame_features_nan=int(counts["nan"]),
        frame_features_inf=int(counts["inf"]),
        n_features_any_nan=any_nan,
        n_features_all_nan=all_nan,
        all_nan_feature_names=sorted(list(set(all_nan_names))),
        present_ratio_max_abs_diff_vs_computed=diff,
        stage_timings_ms=st,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for shot_quality (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to shot_quality.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    doc = {
        "component": "shot_quality",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "S_set": sorted({int(r.S) for r in runs}),
            "F_set": sorted({int(r.F) for r in runs}),
            "P_set": sorted({int(r.P) for r in runs}),
            "K_set": sorted({int(r.K) for r in runs}),
            "quality_row_sum_min_min": float(min([r.quality_row_sum.get("min") for r in runs if r.quality_row_sum.get("min") is not None], default=0.0)),
            "quality_row_sum_max_max": float(max([r.quality_row_sum.get("max") for r in runs if r.quality_row_sum.get("max") is not None], default=0.0)),
            "shot_topk_sum_min_min": float(min([r.shot_topk_sum.get("min") for r in runs if r.shot_topk_sum.get("min") is not None], default=0.0)),
            "shot_topk_sum_max_max": float(max([r.shot_topk_sum.get("max") for r in runs if r.shot_topk_sum.get("max") is not None], default=0.0)),
            "present_ratio_max_abs_diff_vs_computed_max": float(
                max([r.present_ratio_max_abs_diff_vs_computed for r in runs if r.present_ratio_max_abs_diff_vs_computed is not None], default=0.0)
            ),
            "all_nan_feature_names_union": sorted({n for r in runs for n in (r.all_nan_feature_names or [])}),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

