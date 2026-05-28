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


def _corr_top_pairs(mat: np.ndarray, names: List[str], top_k: int = 40) -> Dict[str, Any]:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        return {"n_runs": int(mat.shape[0]) if mat.ndim == 2 else 0, "n_features": int(mat.shape[1]) if mat.ndim == 2 else 0, "top_pairs": []}

    finite_all = np.all(np.isfinite(mat), axis=0)
    sub = mat[:, finite_all]
    sub_names = [names[i] for i, ok in enumerate(finite_all.tolist()) if ok]
    if sub.shape[1] < 2:
        return {"n_runs": int(mat.shape[0]), "n_features": int(mat.shape[1]), "n_finite_all": int(sub.shape[1]), "top_pairs": []}

    v = np.var(sub, axis=0)
    keep = v > 0.0
    sub2 = sub[:, keep]
    sub2_names = [sub_names[i] for i, ok in enumerate(keep.tolist()) if ok]
    if sub2.shape[1] < 2:
        return {
            "n_runs": int(mat.shape[0]),
            "n_features": int(mat.shape[1]),
            "n_finite_all": int(sub.shape[1]),
            "n_non_constant": int(sub2.shape[1]),
            "top_pairs": [],
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(sub2, rowvar=False)
    c = np.asarray(c, dtype=np.float64)

    pairs: List[Tuple[float, int, int]] = []
    F2 = int(c.shape[0])
    for i in range(F2):
        for j in range(i + 1, F2):
            r = float(c[i, j])
            if not np.isfinite(r):
                continue
            pairs.append((abs(r), i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    top = []
    for a, i, j in pairs[: max(0, int(top_k))]:
        top.append({"abs_corr": float(a), "corr": float(c[i, j]), "a": str(sub2_names[i]), "b": str(sub2_names[j])})

    return {
        "n_runs": int(mat.shape[0]),
        "n_features": int(mat.shape[1]),
        "n_finite_all": int(sub.shape[1]),
        "n_non_constant": int(sub2.shape[1]),
        "top_pairs": top,
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

    keys: List[str]
    N: int
    F: int
    V: int
    K: int

    face_present_true: int
    face_present_ratio: float
    frames_processed_openface: Optional[int]

    frame_features_nan_ratio: Optional[float]
    compact22_nan_ratio: Optional[float]
    video_feature_values_nan: Optional[int]

    video_feature_values_summary: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    fi = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    face_any = np.asarray(d.get("face_present_any", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    ff_names = _as_obj_str_list(d.get("frame_feature_names"))
    ff = np.asarray(d.get("frame_features", np.asarray([], dtype=np.float32)), dtype=np.float32)
    comp = np.asarray(d.get("compact22", np.asarray([], dtype=np.float32)), dtype=np.float32)
    v_names = _as_obj_str_list(d.get("feature_names"))
    v_vals = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    ev_times = np.asarray(d.get("event_times_s", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    K = int(ev_times.size)

    N = int(fi.size)
    F = int(len(ff_names))
    V = int(len(v_names))

    face_true = int(face_any.sum()) if face_any.size else 0
    face_ratio = float(face_true / max(N, 1))

    frames_processed_openface = None
    summary = d.get("summary")
    if isinstance(summary, np.ndarray) and summary.dtype == object and summary.shape == ():
        try:
            summary = summary.item()
        except Exception:
            summary = None
    if isinstance(summary, dict):
        v = summary.get("frames_processed_openface")
        try:
            frames_processed_openface = int(v) if v is not None else None
        except Exception:
            frames_processed_openface = None

    ff_nan_ratio = None
    if ff.size:
        ff_nan_ratio = float(np.isnan(ff.astype(np.float64, copy=False)).mean())
    comp_nan_ratio = None
    if comp.size:
        comp_nan_ratio = float(np.isnan(comp.astype(np.float64, copy=False)).mean())

    v_nan = None
    if v_vals.size:
        v_nan = int(np.isnan(v_vals.astype(np.float64, copy=False)).sum())

    # stage timings
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
        F=F,
        V=V,
        K=K,
        face_present_true=face_true,
        face_present_ratio=face_ratio,
        frames_processed_openface=frames_processed_openface,
        frame_features_nan_ratio=ff_nan_ratio,
        compact22_nan_ratio=comp_nan_ratio,
        video_feature_values_nan=v_nan,
        video_feature_values_summary=_numeric_summary(v_vals),
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for micro_emotion (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to micro_emotion.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    # Cross-run correlations for video-level feature_values (only if names align)
    names_common: Optional[List[str]] = None
    mat: List[np.ndarray] = []
    names_ok = True
    for r in runs:
        d = _load_npz(r.npz_path)
        names = _as_obj_str_list(d.get("feature_names"))
        vals = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if names_common is None:
            names_common = names
        else:
            if names != names_common:
                names_ok = False
        mat.append(vals.astype(np.float64, copy=False))

    corr = {}
    if names_common is not None and names_ok and len(runs) >= 2 and len(names_common) >= 2:
        try:
            M = np.stack(mat, axis=0) if mat else np.asarray([], dtype=np.float64).reshape(0, 0)
            corr = _corr_top_pairs(M, names_common, top_k=40)
        except Exception:
            corr = {"error": "failed_to_compute"}
    else:
        corr = {"skipped": True, "reason": "feature_names_mismatch_or_insufficient_runs"}

    doc = {
        "component": "micro_emotion",
        "level": "L2 (A+B, ok artifacts only)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "F_set": sorted({int(r.F) for r in runs}),
            "V_set": sorted({int(r.V) for r in runs}),
            "K_total": int(sum(r.K for r in runs)),
            "face_present_true_total": int(sum(r.face_present_true for r in runs)),
            "face_present_ratio_weighted": float(sum(r.face_present_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "video_feature_values_nan_total": int(sum(int(r.video_feature_values_nan or 0) for r in runs)),
        },
        "cross_run": {
            "video_feature_corr_top_pairs": corr,
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

