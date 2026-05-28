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


def _numeric_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out: Dict[str, Any] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "size": int(arr.size)}
    if arr.size == 0:
        out.update({"nan": 0, "inf": 0, "n_valid": 0, "min": None, "max": None, "mean": None, "std": None})
        return out

    a = arr.astype(np.float64, copy=False) if not np.issubdtype(arr.dtype, np.integer) else arr.astype(np.int64, copy=False)
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_):
        out.update(
            {
                "nan": 0,
                "inf": 0,
                "n_valid": int(a.size),
                "min": int(np.min(a)) if a.size else None,
                "max": int(np.max(a)) if a.size else None,
                "mean": float(np.mean(a)) if a.size else None,
                "std": float(np.std(a)) if a.size else None,
            }
        )
        return out

    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    finite = a[np.isfinite(a)]
    out.update(
        {
            "nan": nan,
            "inf": inf,
            "n_valid": int(finite.size),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
        }
    )
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


def _as_obj_str_list(arr: Any) -> List[str]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [str(x) for x in arr.reshape(-1).tolist()]
    if isinstance(arr, (list, tuple)):
        return [str(x) for x in arr]
    return []


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
    S: int
    D: int
    F: int
    E: int
    T: int

    axis_ok: bool
    times_monotonic: bool

    frame_features_nan_ratio: Optional[float]
    text_feature_nan_ratio: Optional[float]
    all_nan_frame_feature_names: List[str]

    present_ratio_summary: Dict[str, Any]
    present_ratio_max_abs_diff_vs_computed: Optional[float]

    scene_embeddings_row_norm_max_abs_err: Optional[float]
    scene_embedding_mean_norm_summary: Dict[str, Any]

    event_type_counts: Dict[str, int]
    models_used_n: int

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    fi = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    ts = np.asarray(d.get("times_s", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    scene_id = np.asarray(d.get("scene_id", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    scene_emb = np.asarray(d.get("scene_embeddings", np.asarray([], dtype=np.float32)), dtype=np.float32)

    ff_names = _as_obj_str_list(d.get("frame_feature_names"))
    ff = np.asarray(d.get("frame_features", np.asarray([], dtype=np.float32)), dtype=np.float32)
    ffpr = np.asarray(d.get("frame_feature_present_ratio", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    ev_type = np.asarray(d.get("event_type_id", np.asarray([], dtype=np.int16)), dtype=np.int16).reshape(-1)
    text_names = _as_obj_str_list(d.get("text_feature_names"))
    text_vals = np.asarray(d.get("text_feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    N = int(fi.size)
    S = int(scene_emb.shape[0]) if scene_emb.ndim == 2 else 0
    D = int(scene_emb.shape[1]) if scene_emb.ndim == 2 else 0
    F = int(len(ff_names))
    E = int(ev_type.size)
    T = int(len(text_names))

    axis_ok = bool(
        (fi.size == ts.size == scene_id.size)
        and (ff.ndim == 2 and ff.shape[0] == fi.size and ff.shape[1] == F)
        and (ffpr.size == F)
        and (text_vals.size == T)
    )
    times_monotonic = bool(ts.size < 2 or np.all(np.diff(ts.astype(np.float64, copy=False)) >= 0))

    # NaN ratios
    frame_nan_ratio: Optional[float] = None
    if ff.size:
        frame_nan_ratio = float(np.isnan(ff.astype(np.float64, copy=False)).mean())
    text_nan_ratio: Optional[float] = None
    if text_vals.size:
        text_nan_ratio = float(np.isnan(text_vals.astype(np.float64, copy=False)).mean())

    # present_ratio sanity
    pr_summary: Dict[str, Any] = {}
    pr_max_abs_diff: Optional[float] = None
    all_nan_cols: List[str] = []
    if ffpr.size:
        pr_summary = {"min": float(np.min(ffpr)), "max": float(np.max(ffpr)), "mean": float(np.mean(ffpr))}
    if ff.ndim == 2 and ff.shape[0] == N and ff.shape[1] == F and F > 0:
        pr_comp = np.isfinite(ff.astype(np.float64, copy=False)).mean(axis=0).astype(np.float64)
        if ffpr.size == F:
            pr_max_abs_diff = float(np.max(np.abs(pr_comp - ffpr.astype(np.float64, copy=False))))
        for j in range(F):
            if float(pr_comp[j]) == 0.0:
                all_nan_cols.append(str(ff_names[j]) if j < len(ff_names) else str(j))

    # scene embeddings row norms
    norm_err: Optional[float] = None
    if scene_emb.ndim == 2 and scene_emb.size:
        norms = np.linalg.norm(scene_emb.astype(np.float64, copy=False), axis=1)
        if norms.size:
            norm_err = float(np.max(np.abs(norms - 1.0)))

    mean_norm = np.asarray(d.get("scene_embedding_mean_norm", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    mean_norm_summary = _numeric_summary(mean_norm)

    # events
    ev_counts: Dict[str, int] = {}
    if ev_type.size:
        for x in ev_type.astype(np.int64).tolist():
            k = str(int(x))
            ev_counts[k] = int(ev_counts.get(k, 0) + 1)

    models_used = meta.get("models_used")
    models_used_n = int(len(models_used)) if isinstance(models_used, list) else 0

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
        S=S,
        D=D,
        F=F,
        E=E,
        T=T,
        axis_ok=axis_ok,
        times_monotonic=times_monotonic,
        frame_features_nan_ratio=frame_nan_ratio,
        text_feature_nan_ratio=text_nan_ratio,
        all_nan_frame_feature_names=sorted(all_nan_cols),
        present_ratio_summary=pr_summary,
        present_ratio_max_abs_diff_vs_computed=pr_max_abs_diff,
        scene_embeddings_row_norm_max_abs_err=norm_err,
        scene_embedding_mean_norm_summary=mean_norm_summary,
        event_type_counts={k: int(v) for k, v in sorted(ev_counts.items(), key=lambda kv: int(kv[0]))},
        models_used_n=models_used_n,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for high_level_semantic (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to high_level_semantic.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    n_runs = int(len(runs))

    # Cross-run correlations for text_feature_values (only if names align)
    text_names_common: Optional[List[str]] = None
    mat: List[np.ndarray] = []
    names_ok = True
    for r in runs:
        d = _load_npz(r.npz_path)
        names = _as_obj_str_list(d.get("text_feature_names"))
        vals = np.asarray(d.get("text_feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if text_names_common is None:
            text_names_common = names
        else:
            if names != text_names_common:
                names_ok = False
        mat.append(vals.astype(np.float64, copy=False))

    corr = {}
    if text_names_common is not None and names_ok and n_runs >= 2 and len(text_names_common) >= 2:
        try:
            M = np.stack(mat, axis=0) if mat else np.asarray([], dtype=np.float64).reshape(0, 0)
            corr = _corr_top_pairs(M, text_names_common, top_k=40)
        except Exception:
            corr = {"error": "failed_to_compute"}
    else:
        corr = {"skipped": True, "reason": "text_feature_names_mismatch_or_insufficient_runs"}

    doc = {
        "component": "high_level_semantic",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": n_runs,
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "S_set": sorted({int(r.S) for r in runs}),
            "D_set": sorted({int(r.D) for r in runs}),
            "F_set": sorted({int(r.F) for r in runs}),
            "T_set": sorted({int(r.T) for r in runs}),
            "axis_ok_all": bool(all(r.axis_ok for r in runs)),
            "times_monotonic_all": bool(all(r.times_monotonic for r in runs)),
            "all_nan_frame_feature_names_union": sorted({x for r in runs for x in r.all_nan_frame_feature_names}),
            "models_used_n_set": sorted({int(r.models_used_n) for r in runs}),
        },
        "cross_run": {
            "text_feature_corr_top_pairs": corr,
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

