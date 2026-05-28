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


def _as_obj_str_list(arr: Any) -> List[str]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [str(x) for x in arr.reshape(-1).tolist()]
    if isinstance(arr, (list, tuple)):
        return [str(x) for x in arr]
    return []


def _corr_top_pairs(mat: np.ndarray, names: List[str], top_k: int = 40) -> Dict[str, Any]:
    """
    Cross-run correlation across video-level features.
    mat: (R, F)
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        return {"n_runs": int(mat.shape[0]) if mat.ndim == 2 else 0, "n_features": int(mat.shape[1]) if mat.ndim == 2 else 0, "top_pairs": []}

    finite_all = np.all(np.isfinite(mat), axis=0)
    sub = mat[:, finite_all]
    sub_names = [names[i] for i, ok in enumerate(finite_all.tolist()) if ok]
    if sub.shape[1] < 2:
        return {"n_runs": int(mat.shape[0]), "n_features": int(mat.shape[1]), "n_finite_all": int(sub.shape[1]), "top_pairs": []}

    # Drop constant columns (corr undefined)
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
        top.append(
            {
                "abs_corr": float(a),
                "corr": float(c[i, j]),
                "a": str(sub2_names[i]),
                "b": str(sub2_names[j]),
            }
        )

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

    feature_set: Optional[str]
    features: Optional[str]
    num_workers: Optional[int]

    keys: List[str]
    N: int
    D: int
    F: int

    axis_ok: bool
    times_monotonic: bool

    frame_feature_values_summary: Dict[str, Any]
    feature_values_summary: Dict[str, Any]

    present_ratio_summary: Dict[str, Any]
    present_ratio_max_abs_diff_vs_computed: Optional[float]
    present_ratio_top_missing: List[Dict[str, Any]]

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    fi = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    ts = np.asarray(d.get("times_s", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    ff_names = _as_obj_str_list(d.get("frame_feature_names"))
    vf_names = _as_obj_str_list(d.get("feature_names"))
    ffv = np.asarray(d.get("frame_feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32)
    vfv = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    ffpr = np.asarray(d.get("frame_feature_present_ratio", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    N = int(fi.size)
    D = int(len(ff_names))
    F = int(len(vf_names))

    axis_ok = bool(fi.size == ts.size and ffv.ndim == 2 and ffv.shape[0] == fi.size and ffv.shape[1] == D and vfv.size == F and ffpr.size == D)
    times_monotonic = bool(ts.size < 2 or np.all(np.diff(ts.astype(np.float64, copy=False)) >= 0))

    ffv_summary = _numeric_summary(ffv.astype(np.float64, copy=False) if ffv.size else np.asarray([], dtype=np.float64))
    vfv_summary = _numeric_summary(vfv.astype(np.float64, copy=False) if vfv.size else np.asarray([], dtype=np.float64))

    pr_summary: Dict[str, Any] = {}
    pr_max_abs_diff: Optional[float] = None
    top_missing: List[Dict[str, Any]] = []
    if ffpr.size:
        pr_summary = {
            "min": float(np.min(ffpr)) if ffpr.size else None,
            "max": float(np.max(ffpr)) if ffpr.size else None,
            "mean": float(np.mean(ffpr)) if ffpr.size else None,
        }

    if ffv.ndim == 2 and ffv.shape[0] == N and ffv.shape[1] == D and D > 0:
        finite = np.isfinite(ffv.astype(np.float64, copy=False))
        pr_comp = finite.mean(axis=0).astype(np.float64)
        if ffpr.size == D:
            pr_max_abs_diff = float(np.max(np.abs(pr_comp - ffpr.astype(np.float64, copy=False))))
        missing = 1.0 - pr_comp
        order = np.argsort(-missing)  # most missing first
        for j in order[: min(10, D)].tolist():
            top_missing.append(
                {
                    "name": str(ff_names[j]) if j < len(ff_names) else str(j),
                    "present_ratio_computed": float(pr_comp[j]),
                    "present_ratio_reported": float(ffpr[j]) if ffpr.size == D else None,
                    "missing_ratio_computed": float(missing[j]),
                }
            )

    # stage timings
    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    num_workers = meta.get("num_workers")
    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        feature_set=str(meta.get("feature_set")) if meta.get("feature_set") is not None else None,
        features=str(meta.get("features")) if meta.get("features") is not None else None,
        num_workers=int(num_workers) if isinstance(num_workers, (int, np.integer)) else None,
        keys=keys,
        N=N,
        D=D,
        F=F,
        axis_ok=axis_ok,
        times_monotonic=times_monotonic,
        frame_feature_values_summary=ffv_summary,
        feature_values_summary=vfv_summary,
        present_ratio_summary=pr_summary,
        present_ratio_max_abs_diff_vs_computed=pr_max_abs_diff,
        present_ratio_top_missing=top_missing,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for frames_composition (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to frames_composition.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    n_runs = int(len(runs))

    # Cross-run: video-level feature correlations (only if names align)
    vf_names_common: Optional[List[str]] = None
    mat: List[np.ndarray] = []
    names_ok = True
    for r in runs:
        d = _load_npz(r.npz_path)
        names = _as_obj_str_list(d.get("feature_names"))
        vals = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        if vf_names_common is None:
            vf_names_common = names
        else:
            if names != vf_names_common:
                names_ok = False
        mat.append(vals.astype(np.float64, copy=False))

    corr = {}
    if vf_names_common is not None and names_ok and n_runs >= 2:
        try:
            M = np.stack(mat, axis=0) if mat else np.asarray([], dtype=np.float64).reshape(0, 0)
            corr = _corr_top_pairs(M, vf_names_common, top_k=40)
        except Exception:
            corr = {"error": "failed_to_compute"}
    else:
        corr = {"skipped": True, "reason": "feature_names_mismatch_or_insufficient_runs"}

    doc = {
        "component": "frames_composition",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": n_runs,
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "D_set": sorted({int(r.D) for r in runs}),
            "F_set": sorted({int(r.F) for r in runs}),
            "axis_ok_all": bool(all(r.axis_ok for r in runs)),
            "times_monotonic_all": bool(all(r.times_monotonic for r in runs)),
            "present_ratio_max_abs_diff_vs_computed_max": float(
                max([r.present_ratio_max_abs_diff_vs_computed for r in runs if isinstance(r.present_ratio_max_abs_diff_vs_computed, float)], default=0.0)
            ),
        },
        "cross_run": {
            "video_level_feature_corr_top_pairs": corr,
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

