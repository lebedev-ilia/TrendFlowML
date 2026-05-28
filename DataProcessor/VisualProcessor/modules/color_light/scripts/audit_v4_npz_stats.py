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


def _corr_summary(mat: np.ndarray) -> Dict[str, Any]:
    """
    Correlation summary for compact frame features (M, 16).
    Uses Pearson correlation on finite rows only.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return {"note": "invalid_shape", "shape": list(mat.shape)}
    finite_row = np.all(np.isfinite(mat), axis=1)
    sub = mat[finite_row]
    if sub.shape[0] < 3:
        return {"note": "too_few_rows", "n_rows": int(sub.shape[0]), "shape": list(mat.shape)}
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(sub, rowvar=False)
    # Numeric safety: corrcoef may produce NaNs for constant columns
    c = np.asarray(c, dtype=np.float64)
    off = c[~np.eye(c.shape[0], dtype=bool)]
    off_f = off[np.isfinite(off)]
    return {
        "n_rows_used": int(sub.shape[0]),
        "shape": [int(c.shape[0]), int(c.shape[1])],
        "offdiag_abs_mean": float(np.mean(np.abs(off_f))) if off_f.size else None,
        "offdiag_abs_p95": float(np.percentile(np.abs(off_f), 95)) if off_f.size else None,
        "nan": int(np.isnan(c).sum()),
        "inf": int(np.isinf(c).sum()),
    }


def _video_features_corr_across_runs(vfs: List[Dict[str, float]], top_k: int = 40) -> Dict[str, Any]:
    """
    Compute correlation of scalar video_features across runs.
    N is tiny (typically 5); treat as a navigation aid, not as stable statistics.
    """
    if not vfs:
        return {"note": "no_runs"}
    keys = sorted({k for d in vfs for k in d.keys()})
    if not keys:
        return {"note": "no_keys"}

    # Keep only keys present and finite in all runs.
    kept: List[str] = []
    cols: List[List[float]] = []
    for k in keys:
        vals = []
        ok = True
        for d in vfs:
            if k not in d:
                ok = False
                break
            v = float(d[k])
            if not np.isfinite(v):
                ok = False
                break
            vals.append(v)
        if ok:
            kept.append(k)
            cols.append(vals)

    if len(kept) < 2 or len(vfs) < 3:
        return {"note": "too_small", "n_runs": int(len(vfs)), "n_keys": int(len(kept))}

    X = np.asarray(cols, dtype=np.float64).T  # [n_runs, n_keys]
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.corrcoef(X, rowvar=False)  # [n_keys, n_keys]
    C = np.asarray(C, dtype=np.float64)

    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"note": "all_nan", "n_runs": int(len(vfs)), "n_keys": int(len(kept))}

    order = np.argsort(np.abs(finite))[::-1]
    # Map back to original indices: we need to rank pairs by abs corr.
    # We'll compute abs ordering on the full upper triangle but skip non-finite.
    abs_vals = np.abs(vals)
    finite_mask = np.isfinite(vals)
    finite_idx = np.flatnonzero(finite_mask)
    top = finite_idx[np.argsort(abs_vals[finite_mask])[::-1][: max(0, int(top_k))]]

    pairs = []
    for flat in top.tolist():
        i = int(iu[0][flat])
        j = int(iu[1][flat])
        pairs.append({"k1": kept[i], "k2": kept[j], "corr": float(C[i, j])})

    return {
        "n_runs": int(len(vfs)),
        "n_keys_used": int(len(kept)),
        "abs_corr_summary": _numeric_summary(finite),
        "top_abs_pairs": pairs,
        "note": "N is small; interpret cautiously",
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
    store_debug_objects: Optional[bool]
    processed_frames: Optional[int]

    keys: List[str]
    N_frame_indices: int
    M_sequence: int
    N_equals_M: Optional[bool]
    indices_equal: Optional[bool]

    compact_summary: Dict[str, Any]
    compact_per_dim: Dict[str, Any]
    compact_corr: Dict[str, Any]

    video_features: Dict[str, Any]
    aggregated_frame_compact: Dict[str, Any]
    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))

    meta = _unbox_meta(d.get("meta"))
    platform_id = meta.get("platform_id")
    video_id = meta.get("video_id")
    run_id = meta.get("run_id")
    config_hash = meta.get("config_hash")
    sampling_policy_version = meta.get("sampling_policy_version")
    schema_version = meta.get("schema_version")
    producer_version = meta.get("producer_version")
    store_debug_objects = meta.get("store_debug_objects")
    processed_frames = meta.get("processed_frames")

    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32))).reshape(-1)
    sequence_frame_indices = np.asarray(d.get("sequence_frame_indices", np.asarray([], dtype=np.int32))).reshape(-1)
    N = int(frame_indices.size)
    M = int(sequence_frame_indices.size)

    indices_equal = None
    try:
        indices_equal = bool(N == M and np.array_equal(frame_indices.astype(np.int64), sequence_frame_indices.astype(np.int64)))
    except Exception:
        indices_equal = None

    compact = np.asarray(d.get("frame_compact_features", np.asarray([], dtype=np.float32)))
    compact_summary = _numeric_summary(compact)
    compact_corr = _corr_summary(compact)

    compact_per_dim: Dict[str, Any] = {}
    if compact.ndim == 2 and compact.shape[1] > 0:
        for j in range(int(compact.shape[1])):
            compact_per_dim[str(j)] = _numeric_summary(compact[:, j])

    # video_features: dict of scalars
    vf = _unbox_dict_scalar_object(d.get("video_features"))
    vf_numeric = _flatten_numeric_leaves(vf)
    vf_nan_keys: List[str] = []
    for k, v in (vf or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if float(v) != float(v):  # NaN
                vf_nan_keys.append(str(k))
    video_features = {
        "n_keys_top": int(len(vf.keys())) if isinstance(vf, dict) else 0,
        "n_numeric_leaves": int(len(vf_numeric)),
        "nan_numeric_top_keys": sorted(vf_nan_keys),
    }

    aggregated = _unbox_dict_scalar_object(d.get("aggregated"))
    agg_fc = (aggregated or {}).get("frame_compact") if isinstance(aggregated, dict) else None
    agg_fc = dict(agg_fc) if isinstance(agg_fc, dict) else {}
    aggregated_frame_compact = {
        "keys": sorted(list(agg_fc.keys())) if isinstance(agg_fc, dict) else [],
        "rows": int(agg_fc.get("rows")) if isinstance(agg_fc.get("rows"), (int, float)) else None,
        "valid_rows": int(agg_fc.get("valid_rows")) if isinstance(agg_fc.get("valid_rows"), (int, float)) else None,
    }
    for k in ["mean", "std", "p25", "p50", "p75"]:
        if k in agg_fc:
            try:
                aggregated_frame_compact[k] = _numeric_summary(np.asarray(agg_fc[k], dtype=np.float64))
            except Exception:
                aggregated_frame_compact[k] = {"note": "unreadable"}

    return RunStats(
        npz_path=npz_path,
        platform_id=str(platform_id) if platform_id is not None else None,
        video_id=str(video_id) if video_id is not None else None,
        run_id=str(run_id) if run_id is not None else None,
        config_hash=str(config_hash) if config_hash is not None else None,
        sampling_policy_version=str(sampling_policy_version) if sampling_policy_version is not None else None,
        schema_version=str(schema_version) if schema_version is not None else None,
        producer_version=str(producer_version) if producer_version is not None else None,
        store_debug_objects=bool(store_debug_objects) if store_debug_objects is not None else None,
        processed_frames=int(processed_frames) if isinstance(processed_frames, (int, float)) else None,
        keys=keys,
        N_frame_indices=N,
        M_sequence=M,
        N_equals_M=bool(N == M) if (N is not None and M is not None) else None,
        indices_equal=indices_equal,
        compact_summary=compact_summary,
        compact_per_dim=compact_per_dim,
        compact_corr=compact_corr,
        video_features=video_features,
        aggregated_frame_compact=aggregated_frame_compact,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for color_light (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to color_light_features.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    # Aggregate compact vectors across runs
    mats: List[np.ndarray] = []
    vf_numeric_runs: List[Dict[str, float]] = []
    for r in runs:
        try:
            d = _load_npz(r.npz_path)
            mats.append(np.asarray(d.get("frame_compact_features", np.asarray([], dtype=np.float32)), dtype=np.float64))
            vf = _unbox_dict_scalar_object(d.get("video_features"))
            vf_numeric_runs.append(_flatten_numeric_leaves(vf))
        except Exception:
            continue
    if mats:
        cat = np.concatenate([m for m in mats if m.ndim == 2 and m.size], axis=0) if any(m.ndim == 2 and m.size for m in mats) else np.asarray([], dtype=np.float64)
    else:
        cat = np.asarray([], dtype=np.float64)

    compact_all = {
        "rows_total": int(cat.shape[0]) if cat.ndim == 2 else 0,
        "dim": int(cat.shape[1]) if cat.ndim == 2 else 0,
        "summary": _numeric_summary(cat),
    }
    if cat.ndim == 2 and cat.shape[1] > 0 and cat.shape[0] > 0:
        per_dim = {}
        for j in range(int(cat.shape[1])):
            per_dim[str(j)] = _numeric_summary(cat[:, j])
        compact_all["per_dim"] = per_dim
        compact_all["corr"] = _corr_summary(cat)

    # Aggregate video_features NaN keys across runs
    nan_key_counts: Dict[str, int] = {}
    n_video_features_keys = []
    for r in runs:
        for k in r.video_features.get("nan_numeric_top_keys") or []:
            nan_key_counts[str(k)] = int(nan_key_counts.get(str(k), 0) + 1)
        n_video_features_keys.append(int(r.video_features.get("n_keys_top") or 0))

    video_features_corr = _video_features_corr_across_runs(vf_numeric_runs, top_k=40)

    doc = {
        "component": "color_light",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N_frame_indices for r in runs)),
            "M_total": int(sum(r.M_sequence for r in runs)),
            "M_min": int(min((r.M_sequence for r in runs), default=0)),
            "M_max": int(max((r.M_sequence for r in runs), default=0)),
            "video_features_keys_min": int(min(n_video_features_keys, default=0)),
            "video_features_keys_max": int(max(n_video_features_keys, default=0)),
            "video_features_nan_keys_counts": {k: int(v) for k, v in sorted(nan_key_counts.items())},
        },
        "compact_all": compact_all,
        "video_features_corr_across_runs": video_features_corr,
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

