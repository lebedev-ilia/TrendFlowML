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
    xs = np.asarray(xs)
    if xs.size == 0:
        return {f"p{int(p):02d}": None for p in ps}
    out = np.percentile(xs, ps).astype(np.float64)
    return {f"p{int(p):02d}": float(v) for p, v in zip(ps, out)}


def _numeric_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out: Dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "size": int(arr.size),
    }

    if arr.size == 0:
        out.update(
            {
                "nan": 0,
                "inf": 0,
                "n_valid": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        )
        out.update(_percentiles(np.asarray([], dtype=np.float64), [1, 5, 50, 95, 99]))
        return out

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        # Integer arrays: no NaN/Inf by definition
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
    stage_timings_ms: Dict[str, Optional[float]]

    keys: List[str]
    T_tracks: int
    total_clips: int
    tracks_with_multi_clips: int

    metric_summaries: Dict[str, Any]
    embedding_norms: Dict[str, Any]


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
    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    tracks = np.asarray(d.get("tracks", np.asarray([], dtype=np.int32)))
    T = int(tracks.shape[0]) if tracks.ndim == 1 else int(tracks.size)

    metric_summaries: Dict[str, Any] = {}
    for k in keys:
        if not k.startswith("metric__"):
            continue
        v = d[k]
        if isinstance(v, np.ndarray) and v.dtype != object:
            metric_summaries[k] = _numeric_summary(v)
        else:
            metric_summaries[k] = {"dtype": str(getattr(v, "dtype", type(v))), "shape": list(getattr(v, "shape", ()))}

    # num_clips is the main coverage signal for L2
    num_clips = d.get("metric__num_clips")
    if isinstance(num_clips, np.ndarray) and num_clips.size:
        nc = np.asarray(num_clips).astype(np.int64, copy=False)
        total_clips = int(nc.sum())
        tracks_with_multi = int((nc > 1).sum())
    else:
        total_clips = 0
        tracks_with_multi = 0

    # embeddings are stored as object array of per-track [num_clips, 256]
    emb = d.get("embeddings")
    norms_all: List[float] = []
    if isinstance(emb, np.ndarray) and emb.dtype == object and emb.size:
        for e in emb.tolist():
            try:
                ea = np.asarray(e, dtype=np.float32)
                if ea.ndim != 2 or ea.shape[0] == 0:
                    continue
                n = np.linalg.norm(ea, axis=1)
                norms_all.extend([float(x) for x in n.tolist()])
            except Exception:
                continue

    emb_norms = np.asarray(norms_all, dtype=np.float64)
    embedding_norms = {
        "n_vectors": int(emb_norms.size),
        "summary": _numeric_summary(emb_norms),
    }

    return RunStats(
        npz_path=npz_path,
        platform_id=str(platform_id) if platform_id is not None else None,
        video_id=str(video_id) if video_id is not None else None,
        run_id=str(run_id) if run_id is not None else None,
        config_hash=str(config_hash) if config_hash is not None else None,
        sampling_policy_version=str(sampling_policy_version) if sampling_policy_version is not None else None,
        schema_version=str(schema_version) if schema_version is not None else None,
        producer_version=str(producer_version) if producer_version is not None else None,
        stage_timings_ms=stage_timings_ms,
        keys=keys,
        T_tracks=T,
        total_clips=total_clips,
        tracks_with_multi_clips=tracks_with_multi,
        metric_summaries=metric_summaries,
        embedding_norms=embedding_norms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for action_recognition (VisualProcessor)")
    ap.add_argument(
        "--npz",
        action="append",
        required=True,
        help="Path to action_recognition_features.npz (repeatable)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path (will create parent dirs)",
    )
    args = ap.parse_args()

    npz_paths: List[str] = [os.fspath(p) for p in (args.npz or [])]
    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs: List[RunStats] = []
    for p in npz_paths:
        runs.append(compute_run_stats(p))

    # Aggregate across runs (tracks and embedding vectors)
    agg_tracks = int(sum(r.T_tracks for r in runs))
    agg_total_clips = int(sum(r.total_clips for r in runs))
    agg_tracks_multi = int(sum(r.tracks_with_multi_clips for r in runs))
    agg_stage_total_ms = float(
        sum((r.stage_timings_ms.get("total") or 0.0) for r in runs)
    )

    # Aggregate metric summaries across all runs for common keys
    all_metric_keys = sorted({k for r in runs for k in r.metric_summaries.keys()})
    agg_metrics: Dict[str, Any] = {}
    for mk in all_metric_keys:
        arrays: List[np.ndarray] = []
        for r in runs:
            ms = r.metric_summaries.get(mk)
            # only aggregate numeric arrays we can reload reliably
            # (we keep per-run summaries anyway)
            _ = ms
        # Keep this section minimal for now; per-run summaries + high-level aggregates are enough for L2 navigation.
        agg_metrics[mk] = {"note": "see per_run.metric_summaries"}

    doc = {
        "component": "action_recognition",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "tracks_total": agg_tracks,
            "clips_total": agg_total_clips,
            "tracks_with_multi_clips_total": agg_tracks_multi,
            "stage_total_ms_sum": agg_stage_total_ms,
        },
        "per_run": [asdict(r) for r in runs],
        "metric_keys": all_metric_keys,
        "agg_metric_summaries": agg_metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

