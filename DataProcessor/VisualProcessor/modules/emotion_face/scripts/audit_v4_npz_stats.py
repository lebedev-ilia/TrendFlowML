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
    K_keyframes: int

    face_present_true: int
    processed_mask_true: int
    face_present_ratio: float
    processed_ratio: float

    # Numeric sanity
    probs_row_sum_when_processed: Dict[str, Any]
    valence_when_processed: Dict[str, Any]
    arousal_when_processed: Dict[str, Any]
    intensity_when_processed: Dict[str, Any]
    confidence_when_processed: Dict[str, Any]

    dominant_emotion_id_counts: Dict[str, int]
    dominant_emotion_id_neg1_ratio: float

    stage_timings_ms: Dict[str, Optional[float]]
    features_numeric: Dict[str, Any]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    processed = np.asarray(d.get("processed_mask", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    face_present = np.asarray(d.get("face_present", np.asarray([], dtype=bool)), dtype=bool).reshape(-1)
    N = int(processed.size)
    pm_true = int(processed.sum())
    fp_true = int(face_present.sum())
    pm_ratio = float(pm_true / max(N, 1))
    fp_ratio = float(fp_true / max(N, 1))

    # keyframes K (object array)
    kf = d.get("keyframes")
    K = int(np.asarray(kf).size) if isinstance(kf, np.ndarray) else 0

    # time series
    valence = np.asarray(d.get("valence", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    arousal = np.asarray(d.get("arousal", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    intensity = np.asarray(d.get("intensity", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    conf = np.asarray(d.get("emotion_confidence", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    probs = np.asarray(d.get("emotion_probs", np.asarray([], dtype=np.float32)), dtype=np.float32)

    if probs.ndim == 2 and probs.shape[0] == N:
        row_sum = np.sum(probs.astype(np.float64), axis=1)
    else:
        row_sum = np.asarray([], dtype=np.float64)

    # Stats only when processed
    valence_proc = valence[processed] if valence.size == N else np.asarray([], dtype=np.float32)
    arousal_proc = arousal[processed] if arousal.size == N else np.asarray([], dtype=np.float32)
    intensity_proc = intensity[processed] if intensity.size == N else np.asarray([], dtype=np.float32)
    conf_proc = conf[processed] if conf.size == N else np.asarray([], dtype=np.float32)
    row_sum_proc = row_sum[processed] if row_sum.size == N else np.asarray([], dtype=np.float64)

    # dominant id
    dom = np.asarray(d.get("dominant_emotion_id", np.asarray([], dtype=np.int8)), dtype=np.int8).reshape(-1)
    dom_counts: Dict[str, int] = {}
    if dom.size:
        for x in dom.astype(np.int64).tolist():
            k = str(int(x))
            dom_counts[k] = int(dom_counts.get(k, 0) + 1)
    neg1_ratio = float((dom == -1).mean()) if dom.size else 1.0

    # stage timings
    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    # features dict: numeric leaves only
    feats = _unbox_dict_scalar_object(d.get("features"))
    feats_num: Dict[str, float] = {}
    for k, v in (feats or {}).items():
        fv = _as_float(v)
        if fv is not None:
            feats_num[str(k)] = float(fv)
    features_numeric = {
        "n_keys_top": int(len(feats.keys())) if isinstance(feats, dict) else 0,
        "n_numeric": int(len(feats_num)),
        "numeric": {k: feats_num[k] for k in sorted(feats_num.keys())},
    }

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
        K_keyframes=K,
        face_present_true=fp_true,
        processed_mask_true=pm_true,
        face_present_ratio=fp_ratio,
        processed_ratio=pm_ratio,
        probs_row_sum_when_processed=_numeric_summary(row_sum_proc),
        valence_when_processed=_numeric_summary(valence_proc),
        arousal_when_processed=_numeric_summary(arousal_proc),
        intensity_when_processed=_numeric_summary(intensity_proc),
        confidence_when_processed=_numeric_summary(conf_proc),
        dominant_emotion_id_counts={k: int(v) for k, v in sorted(dom_counts.items(), key=lambda kv: int(kv[0]))},
        dominant_emotion_id_neg1_ratio=neg1_ratio,
        stage_timings_ms=stage_timings_ms,
        features_numeric=features_numeric,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for emotion_face (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to emotion_face.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc = {
        "component": "emotion_face",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "K_keyframes_total": int(sum(r.K_keyframes for r in runs)),
            "face_present_true_total": int(sum(r.face_present_true for r in runs)),
            "processed_mask_true_total": int(sum(r.processed_mask_true for r in runs)),
            "face_present_ratio_weighted": float(sum(r.face_present_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "processed_ratio_weighted": float(sum(r.processed_mask_true for r in runs) / max(sum(r.N for r in runs), 1)),
            "dominant_emotion_id_counts_total": {
                k: int(sum(int(r.dominant_emotion_id_counts.get(k, 0)) for r in runs))
                for k in sorted({k for r in runs for k in r.dominant_emotion_id_counts.keys()}, key=lambda x: int(x))
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

