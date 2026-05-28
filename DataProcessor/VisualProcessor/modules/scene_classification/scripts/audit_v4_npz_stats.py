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
    topk: int

    label_fusion: Optional[str]
    min_scene_seconds: Optional[float]

    frame_scene_id_unique: int
    frame_scene_id_min: Optional[int]
    frame_scene_id_max: Optional[int]

    frame_topk_sum_summary: Dict[str, Any]
    frame_top1_prob_summary: Dict[str, Any]
    frame_entropy_summary: Dict[str, Any]

    scene_label_counts: Dict[str, int]
    prompts_len: Dict[str, int]

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    keys = sorted(list(d.keys()))
    meta = _unbox_meta(d.get("meta"))

    frame_topk_probs = np.asarray(d.get("frame_topk_probs", np.asarray([], dtype=np.float32)), dtype=np.float32)
    frame_top1 = np.asarray(d.get("frame_top1_prob", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    frame_entropy = np.asarray(d.get("frame_entropy", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    frame_scene_id = np.asarray(d.get("frame_scene_id", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    scene_label = _as_obj_str_list(d.get("scene_label"))

    N = int(frame_scene_id.size)
    topk = int(frame_topk_probs.shape[1]) if frame_topk_probs.ndim == 2 else 0
    S = int(len(scene_label))

    # top-k sum per frame (finite only)
    if frame_topk_probs.ndim == 2 and frame_topk_probs.shape[0] == N:
        row_sum = np.sum(frame_topk_probs.astype(np.float64), axis=1)
    else:
        row_sum = np.asarray([], dtype=np.float64)

    # scene id stats
    uniq = int(np.unique(frame_scene_id).size) if frame_scene_id.size else 0
    sid_min = int(frame_scene_id.min()) if frame_scene_id.size else None
    sid_max = int(frame_scene_id.max()) if frame_scene_id.size else None

    # scene label counts
    counts: Dict[str, int] = {}
    for s in scene_label:
        counts[s] = int(counts.get(s, 0) + 1)

    # prompts lengths
    prompts_len = {
        "scene_aesthetic_prompts": int(np.asarray(d.get("scene_aesthetic_prompts", np.asarray([], dtype=object))).size),
        "scene_luxury_prompts": int(np.asarray(d.get("scene_luxury_prompts", np.asarray([], dtype=object))).size),
        "scene_atmosphere_prompts": int(np.asarray(d.get("scene_atmosphere_prompts", np.asarray([], dtype=object))).size),
        "places365_prompts": int(np.asarray(d.get("places365_prompts", np.asarray([], dtype=object))).size),
    }

    # stage timings
    stage_timings_raw = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    stage_timings_ms: Dict[str, Optional[float]] = {}
    if isinstance(stage_timings_raw, dict):
        for k, v in stage_timings_raw.items():
            if isinstance(k, str) and k:
                stage_timings_ms[k] = _as_float(v)

    label_fusion = d.get("label_fusion")
    if isinstance(label_fusion, np.ndarray) and label_fusion.dtype == object and label_fusion.shape == ():
        try:
            label_fusion = label_fusion.item()
        except Exception:
            pass

    min_scene_seconds = d.get("min_scene_seconds")
    if isinstance(min_scene_seconds, np.ndarray) and min_scene_seconds.shape == ():
        try:
            min_scene_seconds = float(min_scene_seconds.item())
        except Exception:
            min_scene_seconds = None
    else:
        min_scene_seconds = _as_float(min_scene_seconds)

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
        topk=topk,
        label_fusion=str(label_fusion) if label_fusion is not None else None,
        min_scene_seconds=min_scene_seconds,
        frame_scene_id_unique=uniq,
        frame_scene_id_min=sid_min,
        frame_scene_id_max=sid_max,
        frame_topk_sum_summary=_numeric_summary(row_sum.astype(np.float32, copy=False) if row_sum.size else np.asarray([], dtype=np.float32)),
        frame_top1_prob_summary=_numeric_summary(frame_top1),
        frame_entropy_summary=_numeric_summary(frame_entropy),
        scene_label_counts={k: int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))},
        prompts_len=prompts_len,
        stage_timings_ms=stage_timings_ms,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for scene_classification (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to scene_classification_features.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    doc = {
        "component": "scene_classification",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "S_set": sorted({int(r.S) for r in runs}),
            "topk_set": sorted({int(r.topk) for r in runs}),
            "label_fusion_set": sorted({str(r.label_fusion) for r in runs if r.label_fusion is not None}),
            "places365_prompts_len_set": sorted({int(r.prompts_len.get('places365_prompts', 0)) for r in runs}),
            "frame_topk_sum_min_min": float(min([r.frame_topk_sum_summary.get("min") for r in runs if r.frame_topk_sum_summary.get("min") is not None], default=0.0)),
            "frame_topk_sum_max_max": float(max([r.frame_topk_sum_summary.get("max") for r in runs if r.frame_topk_sum_summary.get("max") is not None], default=0.0)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

