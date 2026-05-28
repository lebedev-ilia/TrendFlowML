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


def _l2_norm_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(a, axis=1)


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
    D: int
    K_places365: int
    prompts_lens: Dict[str, int]

    frame_embeddings_l2: Dict[str, Any]
    consecutive_cosine_prev: Dict[str, Any]
    places365_topk_scores_row_sum: Dict[str, Any]
    shot_quality_scores_row_sum: Dict[str, Any]

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_embeddings = np.asarray(d.get("frame_embeddings", np.asarray([], dtype=np.float32)), dtype=np.float32)
    consecutive = np.asarray(d.get("consecutive_cosine_prev", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)
    p365_topk_scores = np.asarray(d.get("places365_topk_scores", np.asarray([], dtype=np.float32)), dtype=np.float32)
    shot_scores = np.asarray(d.get("shot_quality_scores", np.asarray([], dtype=np.float32)), dtype=np.float32)

    shot_prompts = np.asarray(d.get("shot_quality_prompts", np.asarray([], dtype=object)))
    aes_prompts = np.asarray(d.get("scene_aesthetic_prompts", np.asarray([], dtype=object)))
    lux_prompts = np.asarray(d.get("scene_luxury_prompts", np.asarray([], dtype=object)))
    atm_prompts = np.asarray(d.get("scene_atmosphere_prompts", np.asarray([], dtype=object)))
    cut_prompts = np.asarray(d.get("cut_detection_transition_prompts", np.asarray([], dtype=object)))
    pop_prompts = np.asarray(d.get("popularity_topic_prompts", np.asarray([], dtype=object)))
    p365_prompts = np.asarray(d.get("places365_prompts", np.asarray([], dtype=object)))

    N = int(frame_embeddings.shape[0]) if frame_embeddings.ndim == 2 else 0
    D = int(frame_embeddings.shape[1]) if frame_embeddings.ndim == 2 else 0
    K = int(p365_topk_scores.shape[1]) if p365_topk_scores.ndim == 2 else 0

    norms = _l2_norm_rows(frame_embeddings)

    if p365_topk_scores.ndim == 2 and p365_topk_scores.shape[0] > 0:
        p365_sum = p365_topk_scores.astype(np.float64).sum(axis=1)
    else:
        p365_sum = np.asarray([], dtype=np.float64)

    if shot_scores.ndim == 2 and shot_scores.shape[0] > 0:
        shot_sum = shot_scores.astype(np.float64).sum(axis=1)
    else:
        shot_sum = np.asarray([], dtype=np.float64)

    prompts_lens = {
        "shot_quality_prompts": int(shot_prompts.size),
        "scene_aesthetic_prompts": int(aes_prompts.size),
        "scene_luxury_prompts": int(lux_prompts.size),
        "scene_atmosphere_prompts": int(atm_prompts.size),
        "cut_detection_transition_prompts": int(cut_prompts.size),
        "popularity_topic_prompts": int(pop_prompts.size),
        "places365_prompts": int(p365_prompts.size),
    }

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
        D=D,
        K_places365=K,
        prompts_lens=prompts_lens,
        frame_embeddings_l2=_summary_finite_1d(norms),
        consecutive_cosine_prev=_summary_finite_1d(consecutive),
        places365_topk_scores_row_sum=_summary_finite_1d(p365_sum),
        shot_quality_scores_row_sum=_summary_finite_1d(shot_sum),
        stage_timings_ms=st,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for core_clip (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to core_clip/embeddings.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc = {
        "component": "core_clip",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "D_set": sorted({int(r.D) for r in runs}),
            "K_places365_set": sorted({int(r.K_places365) for r in runs}),
            "prompts_lens_set": {k: sorted({int(r.prompts_lens.get(k, 0)) for r in runs}) for k in (runs[0].prompts_lens.keys() if runs else [])},
            "frame_embeddings_l2_mean_min": float(min([r.frame_embeddings_l2.get("mean") for r in runs if r.frame_embeddings_l2.get("mean") is not None], default=0.0)),
            "frame_embeddings_l2_mean_max": float(max([r.frame_embeddings_l2.get("mean") for r in runs if r.frame_embeddings_l2.get("mean") is not None], default=0.0)),
            "consecutive_cosine_prev_nan_total": int(sum(int(r.consecutive_cosine_prev.get("nan") or 0) for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

