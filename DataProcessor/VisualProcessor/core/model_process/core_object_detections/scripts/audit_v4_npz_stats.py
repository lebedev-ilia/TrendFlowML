#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def _finite_min_max_1d(x: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "finite": int(finite.size),
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
    M: int
    class_names_len: int
    det_count_min: int
    det_count_max: int
    det_count_sum: int
    valid_mask_true_ratio: Optional[float]

    score_valid: Dict[str, Any]
    score_invalid: Dict[str, Any]
    det_count_matches_mask: bool


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    valid_mask = np.asarray(d.get("valid_mask", np.asarray([], dtype=bool)), dtype=bool)
    scores = np.asarray(d.get("scores", np.asarray([], dtype=np.float32)), dtype=np.float32)
    det_count = np.asarray(d.get("det_count", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    class_names = np.asarray(d.get("class_names", np.asarray([], dtype="U")), dtype="U").reshape(-1)

    N = int(valid_mask.shape[0]) if valid_mask.ndim == 2 else 0
    M = int(valid_mask.shape[1]) if valid_mask.ndim == 2 else 0

    det_count_from_mask = np.sum(valid_mask, axis=1).astype(np.int32) if valid_mask.ndim == 2 else np.asarray([], dtype=np.int32)
    det_count_matches_mask = bool(det_count_from_mask.shape == det_count.shape and np.all(det_count_from_mask == det_count))

    valid_ratio = (float(np.mean(valid_mask.astype(np.float32))) if valid_mask.size else None)

    score_valid = scores[valid_mask] if (scores.shape == valid_mask.shape and valid_mask.size) else np.asarray([], dtype=np.float32)
    score_invalid = scores[~valid_mask] if (scores.shape == valid_mask.shape and valid_mask.size) else np.asarray([], dtype=np.float32)

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
        M=M,
        class_names_len=int(class_names.size),
        det_count_min=int(det_count.min()) if det_count.size else 0,
        det_count_max=int(det_count.max()) if det_count.size else 0,
        det_count_sum=int(det_count.sum()) if det_count.size else 0,
        valid_mask_true_ratio=valid_ratio,
        score_valid=_finite_min_max_1d(score_valid),
        score_invalid=_finite_min_max_1d(score_invalid),
        det_count_matches_mask=det_count_matches_mask,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for core_object_detections (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to core_object_detections/detections.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc: Dict[str, Any] = {
        "component": "core_object_detections",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "N_set": sorted({int(r.N) for r in runs}),
            "M_set": sorted({int(r.M) for r in runs}),
            "class_names_len_set": sorted({int(r.class_names_len) for r in runs}),
            "valid_mask_true_ratio_min": min([r.valid_mask_true_ratio for r in runs if r.valid_mask_true_ratio is not None], default=None),
            "valid_mask_true_ratio_max": max([r.valid_mask_true_ratio for r in runs if r.valid_mask_true_ratio is not None], default=None),
            "det_count_sum_total": int(sum(r.det_count_sum for r in runs)),
            "det_count_min_min": int(min([r.det_count_min for r in runs], default=0)),
            "det_count_max_max": int(max([r.det_count_max for r in runs], default=0)),
            "det_count_matches_mask_all": bool(all(r.det_count_matches_mask for r in runs)),
            "score_valid_min_min": min([r.score_valid.get("min") for r in runs if r.score_valid.get("min") is not None], default=None),
            "score_valid_max_max": max([r.score_valid.get("max") for r in runs if r.score_valid.get("max") is not None], default=None),
            "score_invalid_min_min": min([r.score_invalid.get("min") for r in runs if r.score_invalid.get("min") is not None], default=None),
            "score_invalid_max_max": max([r.score_invalid.get("max") for r in runs if r.score_invalid.get("max") is not None], default=None),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

