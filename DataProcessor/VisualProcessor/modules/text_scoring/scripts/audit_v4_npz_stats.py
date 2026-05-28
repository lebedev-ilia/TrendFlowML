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


def _as_bool0(x: Any) -> Optional[bool]:
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            x = x.item()
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
    except Exception:
        return None
    return None


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


def _nan_count(x: np.ndarray) -> int:
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        return 0
    af = a.astype(np.float64, copy=False)
    return int(np.isnan(af).sum())


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
    F: int
    text_present: Optional[bool]
    text_presence_true: int
    text_presence_ratio: Optional[float]
    text_count_sum: int

    feature_nan: int
    feature_total: int

    ocr_raw_len: int
    ocr_unique_elements_len: int

    stage_timings_ms: Dict[str, Optional[float]]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    text_presence = np.asarray(d.get("text_presence", np.asarray([], dtype=np.bool_)), dtype=np.bool_).reshape(-1)
    text_count = np.asarray(d.get("text_count_per_frame", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    feature_names = np.asarray(d.get("feature_names", np.asarray([], dtype=object)))
    feature_values = np.asarray(d.get("feature_values", np.asarray([], dtype=np.float32)), dtype=np.float32).reshape(-1)

    ocr_raw = d.get("ocr_raw")
    ocr_unique = d.get("ocr_unique_elements")

    N = int(frame_indices.size)
    F = int(feature_names.size)

    tp = _as_bool0(d.get("text_present"))
    tpt = int(text_presence.sum()) if text_presence.size else 0
    tpr = (float(tpt) / float(text_presence.size)) if text_presence.size else None

    tcs = int(text_count.sum()) if text_count.size else 0

    ocr_raw_len = int(np.asarray(ocr_raw, dtype=object).size) if ocr_raw is not None else 0
    ocr_unique_len = int(np.asarray(ocr_unique, dtype=object).size) if ocr_unique is not None else 0

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
        F=F,
        text_present=tp,
        text_presence_true=tpt,
        text_presence_ratio=tpr,
        text_count_sum=tcs,
        feature_nan=_nan_count(feature_values),
        feature_total=int(feature_values.size),
        ocr_raw_len=ocr_raw_len,
        ocr_unique_elements_len=ocr_unique_len,
        stage_timings_ms=st,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for text_scoring (VisualProcessor)")
    ap.add_argument("--npz", action="append", required=True, help="Path to text_scoring.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]
    ratios = [r.text_presence_ratio for r in runs if r.text_presence_ratio is not None]

    doc = {
        "component": "text_scoring",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "F_set": sorted({int(r.F) for r in runs}),
            "text_present_true_count": int(sum(1 for r in runs if r.text_present is True)),
            "text_present_false_count": int(sum(1 for r in runs if r.text_present is False)),
            "text_presence_true_total": int(sum(r.text_presence_true for r in runs)),
            "text_count_sum_total": int(sum(r.text_count_sum for r in runs)),
            "text_presence_ratio_min": float(min(ratios)) if ratios else None,
            "text_presence_ratio_max": float(max(ratios)) if ratios else None,
            "feature_nan_total": int(sum(r.feature_nan for r in runs)),
            "feature_total_total": int(sum(r.feature_total for r in runs)),
            "ocr_raw_len_total": int(sum(r.ocr_raw_len for r in runs)),
            "ocr_unique_elements_len_total": int(sum(r.ocr_unique_elements_len for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

