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
    # best-effort: meta_json (0-dim str) might exist, but for stats we keep it simple
    return {}


def _as_rows(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    arr = np.asarray(x, dtype=object).reshape(-1)
    out: List[Dict[str, Any]] = []
    for v in arr.tolist():
        if isinstance(v, dict):
            out.append(v)
        else:
            try:
                if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
                    vv = v.item()
                    if isinstance(vv, dict):
                        out.append(vv)
            except Exception:
                continue
    return out


def _finite_stats_int(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None}
    a = np.asarray(vals, dtype=np.int64)
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
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
    status: Optional[str]
    empty_reason: Optional[str]

    N: int
    R: int
    frames_with_ocr: int
    frames_subset_ok: bool
    max_rows_per_frame: int

    engine: Optional[str]
    retain_raw_ocr_text: Optional[bool]
    raw_text_keys_present: bool

    text_len_stats: Dict[str, Any]
    text_sha256_unique: int
    row_key_union: List[str]


def compute_run_stats(npz_path: str) -> RunStats:
    d = _load_npz(npz_path)
    meta = _unbox_meta(d.get("meta"))

    frame_indices = np.asarray(d.get("frame_indices", np.asarray([], dtype=np.int32)), dtype=np.int32).reshape(-1)
    rows = _as_rows(d.get("ocr_raw"))

    N = int(frame_indices.size)
    R = int(len(rows))

    fi_set = set(int(x) for x in frame_indices.tolist())
    per_frame_counts: Dict[int, int] = {}
    row_key_union: set[str] = set()

    text_lens: List[int] = []
    sha_set: set[str] = set()
    raw_text_keys_present = False

    frames_subset_ok = True
    for r in rows:
        row_key_union.update(str(k) for k in r.keys())
        if ("text_raw" in r) or ("text_norm" in r):
            raw_text_keys_present = True
        fr = r.get("frame")
        if fr is None:
            continue
        try:
            fri = int(fr)
        except Exception:
            continue
        if fi_set and (fri not in fi_set):
            frames_subset_ok = False
        per_frame_counts[fri] = int(per_frame_counts.get(fri, 0) + 1)

        tl = r.get("text_len")
        if tl is not None:
            try:
                text_lens.append(int(tl))
            except Exception:
                pass
        sh = r.get("text_sha256")
        if sh is not None:
            try:
                sha_set.add(str(sh))
            except Exception:
                pass

    frames_with_ocr = int(len(per_frame_counts))
    max_rows_per_frame = int(max(per_frame_counts.values(), default=0))

    retain = meta.get("retain_raw_ocr_text")
    retain_bool: Optional[bool]
    if retain is None:
        retain_bool = None
    else:
        retain_bool = bool(retain)

    return RunStats(
        npz_path=npz_path,
        platform_id=str(meta.get("platform_id")) if meta.get("platform_id") is not None else None,
        video_id=str(meta.get("video_id")) if meta.get("video_id") is not None else None,
        run_id=str(meta.get("run_id")) if meta.get("run_id") is not None else None,
        config_hash=str(meta.get("config_hash")) if meta.get("config_hash") is not None else None,
        sampling_policy_version=str(meta.get("sampling_policy_version")) if meta.get("sampling_policy_version") is not None else None,
        schema_version=str(meta.get("schema_version")) if meta.get("schema_version") is not None else None,
        producer_version=str(meta.get("producer_version")) if meta.get("producer_version") is not None else None,
        status=str(meta.get("status")) if meta.get("status") is not None else None,
        empty_reason=str(meta.get("empty_reason")) if meta.get("empty_reason") is not None else None,
        N=N,
        R=R,
        frames_with_ocr=frames_with_ocr,
        frames_subset_ok=bool(frames_subset_ok),
        max_rows_per_frame=max_rows_per_frame,
        engine=str(meta.get("engine")) if meta.get("engine") is not None else None,
        retain_raw_ocr_text=retain_bool,
        raw_text_keys_present=bool(raw_text_keys_present),
        text_len_stats=_finite_stats_int(text_lens),
        text_sha256_unique=int(len(sha_set)),
        row_key_union=sorted(row_key_union),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v4 stats for ocr_extractor (VisualProcessor core)")
    ap.add_argument("--npz", action="append", required=True, help="Path to ocr_extractor/ocr.npz (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = os.fspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    runs = [compute_run_stats(os.fspath(p)) for p in (args.npz or [])]

    doc: Dict[str, Any] = {
        "component": "ocr_extractor",
        "level": "L2 (A+B, 5 runs)",
        "n_runs": int(len(runs)),
        "aggregates": {
            "N_total": int(sum(r.N for r in runs)),
            "N_set": sorted({int(r.N) for r in runs}),
            "R_total": int(sum(r.R for r in runs)),
            "R_set": sorted({int(r.R) for r in runs}),
            "frames_with_ocr_total": int(sum(r.frames_with_ocr for r in runs)),
            "frames_subset_ok_all": bool(all(r.frames_subset_ok for r in runs)),
            "max_rows_per_frame_max": int(max([r.max_rows_per_frame for r in runs], default=0)),
            "engine_set": sorted({str(r.engine) for r in runs if r.engine is not None}),
            "retain_raw_ocr_text_set": sorted({r.retain_raw_ocr_text for r in runs}),
            "raw_text_keys_present_any": bool(any(r.raw_text_keys_present for r in runs)),
            "text_sha256_unique_total": int(sum(r.text_sha256_unique for r in runs)),
        },
        "per_run": [asdict(r) for r in runs],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

