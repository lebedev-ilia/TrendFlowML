#!/usr/bin/env python3
from __future__ import annotations

"""
V0: Build model-ready dataset index for v1.

Output: v1_dataset_index.(parquet|csv|jsonl fallback)

The index is intentionally "light":
- per row: pointers to per-run artifacts (manifest + key NPZ paths) and snapshot_0/meta fields
- targets/masks are included (log1p deltas)
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from Models.v1.common.utils_bigjson import load_video_records_subset


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _parse_iso8601(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_run_manifests(rs_base: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = Path(rs_base)
    for manifest_path in base.glob("*/*/*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            yield str(manifest_path), payload
        except Exception:
            continue


def _find_component_npz(manifest: Dict[str, Any], component_name: str) -> str:
    comps = manifest.get("components") or []
    if not isinstance(comps, list):
        return ""
    for c in comps:
        if not isinstance(c, dict):
            continue
        if c.get("name") != component_name:
            continue
        arts = c.get("artifacts") or []
        if not isinstance(arts, list):
            continue
        for a in arts:
            if isinstance(a, dict) and isinstance(a.get("path"), str) and str(a["path"]).lower().endswith(".npz"):
                return str(a["path"])
    return ""


def _log1p_delta(curr: Any, base: Any) -> Tuple[float, float]:
    try:
        if curr is None or base is None:
            return float("nan"), 0.0
        c = int(float(curr))
        b = int(float(base))
        d = c - b
        if d < 0:
            d = 0
        import math

        return float(math.log1p(d)), 1.0
    except Exception:
        return float("nan"), 0.0


def compute_targets_from_record(rec: Dict[str, Any]) -> Dict[str, float]:
    s0 = rec.get("snapshot_0") or {}
    s1 = rec.get("snapshot_1") or {}
    s2 = rec.get("snapshot_2") or {}
    s3 = rec.get("snapshot_3") or {}

    v0 = s0.get("viewCount")
    l0 = s0.get("likeCount")

    yv7, m7v = _log1p_delta(s1.get("viewCount"), v0)
    yl7, m7l = _log1p_delta(s1.get("likeCount"), l0)
    yv14, m14v = _log1p_delta(s2.get("viewCount"), v0)
    yl14, m14l = _log1p_delta(s2.get("likeCount"), l0)
    yv21, m21v = _log1p_delta(s3.get("viewCount"), v0)
    yl21, m21l = _log1p_delta(s3.get("likeCount"), l0)

    m7 = 1.0 if (m7v > 0 and m7l > 0) else 0.0
    m14 = 1.0 if (m14v > 0 and m14l > 0) else 0.0
    m21 = 1.0 if (m21v > 0 and m21l > 0) else 0.0

    return {
        "target_views_7d": yv7,
        "target_views_14d": yv14,
        "target_views_21d": yv21,
        "target_likes_7d": yl7,
        "target_likes_14d": yl14,
        "target_likes_21d": yl21,
        "mask_7d": m7,
        "mask_14d": m14,
        "mask_21d": m21,
    }


def write_table(rows: List[Dict[str, Any]], out_path: str) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # prefer parquet if pandas available
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
            fmt = "parquet"
        else:
            df.to_csv(out, index=False)
            fmt = "csv"
        return {"path": str(out), "format": fmt, "rows": int(df.shape[0]), "cols": list(df.columns)}
    except Exception:
        jsonl = out.with_suffix(out.suffix + ".jsonl")
        with jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return {"path": str(jsonl), "format": "jsonl_fallback", "rows": len(rows)}


def main() -> int:
    p = argparse.ArgumentParser(description="Build v1_dataset_index (V0)")
    p.add_argument("--rs-base", type=str, required=True, help="result_store base")
    p.add_argument("--data-json", type=str, required=True, help="data_00.json with snapshot_0..3")
    p.add_argument("--out-index", type=str, required=True)
    p.add_argument("--out-metadata", type=str, required=True)
    args = p.parse_args()

    # First pass: gather runs
    rows: List[Dict[str, Any]] = []
    video_ids: set[str] = set()
    for manifest_path, manifest in _iter_run_manifests(args.rs_base):
        run = manifest.get("run") or {}
        platform_id = _safe_str(run.get("platform_id"))
        video_id = _safe_str(run.get("video_id"))
        run_id = _safe_str(run.get("run_id"))
        if not video_id or not run_id:
            continue
        video_ids.add(video_id)

        frames_dir = _safe_str(run.get("frames_dir"))
        segmenter_metadata = str(Path(frames_dir) / "metadata.json") if frames_dir else ""

        row = {
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": _safe_str(run.get("config_hash")),
            "sampling_policy_version": _safe_str(run.get("sampling_policy_version")),
            "manifest_path": manifest_path,
            "manifest_created_at": _safe_str(run.get("created_at")),
            "frames_dir": frames_dir,
            "segmenter_metadata_path": segmenter_metadata,
            # Key seq sources (Tier-0)
            "core_clip_npz_path": _find_component_npz(manifest, "core_clip"),
            "clap_extractor_npz_path": _find_component_npz(manifest, "clap_extractor"),
            # snapshot/meta placeholders (filled from data_json)
        }
        rows.append(row)

    subset, stats = load_video_records_subset(args.data_json, include_video_ids=video_ids)

    out_rows: List[Dict[str, Any]] = []
    missing_data_json = 0
    for r in rows:
        rec = subset.get(r["video_id"])
        if rec is None:
            missing_data_json += 1
            continue
        meta = rec.get("metadata") or {}
        s0 = rec.get("snapshot_0") or {}

        # snapshot_0 numeric/meta
        r["publishedAt"] = _safe_str(meta.get("publishedAt"))
        r["language"] = _safe_str(meta.get("language"))
        r["channelTitle"] = _safe_str(meta.get("channelTitle"))
        dur = meta.get("duration_seconds", meta.get("duration"))
        r["duration_sec"] = _safe_float(dur)

        # derived age (snapshot time approximated by manifest created_at)
        dt_pub = _parse_iso8601(r["publishedAt"])
        dt_snap = _parse_iso8601(r.get("manifest_created_at"))
        if dt_pub and dt_snap:
            r["video_age_hours_at_snapshot1"] = (dt_snap - dt_pub).total_seconds() / 3600.0
        else:
            r["video_age_hours_at_snapshot1"] = float("nan")

        r["views_0"] = _safe_float(s0.get("viewCount"))
        r["likes_0"] = _safe_float(s0.get("likeCount"))
        r["comments_0"] = _safe_float(s0.get("commentCount"))
        try:
            r["comments_list_len_0"] = float(len(s0.get("comments") or []))
        except Exception:
            r["comments_list_len_0"] = float("nan")
        r["channel_subscribers_0"] = _safe_float(s0.get("subscriberCount"))
        r["channel_total_views_0"] = _safe_float(s0.get("viewCount_channel"))
        r["channel_total_videos_0"] = _safe_float(s0.get("videoCount"))

        r.update(compute_targets_from_record(rec))
        out_rows.append(r)

    wrote = write_table(out_rows, args.out_index)
    parts = [f"{r['platform_id']}|{r['video_id']}|{r['run_id']}" for r in out_rows]
    fingerprint = _sha256_text("\n".join(sorted(parts)))

    meta_out = {
        "created_at": _now_utc(),
        "v1_dataset_fingerprint": fingerprint,
        "rs_base": args.rs_base,
        "data_json": args.data_json,
        "stats": {
            "runs_found": len(rows),
            "rows_written": len(out_rows),
            "missing_data_json_records": missing_data_json,
            "data_json_seen": stats.total_records_seen,
            "data_json_loaded": stats.total_records_yielded,
        },
        "index": wrote,
    }
    Path(args.out_metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metadata).write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote v1 index -> {wrote['path']}")
    print(f"[ok] wrote v1 metadata -> {args.out_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


