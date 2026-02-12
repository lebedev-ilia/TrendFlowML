#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils_bigjson import load_video_records_subset


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _log1p_delta(curr: Optional[int], base: Optional[int]) -> Tuple[float, float]:
    """
    Returns (y, present_mask) where y = log1p(max(curr-base, 0)).
    """
    if curr is None or base is None:
        return float("nan"), 0.0
    d = curr - base
    if d < 0:
        d = 0
    return float(math.log1p(d)), 1.0


@dataclass(frozen=True)
class Targets:
    y_views_7d: float
    y_views_14d: float
    y_views_21d: float
    y_likes_7d: float
    y_likes_14d: float
    y_likes_21d: float
    m_7d: float
    m_14d: float
    m_21d: float


def compute_targets_from_record(rec: Dict[str, Any]) -> Targets:
    """
    Contract for data_00.json-style records:
      - snapshot_0: base (t0)
      - snapshot_1: +7d (optional, masked)
      - snapshot_2: +14d (required)
      - snapshot_3: +21d (required)
    """
    s0 = rec.get("snapshot_0") or {}
    s1 = rec.get("snapshot_1") or {}
    s2 = rec.get("snapshot_2") or {}
    s3 = rec.get("snapshot_3") or {}

    v0 = _to_int(s0.get("viewCount"))
    l0 = _to_int(s0.get("likeCount"))

    yv7, m7v = _log1p_delta(_to_int(s1.get("viewCount")), v0)
    yl7, m7l = _log1p_delta(_to_int(s1.get("likeCount")), l0)
    yv14, m14v = _log1p_delta(_to_int(s2.get("viewCount")), v0)
    yl14, m14l = _log1p_delta(_to_int(s2.get("likeCount")), l0)
    yv21, m21v = _log1p_delta(_to_int(s3.get("viewCount")), v0)
    yl21, m21l = _log1p_delta(_to_int(s3.get("likeCount")), l0)

    # horizon present mask is per-horizon (we require both views+likes for that horizon)
    m7 = 1.0 if (m7v > 0.0 and m7l > 0.0) else 0.0
    m14 = 1.0 if (m14v > 0.0 and m14l > 0.0) else 0.0
    m21 = 1.0 if (m21v > 0.0 and m21l > 0.0) else 0.0

    return Targets(
        y_views_7d=yv7,
        y_views_14d=yv14,
        y_views_21d=yv21,
        y_likes_7d=yl7,
        y_likes_14d=yl14,
        y_likes_21d=yl21,
        m_7d=m7,
        m_14d=m14,
        m_21d=m21,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Compute baseline targets from snapshot JSON (data_00.json)")
    p.add_argument("--data-json", type=str, required=True, help="Path to large JSON: video_id -> {snapshot_0..3, metadata,...}")
    p.add_argument("--video-ids-json", type=str, required=True, help="JSON list of video_ids to compute targets for")
    p.add_argument("--out-json", type=str, required=True, help="Output path (JSON dict video_id -> targets)")
    args = p.parse_args()

    with open(args.video_ids_json, "r", encoding="utf-8") as f:
        video_ids = set(json.load(f))

    subset, stats = load_video_records_subset(args.data_json, include_video_ids=video_ids)

    out: Dict[str, Any] = {}
    missing = 0
    for vid in video_ids:
        rec = subset.get(vid)
        if rec is None:
            missing += 1
            continue
        t = compute_targets_from_record(rec)
        out[vid] = {
            "target_views_7d": t.y_views_7d,
            "target_views_14d": t.y_views_14d,
            "target_views_21d": t.y_views_21d,
            "target_likes_7d": t.y_likes_7d,
            "target_likes_14d": t.y_likes_14d,
            "target_likes_21d": t.y_likes_21d,
            "mask_7d": t.m_7d,
            "mask_14d": t.m_14d,
            "mask_21d": t.m_21d,
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "targets_by_video_id": out,
                "stats": {
                    "video_ids_requested": len(video_ids),
                    "records_seen_in_json": stats.total_records_seen,
                    "records_loaded": stats.total_records_yielded,
                    "video_ids_missing_in_json": missing,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ok] wrote targets for {len(out)}/{len(video_ids)} video_ids -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


