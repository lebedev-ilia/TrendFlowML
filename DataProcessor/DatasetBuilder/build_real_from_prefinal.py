#!/usr/bin/env python3
"""
build_real_from_prefinal.py  —  DatasetBuilder: REAL labelled table from pre_final_data

The `Ilialebedev/pre_final_data/main_ready` dataset already carries real multi-horizon
targets (snapshot_0..3 at ~0/7/14/21 days) AND the snapshot_0 input fields + video
metadata for tens of thousands of videos. Our local `result_store` only has NPZ
content features for a handful of ad-hoc test videos, so to get a MEANINGFUL first
real baseline we build the training table directly from pre_final_data using the
leakage-safe feature families that are available at prediction time (snapshot_0):

  FEATURES (all known at/ before snapshot_0 — no future leakage):
    snapshot_0 numeric : views_0, likes_0, comments_0, channel_subscribers_0,
                         channel_total_views_0, channel_total_videos_0
    engagement ratios  : likes_0/views_0, comments_0/views_0            (t0 only)
    metadata           : duration_seconds, title_len, title_word_count,
                         title_has_question, title_upper_ratio, description_len,
                         tag_count, made_for_kids, has_chapters, chapter_count,
                         n_thumbnails, n_subtitle_langs
    temporal           : video_age_hours_at_snapshot1
  GROUPING/SPLIT       : channelTitle (channel-group), publishedAt (time)
  TARGETS              : y_h = log1p(max(viewCount_h - viewCount_0, 0)) for views &
                         likes, horizons 7/14/21d, with per-horizon masks.

We DELIBERATELY do NOT put any snapshot_1/2/3 raw value into the feature matrix — an
explicit leakage guard asserts this before writing.

Usage:
    python build_real_from_prefinal.py \
        --prefinal storage/pre_final_data/data_00.json \
        --out Models/baseline/artifacts/<tag>/dataset_real_prefinal.parquet \
        [--max-videos N] [--require-full4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

import add_targets as C2  # noqa: E402

REPO_ROOT = THIS.parents[2]

# feature columns that must be present but are NOT model inputs
META_COLS = {"video_id", "channelTitle", "channel_id", "publishedAt", "language", "country"}
TARGET_MASK_COLS = {
    "target_views_7d", "target_views_14d", "target_views_21d",
    "target_likes_7d", "target_likes_14d", "target_likes_21d",
    "mask_7d", "mask_14d", "mask_21d",
}


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _f(x) -> float:
    return C2._to_float(x)


def _ratio(a: float, b: float) -> float:
    if math.isfinite(a) and math.isfinite(b) and b > 0:
        return a / b
    return float("nan")


def metadata_features(meta: dict) -> dict:
    title = meta.get("title") or ""
    desc = meta.get("description") or ""
    tags = meta.get("tags") or []
    chapters = meta.get("chapters") or []
    thumbs = meta.get("thumbnails") or meta.get("thumbnails_ytdlp") or []
    subs = meta.get("subtitles") or {}
    n_upper = sum(1 for c in title if c.isupper())
    return {
        "duration_seconds": _f(meta.get("duration_seconds") or meta.get("duration")),
        "title_len": float(len(title)),
        "title_word_count": float(len(title.split())),
        "title_has_question": 1.0 if "?" in title else 0.0,
        "title_upper_ratio": (n_upper / len(title)) if title else 0.0,
        "description_len": float(len(desc)),
        "tag_count": float(len(tags)) if isinstance(tags, list) else 0.0,
        "made_for_kids": 1.0 if meta.get("madeForKids") else 0.0,
        "has_chapters": 1.0 if chapters else 0.0,
        "chapter_count": float(len(chapters)) if isinstance(chapters, list) else 0.0,
        "n_thumbnails": float(len(thumbs)) if isinstance(thumbs, (list, dict)) else 0.0,
        "n_subtitle_langs": float(len(subs)) if isinstance(subs, dict) else 0.0,
    }


def build_row(vid: str, rec: dict, index_map: dict) -> dict | None:
    snaps = C2._collect_snapshots(rec)
    if 0 not in snaps and not rec.get("snapshot_0"):
        return None
    # at least one follow-up target
    if not any(i in snaps for i in index_map):
        return None

    tr = C2.compute_targets_for_video(rec, index_map)  # snapshot_0 fields + targets + masks
    meta = rec.get("metadata") or {}

    row: dict = {"video_id": vid, "channel_id": meta.get("channelTitle") or rec.get("channelTitle")}
    row.update(tr)  # views_0/likes_0/... + publishedAt/channelTitle + targets/masks + video_age
    row.update(metadata_features(meta))
    # engagement ratios at t0 (leakage-safe)
    row["likes_per_view_0"] = _ratio(row.get("likes_0", float("nan")), row.get("views_0", float("nan")))
    row["comments_per_view_0"] = _ratio(row.get("comments_0", float("nan")), row.get("views_0", float("nan")))
    return row


def leakage_guard(df) -> list:
    """Assert no snapshot_1/2/3 raw metric leaked into the feature matrix."""
    forbidden_tokens = ("_1", "_2", "_3", "snapshot1", "snapshot2", "snapshot3")
    feats = [c for c in df.columns if c not in META_COLS and c not in TARGET_MASK_COLS]
    hits = []
    for c in feats:
        low = c.lower()
        # feature columns are snapshot_0 / metadata / temporal only; anything that
        # references a follow-up index by name is suspicious
        if any(low.endswith(t) or f"snapshot_{t[-1]}" in low for t in ("_1", "_2", "_3")):
            if "video_age_hours_at_snapshot1" == c:  # temporal ref time, not a metric value -> allowed
                continue
            hits.append(c)
    return hits


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description="Build REAL labelled dataset from pre_final_data")
    ap.add_argument("--prefinal", nargs="+", required=True)
    ap.add_argument("--index-map", default="1:7d,2:14d,3:21d")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all")
    ap.add_argument("--require-full4", action="store_true", help="only videos with all 4 snapshots")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index_map = {int(p.split(":")[0]): p.split(":")[1].strip() for p in args.index_map.split(",")}
    horizons = list(index_map.values())

    rows = []
    n_seen = n_kept = n_nonnum = n_decr = n_full4 = 0
    for vid, rec in C2.stream_prefinal_records(args.prefinal):
        n_seen += 1
        snaps = C2._collect_snapshots(rec)
        if args.require_full4 and not all(i in snaps for i in (0, 1, 2, 3)):
            continue
        row = build_row(vid, rec, index_map)
        if row is None:
            continue
        ec = C2.target_edge_cases(rec, index_map)
        n_nonnum += int(ec["nonnumeric_views"])
        n_decr += int(ec["decreasing_views"])
        if all(i in snaps for i in (0, 1, 2, 3)):
            n_full4 += 1
        rows.append(row)
        n_kept += 1
        if args.max_videos and n_kept >= args.max_videos:
            break
        if n_seen % 5000 == 0:
            print(f"  ... seen={n_seen} kept={n_kept}", flush=True)

    df = pd.DataFrame(rows)
    for hz in horizons:
        df[f"mask_{hz}"] = df[f"mask_{hz}"].fillna(0.0)

    leaks = leakage_guard(df)
    if leaks:
        raise SystemExit(f"[LEAKAGE] forbidden follow-up features in matrix: {leaks}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    feat_cols = [c for c in df.columns if c not in META_COLS and c not in TARGET_MASK_COLS]
    real_rows = int((df[[f"mask_{hz}" for hz in horizons]].max(axis=1) > 0).sum())
    meta = {
        "created_at": _now_utc(),
        "feature_schema_version": "v0-prefinal-meta",
        "source": "Ilialebedev/pre_final_data/main_ready",
        "shards": [str(p) for p in args.prefinal],
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_channels": int(df["channel_id"].nunique()),
        "n_feature_columns": len(feat_cols),
        "feature_columns": feat_cols,
        "rows_with_real_targets": real_rows,
        "rows_full4_snapshots": n_full4,
        "synthetic_targets": False,
        "horizons": horizons,
        "mask_coverage": {hz: int(df[f"mask_{hz}"].sum()) for hz in horizons},
        "edge_cases": {
            "nonnumeric_viewCount": n_nonnum,
            "decreasing_views_between_snapshots": n_decr,
        },
    }
    (out.parent / "dataset_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] REAL dataset -> {out}")
    print(f"     rows={len(df)} videos={meta['n_videos']} channels={meta['n_channels']} "
          f"feat={len(feat_cols)} real_target_rows={real_rows} full4={n_full4}")
    print(f"     mask coverage: {meta['mask_coverage']}")
    print(f"     edge: nonnumeric_views={n_nonnum} decreasing_views={n_decr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
