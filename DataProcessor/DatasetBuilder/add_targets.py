#!/usr/bin/env python3
"""
add_targets.py  —  DatasetBuilder stage C2

Attach multi-horizon targets + masks + snapshot_0 fields to the feature table.

Target contract (TARGETS_SPLITS_METRICS.md):
    delta_h = x_h - x_0                       (x in {viewCount, likeCount})
    y_h     = log(1 + max(delta_h, 0))        (log1p of the positive delta)
    horizons: 7d (masked/optional), 14d, 21d
    snapshot_0 fields are FEATURES (comments_0 too), never targets.

Snapshot sources (in priority order):
  1. `--prefinal` : the READY labelled dataset `Ilialebedev/pre_final_data`
     (`main_ready/data_00.json … data_21.json`). Top level is a dict
     {video_id: {time_interval, metadata, snapshot_0..3, _enriched, ...}}. Metric
     fields live directly in each `snapshot_N` as STRINGS (viewCount, likeCount,
     commentCount, subscriberCount, videoCount, viewCount_channel, comments).
     ~81.5% of videos carry all 4 snapshots; intervals measured at ~7d steps, so
     index 1->7d, 2->14d, 3->21d. This is the PRIMARY real-target source.
  2. `--snapshots` : the live Fetcher HF collection (`Ilialebedev/*` monthly
     shards). Same record shapes accepted; used as fallback / newer data.

A video's snapshot record may arrive in several shapes; we accept all:
    (a) {"snapshots": [{"snapshot_index":0,"viewCount":..}, {"snapshot_index":1,..}, ...]}
    (b) {"snapshot_0": {...}, "snapshot_1": {...}, "snapshot_2": {...}, ...}  <- pre_final
    (c) {"snapshot_0": {...}}   -> only snapshot_0 filled, all horizons masked=0
Index->horizon mapping follows the collection schedule [0,7,14,21,28]:
    index 1 -> 7d, index 2 -> 14d, index 3 -> 21d   (override via --index-map).

Data-integrity handling (measured on pre_final_data):
  * metric fields are strings -> parsed via _to_float (try/except -> NaN);
  * ~0.7% records have a non-numeric viewCount -> counted, target masked;
  * ~6.6% records show views DECREASING between snapshots (YouTube API noise, not a
    builder bug) -> counted + logged as edge-case, NOT dropped; log1p(max(delta,0))
    already floors negative deltas at 0.

--synthetic still fabricates clearly-labelled targets for a pure smoke run when NO
real source is available.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HORIZON_INDEX_DEFAULT = {1: "7d", 2: "14d", 3: "21d"}


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _parse_time_get(s: Any) -> Optional[datetime]:
    """Snapshot time_get format: '2026_06_28_04_09' (Y_m_d_H_M), UTC."""
    if not isinstance(s, str):
        return None
    for fmt in ("%Y_%m_%d_%H_%M", "%Y_%m_%d_%H_%M_%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _parse_iso(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# snapshot source loading
# ---------------------------------------------------------------------------
def load_snapshot_records(paths: List[str]) -> Dict[str, dict]:
    """Load {video_id: record} from one or more shard JSON files (dict-keyed or list)."""
    out: Dict[str, dict] = {}
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        else:
            files.extend(glob.glob(p))
    for f in files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for vid, rec in data.items():
                if isinstance(rec, dict):
                    out[str(vid)] = rec
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict) and rec.get("video_id"):
                    out[str(rec["video_id"])] = rec
    return out


def _iter_prefinal_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "data_*.json"), recursive=True)))
        else:
            files.extend(sorted(glob.glob(p)))
    return files


def stream_prefinal_records(paths: List[str], wanted_ids: Optional[set] = None):
    """
    Stream {video_id: record} from pre_final_data shards WITHOUT loading a whole
    390MB shard into memory (ijson.kvitems over the top-level dict). If `wanted_ids`
    is given, only those video_ids are yielded (memory-safe join for a small feature
    table). Yields (video_id, record).
    """
    import ijson  # local import: only needed for the pre_final path

    for f in _iter_prefinal_files(paths):
        with open(f, "rb") as fh:
            for vid, rec in ijson.kvitems(fh, ""):
                if not isinstance(rec, dict):
                    continue
                if wanted_ids is not None and str(vid) not in wanted_ids:
                    continue
                yield str(vid), rec


def load_prefinal_records(paths: List[str], wanted_ids: Optional[set] = None) -> Dict[str, dict]:
    """Materialize {video_id: record} from pre_final_data shards (optionally filtered)."""
    out: Dict[str, dict] = {}
    for vid, rec in stream_prefinal_records(paths, wanted_ids=wanted_ids):
        out[vid] = rec
    return out


def target_edge_cases(rec: dict, index_map: Dict[int, str]) -> Dict[str, bool]:
    """Flag data-integrity edge-cases per the owner's audit (log, don't drop)."""
    snaps = _collect_snapshots(rec)
    s0 = snaps.get(0) or rec.get("snapshot_0") or {}
    raw_v0 = s0.get("viewCount")
    nonnumeric_views = not math.isfinite(_to_float(raw_v0))
    v0 = _to_float(raw_v0)
    decreasing = False
    prev = v0
    for idx in sorted(index_map):
        s = snaps.get(idx)
        if not s:
            continue
        vh = _to_float(s.get("viewCount"))
        if math.isfinite(vh) and math.isfinite(prev) and vh < prev - 1e-9:
            decreasing = True
        if math.isfinite(vh):
            prev = vh
    return {"nonnumeric_views": bool(nonnumeric_views), "decreasing_views": bool(decreasing)}


def _collect_snapshots(rec: dict) -> Dict[int, dict]:
    """Normalize any record shape into {index: snapshot_dict}."""
    snaps: Dict[int, dict] = {}
    if isinstance(rec.get("snapshots"), list):
        for s in rec["snapshots"]:
            if isinstance(s, dict) and s.get("snapshot_index") is not None:
                snaps[int(s["snapshot_index"])] = s
    for k, v in rec.items():
        if k.startswith("snapshot_") and isinstance(v, dict):
            try:
                idx = int(k.split("_", 1)[1])
                snaps.setdefault(idx, v)
            except Exception:
                pass
    return snaps


# ---------------------------------------------------------------------------
# target computation
# ---------------------------------------------------------------------------
def _log1p_delta(x_h: float, x_0: float) -> float:
    if not (math.isfinite(x_h) and math.isfinite(x_0)):
        return float("nan")
    return math.log1p(max(x_h - x_0, 0.0))


def compute_targets_for_video(
    rec: dict, index_map: Dict[int, str]
) -> Dict[str, Any]:
    snaps = _collect_snapshots(rec)
    s0 = snaps.get(0) or rec.get("snapshot_0") or {}
    meta = rec.get("metadata") or {}

    views_0 = _to_float(s0.get("viewCount"))
    likes_0 = _to_float(s0.get("likeCount"))
    comments_0 = _to_float(s0.get("commentCount"))

    row: Dict[str, Any] = {
        "views_0": views_0,
        "likes_0": likes_0,
        "comments_0": comments_0,
        "channel_subscribers_0": _to_float(s0.get("subscriberCount")),
        "channel_total_views_0": _to_float(s0.get("viewCount_channel")),
        "channel_total_videos_0": _to_float(s0.get("videoCount")),
        "publishedAt": meta.get("publishedAt") or rec.get("publishedAt"),
        "channelTitle": meta.get("channelTitle") or rec.get("channelTitle"),
    }

    # video_age_hours_at_snapshot1: from earliest follow-up (or snapshot_0) vs publishedAt
    pub = _parse_iso(row["publishedAt"])
    ref_snap = snaps.get(1) or s0
    ref_t = _parse_time_get(ref_snap.get("time_get")) or _parse_iso(ref_snap.get("collected_at"))
    if pub and ref_t:
        row["video_age_hours_at_snapshot1"] = max((ref_t - pub).total_seconds() / 3600.0, 0.0)
    else:
        row["video_age_hours_at_snapshot1"] = float("nan")

    for idx, hz in index_map.items():
        s = snaps.get(idx)
        if s is None:
            row[f"target_views_{hz}"] = float("nan")
            row[f"target_likes_{hz}"] = float("nan")
            row[f"mask_{hz}"] = 0.0
            continue
        vh = _to_float(s.get("viewCount"))
        lh = _to_float(s.get("likeCount"))
        yv = _log1p_delta(vh, views_0)
        yl = _log1p_delta(lh, likes_0)
        row[f"target_views_{hz}"] = yv
        row[f"target_likes_{hz}"] = yl
        row[f"mask_{hz}"] = 1.0 if (math.isfinite(yv) and math.isfinite(yl)) else 0.0
    return row


# ---------------------------------------------------------------------------
# synthetic targets (smoke only, clearly labelled)
# ---------------------------------------------------------------------------
def synthesize_targets(df: "pd.DataFrame", *, seed: int = 1337) -> "pd.DataFrame":  # type: ignore # noqa: F821
    """
    Fabricate plausible, leakage-free targets so the TRAIN pipeline can be
    smoke-tested end-to-end. Targets = monotone f(a few real features) + noise,
    which lets Spearman / feature-importance / leakage-audit exercise real code
    paths and return non-NaN numbers. NOT real ground truth — dataset metadata
    marks synthetic_targets=true.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = len(df)

    # pick up to 3 real, mostly-populated numeric feature columns as drivers
    feat_cols = [c for c in df.columns if "__" in c]
    Xnum = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    good = [c for c in feat_cols if Xnum[c].notna().mean() > 0.6 and Xnum[c].nunique(dropna=True) > 2]
    drivers = good[:3] if good else []

    base = np.zeros(n)
    for c in drivers:
        v = Xnum[c].to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
        std = v.std() or 1.0
        base += (v - v.mean()) / std

    # snapshot_0 scale (log views_0) also drives growth if present
    if "views_0" in df.columns:
        v0 = np.nan_to_num(df["views_0"].to_numpy(dtype=float), nan=0.0)
        base += 0.5 * np.log1p(np.clip(v0, 0, None)) / (np.log1p(np.clip(v0, 0, None)).std() or 1.0)

    for hz, scale in [("7d", 4.0), ("14d", 6.0), ("21d", 7.0)]:
        noise = rng.normal(0, 0.5, n)
        yv = np.clip(scale + 1.2 * base + noise, 0, None)
        yl = np.clip(scale - 1.5 + 1.0 * base + rng.normal(0, 0.5, n), 0, None)
        df[f"target_views_{hz}"] = yv
        df[f"target_likes_{hz}"] = yl
        df[f"mask_{hz}"] = 1.0
    if "video_age_hours_at_snapshot1" not in df.columns or df["video_age_hours_at_snapshot1"].isna().all():
        df["video_age_hours_at_snapshot1"] = rng.uniform(1, 500, n)
    return df


# ---------------------------------------------------------------------------
def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description="Attach targets/masks/snapshot_0 to feature table (DatasetBuilder C2)")
    ap.add_argument("--features", required=True, help="Input feature table (.parquet/.csv)")
    ap.add_argument("--prefinal", nargs="*", default=[], help="pre_final_data shard files/dirs/globs (PRIMARY real-target source)")
    ap.add_argument("--snapshots", nargs="*", default=[], help="Fetcher HF snapshot shard JSON files/dirs/globs (fallback)")
    ap.add_argument("--out", required=True, help="Output path (.parquet/.csv)")
    ap.add_argument("--synthetic", action="store_true", help="Fabricate labelled targets (smoke only)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--index-map", default="1:7d,2:14d,3:21d", help="snapshot_index->horizon map")
    args = ap.parse_args()

    index_map = {}
    for pair in args.index_map.split(","):
        i, hz = pair.split(":")
        index_map[int(i)] = hz.strip()

    fp = Path(args.features)
    df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
    n0 = len(df)

    wanted = set(df["video_id"].astype(str).tolist())
    if args.prefinal:
        records = load_prefinal_records(args.prefinal, wanted_ids=wanted)
        source = "pre_final"
    elif args.snapshots:
        records = load_snapshot_records(args.snapshots)
        source = "hf_shards"
    else:
        records = {}
        source = "none"
    matched = 0
    n_nonnumeric_views = 0
    n_decreasing_views = 0
    horizons = list(index_map.values())

    # ensure target/mask/snapshot columns exist (NaN default => native missing)
    for hz in horizons:
        for col in (f"target_views_{hz}", f"target_likes_{hz}", f"mask_{hz}"):
            if col not in df.columns:
                df[col] = float("nan")

    real_target_rows = 0
    for i, vid in enumerate(df["video_id"].astype(str).tolist()):
        rec = records.get(vid)
        if rec is None:
            continue
        matched += 1
        ec = target_edge_cases(rec, index_map)
        n_nonnumeric_views += int(ec["nonnumeric_views"])
        n_decreasing_views += int(ec["decreasing_views"])
        tr = compute_targets_for_video(rec, index_map)
        for k, v in tr.items():
            df.at[df.index[i], k] = v
        if any(df.at[df.index[i], f"mask_{hz}"] == 1.0 for hz in horizons):
            real_target_rows += 1

    # masks NaN -> 0
    for hz in horizons:
        df[f"mask_{hz}"] = df[f"mask_{hz}"].fillna(0.0)

    synthetic = False
    if args.synthetic and real_target_rows == 0:
        df = synthesize_targets(df, seed=args.seed)
        synthetic = True

    labelled_rows = int(sum((df[[f"mask_{hz}" for hz in horizons]].max(axis=1) > 0)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    meta = {
        "created_at": _now_utc(),
        "n_rows": int(len(df)),
        "source": source,
        "snapshot_files_matched_videos": matched,
        "rows_with_real_targets": real_target_rows,
        "rows_with_any_target": labelled_rows,
        "synthetic_targets": synthetic,
        "edge_cases": {
            "nonnumeric_viewCount": n_nonnumeric_views,
            "decreasing_views_between_snapshots": n_decreasing_views,
        },
        "index_map": index_map,
        "horizons": horizons,
    }
    (out.parent / (out.stem + ".targets_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tag = "SYNTHETIC" if synthetic else "real"
    print(f"[ok] targets attached ({tag}, source={source}): {labelled_rows}/{n0} rows labelled, "
          f"real={real_target_rows}, matched_videos={matched} -> {out}")
    if matched:
        print(f"[edge] non-numeric viewCount={n_nonnumeric_views}, "
              f"decreasing-views records={n_decreasing_views} (logged, not dropped)")
    if real_target_rows == 0 and not synthetic:
        print("[warn] no real follow-up snapshots (day 7/14/21) found -> all targets masked. "
              "This is expected until the HF collection matures. Use --synthetic for a smoke run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
