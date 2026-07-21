#!/usr/bin/env python3
"""
build_corpus_content_dataset.py — join Agent A's 300-video CONTENT features (NPZ on
the Network Volume, read via S3 into storage/corpus_npz) with REAL targets +
snapshot_0 inputs + video metadata from pre_final_data.

This is the "content_features_plus_real_targets" v1 step: the first training table
that carries BOTH real video content features (7 VP components) AND real
multi-horizon targets on the SAME videos.

Feature families (all leakage-safe — content + snapshot_0/metadata at prediction time):
  - content : build_from_rs() over corpus_npz rs/<component>/*.npz (feature_spec v0-real filtering)
  - snapshot_0 : views_0, likes_0, comments_0, channel_subscribers_0, channel_total_*
  - metadata : duration_seconds, title/description/tag stats, made_for_kids, chapters, ...
  - temporal : video_age_hours_at_snapshot1
  - targets : y=log1p(max(delta,0)) views/likes @7/14/21d + per-horizon masks

Usage:
  python build_corpus_content_dataset.py \
      --rs-root storage/corpus_npz \
      --prefinal storage/pre_final_data/data_00.json storage/pre_final_data/data_01.json storage/pre_final_data/data_02.json \
      --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
      --out Models/baseline/artifacts/<tag>/dataset_corpus_content.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
REPO_ROOT = THIS.parents[2]

import build_training_table as C1  # noqa: E402
import add_targets as C2  # noqa: E402
from build_real_from_prefinal import metadata_features, _ratio  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--rs-root", default=str(REPO_ROOT / "storage/corpus_npz"))
    ap.add_argument("--prefinal", nargs="+", required=True)
    ap.add_argument("--feature-spec", default=str(THIS.parent / "feature_spec.yaml"))
    ap.add_argument("--index-map", default="1:7d,2:14d,3:21d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index_map = {int(p.split(":")[0]): p.split(":")[1].strip() for p in args.index_map.split(",")}
    horizons = list(index_map.values())

    # ---- C1: content features from rs/ ----
    df, feat_meta = C1.build_from_rs(args.rs_root, args.feature_spec)
    print(f"[content] {feat_meta['n_rows']} videos x {feat_meta['n_feature_columns']} content-feat cols")

    # ---- targets + snapshot_0 + metadata from pre_final ----
    wanted = set(df["video_id"].astype(str))
    recs = C2.load_prefinal_records(args.prefinal, wanted_ids=wanted)
    print(f"[targets] pre_final records matched: {len(recs)}/{len(wanted)}")

    add_cols = {}
    real_rows = 0
    n_dec = n_nn = 0
    for i, vid in enumerate(df["video_id"].astype(str)):
        rec = recs.get(vid)
        if rec is None:
            continue
        tr = C2.compute_targets_for_video(rec, index_map)   # snapshot_0 + targets + masks + publishedAt/channelTitle
        mf = metadata_features(rec.get("metadata") or {})
        row = dict(tr)
        row.update(mf)
        row["likes_per_view_0"] = _ratio(tr.get("likes_0", float("nan")), tr.get("views_0", float("nan")))
        row["comments_per_view_0"] = _ratio(tr.get("comments_0", float("nan")), tr.get("views_0", float("nan")))
        row["channel_id"] = (rec.get("metadata") or {}).get("channelTitle") or tr.get("channelTitle")
        add_cols[i] = row
        ec = C2.target_edge_cases(rec, index_map)
        n_dec += int(ec["decreasing_views"]); n_nn += int(ec["nonnumeric_views"])
        if any(row.get(f"mask_{hz}") == 1.0 for hz in horizons):
            real_rows += 1

    # merge add_cols into df
    all_keys = sorted({k for r in add_cols.values() for k in r})
    for k in all_keys:
        df[k] = [add_cols.get(i, {}).get(k, float("nan")) for i in range(len(df))]
    for hz in horizons:
        df[f"mask_{hz}"] = df[f"mask_{hz}"].fillna(0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    content_cols = [c for c in df.columns if "__" in c]
    meta = {
        "created_at": _now_utc(),
        "feature_schema_version": "v0-content+prefinal",
        "rs_root": str(args.rs_root),
        "prefinal_shards": list(args.prefinal),
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_channels": int(df["channel_id"].nunique()) if "channel_id" in df else 0,
        "n_content_feature_columns": len(content_cols),
        "rows_with_real_targets": real_rows,
        "synthetic_targets": False,
        "horizons": horizons,
        "mask_coverage": {hz: int(df[f"mask_{hz}"].sum()) for hz in horizons},
        "edge_cases": {"nonnumeric_viewCount": n_nn, "decreasing_views_between_snapshots": n_dec},
        "included_components": feat_meta["included_components"],
    }
    (out.parent / "dataset_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {out}  rows={len(df)} content_feat={len(content_cols)} real_targets={real_rows}")
    print(f"     mask coverage: {meta['mask_coverage']}  channels={meta['n_channels']}")
    print(f"     edge: decreasing_views={n_dec} nonnumeric={n_nn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
