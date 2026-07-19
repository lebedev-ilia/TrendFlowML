#!/usr/bin/env python3
"""
build_full_dataset.py  —  DatasetBuilder stage C4 (orchestrator)

One command: result_store --> features (C1) --> targets (C2) --> channel enrichment
(C3) --> final dataset parquet + dataset_metadata.json.

    python build_full_dataset.py \
        --result-store storage/result_store \
        --snapshots <shard_dir_or_glob> \
        --out out/dataset.parquet

Add --synthetic to fabricate labelled targets for a smoke run when real
follow-up snapshots (day 7/14/21) are not collected yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

import build_training_table as C1  # noqa: E402
import add_targets as C2  # noqa: E402
from enrichment import enrich_channel_id  # noqa: E402


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description="Build full baseline dataset (DatasetBuilder C4)")
    ap.add_argument("--result-store", default=str(C1.REPO_ROOT / "storage" / "result_store"))
    ap.add_argument("--feature-spec", default=str(THIS.parent / "feature_spec.yaml"))
    ap.add_argument("--snapshots", nargs="*", default=[])
    ap.add_argument("--platform", default="youtube")
    ap.add_argument("--all-runs", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--index-map", default="1:7d,2:14d,3:21d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # C1 -----------------------------------------------------------------
    df, feat_meta = C1.build(
        args.result_store, args.feature_spec,
        platform=args.platform, all_runs=args.all_runs,
    )
    print(f"[C1] features: {feat_meta['n_rows']} rows x {feat_meta['n_feature_columns']} cols")

    # C2 -----------------------------------------------------------------
    index_map = {int(p.split(':')[0]): p.split(':')[1].strip() for p in args.index_map.split(',')}
    horizons = list(index_map.values())
    records = C2.load_snapshot_records(args.snapshots) if args.snapshots else {}

    for hz in horizons:
        for col in (f"target_views_{hz}", f"target_likes_{hz}", f"mask_{hz}"):
            df[col] = float("nan")

    real_rows = 0
    for i, vid in enumerate(df["video_id"].astype(str).tolist()):
        rec = records.get(vid)
        if rec is None:
            continue
        tr = C2.compute_targets_for_video(rec, index_map)
        for k, v in tr.items():
            df.at[df.index[i], k] = v
        if any(df.at[df.index[i], f"mask_{hz}"] == 1.0 for hz in horizons):
            real_rows += 1
    for hz in horizons:
        df[f"mask_{hz}"] = df[f"mask_{hz}"].fillna(0.0)

    synthetic = False
    if args.synthetic and real_rows == 0:
        df = C2.synthesize_targets(df, seed=args.seed)
        synthetic = True
    print(f"[C2] targets: real={real_rows}, synthetic={synthetic}")

    # C3 -----------------------------------------------------------------
    df, ch_stats = enrich_channel_id(df)
    print(f"[C3] channel_id: {ch_stats}")

    # save ---------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    meta = {
        "created_at": _now_utc(),
        "feature_schema_version": feat_meta["feature_schema_version"],
        "feature_spec_hash": feat_meta["feature_spec_hash"],
        "dataset_fingerprint": C1._fingerprint(df),
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_channels": int(df["channel_id"].nunique()),
        "n_feature_columns": feat_meta["n_feature_columns"],
        "rows_with_real_targets": real_rows,
        "synthetic_targets": synthetic,
        "channel_enrichment": ch_stats,
        "horizons": horizons,
        "included_components": feat_meta["included_components"],
    }
    (out.parent / "dataset_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[C4] wrote dataset -> {out}  ({meta['n_rows']} rows, {meta['n_channels']} channels, "
          f"synthetic_targets={synthetic})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
