#!/usr/bin/env python3
"""
make_smoke_dataset.py  —  smoke fixture generator (NOT a real dataset)

Purpose: exercise the TRAIN pipeline (train_baseline.py) end-to-end on a dataset
that has enough videos/channels/time-spread for the hybrid split + metrics to be
meaningful (non-NaN Spearman on val AND test), while reusing the REAL v0-real
feature schema and real feature-value distributions.

Why this exists: real features currently cover only ~6 videos, and real
follow-up snapshots (day 7/14/21) are not collected yet, so a genuine trained
baseline is impossible today. This fixture proves the code path works and the
leakage audit + feature importance produce sane output; it is CLEARLY labelled
synthetic (dataset_metadata.synthetic=true) and MUST NOT be read as real quality.

Construction:
  - bootstrap N synthetic videos from the real feature rows (sample + gaussian
    jitter on numeric features),
  - assign K channels and spread publishedAt over ~180 days (drives time-split),
  - synthetic snapshot_0 (views_0/likes_0/subscribers_0 log-normal),
  - targets = monotone f(a few real features + log views_0) + noise  (leakage-free:
    only snapshot_0 / content features feed the target).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Real feature table (.parquet) to bootstrap schema from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-videos", type=int, default=240)
    ap.add_argument("--n-channels", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    real = pd.read_parquet(args.features)

    feat_cols = [c for c in real.columns if "__" in c]
    Xr = real[feat_cols].apply(pd.to_numeric, errors="coerce")
    # keep features that are populated enough to bootstrap
    keep = [c for c in feat_cols if Xr[c].notna().mean() > 0.3]
    Xr = Xr[keep]
    col_med = Xr.median()
    col_std = Xr.std().fillna(0.0)

    n = args.n_videos
    base_idx = rng.integers(0, len(real), n)
    rows = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        src = Xr.iloc[base_idx[i]]
        row = {}
        for c in keep:
            v = src[c]
            if not np.isfinite(v):
                v = col_med[c]
            jitter = rng.normal(0, 0.15 * (col_std[c] if np.isfinite(col_std[c]) else 0.0))
            row[c] = float(v + jitter) if np.isfinite(v) else float("nan")
        ch = int(rng.integers(0, args.n_channels))
        row["video_id"] = f"smk_{i:04d}"
        row["channel_id"] = f"smkch_{ch:02d}"
        row["channelTitle"] = f"SmokeChannel {ch:02d}"
        row["platform_id"] = "youtube"
        # publishedAt spread over ~180 days, correlated with channel to make the
        # channel-group split and time-split interact realistically
        day = int(rng.integers(0, 180)) + ch  # channels start at staggered times
        row["publishedAt"] = (t0 + timedelta(days=day, hours=int(rng.integers(0, 24)))).isoformat().replace("+00:00", "Z")
        # snapshot_0
        v0 = float(np.exp(rng.normal(8, 2)))  # log-normal views
        row["views_0"] = v0
        row["likes_0"] = v0 * rng.uniform(0.01, 0.08)
        row["comments_0"] = v0 * rng.uniform(0.0005, 0.01)
        row["channel_subscribers_0"] = float(np.exp(rng.normal(10, 1.5)))
        row["channel_total_views_0"] = row["channel_subscribers_0"] * rng.uniform(50, 500)
        row["channel_total_videos_0"] = float(rng.integers(10, 2000))
        row["video_age_hours_at_snapshot1"] = float(rng.uniform(1, 500))
        rows.append(row)

    df = pd.DataFrame(rows)

    # leakage-free synthetic targets: driven by a few content features + log(views_0)
    drivers = keep[:3]
    signal = np.zeros(n)
    for c in drivers:
        v = df[c].to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
        s = v.std() or 1.0
        signal += (v - v.mean()) / s
    lv0 = np.log1p(df["views_0"].to_numpy(dtype=float))
    signal += 0.8 * (lv0 - lv0.mean()) / (lv0.std() or 1.0)

    for hz, scale in [("7d", 4.0), ("14d", 6.0), ("21d", 7.0)]:
        df[f"target_views_{hz}"] = np.clip(scale + 1.2 * signal + rng.normal(0, 0.5, n), 0, None)
        df[f"target_likes_{hz}"] = np.clip(scale - 1.5 + 1.0 * signal + rng.normal(0, 0.5, n), 0, None)
        df[f"mask_{hz}"] = 1.0
    # 7d partially masked to exercise mask handling
    mask7 = rng.random(n) > 0.3
    df.loc[~mask7, "mask_7d"] = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    meta = {
        "synthetic": True,
        "note": "SMOKE FIXTURE — not real ground truth. For pipeline integration test only.",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_videos": int(df["video_id"].nunique()),
        "n_channels": int(df["channel_id"].nunique()),
        "n_feature_columns": len(keep),
        "target_drivers": drivers,
        "bootstrapped_from": str(Path(args.features).resolve()),
    }
    (out.parent / (out.stem + ".smoke_meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] smoke dataset: {len(df)} rows, {df['channel_id'].nunique()} channels, "
          f"{len(keep)} feat cols -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
