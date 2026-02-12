#!/usr/bin/env python3
from __future__ import annotations

"""
Generate fixed golden sets for v1 index:
- holdout: 2000 video_ids
- regression_mini: 200 video_ids

Stored under:
  Models/v1/training/golden_sets/<v1_dataset_fingerprint>/{holdout,regression_mini}.json
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_index(path: str) -> "pd.DataFrame":
    import pandas as pd  # type: ignore

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported v1 index format: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate v1 golden sets")
    p.add_argument("--v1-index", type=str, required=True)
    p.add_argument("--v1-metadata", type=str, required=True, help="v1_dataset_metadata.json")
    p.add_argument("--out-root", type=str, default="Models/v1/training/golden_sets")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--holdout-size", type=int, default=2000)
    p.add_argument("--regression-size", type=int, default=200)
    args = p.parse_args()

    meta = json.loads(Path(args.v1_metadata).read_text(encoding="utf-8"))
    fp = meta.get("v1_dataset_fingerprint")
    if not isinstance(fp, str) or not fp:
        raise ValueError("v1_dataset_metadata.json missing v1_dataset_fingerprint")

    df = load_index(args.v1_index)
    if "video_id" not in df.columns:
        raise ValueError("v1 index missing video_id")
    video_ids = sorted(set([str(v) for v in df["video_id"].astype(str).tolist()]))

    def key(v: str) -> str:
        return hashlib.sha256((str(args.seed) + "::" + v).encode("utf-8")).hexdigest()

    ordered = sorted(video_ids, key=key)
    holdout = ordered[: min(args.holdout_size, len(ordered))]
    remaining = ordered[min(args.holdout_size, len(ordered)) :]
    regression = remaining[: min(args.regression_size, len(remaining))]

    out_dir = Path(args.out_root) / fp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "regression_mini.json").write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "created_at": _now_utc(),
                "v1_dataset_fingerprint": fp,
                "seed": args.seed,
                "holdout_size": len(holdout),
                "regression_mini_size": len(regression),
                "v1_index": args.v1_index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] wrote v1 golden sets -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


