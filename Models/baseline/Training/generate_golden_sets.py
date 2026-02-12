#!/usr/bin/env python3
from __future__ import annotations

"""
Golden sets generator (quality gate inputs):
- holdout: 2000 video_ids
- regression_mini: 200 video_ids

We store the sets under:
  Models/baseline/Training/golden_sets/<dataset_fingerprint>/{holdout,regression_mini}.json

Why:
- fixed evaluation sets prevent metric drift from accidental data leakage/sampling changes.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_dataset(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Golden sets generation requires pandas. Install it in your venv.") from e

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported dataset format: {path}")


def _stable_hash_order(video_ids: List[str], seed: int) -> List[str]:
    def key(v: str) -> str:
        return hashlib.sha256((str(seed) + "::" + v).encode("utf-8")).hexdigest()

    return sorted(video_ids, key=key)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate fixed golden sets (holdout/regression) for a dataset")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset-metadata", type=str, required=True, help="dataset_metadata.json (contains dataset_fingerprint)")
    p.add_argument("--out-root", type=str, default="Models/baseline/Training/golden_sets")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--holdout-size", type=int, default=2000)
    p.add_argument("--regression-size", type=int, default=200)
    args = p.parse_args()

    meta = json.loads(Path(args.dataset_metadata).read_text(encoding="utf-8"))
    fingerprint = meta.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("dataset_metadata.json missing dataset_fingerprint")

    df = load_dataset(args.dataset)
    if "video_id" not in df.columns:
        raise ValueError("Dataset missing video_id column")

    video_ids = [str(v) for v in df["video_id"].astype(str).tolist()]
    video_ids = sorted(set(video_ids))

    ordered = _stable_hash_order(video_ids, args.seed)
    holdout = ordered[: min(args.holdout_size, len(ordered))]
    remaining = ordered[min(args.holdout_size, len(ordered)) :]
    regression = remaining[: min(args.regression_size, len(remaining))]

    out_dir = Path(args.out_root) / fingerprint
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "holdout.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "regression_mini.json").write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "created_at": _now_utc(),
                "dataset_fingerprint": fingerprint,
                "seed": args.seed,
                "holdout_size": len(holdout),
                "regression_mini_size": len(regression),
                "dataset_path": args.dataset,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] wrote golden sets -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


