#!/usr/bin/env python3
from __future__ import annotations

"""
Enrichment stub for baseline (M4): video_id -> channel_id (+ optional channel stats).

Project contract:
- We do NOT rely on channelTitle/authorName for channel-group split.
- Enrichment should be done out-of-band (YouTube API or a prebuilt index), and
  stored as a stable mapping file that DatasetBuilder can join.

This file intentionally ships as a minimal interface so you can plug in:
- a local JSON/CSV mapping (recommended for offline reproducibility), OR
- a YouTube API fetcher (requires network + credentials; not enabled by default).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_video_to_channel_mapping(path: str) -> Dict[str, str]:
    """
    Supported formats:
      - JSON dict: {"<video_id>": "<channel_id>", ...}
      - CSV with headers: video_id,channel_id
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (data or {}).items()}
    if p.suffix.lower() == ".csv":
        out: Dict[str, str] = {}
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                vid = (row.get("video_id") or "").strip()
                cid = (row.get("channel_id") or "").strip()
                if vid and cid:
                    out[vid] = cid
        return out
    raise ValueError(f"Unsupported mapping format: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Validate/normalize video_id -> channel_id mapping for dataset enrichment")
    p.add_argument("--in-mapping", type=str, required=True, help="Input mapping (.json or .csv)")
    p.add_argument("--out-json", type=str, required=True, help="Output normalized mapping (json)")
    args = p.parse_args()

    mapping = load_video_to_channel_mapping(args.in_mapping)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(mapping)} mappings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


