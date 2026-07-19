#!/usr/bin/env python3
"""
enrichment.py  —  DatasetBuilder stage C3

Ensure every row has a channel_id usable for the channel-group split (the split
that PROTECTS against channel leakage — TARGETS_SPLITS_METRICS.md). Without a
real channel_id the hybrid split silently degrades to per-video, which leaks.

Resolution order per video:
  1) real channel_id already on the row (from manifest/HF metadata)
  2) stable hash of channelTitle  -> "ch_<hash>"   (degraded but deterministic)
  3) fallback to video_id         -> "vid_<video_id>" (last resort; logged)

Real YouTube-API video->channel resolution is a future upgrade; for the first
pass we only need a deterministic, non-leaking grouping key + an honest count of
how many rows fell back.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Tuple


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def enrich_channel_id(df: "pd.DataFrame") -> Tuple["pd.DataFrame", Dict[str, int]]:  # type: ignore # noqa: F821
    import pandas as pd  # noqa: F401

    stats = {"real": 0, "from_title": 0, "from_video_id": 0}

    def resolve(row) -> str:
        cid = row.get("channel_id")
        if isinstance(cid, str) and cid.strip():
            stats["real"] += 1
            return cid
        title = row.get("channelTitle")
        if isinstance(title, str) and title.strip():
            stats["from_title"] += 1
            return f"ch_{_hash(title)}"
        stats["from_video_id"] += 1
        return f"vid_{row.get('video_id')}"

    df["channel_id"] = df.apply(resolve, axis=1)
    return df, stats
