from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fetcher.dataset_collector.training_format import iter_metadata_shard_entries


def _percentile(values: List[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def aggregate_shard_distributions(run_root: str | Path) -> Dict[str, Any]:
    root = Path(run_root)
    metadata_dir = root / "shards" / "metadata"
    views: List[int] = []
    likes: List[int] = []
    comments: List[int] = []
    durations: List[int] = []
    time_intervals: Dict[str, int] = {}
    rejected_reasons: Dict[str, int] = {}

    if metadata_dir.exists():
        for shard_path in metadata_dir.glob("**/part_*.json"):
            data = json.loads(shard_path.read_text(encoding="utf-8"))
            for record in iter_metadata_shard_entries(data):
                snapshot = record.get("snapshot_0") or {}
                metadata = record.get("metadata") or {}
                interval = record.get("time_interval") or "unknown"
                time_intervals[interval] = time_intervals.get(interval, 0) + 1
                for target, key in ((views, "viewCount"), (likes, "likeCount"), (comments, "commentCount")):
                    raw = snapshot.get(key) or metadata.get(key.replace("Count", "_count"))
                    try:
                        if raw is not None:
                            target.append(int(raw))
                    except (TypeError, ValueError):
                        pass
                try:
                    duration = metadata.get("duration_seconds")
                    if duration is not None:
                        durations.append(int(duration))
                except (TypeError, ValueError):
                    pass

    rejected_dir = root / "rejected"
    if rejected_dir.exists():
        for shard_path in rejected_dir.glob("part_*.json"):
            records = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for record in records:
                reason = str(record.get("reason") or "unknown")
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    def _summary(values: List[int]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
        }

    return {
        "view_count": _summary(views),
        "like_count": _summary(likes),
        "comment_count": _summary(comments),
        "duration_seconds": _summary(durations),
        "time_interval": time_intervals,
        "rejected_reasons": rejected_reasons,
    }
