from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from fetcher.dataset_collector.state import DatasetState


def iter_legacy_ids(path: str | Path) -> Iterable[str]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in data.keys():
                yield str(key)
        elif isinstance(data, list):
            for item in data:
                yield str(item.get("id") or item.get("video_id") or item) if item else ""
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row.get("video_id") or row.get("id") or next(iter(row.values()))
    else:
        with source.open("r", encoding="utf-8") as fh:
            for line in fh:
                value = line.strip()
                if not value:
                    continue
                if value.startswith("{"):
                    row = json.loads(value)
                    yield str(row.get("key") or row.get("video_id") or row.get("id") or "")
                else:
                    yield value


def import_seen_ids(state: DatasetState, path: str | Path, *, platform: str = "youtube", category: str = "legacy") -> int:
    imported = 0
    for raw_id in iter_legacy_ids(path):
        video_id = raw_id.strip()
        if not video_id:
            continue
        key = video_id if ":" in video_id else f"{platform}:{video_id}"
        if state.is_seen(key):
            continue
        state.mark_seen(key, category=category)
        imported += 1
    return imported
