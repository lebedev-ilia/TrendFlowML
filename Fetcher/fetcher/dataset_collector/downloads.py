from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from fetcher.platforms.registry import get_adapter


def iter_download_queue(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def run_download_queue(queue_path: Path, *, run_id_prefix: str = "dataset") -> dict[str, int]:
    results = {"downloaded": 0, "failed": 0, "skipped": 0}
    for item in iter_download_queue(queue_path):
        platform = item.get("platform")
        url = item.get("url")
        video_id = item.get("video_id")
        if not platform or not url:
            results["skipped"] += 1
            continue
        try:
            adapter = get_adapter(platform)
            adapter.download_video(url, run_id=f"{run_id_prefix}-{platform}-{video_id}")
            results["downloaded"] += 1
        except Exception:
            results["failed"] += 1
    return results
