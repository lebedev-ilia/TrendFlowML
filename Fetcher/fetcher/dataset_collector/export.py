from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fetcher.dataset_collector.state import atomic_write_json


def iter_json_files(directory: Path):
    if not directory.exists():
        return
    for path in sorted(directory.glob("**/*.json")):
        if path.name.endswith(".tmp"):
            continue
        yield path


def load_metadata_records(root: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for path in iter_json_files(root / "shards" / "metadata"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            key = f"{record['platform']}:{record['video_id']}"
            records[key] = record
    return records


def load_snapshot_records(root: Path) -> Dict[str, Dict[str, dict]]:
    snapshots: Dict[str, Dict[str, dict]] = {}
    for path in iter_json_files(root / "shards" / "snapshots"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, snapshot in data.items():
            index = str(snapshot["snapshot_index"])
            snapshots.setdefault(key, {})[f"snapshot_{index}"] = snapshot
    return snapshots


def _legacy_metadata(record: dict) -> dict:
    metadata = record.get("metadata") or {}
    raw = metadata.get("raw") or {}
    snippet = raw.get("snippet") or {}
    status = raw.get("status") or {}
    recording = raw.get("recordingDetails") or {}
    content_details = raw.get("contentDetails") or {}
    thumbnails = snippet.get("thumbnails") or {}
    standard_thumb = thumbnails.get("standard") or thumbnails.get("high") or thumbnails.get("medium") or {}
    return {
        "title": metadata.get("title") or snippet.get("title") or "",
        "description": metadata.get("description") or snippet.get("description") or "",
        "tags": snippet.get("tags") or [],
        "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
        "madeForKids": status.get("madeForKids", False),
        "duration": metadata.get("duration_seconds"),
        "duration_seconds": metadata.get("duration_seconds"),
        "publishedAt": snippet.get("publishedAt") or metadata.get("publishedAt"),
        "channelTitle": metadata.get("channelTitle") or snippet.get("channelTitle"),
        "country": recording.get("locationDescription") or recording.get("recordingDate"),
        "thumbnails": {"standard": standard_thumb} if standard_thumb else {},
        "subtitles": {},
        "automatic_captions": {},
        "chapters": None,
        "formats": [],
        "thumbnails_ytdlp": [],
    }


def _legacy_snapshot(snapshot: dict) -> dict:
    return {
        "time_get": snapshot.get("time_get"),
        "viewCount": snapshot.get("viewCount"),
        "likeCount": snapshot.get("likeCount"),
        "commentCount": snapshot.get("commentCount"),
        "subscriberCount": snapshot.get("subscriberCount"),
        "videoCount": snapshot.get("videoCount"),
        "viewCount_channel": snapshot.get("viewCount_channel"),
        "comments": snapshot.get("comments") or [],
    }


def build_legacy_record(record: dict, snapshots: Dict[str, dict], *, youtube_plain_keys: bool) -> tuple[str, dict]:
    key = record["video_id"] if youtube_plain_keys and record["platform"] == "youtube" else f"{record['platform']}:{record['video_id']}"
    payload: Dict[str, Any] = {
        "platform": record.get("platform"),
        "category": record.get("category"),
        "query": record.get("query"),
        "collected_at": (record.get("snapshot_0") or {}).get("collected_at") or record.get("discovered_at"),
        "time_interval": record.get("time_interval"),
        "metadata": _legacy_metadata(record),
        "snapshot_0": _legacy_snapshot(record["snapshot_0"]),
        "_enriched": {
            "at": record.get("discovered_at"),
            "source": "dataset_collector",
        },
    }
    payload.update({name: _legacy_snapshot(snapshot) for name, snapshot in snapshots.items()})
    return key, payload


def export_legacy_json(
    output_dir: str | Path,
    export_dir: str | Path,
    *,
    split_count: int = 20,
    youtube_plain_keys: bool = True,
) -> dict[str, int]:
    root = Path(output_dir)
    target = Path(export_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata_records(root)
    snapshots = load_snapshot_records(root)
    merged: Dict[str, dict] = {}
    for key, record in metadata.items():
        export_key, payload = build_legacy_record(
            record,
            snapshots.get(key, {}),
            youtube_plain_keys=youtube_plain_keys,
        )
        merged[export_key] = payload

    items = list(merged.items())
    if not items:
        atomic_write_json(target / "data_00.json", {})
        return {"records": 0, "files": 1}

    split_count = max(1, split_count)
    chunk_size = max(1, (len(items) + split_count - 1) // split_count)
    written = 0
    for index, start in enumerate(range(0, len(items), chunk_size)):
        chunk = dict(items[start : start + chunk_size])
        atomic_write_json(target / f"data_{index:02d}.json", chunk)
        written += 1
    atomic_write_json(
        target / "export_manifest.json",
        {"records": len(items), "files": written, "source": str(root)},
    )
    return {"records": len(items), "files": written}


def validate_export(output_dir: str | Path, *, required_snapshots: int = 1) -> dict[str, int]:
    root = Path(output_dir)
    metadata = load_metadata_records(root)
    snapshots = load_snapshot_records(root)
    complete = 0
    incomplete = 0
    for key in metadata:
        available = snapshots.get(key, {})
        if len(available) + 1 >= required_snapshots:
            complete += 1
        else:
            incomplete += 1
    return {"total": len(metadata), "complete": complete, "incomplete": incomplete}
