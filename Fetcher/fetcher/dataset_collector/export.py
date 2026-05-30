from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fetcher.dataset_collector.state import atomic_write_json
from fetcher.dataset_collector.training_format import (
    format_training_record,
    format_training_snapshot,
    load_metadata_records,
)


def iter_json_files(directory: Path):
    if not directory.exists():
        return
    for path in sorted(directory.glob("**/*.json")):
        if path.name.endswith(".tmp"):
            continue
        yield path


def load_snapshot_records(root: Path) -> Dict[str, Dict[str, dict]]:
    snapshots: Dict[str, Dict[str, dict]] = {}
    for path in iter_json_files(root / "shards" / "snapshots"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, snapshot in data.items():
            index = str(snapshot["snapshot_index"])
            snapshots.setdefault(key, {})[f"snapshot_{index}"] = snapshot
    return snapshots


def build_legacy_record(
    record: dict,
    snapshots: Dict[str, dict],
    *,
    youtube_plain_keys: bool,
) -> tuple[str, dict]:
    key, payload = format_training_record(record)
    if not (youtube_plain_keys and record.get("platform") == "youtube"):
        key = f"{record.get('platform', 'youtube')}:{record['video_id']}"
    for name, snapshot in snapshots.items():
        payload[name] = format_training_snapshot(snapshot)
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
