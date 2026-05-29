from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

from fetcher.dataset_collector.schemas import (
    CampaignConfig,
    CampaignManifest,
    CollectedVideo,
    RejectedRecord,
    ScheduleEntry,
    Snapshot,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_time_get(value: datetime | None = None) -> str:
    return (value or utcnow()).strftime("%Y_%m_%d_%H_%M")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(payload), ensure_ascii=False))
        fh.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class DatasetState:
    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        self.root = Path(config.output_dir)
        self.state_dir = self.root / "state"
        self.shards_dir = self.root / "shards"
        self.rejected_dir = self.root / "rejected"
        self.download_dir = self.root / "downloads"
        self.manifest_path = self.root / "manifest.json"
        self.seen_path = self.state_dir / "seen_ids.jsonl"
        self.schedule_path = self.state_dir / "video_schedule.jsonl"
        self.snapshot_completion_path = self.state_dir / "snapshot_completion.jsonl"
        self.api_keys_path = self.state_dir / "api_keys.json"
        self.lock_path = self.state_dir / ".collector.lock"
        self._seen: Set[str] | None = None

    def initialize(self) -> CampaignManifest:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            return self.load_manifest()
        manifest = CampaignManifest(
            name=self.config.name,
            created_at=utcnow(),
            updated_at=utcnow(),
            output_dir=str(self.root),
            counters={"accepted": 0, "rejected": 0, "snapshots": 0, "downloads": 0},
            category_counters={category.name: {"accepted": 0, "rejected": 0} for category in self.config.categories},
        )
        atomic_write_json(self.manifest_path, jsonable(manifest.dict()))
        return manifest

    def load_manifest(self) -> CampaignManifest:
        return CampaignManifest.parse_obj(json.loads(self.manifest_path.read_text(encoding="utf-8")))

    def save_manifest(self, manifest: CampaignManifest) -> None:
        manifest.updated_at = utcnow()
        atomic_write_json(self.manifest_path, jsonable(manifest.dict()))

    def load_seen(self) -> Set[str]:
        if self._seen is None:
            self._seen = {
                str(row["key"])
                for row in iter_jsonl(self.seen_path)
                if row.get("key")
            }
        return self._seen

    def is_seen(self, key: str) -> bool:
        return key in self.load_seen()

    def mark_seen(self, key: str, *, category: str) -> None:
        seen = self.load_seen()
        if key in seen:
            return
        append_jsonl(self.seen_path, {"key": key, "category": category, "seen_at": utcnow().isoformat()})
        seen.add(key)

    def write_metadata_shard(self, category: str, records: Iterable[CollectedVideo]) -> Path:
        records_list = [jsonable(record.dict()) for record in records]
        if not records_list:
            raise ValueError("cannot write empty metadata shard")
        shard_path = self._next_shard_path("metadata", category)
        atomic_write_json(shard_path, records_list)
        manifest = self.load_manifest()
        manifest.shards.append(str(shard_path.relative_to(self.root)))
        manifest.counters["accepted"] = manifest.counters.get("accepted", 0) + len(records_list)
        cat_counter = manifest.category_counters.setdefault(category, {"accepted": 0, "rejected": 0})
        cat_counter["accepted"] = cat_counter.get("accepted", 0) + len(records_list)
        self.save_manifest(manifest)
        return shard_path

    def write_snapshot_shard(self, snapshot_index: int, records: Dict[str, Snapshot]) -> Path:
        if not records:
            raise ValueError("cannot write empty snapshot shard")
        shard_path = self._next_snapshot_path(snapshot_index)
        atomic_write_json(shard_path, {key: jsonable(value.dict()) for key, value in records.items()})
        manifest = self.load_manifest()
        manifest.snapshot_shards.append(str(shard_path.relative_to(self.root)))
        manifest.counters["snapshots"] = manifest.counters.get("snapshots", 0) + len(records)
        self.save_manifest(manifest)
        return shard_path

    def write_rejected(self, records: Iterable[RejectedRecord]) -> Path | None:
        records_list = [jsonable(record.dict()) for record in records]
        if not records_list:
            return None
        shard_path = self._next_rejected_path()
        atomic_write_json(shard_path, records_list)
        manifest = self.load_manifest()
        manifest.rejected_shards.append(str(shard_path.relative_to(self.root)))
        manifest.counters["rejected"] = manifest.counters.get("rejected", 0) + len(records_list)
        for record in records_list:
            category = str(record.get("category") or "unknown")
            cat_counter = manifest.category_counters.setdefault(category, {"accepted": 0, "rejected": 0})
            cat_counter["rejected"] = cat_counter.get("rejected", 0) + 1
        self.save_manifest(manifest)
        return shard_path

    def append_schedule(self, entry: ScheduleEntry) -> None:
        append_jsonl(self.schedule_path, jsonable(entry.dict()))

    def load_schedule(self) -> List[ScheduleEntry]:
        return [ScheduleEntry.parse_obj(row) for row in iter_jsonl(self.schedule_path)]

    def mark_snapshot_completed(self, key: str, snapshot_index: int) -> None:
        append_jsonl(
            self.snapshot_completion_path,
            {
                "key": key,
                "snapshot_index": snapshot_index,
                "completed_at": utcnow().isoformat(),
            },
        )

    def load_completed_snapshots(self) -> set[tuple[str, int]]:
        return {
            (str(row.get("key")), int(row.get("snapshot_index")))
            for row in iter_jsonl(self.snapshot_completion_path)
            if row.get("key") and row.get("snapshot_index") is not None
        }

    def enqueue_download(self, video: CollectedVideo) -> None:
        append_jsonl(
            self.download_dir / "queue.jsonl",
            {
                "platform": video.platform,
                "video_id": video.video_id,
                "url": video.url,
                "category": video.category,
                "queued_at": utcnow().isoformat(),
            },
        )
        manifest = self.load_manifest()
        manifest.counters["downloads"] = manifest.counters.get("downloads", 0) + 1
        self.save_manifest(manifest)

    def _next_shard_path(self, kind: str, category: str) -> Path:
        base = self.shards_dir / kind / f"category={category}"
        return base / f"part_{self._next_index(base):06d}.json"

    def _next_snapshot_path(self, snapshot_index: int) -> Path:
        base = self.shards_dir / "snapshots" / f"snapshot={snapshot_index}"
        return base / f"part_{self._next_index(base):06d}.json"

    def _next_rejected_path(self) -> Path:
        return self.rejected_dir / f"part_{self._next_index(self.rejected_dir):06d}.json"

    @staticmethod
    def _next_index(directory: Path) -> int:
        directory.mkdir(parents=True, exist_ok=True)
        indices = []
        for path in directory.glob("part_*.json"):
            try:
                indices.append(int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return (max(indices) + 1) if indices else 0
