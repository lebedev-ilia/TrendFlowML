from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

from fetcher.dataset_collector.checkpoint import DiscoveryCheckpoint
from fetcher.dataset_collector.keyword_progress import KeywordProgressEntry
from fetcher.dataset_collector.schemas import (
    CampaignConfig,
    CampaignManifest,
    CollectedVideo,
    RejectedRecord,
    ScheduleEntry,
    Snapshot,
)
from fetcher.dataset_collector.training_format import format_training_shard


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_time_get(value: datetime | None = None) -> str:
    return (value or utcnow()).strftime("%Y_%m_%d_%H_%M")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per writer — several workers refresh inventory/summary in parallel.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{utcnow().timestamp():.0f}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


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
        self.checkpoint_path = self.state_dir / "discovery_checkpoint.json"
        self.keyword_progress_path = self.state_dir / "keyword_progress.jsonl"
        self.metadata_enrich_queue_path = self.state_dir / "metadata_enrich_queue.jsonl"
        self.metadata_enrich_done_path = self.state_dir / "metadata_enrich_done.jsonl"
        self.hf_video_upload_queue_path = self.state_dir / "hf_video_upload_queue.jsonl"
        self.hf_video_upload_done_path = self.state_dir / "hf_video_upload_done.jsonl"
        self.hf_shard_upload_queue_path = self.state_dir / "hf_shard_upload_queue.jsonl"
        self.hf_shard_upload_done_path = self.state_dir / "hf_shard_upload_done.jsonl"
        self.download_done_path = self.state_dir / "download_done.jsonl"
        self.post_enrich_rejected_path = self.state_dir / "post_enrich_rejected.jsonl"
        self.lock_path = self.state_dir / ".collector.lock"
        self._seen: Set[str] | None = None
        self._pending_accepted: Dict[str, List[CollectedVideo]] = {}
        self._pending_rejected: List[RejectedRecord] = []

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
            baseline_accepted=self.config.baseline_accepted,
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

    def start_session(self) -> CampaignManifest:
        manifest = self.initialize()
        if manifest.baseline_accepted == 0 and self.config.baseline_accepted:
            manifest.baseline_accepted = self.config.baseline_accepted
        manifest.session_started_at = utcnow()
        manifest.session_counters = {"accepted": 0, "rejected": 0, "quota_units": 0}
        self.save_manifest(manifest)
        return manifest

    def increment_session(self, *, accepted: int = 0, rejected: int = 0) -> None:
        manifest = self.load_manifest()
        session = dict(manifest.session_counters or {})
        if accepted:
            session["accepted"] = session.get("accepted", 0) + accepted
        if rejected:
            session["rejected"] = session.get("rejected", 0) + rejected
        manifest.session_counters = session
        self.save_manifest(manifest)

    def category_accepted(self, category: str) -> int:
        manifest = self.load_manifest()
        return int(manifest.category_counters.get(category, {}).get("accepted", 0))

    def pending_accepted_count(self, category: str | None = None) -> int:
        if category is None:
            return sum(len(items) for items in self._pending_accepted.values())
        return len(self._pending_accepted.get(category, []))

    def live_category_accepted(self, category: str) -> int:
        return self.category_accepted(category) + self.pending_accepted_count(category)

    def live_run_accepted(self) -> int:
        manifest = self.load_manifest()
        return int(manifest.counters.get("accepted", 0)) + self.pending_accepted_count()

    def is_category_complete(self, category: str, collect_count: int) -> bool:
        return self.live_category_accepted(category) >= collect_count

    def load_checkpoint(self) -> DiscoveryCheckpoint | None:
        if not self.checkpoint_path.exists():
            return None
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return DiscoveryCheckpoint.parse_obj(data)

    def save_checkpoint(self, checkpoint: DiscoveryCheckpoint) -> None:
        checkpoint.updated_at = utcnow()
        atomic_write_json(self.checkpoint_path, jsonable(checkpoint.dict()))

    def clear_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def append_keyword_progress(self, entry: KeywordProgressEntry) -> None:
        append_jsonl(self.keyword_progress_path, entry.dict())

    def load_completed_keyword_indices(
        self,
        *,
        category: str,
        bucket_name: str | None,
        platform: str,
    ) -> Set[int]:
        """Keyword indices that already reached min_required in a prior run."""
        completed: Set[int] = set()
        for row in iter_jsonl(self.keyword_progress_path):
            try:
                entry = KeywordProgressEntry.parse_obj(row)
            except Exception:
                continue
            if not entry.is_done:
                continue
            if entry.category != category:
                continue
            if entry.bucket_name != bucket_name:
                continue
            if entry.platform != platform:
                continue
            completed.add(entry.keyword_index)
        return completed

    def buffer_accepted(self, category: str, video: CollectedVideo) -> None:
        self._pending_accepted.setdefault(category, []).append(video)

    def buffer_rejected(self, record: RejectedRecord) -> None:
        self._pending_rejected.append(record)

    def flush_pending(
        self,
        category: str,
        *,
        shard_size: int,
        force: bool = False,
    ) -> List[Path]:
        written: List[Path] = []
        pending = self._pending_accepted.get(category, [])
        while pending and (force or len(pending) >= shard_size):
            chunk = pending[:shard_size]
            del pending[:shard_size]
            written.append(self.write_metadata_shard(category, chunk))
        if force and self._pending_rejected:
            self.write_rejected(self._pending_rejected)
            self._pending_rejected = []
        return written

    def flush_all_pending(self, *, shard_size: int) -> List[Path]:
        written: List[Path] = []
        for category in list(self._pending_accepted.keys()):
            written.extend(self.flush_pending(category, shard_size=shard_size, force=True))
        if self._pending_rejected:
            self.write_rejected(self._pending_rejected)
            self._pending_rejected = []
        return written

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
        records_list = list(records)
        if not records_list:
            raise ValueError("cannot write empty metadata shard")
        shard_path = self._next_shard_path("metadata", category)
        shard_relpath = str(shard_path.relative_to(self.root))
        atomic_write_json(shard_path, format_training_shard(records_list))
        from fetcher.dataset_collector.inventory import register_shard

        register_shard(
            self,
            shard_relpath=shard_relpath,
            category=category,
            video_ids=[r.video_id for r in records_list],
            platform=records_list[0].platform if records_list else "youtube",
        )
        for record in records_list:
            if record.platform == "youtube":
                self.enqueue_metadata_enrichment(
                    platform=record.platform,
                    video_id=record.video_id,
                    url=record.url,
                    category=record.category,
                    shard_relpath=shard_relpath,
                )
        self.enqueue_hf_shard_upload(shard_relpath=shard_relpath, category=category)
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

    def enqueue_metadata_enrichment(
        self,
        *,
        platform: str,
        video_id: str,
        url: str,
        category: str,
        shard_relpath: str,
    ) -> None:
        append_jsonl(
            self.metadata_enrich_queue_path,
            {
                "platform": platform,
                "video_id": video_id,
                "url": url,
                "category": category,
                "shard": shard_relpath,
                "queued_at": utcnow().isoformat(),
            },
        )

    def load_metadata_enrich_done(self) -> set[str]:
        return {
            str(row.get("key"))
            for row in iter_jsonl(self.metadata_enrich_done_path)
            if row.get("key")
        }

    def mark_metadata_enrich_done(self, key: str, *, video_id: str, shard: str) -> None:
        append_jsonl(
            self.metadata_enrich_done_path,
            {
                "key": key,
                "video_id": video_id,
                "shard": shard,
                "completed_at": utcnow().isoformat(),
            },
        )

    def enqueue_hf_video_upload(
        self,
        *,
        platform: str,
        video_id: str,
        category: str,
        local_path: str,
    ) -> None:
        append_jsonl(
            self.hf_video_upload_queue_path,
            {
                "platform": platform,
                "video_id": video_id,
                "category": category,
                "local_path": local_path,
                "queued_at": utcnow().isoformat(),
            },
        )

    def load_hf_video_upload_done(self) -> set[str]:
        return {
            str(row.get("key"))
            for row in iter_jsonl(self.hf_video_upload_done_path)
            if row.get("key")
        }

    def mark_hf_video_upload_done(
        self,
        key: str,
        *,
        video_id: str,
        category: str,
        local_path: str,
    ) -> None:
        append_jsonl(
            self.hf_video_upload_done_path,
            {
                "key": key,
                "video_id": video_id,
                "category": category,
                "local_path": local_path,
                "completed_at": utcnow().isoformat(),
            },
        )

    def enqueue_hf_shard_upload(self, *, shard_relpath: str, category: str) -> None:
        append_jsonl(
            self.hf_shard_upload_queue_path,
            {
                "shard": shard_relpath,
                "category": category,
                "queued_at": utcnow().isoformat(),
            },
        )

    def load_hf_shard_upload_done(self) -> set[str]:
        return {
            str(row.get("key"))
            for row in iter_jsonl(self.hf_shard_upload_done_path)
            if row.get("key")
        }

    def mark_hf_shard_upload_done(self, key: str, *, shard_relpath: str) -> None:
        append_jsonl(
            self.hf_shard_upload_done_path,
            {
                "key": key,
                "shard": shard_relpath,
                "completed_at": utcnow().isoformat(),
            },
        )

    def load_download_done(self) -> set[str]:
        return {
            str(row.get("key"))
            for row in iter_jsonl(self.download_done_path)
            if row.get("key")
        }

    def mark_download_done(
        self,
        key: str,
        *,
        video_id: str,
        category: str,
        local_path: str,
    ) -> None:
        append_jsonl(
            self.download_done_path,
            {
                "key": key,
                "video_id": video_id,
                "category": category,
                "local_path": local_path,
                "completed_at": utcnow().isoformat(),
            },
        )

    def load_post_enrich_rejected_video_ids(self) -> set[str]:
        return {
            str(row.get("video_id"))
            for row in iter_jsonl(self.post_enrich_rejected_path)
            if row.get("video_id")
        }

    def record_post_enrich_rejection(
        self,
        *,
        platform: str,
        video_id: str,
        category: str,
        query: str,
        reason: str,
        record: dict,
    ) -> None:
        self.buffer_rejected(
            RejectedRecord(
                platform=platform,
                video_id=video_id,
                category=category,
                query=query,
                reason=reason,
                record=record,
                rejected_at=utcnow(),
            ),
        )
        if len(self._pending_rejected) >= self.config.shard_size:
            self.flush_pending(force=True)
        append_jsonl(
            self.post_enrich_rejected_path,
            {
                "key": f"{platform}:{video_id}",
                "video_id": video_id,
                "category": category,
                "reason": reason,
                "rejected_at": utcnow().isoformat(),
            },
        )
        manifest = self.load_manifest()
        manifest.counters["rejected"] = manifest.counters.get("rejected", 0) + 1
        manifest.counters["accepted"] = max(0, manifest.counters.get("accepted", 0) - 1)
        cat_counter = manifest.category_counters.setdefault(category, {"accepted": 0, "rejected": 0})
        cat_counter["rejected"] = cat_counter.get("rejected", 0) + 1
        cat_counter["accepted"] = max(0, cat_counter.get("accepted", 0) - 1)
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
