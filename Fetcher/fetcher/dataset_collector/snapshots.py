from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Iterable, List

from fetcher.dataset_collector.discovery.base import DiscoveryAdapter
from fetcher.dataset_collector.schemas import ScheduleEntry, Snapshot
from fetcher.dataset_collector.state import DatasetState, utcnow


def _schedule_offsets(
    *,
    schedule_days: Iterable[int] | None = None,
    schedule_hours: Iterable[int] | None = None,
    schedule_minutes: Iterable[int] | None = None,
) -> tuple[str, List[int]]:
    minutes = list(schedule_minutes or [])
    if minutes:
        return "minutes", minutes
    hours = list(schedule_hours or [])
    if hours:
        return "hours", hours
    return "days", list(schedule_days or [0])


def build_schedule_entry(
    video,
    schedule_days: Iterable[int] | None = None,
    *,
    schedule_hours: Iterable[int] | None = None,
    schedule_minutes: Iterable[int] | None = None,
) -> ScheduleEntry:
    unit, offsets = _schedule_offsets(
        schedule_days=schedule_days,
        schedule_hours=schedule_hours,
        schedule_minutes=schedule_minutes,
    )
    due_at = {}
    for index, offset in enumerate(offsets):
        if index == 0:
            continue
        if unit == "minutes":
            due_at[str(index)] = video.snapshot_0.collected_at + timedelta(minutes=offset)
        elif unit == "hours":
            due_at[str(index)] = video.snapshot_0.collected_at + timedelta(hours=offset)
        else:
            due_at[str(index)] = video.snapshot_0.collected_at + timedelta(days=offset)
    return ScheduleEntry(
        platform=video.platform,
        video_id=video.video_id,
        category=video.category,
        url=video.url,
        baseline_collected_at=video.snapshot_0.collected_at,
        due_at=due_at,
    )


def snapshot_follow_up_indices(config) -> List[int]:
    """Indices 1..N for scheduled follow-up snapshots (index 0 is snapshot_0 at discover)."""
    if getattr(config, "snapshot_schedule_minutes", None):
        return list(range(1, len(config.snapshot_schedule_minutes)))
    if getattr(config, "snapshot_schedule_hours", None):
        return list(range(1, len(config.snapshot_schedule_hours)))
    return list(range(1, len(config.snapshot_schedule_days)))


def snapshot_loop_wait_seconds(
    config,
    snapshot_index: int,
    *,
    override_seconds: int | None = None,
) -> int:
    """Seconds to wait before collecting snapshot_index (1-based follow-up index)."""
    if override_seconds is not None:
        return max(0, override_seconds)
    minutes = list(getattr(config, "snapshot_schedule_minutes", None) or [])
    if minutes and snapshot_index < len(minutes):
        return max(0, (minutes[snapshot_index] - minutes[snapshot_index - 1]) * 60)
    hours = list(getattr(config, "snapshot_schedule_hours", None) or [])
    if hours and snapshot_index < len(hours):
        return max(0, (hours[snapshot_index] - hours[snapshot_index - 1]) * 3600)
    return 0


class SnapshotRunner:
    def __init__(
        self,
        state: DatasetState,
        adapters: Dict[str, DiscoveryAdapter],
        *,
        comments_limit: int,
    ) -> None:
        self.state = state
        self.adapters = adapters
        self.comments_limit = comments_limit

    def _now(self):
        override = os.environ.get("DATASET_SNAPSHOT_TEST_NOW", "").strip()
        if override:
            from datetime import datetime, timezone

            return datetime.fromisoformat(override.replace("Z", "+00:00"))
        return utcnow()

    def due_entries(self, *, snapshot_index: int) -> List[ScheduleEntry]:
        now = self._now()
        due_key = str(snapshot_index)
        completed = self.state.load_completed_snapshots()
        result = []
        for entry in self.state.load_schedule():
            if (entry.dedup_key, snapshot_index) in completed:
                continue
            if due_key in entry.completed:
                continue
            due_at = entry.due_at.get(due_key)
            if due_at and due_at <= now:
                result.append(entry)
        return result

    def collect_due(self, *, snapshot_index: int, limit: int | None = None) -> dict[str, Snapshot]:
        entries = self.due_entries(snapshot_index=snapshot_index)
        if limit is not None:
            entries = entries[:limit]
        snapshots: dict[str, Snapshot] = {}
        for entry in entries:
            adapter = self.adapters.get(entry.platform)
            if adapter is None or not adapter.capabilities.snapshots:
                continue
            snapshot = adapter.collect_snapshot(
                entry.video_id,
                snapshot_index=snapshot_index,
                comments_limit=self.comments_limit,
            )
            snapshots[entry.dedup_key] = snapshot
            self.state.mark_snapshot_completed(entry.dedup_key, snapshot_index)
        if snapshots:
            self.state.write_snapshot_shard(snapshot_index, snapshots)
        return snapshots
