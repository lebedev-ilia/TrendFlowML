from __future__ import annotations

from datetime import timedelta
from typing import Dict, Iterable, List

from fetcher.dataset_collector.discovery.base import DiscoveryAdapter
from fetcher.dataset_collector.schemas import ScheduleEntry, Snapshot
from fetcher.dataset_collector.state import DatasetState, utcnow


def build_schedule_entry(video, schedule_days: Iterable[int]) -> ScheduleEntry:
    due_at = {}
    for index, days in enumerate(schedule_days):
        if index == 0:
            continue
        due_at[str(index)] = video.snapshot_0.collected_at + timedelta(days=days)
    return ScheduleEntry(
        platform=video.platform,
        video_id=video.video_id,
        category=video.category,
        url=video.url,
        baseline_collected_at=video.snapshot_0.collected_at,
        due_at=due_at,
    )


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

    def due_entries(self, *, snapshot_index: int) -> List[ScheduleEntry]:
        now = utcnow()
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
