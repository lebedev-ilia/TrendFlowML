from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

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
    snapshot_sleep_seconds: int | None = None,
    snapshot_follow_up_count: int | None = None,
) -> ScheduleEntry:
    due_at: Dict[str, datetime] = {}
    if snapshot_sleep_seconds and snapshot_follow_up_count:
        base = video.snapshot_0.collected_at
        for index in range(1, snapshot_follow_up_count + 1):
            due_at[str(index)] = base + timedelta(seconds=index * snapshot_sleep_seconds)
    else:
        unit, offsets = _schedule_offsets(
            schedule_days=schedule_days,
            schedule_hours=schedule_hours,
            schedule_minutes=schedule_minutes,
        )
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
    count = getattr(config, "snapshot_follow_up_count", None)
    if count:
        return list(range(1, int(count) + 1))
    if getattr(config, "snapshot_schedule_minutes", None):
        return list(range(1, len(config.snapshot_schedule_minutes)))
    if getattr(config, "snapshot_schedule_hours", None):
        return list(range(1, len(config.snapshot_schedule_hours)))
    return list(range(1, len(config.snapshot_schedule_days)))


def _normalize_now(now: datetime | None = None) -> datetime:
    if now is None:
        return utcnow()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def pending_snapshot_indices(state: DatasetState, indices: List[int]) -> List[int]:
    completed = state.load_completed_snapshots()
    pending: set[int] = set()
    for entry in state.load_schedule():
        for index in indices:
            key = str(index)
            if (entry.dedup_key, index) in completed:
                continue
            if key in entry.completed:
                continue
            if key in entry.due_at:
                pending.add(index)
    return sorted(pending)


def seconds_until_next_snapshot_due(
    state: DatasetState,
    indices: List[int],
    *,
    now: datetime | None = None,
) -> Optional[float]:
    """Seconds until the earliest not-yet-due follow-up snapshot across all videos."""
    now = _normalize_now(now)
    completed = state.load_completed_snapshots()
    next_due: datetime | None = None
    for entry in state.load_schedule():
        for index in indices:
            key = str(index)
            if (entry.dedup_key, index) in completed:
                continue
            if key in entry.completed:
                continue
            due_at = entry.due_at.get(key)
            if due_at is None:
                continue
            if due_at <= now:
                return 0.0
            if next_due is None or due_at < next_due:
                next_due = due_at
    if next_due is None:
        return None
    return max(0.0, (next_due - now).total_seconds())


def run_snapshot_poll_loop(
    runner: "SnapshotRunner",
    state: DatasetState,
    config,
    *,
    poll_interval_seconds: int = 30,
) -> dict[str, int]:
    """Collect follow-up snapshots per video when each video's due_at elapses (never early)."""
    indices = snapshot_follow_up_indices(config)
    totals = {"snapshots": 0, "passes": 0}
    if not indices:
        return totals

    while pending_snapshot_indices(state, indices):
        totals["passes"] += 1
        collected_this_pass = 0
        for index in indices:
            due = runner.due_entries(snapshot_index=index)
            if not due:
                continue
            result = runner.collect_due(snapshot_index=index)
            collected_this_pass += len(result)
            totals["snapshots"] += len(result)

        if not pending_snapshot_indices(state, indices):
            break

        wait_sec = seconds_until_next_snapshot_due(state, indices)
        if wait_sec is None:
            break
        if wait_sec > 0:
            print(
                f"snapshot-poll: waiting {wait_sec:.0f}s until next per-video due_at "
                f"(never collecting early)",
                flush=True,
            )
            time.sleep(wait_sec)
        elif collected_this_pass == 0:
            time.sleep(max(5, poll_interval_seconds))

    return totals


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
