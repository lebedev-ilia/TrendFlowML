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


def snapshot_poll_report(
    state: DatasetState,
    config,
    *,
    now: datetime | None = None,
) -> dict:
    """Summary for logging: pending/due_now/done per snapshot index and next due_at."""
    now = _normalize_now(now)
    indices = snapshot_follow_up_indices(config)
    completed = state.load_completed_snapshots()
    schedule = state.load_schedule()
    per_index: dict[int, dict] = {}
    next_due_at: datetime | None = None

    for index in indices:
        key = str(index)
        pending = due_now = done = 0
        earliest_future: datetime | None = None
        for entry in schedule:
            if (entry.dedup_key, index) in completed or key in entry.completed:
                done += 1
                continue
            due_at = entry.due_at.get(key)
            if due_at is None:
                continue
            pending += 1
            if due_at <= now:
                due_now += 1
            elif earliest_future is None or due_at < earliest_future:
                earliest_future = due_at
        per_index[index] = {
            "pending": pending,
            "due_now": due_now,
            "done": done,
            "earliest_future": earliest_future,
        }
        if earliest_future and (next_due_at is None or earliest_future < next_due_at):
            next_due_at = earliest_future

    return {
        "now": now,
        "schedule_entries": len(schedule),
        "indices": indices,
        "per_index": per_index,
        "wait_seconds": seconds_until_next_snapshot_due(state, indices, now=now),
        "next_due_at": next_due_at,
        "snapshot_sleep_seconds": getattr(config, "snapshot_sleep_seconds", None),
        "snapshot_follow_up_count": getattr(config, "snapshot_follow_up_count", None),
    }


def _fmt_utc_short(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%H:%M:%S UTC")


def format_snapshot_poll_report(report: dict) -> str:
    sleep = report.get("snapshot_sleep_seconds")
    follow = report.get("snapshot_follow_up_count")
    interval = f", интервал {sleep} с" if sleep else ""
    lines = [
        f"[снапшоты] видео в расписании: {report['schedule_entries']}"
        f"{interval}, снапшоты 1–{follow or '?'}"
    ]
    parts = []
    for index in report.get("indices", []):
        row = report["per_index"][index]
        parts.append(f"#{index}: готово {row['done']}, ждут {row['pending']}, пора {row['due_now']}")
    if parts:
        lines.append("[снапшоты] " + " | ".join(parts))
    wait = report.get("wait_seconds")
    next_due = report.get("next_due_at")
    if wait is not None and wait > 0 and next_due:
        lines.append(
            f"[снапшоты] следующий сбор не раньше {_fmt_utc_short(next_due)} (ждём ~{wait:.0f} с)"
        )
    elif wait is not None and wait <= 0:
        lines.append("[снапшоты] есть видео с наступившим due — можно собирать")
    elif wait is None:
        lines.append("[снапшоты] всё собрано")
    return "\n".join(lines)


def log_snapshot_poll_report(report: dict) -> None:
    print(format_snapshot_poll_report(report), flush=True)


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


def _done_count_for_index(report: dict, index: int) -> int:
    return report["per_index"].get(index, {}).get("done", 0)


def run_snapshot_poll_loop(
    runner: "SnapshotRunner",
    state: DatasetState,
    config,
    *,
    poll_interval_seconds: int = 30,
    verbose: bool = True,
) -> dict[str, int]:
    """Collect follow-up snapshots per video when each video's due_at elapses (never early)."""
    indices = snapshot_follow_up_indices(config)
    totals = {"snapshots": 0, "passes": 0}
    if not indices:
        if verbose:
            print("[снапшоты] в конфиге нет follow-up снапшотов", flush=True)
        return totals

    schedule_size = len(state.load_schedule())
    if verbose:
        print("[снапшоты] старт", flush=True)
        log_snapshot_poll_report(snapshot_poll_report(state, config, now=runner._now()))

    while pending_snapshot_indices(state, indices):
        totals["passes"] += 1
        collected_this_pass = 0
        for index in indices:
            due = runner.due_entries(snapshot_index=index)
            if not due:
                continue
            result = runner.collect_due(snapshot_index=index)
            n = len(result)
            collected_this_pass += n
            totals["snapshots"] += n
            if verbose and n:
                report = snapshot_poll_report(state, config, now=runner._now())
                done = _done_count_for_index(report, index)
                print(
                    f"[снапшоты] проход {totals['passes']}: снапшот #{index} +{n} "
                    f"(готово {done}/{schedule_size})",
                    flush=True,
                )

        if not pending_snapshot_indices(state, indices):
            break

        wait_sec = seconds_until_next_snapshot_due(state, indices, now=runner._now())
        if wait_sec is None:
            break
        if wait_sec > 0:
            if verbose:
                report = snapshot_poll_report(state, config, now=runner._now())
                next_due = report.get("next_due_at")
                print(
                    f"[снапшоты] пауза ~{wait_sec:.0f} с до {_fmt_utc_short(next_due)}",
                    flush=True,
                )
            time.sleep(wait_sec)
        elif collected_this_pass == 0:
            time.sleep(max(5, poll_interval_seconds))

    if verbose:
        print(
            f"[снапшоты] готово: {totals['snapshots']} снапшотов, {totals['passes']} проходов",
            flush=True,
        )
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
