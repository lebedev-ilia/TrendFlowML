from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reason: str = ""


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class VideoFilter:
    def __init__(self, rules: Dict[str, Any] | None = None) -> None:
        self.rules = rules or {}
        self._channel_counts: Dict[str, int] = {}

    def decide(self, record: Dict[str, Any]) -> FilterDecision:
        metadata = record.get("metadata") or record
        duration = _to_int(metadata.get("duration_seconds") or metadata.get("duration"))
        min_duration = _to_int(self.rules.get("duration_min_seconds"))
        max_duration = _to_int(self.rules.get("duration_max_seconds"))
        if min_duration is not None:
            if duration is None or duration < min_duration:
                return FilterDecision(False, "duration_below_min")
        if max_duration is not None and duration is not None and duration > max_duration:
            return FilterDecision(False, "duration_above_max")

        view_count = _to_int(
            metadata.get("viewCount")
            or metadata.get("view_count")
            or (metadata.get("statistics") or {}).get("viewCount")
        )
        view_max = _to_int(self.rules.get("view_count_max"))
        if view_max is not None and view_count is not None and view_count > view_max:
            return FilterDecision(False, "view_count_above_max")

        comment_count = _to_int(
            metadata.get("commentCount")
            or metadata.get("comment_count")
            or (metadata.get("statistics") or {}).get("commentCount")
        )
        comment_max = _to_int(self.rules.get("comment_count_max"))
        if comment_max is not None and comment_count is not None and comment_count > comment_max:
            return FilterDecision(False, "comment_count_above_max")

        channel_id = (
            metadata.get("channel_id")
            or metadata.get("channelId")
            or metadata.get("uploader_id")
            or record.get("channel_id")
        )
        channel_cap = _to_int(self.rules.get("channel_video_cap"))
        if channel_cap is not None and channel_id:
            current = self._channel_counts.get(str(channel_id), 0)
            if current >= channel_cap:
                return FilterDecision(False, "channel_cap_exceeded")

        return FilterDecision(True)

    def decide_post_enrich(self, *, info: Dict[str, Any], metadata: Dict[str, Any]) -> FilterDecision:
        """Reject live streams and invalid durations revealed by yt-dlp."""
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            return FilterDecision(False, "live_stream")
        duration = _to_int(info.get("duration"))
        if duration is None:
            duration = _to_int(metadata.get("duration_seconds") or metadata.get("duration"))
        return self.decide({"metadata": {**metadata, "duration_seconds": duration, "duration": duration}})

    def accept(self, record: Dict[str, Any]) -> None:
        metadata = record.get("metadata") or record
        channel_id = metadata.get("channel_id") or metadata.get("channelId") or metadata.get("uploader_id")
        if channel_id:
            key = str(channel_id)
            self._channel_counts[key] = self._channel_counts.get(key, 0) + 1


def split_records(
    records: Iterable[Dict[str, Any]],
    video_filter: VideoFilter,
) -> Tuple[list[Dict[str, Any]], list[tuple[Dict[str, Any], str]]]:
    accepted: list[Dict[str, Any]] = []
    rejected: list[tuple[Dict[str, Any], str]] = []
    for record in records:
        decision = video_filter.decide(record)
        if decision.accepted:
            video_filter.accept(record)
            accepted.append(record)
        else:
            rejected.append((record, decision.reason))
    return accepted, rejected
