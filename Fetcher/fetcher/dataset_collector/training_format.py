from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from fetcher.dataset_collector.schemas import CollectedVideo

# Languages kept in shard metadata; full timedtext URLs are omitted to avoid multi-MiB bloat.
CAPTION_LANGS = frozenset({"ru", "en", "en-US", "en-GB"})


def _jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

# Internal bucket names (collector) -> legacy training dataset labels (main_ready / data_00).
LEGACY_TIME_INTERVAL: Dict[str, str] = {
    "lt_1d": "less-1day",
    "1d_1w": "1day-1week",
    "1w_1m": "1week-1month",
    "1m_3m": "1month-3month",
    "3m_6m": "3month-6month",
    "6m_1y": "6month-1year",
    "1y_3y": "1year-3year",
    "gt_3y": "3year-more",
}


def legacy_time_interval(value: str | None) -> str | None:
    if value is None:
        return None
    return LEGACY_TIME_INTERVAL.get(value, value)


def normalize_iso_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    if value.endswith("+00:00"):
        return f"{value[:-6]}Z"
    return value


def format_training_comment(comment: dict) -> dict:
    return {
        "text": comment.get("text") or "",
        "likeCount": int(comment.get("likeCount") or 0),
        "repliesCount": int(comment.get("repliesCount", comment.get("replyCount")) or 0),
        "publishedAt": normalize_iso_timestamp(comment.get("publishedAt")),
        "authorName": comment.get("authorName") or "",
    }


def format_training_snapshot(snapshot: dict) -> dict:
    payload: Dict[str, Any] = {
        "viewCount": snapshot.get("viewCount"),
        "likeCount": snapshot.get("likeCount"),
        "commentCount": snapshot.get("commentCount"),
        "comments": [format_training_comment(item) for item in (snapshot.get("comments") or [])],
        "time_get": snapshot.get("time_get"),
    }
    for key in ("subscriberCount", "videoCount", "viewCount_channel"):
        if snapshot.get(key) is not None:
            payload[key] = snapshot.get(key)
    return payload


def extract_ytdlp_formats(info: dict) -> list[dict]:
    seen: set[tuple[Any, Any]] = set()
    formats: list[dict] = []
    for fmt in info.get("formats") or []:
        if fmt.get("vcodec") in (None, "none"):
            continue
        resolution = fmt.get("resolution")
        if not resolution and fmt.get("width") and fmt.get("height"):
            resolution = f"{fmt['width']}x{fmt['height']}"
        fps = fmt.get("fps")
        key = (fps, resolution)
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {}
        if fps is not None:
            entry["fps"] = fps
        if resolution:
            entry["resolution"] = resolution
        if entry:
            formats.append(entry)
    return formats


def _caption_lang_allowed(lang: str) -> bool:
    normalized = lang.replace("_", "-").lower()
    if lang in CAPTION_LANGS:
        return True
    if normalized == "ru":
        return True
    return normalized.startswith("en")


def slim_caption_tracks(tracks: dict | None) -> dict:
    """Keep ru/en caption tracks without download URLs (ext list only)."""
    if not tracks or not isinstance(tracks, dict):
        return {}
    slim: Dict[str, Any] = {}
    for lang, entries in tracks.items():
        if not _caption_lang_allowed(str(lang)):
            continue
        exts: list[str] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            ext = entry.get("ext")
            if ext and ext not in exts:
                exts.append(str(ext))
        if exts:
            slim[str(lang)] = [{"ext": ext} for ext in exts]
    return slim


def metadata_captions_are_bloated(metadata: dict) -> bool:
    for key in ("automatic_captions", "subtitles"):
        block = metadata.get(key) or {}
        if not isinstance(block, dict):
            continue
        for entries in block.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and entry.get("url"):
                    return True
    return False


def compact_training_metadata(metadata: dict) -> dict:
    merged = dict(metadata)
    merged["automatic_captions"] = slim_caption_tracks(merged.get("automatic_captions"))
    merged["subtitles"] = slim_caption_tracks(merged.get("subtitles"))
    return merged


def extract_ytdlp_thumbnails(info: dict) -> list[dict]:
    thumbnails: list[dict] = []
    for thumb in info.get("thumbnails") or []:
        url = thumb.get("url")
        if not url:
            continue
        entry: Dict[str, Any] = {"url": url}
        for key in ("preference", "id", "height", "width", "resolution"):
            if thumb.get(key) is not None:
                entry[key] = thumb.get(key)
        thumbnails.append(entry)
    return thumbnails


def merge_ytdlp_into_training_metadata(metadata: dict, info: dict) -> dict:
    """Fill training metadata fields that YouTube Data API often omits."""
    merged = dict(metadata)
    if info.get("title") and not merged.get("title"):
        merged["title"] = info["title"]
    if info.get("description"):
        merged["description"] = info["description"]
    if info.get("tags"):
        merged["tags"] = info["tags"]
    if info.get("language"):
        merged["language"] = info["language"]
    duration = info.get("duration")
    if duration is not None:
        try:
            duration_int = int(duration)
            merged["duration"] = duration_int
            merged["duration_seconds"] = duration_int
        except (TypeError, ValueError):
            pass
    if info.get("channel"):
        merged["channelTitle"] = merged.get("channelTitle") or info.get("channel")
    formats = extract_ytdlp_formats(info)
    if formats:
        merged["formats"] = formats
    thumbnails_ytdlp = extract_ytdlp_thumbnails(info)
    if thumbnails_ytdlp:
        merged["thumbnails_ytdlp"] = thumbnails_ytdlp
    if info.get("subtitles") is not None:
        merged["subtitles"] = slim_caption_tracks(info.get("subtitles") or {})
    if info.get("automatic_captions") is not None:
        merged["automatic_captions"] = slim_caption_tracks(info.get("automatic_captions") or {})
    if info.get("chapters") is not None:
        merged["chapters"] = info.get("chapters")
    return compact_training_metadata(merged)


def training_entry_needs_ytdlp_enrichment(entry: dict) -> bool:
    enriched = entry.get("_enriched") or {}
    if enriched.get("source") == "yt_dlp":
        return False
    metadata = entry.get("metadata") or {}
    if metadata.get("formats") or metadata.get("thumbnails_ytdlp"):
        return False
    return True


def format_training_metadata(record: dict) -> dict:
    metadata = record.get("metadata") or {}
    raw = metadata.get("raw") or {}
    snippet = raw.get("snippet") or {}
    status = raw.get("status") or {}
    channel_raw = metadata.get("channel") or {}
    channel_snippet = channel_raw.get("snippet") or {}
    thumbnails = snippet.get("thumbnails") or {}
    standard_thumb = (
        thumbnails.get("standard") or thumbnails.get("high") or thumbnails.get("medium") or {}
    )

    duration_seconds = metadata.get("duration_seconds")
    if duration_seconds is not None:
        try:
            duration_seconds = int(duration_seconds)
        except (TypeError, ValueError):
            duration_seconds = None

    return {
        "title": metadata.get("title") or snippet.get("title") or "",
        "description": metadata.get("description") or snippet.get("description") or "",
        "tags": snippet.get("tags") or [],
        "language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage"),
        "madeForKids": bool(status.get("madeForKids", False)),
        "duration": duration_seconds,
        "publishedAt": normalize_iso_timestamp(snippet.get("publishedAt") or metadata.get("publishedAt")),
        "channelTitle": metadata.get("channelTitle") or snippet.get("channelTitle") or "",
        "country": channel_snippet.get("country"),
        "thumbnails": {"standard": standard_thumb} if standard_thumb else {},
        "subtitles": {},
        "automatic_captions": {},
        "chapters": None,
        "formats": [],
        "thumbnails_ytdlp": [],
        "duration_seconds": duration_seconds,
    }


def format_training_record(record: dict) -> tuple[str, dict]:
    """Convert internal collector record to one training JSON entry (data_XX style)."""
    video_id = record["video_id"]
    payload: Dict[str, Any] = {
        "query": record.get("query") or "",
        "time_interval": legacy_time_interval(record.get("time_interval")),
        "metadata": format_training_metadata(record),
        "snapshot_0": format_training_snapshot(record.get("snapshot_0") or {}),
    }
    return video_id, payload


def format_training_shard(records: Iterable[CollectedVideo | dict]) -> Dict[str, dict]:
    shard: Dict[str, dict] = {}
    for record in records:
        if isinstance(record, CollectedVideo):
            internal = _jsonable(record.dict())
        else:
            internal = record
        video_id, payload = format_training_record(internal)
        shard[video_id] = payload
    return shard


def iter_metadata_shard_entries(data: Any) -> Iterator[dict]:
    """Yield normalized internal-style records from a metadata shard (list or dict)."""
    if isinstance(data, dict):
        for video_id, payload in data.items():
            yield {
                "platform": "youtube",
                "video_id": video_id,
                **payload,
            }
        return
    if isinstance(data, list):
        for record in data:
            yield record


def load_metadata_records(root: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    metadata_root = root / "shards" / "metadata"
    if not metadata_root.exists():
        return records
    for path in sorted(metadata_root.glob("**/part_*.json")):
        if path.name.endswith(".tmp"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in iter_metadata_shard_entries(data):
            platform = record.get("platform") or "youtube"
            video_id = record.get("video_id")
            if not video_id:
                continue
            key = f"{platform}:{video_id}"
            records[key] = record
    return records
