from __future__ import annotations

import json
from collections import Counter as CollectionsCounter
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server

dataset_collector_videos_accepted_total = Counter(
    "dataset_collector_videos_accepted_total",
    "Videos accepted into metadata shards.",
    labelnames=("category", "time_interval", "platform"),
)

dataset_collector_videos_rejected_total = Counter(
    "dataset_collector_videos_rejected_total",
    "Videos rejected during discovery.",
    labelnames=("category", "reason"),
)

dataset_collector_run_rejected_reason = Gauge(
    "dataset_collector_run_rejected_reason",
    "Videos rejected in the current campaign run, grouped by reason.",
    labelnames=("category", "reason"),
)

dataset_collector_session_rejected_reason = Gauge(
    "dataset_collector_session_rejected_reason",
    "Videos rejected in the current CLI session, grouped by reason.",
    labelnames=("category", "reason"),
)

dataset_collector_run_accepted = Gauge(
    "dataset_collector_run_accepted",
    "Accepted videos in the current campaign run (excluding baseline).",
)

dataset_collector_total_with_baseline = Gauge(
    "dataset_collector_total_with_baseline",
    "Total accepted videos including baseline from prior datasets.",
)

dataset_collector_session_accepted = Gauge(
    "dataset_collector_session_accepted",
    "Videos accepted in the current CLI session.",
)

dataset_collector_keys_available = Gauge(
    "dataset_collector_keys_available",
    "YouTube API keys with remaining daily quota.",
)

dataset_collector_quota_session_units = Gauge(
    "dataset_collector_quota_session_units",
    "YouTube API quota units consumed in the current session.",
)

dataset_collector_view_count = Histogram(
    "dataset_collector_view_count",
    "Distribution of view counts at discovery time.",
    labelnames=("category",),
    buckets=(0, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000),
)

dataset_collector_like_count = Histogram(
    "dataset_collector_like_count",
    "Distribution of like counts at discovery time.",
    labelnames=("category",),
    buckets=(0, 10, 100, 1_000, 10_000, 100_000, 1_000_000),
)

dataset_collector_comment_count = Histogram(
    "dataset_collector_comment_count",
    "Distribution of comment counts at discovery time.",
    labelnames=("category",),
    buckets=(0, 1, 10, 100, 1_000, 10_000, 100_000),
)

dataset_collector_duration_seconds = Histogram(
    "dataset_collector_duration_seconds",
    "Video duration in seconds at discovery time.",
    labelnames=("category",),
    buckets=(4, 30, 60, 120, 300, 600, 900, 1500, 3600),
)

NUMERIC_BUCKETS: dict[str, tuple[tuple[str, int | None, int | None], ...]] = {
    "view_count": (
        ("01 0-999", 0, 999),
        ("02 1k-4.9k", 1_000, 4_999),
        ("03 5k-9.9k", 5_000, 9_999),
        ("04 10k-49.9k", 10_000, 49_999),
        ("05 50k-99.9k", 50_000, 99_999),
        ("06 100k-499.9k", 100_000, 499_999),
        ("07 500k-999.9k", 500_000, 999_999),
        ("08 1m-4.9m", 1_000_000, 4_999_999),
        ("09 5m-9.9m", 5_000_000, 9_999_999),
        ("10 10m+", 10_000_000, None),
    ),
    "like_count": (
        ("01 0-9", 0, 9),
        ("02 10-49", 10, 49),
        ("03 50-99", 50, 99),
        ("04 100-499", 100, 499),
        ("05 500-999", 500, 999),
        ("06 1k-4.9k", 1_000, 4_999),
        ("07 5k-9.9k", 5_000, 9_999),
        ("08 10k-49.9k", 10_000, 49_999),
        ("09 50k-99.9k", 50_000, 99_999),
        ("10 100k+", 100_000, None),
    ),
    "comment_count": (
        ("01 0", 0, 0),
        ("02 1-4", 1, 4),
        ("03 5-9", 5, 9),
        ("04 10-49", 10, 49),
        ("05 50-99", 50, 99),
        ("06 100-499", 100, 499),
        ("07 500-999", 500, 999),
        ("08 1k-4.9k", 1_000, 4_999),
        ("09 5k-9.9k", 5_000, 9_999),
        ("10 10k+", 10_000, None),
    ),
    "duration_seconds": (
        ("01 0-29s", 0, 29),
        ("02 30-59s", 30, 59),
        ("03 1-1.9m", 60, 119),
        ("04 2-4.9m", 120, 299),
        ("05 5-9.9m", 300, 599),
        ("06 10-14.9m", 600, 899),
        ("07 15-24.9m", 900, 1_499),
        ("08 25-39.9m", 1_500, 2_399),
        ("09 40-59.9m", 2_400, 3_599),
        ("10 60m+", 3_600, None),
    ),
}

dataset_collector_run_numeric_bucket = Gauge(
    "dataset_collector_run_numeric_bucket",
    "Run-wide non-cumulative bucket counts for numeric video fields.",
    labelnames=("category", "field", "bucket"),
)

dataset_collector_run_field_value = Gauge(
    "dataset_collector_run_field_value",
    "Run-wide counts for categorical video fields.",
    labelnames=("category", "field", "value"),
)

dataset_collector_average_video_collect_seconds = Gauge(
    "dataset_collector_average_video_collect_seconds",
    "Average seconds spent collecting data for a video.",
    labelnames=("service",),
)

dataset_collector_average_video_size_mb = Gauge(
    "dataset_collector_average_video_size_mb",
    "Average downloaded local video file size in MiB.",
)

dataset_collector_average_download_seconds = Gauge(
    "dataset_collector_average_download_seconds",
    "Average seconds spent downloading a video.",
)

dataset_collector_average_quota_units_per_video = Gauge(
    "dataset_collector_average_quota_units_per_video",
    "Average YouTube API quota units spent per accepted video in the current session.",
)

dataset_collector_enrich_on_hf = Gauge(
    "dataset_collector_enrich_on_hf",
    "Enrich JSON records uploaded to Hugging Face.",
    labelnames=("category",),
)

dataset_collector_balancer_decisions_total = Counter(
    "dataset_collector_balancer_decisions_total",
    "Dataset balancer decisions by category and reason.",
    labelnames=("category", "action", "reason"),
)

dataset_collector_balancer_score = Gauge(
    "dataset_collector_balancer_score",
    "Latest dataset balancer score by category.",
    labelnames=("category",),
)

dataset_collector_balancer_field_score = Gauge(
    "dataset_collector_balancer_field_score",
    "Latest dataset balancer field score by category and field.",
    labelnames=("category", "field"),
)

dataset_collector_balancer_bucket_fill_ratio = Gauge(
    "dataset_collector_balancer_bucket_fill_ratio",
    "Current balancer bucket fill ratio by category, field, and value.",
    labelnames=("category", "field", "value"),
)


def _int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bucket_for(field: str, value: int) -> str | None:
    for label, lower, upper in NUMERIC_BUCKETS[field]:
        if lower is not None and value < lower:
            continue
        if upper is not None and value > upper:
            continue
        return label
    return None


def _best_thumbnail_resolution(metadata: dict) -> str:
    thumbnails = metadata.get("thumbnails") or {}
    best_width = 0
    best_height = 0
    if isinstance(thumbnails, dict):
        for item in thumbnails.values():
            if not isinstance(item, dict):
                continue
            width = _int(item.get("width")) or 0
            height = _int(item.get("height")) or 0
            if width * height > best_width * best_height:
                best_width = width
                best_height = height
    if best_width and best_height:
        return f"{best_width}x{best_height}"
    return "unknown"


def _best_video_resolution_from_formats(formats: object) -> str:
    best_width = 0
    best_height = 0
    if isinstance(formats, list):
        for item in formats:
            if not isinstance(item, dict):
                continue
            width = _int(item.get("width")) or 0
            height = _int(item.get("height")) or 0
            if (not width or not height) and isinstance(item.get("resolution"), str):
                parts = item["resolution"].lower().split("x", 1)
                if len(parts) == 2:
                    width = _int(parts[0]) or 0
                    height = _int(parts[1]) or 0
            if width * height > best_width * best_height:
                best_width = width
                best_height = height
    if best_width and best_height:
        return f"{best_width}x{best_height}"
    return "unknown"


def _caption_languages(payload: object) -> set[str]:
    languages: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if value:
                languages.add(str(key))
    return languages


def _iter_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _iter_categories(category: str) -> tuple[str, str]:
    return (category or "unknown", "all")


def record_run_distribution(video: dict, *, category: str) -> None:
    snapshot = video.get("snapshot_0") or {}
    metadata = video.get("metadata") or {}
    numeric_values = {
        "view_count": _int(snapshot.get("viewCount")) or _int(metadata.get("view_count")),
        "like_count": _int(snapshot.get("likeCount")) or _int(metadata.get("like_count")),
        "comment_count": _int(snapshot.get("commentCount")) or _int(metadata.get("comment_count")),
        "duration_seconds": _int(metadata.get("duration_seconds")) or _int(metadata.get("duration")),
    }
    categorical_values = {
        "category": category or "unknown",
        "language": str(metadata.get("language") or "unknown"),
        "country": str(metadata.get("country") or "unknown"),
        "made_for_kids": str(bool(metadata.get("madeForKids"))).lower(),
        "time_interval": str(video.get("time_interval") or "unknown"),
        "thumbnail_resolution": _best_thumbnail_resolution(metadata),
    }
    comments = snapshot.get("comments")
    if isinstance(comments, list):
        categorical_values["comments_available"] = "yes" if comments else "no"

    for cat in _iter_categories(category):
        for field, value in numeric_values.items():
            if value is None:
                continue
            bucket = _bucket_for(field, value)
            if bucket:
                dataset_collector_run_numeric_bucket.labels(
                    category=cat,
                    field=field,
                    bucket=bucket,
                ).inc()
        for field, value in categorical_values.items():
            dataset_collector_run_field_value.labels(
                category=cat,
                field=field,
                value=value,
            ).inc()


def observe_video_metrics(video: dict, *, category: str) -> None:
    snapshot = video.get("snapshot_0") or {}
    metadata = video.get("metadata") or {}

    views = _int(snapshot.get("viewCount")) or _int(metadata.get("view_count"))
    likes = _int(snapshot.get("likeCount")) or _int(metadata.get("like_count"))
    comments = _int(snapshot.get("commentCount")) or _int(metadata.get("comment_count"))
    duration = _int(metadata.get("duration_seconds"))

    if views is not None:
        dataset_collector_view_count.labels(category=category).observe(views)
    if likes is not None:
        dataset_collector_like_count.labels(category=category).observe(likes)
    if comments is not None:
        dataset_collector_comment_count.labels(category=category).observe(comments)
    if duration is not None:
        dataset_collector_duration_seconds.labels(category=category).observe(duration)
    record_run_distribution(video, category=category)


def record_reject_reason(category: str, reason: str) -> None:
    reason = reason or "unknown"
    for cat in _iter_categories(category):
        dataset_collector_run_rejected_reason.labels(category=cat, reason=reason).inc()
        dataset_collector_session_rejected_reason.labels(category=cat, reason=reason).inc()


def _category_from_metadata_shard(path: Path) -> str:
    for part in path.parts:
        if part.startswith("category="):
            return part.split("=", 1)[1] or "unknown"
    return "unknown"


def update_run_distribution_gauges(run_root: str | Path) -> None:
    root = Path(run_root)
    numeric_counts: CollectionsCounter[tuple[str, str, str]] = CollectionsCounter()
    field_counts: CollectionsCounter[tuple[str, str, str]] = CollectionsCounter()
    rejected_counts: CollectionsCounter[tuple[str, str]] = CollectionsCounter()
    local_enrich_counts: CollectionsCounter[str] = CollectionsCounter()
    local_video_sizes_mb: list[float] = []
    download_seconds: list[float] = []
    enrich_seconds: list[float] = []

    metadata_dir = root / "shards" / "metadata"
    if metadata_dir.exists():
        from fetcher.dataset_collector.training_format import iter_metadata_shard_entries

        for shard_path in metadata_dir.glob("**/part_*.json"):
            category = _category_from_metadata_shard(shard_path)
            data = json.loads(shard_path.read_text(encoding="utf-8"))
            for record in iter_metadata_shard_entries(data):
                snapshot = record.get("snapshot_0") or {}
                metadata = record.get("metadata") or {}
                numeric_values = {
                    "view_count": _int(snapshot.get("viewCount")) or _int(metadata.get("view_count")),
                    "like_count": _int(snapshot.get("likeCount")) or _int(metadata.get("like_count")),
                    "comment_count": _int(snapshot.get("commentCount")) or _int(metadata.get("comment_count")),
                    "duration_seconds": _int(metadata.get("duration_seconds")) or _int(metadata.get("duration")),
                }
                categorical_values = {
                    "category": category or "unknown",
                    "language": str(metadata.get("language") or "unknown"),
                    "country": str(metadata.get("country") or "unknown"),
                    "made_for_kids": str(bool(metadata.get("madeForKids"))).lower(),
                    "time_interval": str(record.get("time_interval") or "unknown"),
                    "thumbnail_resolution": _best_thumbnail_resolution(metadata),
                }
                comments = snapshot.get("comments")
                if isinstance(comments, list):
                    categorical_values["comments_available"] = "yes" if comments else "no"
                for cat in _iter_categories(category):
                    for field, value in numeric_values.items():
                        if value is None:
                            continue
                        bucket = _bucket_for(field, value)
                        if bucket:
                            numeric_counts[(cat, field, bucket)] += 1
                    for field, value in categorical_values.items():
                        field_counts[(cat, field, value)] += 1

    enrich_dir = root / "shards" / "enrich"
    if enrich_dir.exists():
        for shard_path in enrich_dir.glob("**/part_*.json"):
            category = _category_from_metadata_shard(shard_path)
            try:
                data = json.loads(shard_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for payload in data.values():
                if not isinstance(payload, dict):
                    continue
                for cat in _iter_categories(category):
                    local_enrich_counts[cat] += 1
                resolution = _best_video_resolution_from_formats(payload.get("formats"))
                languages = _caption_languages(payload.get("subtitles")) | _caption_languages(
                    payload.get("automatic_captions")
                )
                for cat in _iter_categories(category):
                    field_counts[(cat, "video_resolution", resolution)] += 1
                    field_counts[
                        (cat, "subtitle_presence", "with_subtitles" if languages else "without_subtitles")
                    ] += 1
                    for language in languages:
                        field_counts[(cat, "subtitle_language", language)] += 1

    rejected_dir = root / "rejected"
    if rejected_dir.exists():
        for shard_path in rejected_dir.glob("part_*.json"):
            records = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for record in records:
                category = str(record.get("category") or "unknown")
                reason = str(record.get("reason") or "unknown")
                for cat in _iter_categories(category):
                    rejected_counts[(cat, reason)] += 1

    for (category, field, bucket), value in numeric_counts.items():
        dataset_collector_run_numeric_bucket.labels(
            category=category,
            field=field,
            bucket=bucket,
        ).set(value)
    for (category, field, field_value), value in field_counts.items():
        dataset_collector_run_field_value.labels(
            category=category,
            field=field,
            value=field_value,
        ).set(value)
    for (category, reason), value in rejected_counts.items():
        dataset_collector_run_rejected_reason.labels(category=category, reason=reason).set(value)
    for category, value in local_enrich_counts.items():
        dataset_collector_videos_enriched.labels(category=category).set(value)

    hf_enrich_counts: CollectionsCounter[str] = CollectionsCounter()
    for row in _iter_jsonl(root / "state" / "hf_enrich_upload_done.jsonl"):
        category = str(row.get("category") or "unknown")
        for cat in _iter_categories(category):
            hf_enrich_counts[cat] += 1
    for category, value in hf_enrich_counts.items():
        dataset_collector_enrich_on_hf.labels(category=category).set(value)

    for path in (root / "downloads" / "videos").glob("**/*.mp4"):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 0:
            local_video_sizes_mb.append(size / (1024 * 1024))

    for row in _iter_jsonl(root / "state" / "performance_events.jsonl"):
        try:
            seconds = float(row.get("seconds") or 0)
        except (TypeError, ValueError):
            continue
        if seconds <= 0:
            continue
        if row.get("event") == "download":
            download_seconds.append(seconds)
        elif row.get("event") == "enrich":
            enrich_seconds.append(seconds)

    if enrich_seconds:
        dataset_collector_average_video_collect_seconds.labels(service="enrich").set(
            sum(enrich_seconds) / len(enrich_seconds)
        )
    if local_video_sizes_mb:
        dataset_collector_average_video_size_mb.set(sum(local_video_sizes_mb) / len(local_video_sizes_mb))
    if download_seconds:
        dataset_collector_average_download_seconds.set(sum(download_seconds) / len(download_seconds))


dataset_collector_download_queue_pending = Gauge(
    "dataset_collector_download_queue_pending",
    "Videos waiting in local download queue (not yet marked done).",
    labelnames=("category",),
)

dataset_collector_videos_downloaded_local = Gauge(
    "dataset_collector_videos_downloaded_local",
    "Videos with a local mp4 on disk.",
    labelnames=("category",),
)

dataset_collector_videos_on_hf = Gauge(
    "dataset_collector_videos_on_hf",
    "Videos successfully uploaded to Hugging Face.",
    labelnames=("category",),
)

dataset_collector_hf_video_upload_queue_pending = Gauge(
    "dataset_collector_hf_video_upload_queue_pending",
    "Downloaded videos waiting for HF upload.",
    labelnames=("category",),
)

dataset_collector_shards_total = Gauge(
    "dataset_collector_shards_total",
    "Metadata shards written locally.",
    labelnames=("category",),
)

dataset_collector_shards_on_hf = Gauge(
    "dataset_collector_shards_on_hf",
    "Metadata shards uploaded to Hugging Face.",
    labelnames=("category",),
)

dataset_collector_hf_shard_upload_queue_pending = Gauge(
    "dataset_collector_hf_shard_upload_queue_pending",
    "Metadata shards waiting for HF upload.",
    labelnames=("category",),
)

dataset_collector_videos_in_shards = Gauge(
    "dataset_collector_videos_in_shards",
    "Unique videos recorded in metadata shards (inventory index).",
    labelnames=("category",),
)

dataset_collector_metadata_enrich_queue_pending = Gauge(
    "dataset_collector_metadata_enrich_queue_pending",
    "Videos waiting for yt-dlp metadata enrichment.",
    labelnames=("category",),
)

dataset_collector_videos_enriched = Gauge(
    "dataset_collector_videos_enriched",
    "Videos with yt-dlp enrichment completed.",
    labelnames=("category",),
)

dataset_collector_lifecycle_lag = Gauge(
    "dataset_collector_lifecycle_lag",
    "Videos lagging behind a lifecycle stage.",
    labelnames=("category", "stage"),
)

dataset_collector_snapshot_readiness = Gauge(
    "dataset_collector_snapshot_readiness",
    "Videos with required future snapshots completed.",
    labelnames=("category", "snapshot"),
)

dataset_collector_training_ready = Gauge(
    "dataset_collector_training_ready",
    "Videos ready for a training readiness level.",
    labelnames=("category", "level"),
)

dataset_collector_unique_channels = Gauge(
    "dataset_collector_unique_channels",
    "Unique channels represented in metadata shards.",
    labelnames=("category",),
)

dataset_collector_top_channel_share = Gauge(
    "dataset_collector_top_channel_share",
    "Share of videos from the most frequent channel.",
    labelnames=("category",),
)

dataset_collector_queue_dead_letter = Gauge(
    "dataset_collector_queue_dead_letter",
    "Queue items moved to dead letter after repeated failures.",
    labelnames=("category",),
)

dataset_collector_service_pass_total = Counter(
    "dataset_collector_service_pass_total",
    "Completed service passes by service and result.",
    labelnames=("service", "result"),
)

dataset_collector_service_items_total = Counter(
    "dataset_collector_service_items_total",
    "Items processed by long-lived service workers.",
    labelnames=("service", "status"),
)

dataset_collector_hf_commits_total = Counter(
    "dataset_collector_hf_commits_total",
    "HF commits performed by service workers.",
    labelnames=("repo_type",),
)

dataset_collector_hf_commit_files_total = Counter(
    "dataset_collector_hf_commit_files_total",
    "Files included in HF commits.",
    labelnames=("repo_type",),
)

# Multi-Colab HF coordination (coord_sync)
dataset_collector_coord_enabled = Gauge(
    "dataset_collector_coord_enabled",
    "1 when hf_coord_enabled is active on this worker.",
)

dataset_collector_coord_worker_info = Gauge(
    "dataset_collector_coord_worker_info",
    "Coordination worker identity (always 1 for this process).",
    labelnames=("worker_id", "shard_index", "shard_count"),
)

dataset_collector_coord_sync_gauge = Gauge(
    "dataset_collector_coord_sync_gauge",
    "Last HF coordination sync snapshot.",
    labelnames=("worker_id", "service", "metric"),
)

dataset_collector_coord_skip_total = Counter(
    "dataset_collector_coord_skip_total",
    "Queue items skipped by HF coordination.",
    labelnames=("worker_id", "service", "reason"),
)

dataset_collector_coord_claim_total = Counter(
    "dataset_collector_coord_claim_total",
    "HF coordination claim attempts.",
    labelnames=("worker_id", "service", "result"),
)

dataset_collector_coord_sync_errors_total = Counter(
    "dataset_collector_coord_sync_errors_total",
    "HF list_repo_files / sync failures.",
    labelnames=("worker_id", "service"),
)


def record_coord_worker_identity(config, worker_id: str) -> None:
    from fetcher.dataset_collector.hf_coordination import coord_enabled

    if not coord_enabled(config):
        dataset_collector_coord_enabled.set(0)
        return
    dataset_collector_coord_enabled.set(1)
    shard_index = config.worker_shard_index
    shard_count = config.worker_shard_count or 1
    dataset_collector_coord_worker_info.labels(
        worker_id=worker_id,
        shard_index=str(shard_index if shard_index is not None else -1),
        shard_count=str(shard_count),
    ).set(1)


def record_coord_sync(worker_id: str, service: str, stats: dict[str, int], *, active_claims: int, global_done: int) -> None:
    labels = {"worker_id": worker_id, "service": service}
    for metric, value in (
        ("claims_files", stats.get("claims_files", 0)),
        ("done_files", stats.get("done_files", 0)),
        ("metadata_shards", stats.get("metadata_shards", 0)),
        ("active_claims", active_claims),
        ("global_done", global_done),
    ):
        dataset_collector_coord_sync_gauge.labels(metric=metric, **labels).set(value)


def record_coord_skip(worker_id: str, service: str, reason: str) -> None:
    dataset_collector_coord_skip_total.labels(
        worker_id=worker_id,
        service=service,
        reason=reason,
    ).inc()


def record_coord_claim(worker_id: str, service: str, *, ok: bool) -> None:
    dataset_collector_coord_claim_total.labels(
        worker_id=worker_id,
        service=service,
        result="claimed" if ok else "busy",
    ).inc()


def record_coord_sync_error(worker_id: str, service: str) -> None:
    dataset_collector_coord_sync_errors_total.labels(worker_id=worker_id, service=service).inc()


def update_gauges(stats: dict) -> None:
    dataset_collector_run_accepted.set(stats.get("run_accepted", 0))
    dataset_collector_total_with_baseline.set(stats.get("total_with_baseline", 0))
    dataset_collector_session_accepted.set(stats.get("session_accepted", 0))
    dataset_collector_keys_available.set(stats.get("keys_available", 0))
    dataset_collector_quota_session_units.set(stats.get("session_quota_units", 0))
    accepted = int(stats.get("session_accepted") or 0)
    quota = float(stats.get("session_quota_units") or 0)
    if accepted > 0:
        dataset_collector_average_quota_units_per_video.set(quota / accepted)
        elapsed = float(stats.get("session_elapsed_seconds") or 0)
        if elapsed > 0:
            dataset_collector_average_video_collect_seconds.labels(service="discover").set(
                elapsed / accepted
            )


def update_inventory_gauges(inventory: dict) -> None:
    """Update queue/HF/shard gauges from compute_inventory_stats() output."""

    def _set_all(gauge: Gauge, field: str, *, nested: str | None = None) -> None:
        totals = inventory.get("totals") or inventory
        block = totals.get(nested) if nested else totals
        if not isinstance(block, dict):
            return
        value = block.get(field, 0)
        gauge.labels(category="all").set(value)
        for cat, cat_stats in (inventory.get("by_category") or {}).items():
            sub = cat_stats.get(nested) if nested else cat_stats
            if isinstance(sub, dict) and field in sub:
                gauge.labels(category=cat).set(sub[field])

    inv = inventory if "totals" in inventory else {"totals": inventory, "by_category": {}}
    _set_all(dataset_collector_download_queue_pending, "pending_download", nested="videos")
    _set_all(dataset_collector_videos_downloaded_local, "downloaded_local_files", nested="videos")
    _set_all(dataset_collector_videos_on_hf, "on_hf", nested="videos")
    _set_all(
        dataset_collector_hf_video_upload_queue_pending,
        "pending_hf_upload",
        nested="videos",
    )
    _set_all(dataset_collector_shards_total, "total", nested="shards")
    _set_all(dataset_collector_shards_on_hf, "on_hf", nested="shards")
    _set_all(
        dataset_collector_hf_shard_upload_queue_pending,
        "pending_hf_upload",
        nested="shards",
    )
    _set_all(dataset_collector_videos_in_shards, "in_shards", nested="videos")
    _set_all(
        dataset_collector_metadata_enrich_queue_pending,
        "pending_enrich",
        nested="videos",
    )
    _set_all(dataset_collector_videos_enriched, "enriched", nested="videos")
    _set_all(dataset_collector_enrich_on_hf, "enrich_on_hf", nested="videos")

    def _iter_categories():
        totals = inv.get("totals") or {}
        yield "all", totals
        for cat, cat_stats in (inv.get("by_category") or {}).items():
            yield cat, cat_stats

    for cat, stats in _iter_categories():
        lifecycle = stats.get("lifecycle") or {}
        for stage in ("lag_enrich", "lag_download", "lag_hf_video", "lag_hf_enrich"):
            dataset_collector_lifecycle_lag.labels(category=cat, stage=stage).set(
                lifecycle.get(stage, 0)
            )
        dataset_collector_training_ready.labels(category=cat, level="snapshot0").set(
            lifecycle.get("training_ready_snapshot0", 0)
        )
        dataset_collector_training_ready.labels(category=cat, level="14_21").set(
            lifecycle.get("training_ready_14_21", 0)
        )

        snapshots = stats.get("snapshots") or {}
        for name in ("snapshot_7d", "snapshot_14d", "snapshot_21d", "snapshot_28d"):
            dataset_collector_snapshot_readiness.labels(category=cat, snapshot=name).set(
                snapshots.get(name, 0)
            )

        channels = stats.get("channels") or {}
        dataset_collector_unique_channels.labels(category=cat).set(channels.get("unique", 0))
        dataset_collector_top_channel_share.labels(category=cat).set(
            channels.get("top_share", 0.0)
        )
        queues = stats.get("queues") or {}
        dataset_collector_queue_dead_letter.labels(category=cat).set(
            queues.get("dead_letter", 0)
        )


def start_metrics_server(port: int) -> None:
    start_http_server(port)


def record_service_pass(service: str, result: dict) -> None:
    status = "error" if result.get("error") else "ok"
    dataset_collector_service_pass_total.labels(service=service, result=status).inc()
    for key in ("attempted", "downloaded", "uploaded", "enriched", "rejected", "failed", "skipped"):
        value = int(result.get(key) or 0)
        if value:
            dataset_collector_service_items_total.labels(service=service, status=key).inc(value)


def record_hf_commit(repo_type: str, files: int) -> None:
    dataset_collector_hf_commits_total.labels(repo_type=repo_type).inc()
    dataset_collector_hf_commit_files_total.labels(repo_type=repo_type).inc(files)


def record_balancer_decision(
    *,
    category: str,
    accepted: bool,
    reason: str,
    score: float,
    field_scores: dict[str, float],
    field_values: dict[str, str],
    fill_ratios: dict[tuple[str, str], float],
) -> None:
    action = "accepted" if accepted else "rejected"
    dataset_collector_balancer_decisions_total.labels(
        category=category,
        action=action,
        reason=reason or "accepted",
    ).inc()
    dataset_collector_balancer_score.labels(category=category).set(score)
    for field, field_score in field_scores.items():
        dataset_collector_balancer_field_score.labels(category=category, field=field).set(field_score)
    for field, value in field_values.items():
        dataset_collector_balancer_bucket_fill_ratio.labels(
            category=category,
            field=field,
            value=value,
        ).set(fill_ratios.get((field, value), 0.0))
