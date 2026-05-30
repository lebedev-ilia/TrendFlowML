from __future__ import annotations

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


def observe_video_metrics(video: dict, *, category: str) -> None:
    snapshot = video.get("snapshot_0") or {}
    metadata = video.get("metadata") or {}

    def _int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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


def update_gauges(stats: dict) -> None:
    dataset_collector_run_accepted.set(stats.get("run_accepted", 0))
    dataset_collector_total_with_baseline.set(stats.get("total_with_baseline", 0))
    dataset_collector_session_accepted.set(stats.get("session_accepted", 0))
    dataset_collector_keys_available.set(stats.get("keys_available", 0))
    dataset_collector_quota_session_units.set(stats.get("session_quota_units", 0))


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


def start_metrics_server(port: int) -> None:
    start_http_server(port)
