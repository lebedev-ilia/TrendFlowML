from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

"""Prometheus-метрики Fetcher.

Имена и семантика метрик соответствуют `Fetcher/docs/FETCHER_OBSERVABILITY.md`.
"""


fetcher_videos_downloaded_total = Counter(
    "fetcher_videos_downloaded_total",
    "Total number of successfully downloaded videos.",
    labelnames=("platform",),
)

fetcher_videos_failed_total = Counter(
    "fetcher_videos_failed_total",
    "Total number of failed video ingestions.",
    labelnames=("platform", "reason"),
)

fetcher_cache_hits_total = Counter(
    "fetcher_cache_hits_total",
    "Total number of cache hits for video/artifacts.",
    labelnames=("platform",),
)

fetcher_cache_miss_total = Counter(
    "fetcher_cache_miss_total",
    "Total number of cache misses for video/artifacts.",
    labelnames=("platform",),
)

fetcher_download_latency_seconds = Histogram(
    "fetcher_download_latency_seconds",
    "Video download + upload latency in seconds.",
    labelnames=("platform",),
    buckets=(5, 10, 30, 60, 120, 300, 600),
)

fetcher_metadata_latency_seconds = Histogram(
    "fetcher_metadata_latency_seconds",
    "Metadata worker latency in seconds.",
    labelnames=("platform",),
    buckets=(1, 2, 5, 10, 30, 60),
)

fetcher_comments_latency_seconds = Histogram(
    "fetcher_comments_latency_seconds",
    "Comments worker latency in seconds.",
    labelnames=("platform",),
    buckets=(1, 2, 5, 10, 30, 60),
)

fetcher_youtube_429_total = Counter(
    "fetcher_youtube_429_total",
    "Total number of 429 (rate limit) errors from YouTube.",
    labelnames=("operation",),
)

fetcher_youtube_403_total = Counter(
    "fetcher_youtube_403_total",
    "Total number of 403 errors from YouTube.",
    labelnames=("operation", "error_code"),
)

fetcher_provider_fallback_total = Counter(
    "fetcher_provider_fallback_total",
    "Total number of API to SDK fallbacks per platform.",
    labelnames=("platform", "from_provider", "to_provider"),
)

# Proxy / circuit breaker metrics (Phase 2+)
proxy_failure_rate = Gauge(
    "proxy_failure_rate",
    "Fraction of failed requests per proxy.",
    labelnames=("proxy_id", "country"),
)

circuit_breaker_tripped_total = Counter(
    "circuit_breaker_tripped_total",
    "Total number of circuit breaker trips.",
    labelnames=("operation", "reason"),
)

# Backpressure metrics
fetcher_backpressure_detected_total = Counter(
    "fetcher_backpressure_detected_total",
    "Total number of times backpressure was detected from DataProcessor",
)
fetcher_backpressure_check_errors_total = Counter(
    "fetcher_backpressure_check_errors_total",
    "Total number of errors when checking DataProcessor queue size",
    labelnames=["error_type"],
)
fetcher_processor_queue_size = Gauge(
    "fetcher_processor_queue_size",
    "Current size of DataProcessor queue",
)

# Celery queue depth (для GET /api/v1/queue и наблюдаемости)
fetcher_celery_queue_pending = Gauge(
    "fetcher_celery_queue_pending",
    "Number of tasks pending in Celery queue (from Redis LLEN).",
    labelnames=("queue",),
)

# Kafka consumer lag (если Kafka включена; заполняется при наличии consumer group)
fetcher_kafka_consumer_lag = Gauge(
    "fetcher_kafka_consumer_lag",
    "Kafka consumer lag (messages behind) per topic/partition. 0 if Kafka disabled.",
    labelnames=("topic", "partition", "group"),
)


# ============================================================================
# Вспомогательные функции для экспорта метрик
# ============================================================================


def get_metrics() -> bytes:
    """Получить метрики в формате Prometheus.

    Returns:
        Байты с метриками в формате Prometheus text format.
    """
    return generate_latest()


def get_metrics_content_type() -> str:
    """Получить Content-Type для метрик.

    Returns:
        Content-Type для Prometheus метрик (text/plain; version=0.0.4; charset=utf-8).
    """
    return CONTENT_TYPE_LATEST


def get_cache_hit_totals() -> tuple[float, float]:
    """Текущие суммарные значения счётчиков cache hit/miss (для stats).

    Returns:
        (hits_total, miss_total) — сумма по всем лейблам.
    """
    hits_total = 0.0
    miss_total = 0.0
    for m in fetcher_cache_hits_total.collect():
        for s in m.samples:
            hits_total += s.value
    for m in fetcher_cache_miss_total.collect():
        for s in m.samples:
            miss_total += s.value
    return hits_total, miss_total


__all__ = [
    "fetcher_videos_downloaded_total",
    "fetcher_videos_failed_total",
    "fetcher_cache_hits_total",
    "fetcher_cache_miss_total",
    "fetcher_download_latency_seconds",
    "fetcher_metadata_latency_seconds",
    "fetcher_comments_latency_seconds",
    "fetcher_youtube_429_total",
    "fetcher_youtube_403_total",
    "fetcher_provider_fallback_total",
    "proxy_failure_rate",
    "circuit_breaker_tripped_total",
    "fetcher_backpressure_detected_total",
    "fetcher_backpressure_check_errors_total",
    "fetcher_processor_queue_size",
    "fetcher_celery_queue_pending",
    "fetcher_kafka_consumer_lag",
    "get_metrics",
    "get_metrics_content_type",
    "get_cache_hit_totals",
]


