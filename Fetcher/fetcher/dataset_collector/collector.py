from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from fetcher.dataset_collector.age_buckets import allocate_counts, bucket_from_config
from fetcher.dataset_collector.config import merged_filters
from fetcher.dataset_collector.discovery.base import DiscoveryAdapter
from fetcher.dataset_collector.filters import VideoFilter
from fetcher.dataset_collector.hf_upload import maybe_upload_recent_shards
from fetcher.dataset_collector.schemas import CampaignConfig, CollectedVideo, RejectedRecord
from fetcher.dataset_collector.snapshots import build_schedule_entry
from fetcher.dataset_collector.state import DatasetState, jsonable, utcnow


class DatasetCollector:
    def __init__(
        self,
        config: CampaignConfig,
        state: DatasetState,
        adapters: Dict[str, DiscoveryAdapter],
    ) -> None:
        self.config = config
        self.state = state
        self.adapters = adapters

    def discover_category(self, category_name: str, *, limit: int | None = None) -> dict[str, int]:
        category = next(item for item in self.config.categories if item.name == category_name)
        target = min(limit or category.collect_count, category.collect_count)
        accepted: List[CollectedVideo] = []
        rejected: List[RejectedRecord] = []
        video_filter = VideoFilter(merged_filters(self.config, category))
        buckets = [bucket_from_config(raw) for raw in self.config.time_interval_buckets]
        bucket_targets = allocate_counts(buckets, target) if buckets else {None: target}

        for bucket in buckets or [None]:
            bucket_name = bucket.name if bucket else None
            bucket_target = bucket_targets.get(bucket_name, target)
            if bucket_target <= 0:
                continue
            published_after, published_before = bucket.published_window() if bucket else (None, None)
            accepted_before_bucket = len(accepted)
            for platform, platform_limit in self._platform_limits(category.platform_weights, bucket_target):
                adapter = self.adapters.get(platform)
                if adapter is None or not adapter.capabilities.search:
                    continue
                for keyword in category.keywords:
                    platform_accepted = len(
                        [
                            item
                            for item in accepted
                            if item.platform == platform and item.time_interval == bucket_name
                        ]
                    )
                    remaining = platform_limit - platform_accepted
                    if remaining <= 0:
                        break
                    for video in adapter.discover(
                        category=category.name,
                        query=keyword,
                        limit=remaining,
                        published_after=published_after,
                        published_before=published_before,
                        time_interval=bucket_name,
                    ):
                        if self.state.is_seen(video.dedup_key):
                            rejected.append(self._reject(video, "duplicate_seen"))
                            continue
                        decision = video_filter.decide(jsonable(video.dict()))
                        if not decision.accepted:
                            rejected.append(self._reject(video, decision.reason))
                            continue
                        self._attach_initial_comments(adapter, video)
                        video_filter.accept(jsonable(video.dict()))
                        self.state.mark_seen(video.dedup_key, category=category.name)
                        self.state.append_schedule(build_schedule_entry(video, self.config.snapshot_schedule_days))
                        self.state.enqueue_download(video)
                        accepted.append(video)
                        if len(accepted) - accepted_before_bucket >= bucket_target or len(accepted) >= target:
                            break
                    if len(accepted) - accepted_before_bucket >= bucket_target or len(accepted) >= target:
                        break
                if len(accepted) - accepted_before_bucket >= bucket_target or len(accepted) >= target:
                    break
            if len(accepted) >= target:
                break

        written_shards = []
        for chunk_start in range(0, len(accepted), self.config.shard_size):
            written_shards.append(
                self.state.write_metadata_shard(
                    category.name,
                    accepted[chunk_start : chunk_start + self.config.shard_size],
                )
            )
        self.state.write_rejected(rejected)
        maybe_upload_recent_shards(self.config, self.state.root, written_shards)
        return {"accepted": len(accepted), "rejected": len(rejected)}

    @staticmethod
    def _platform_limits(weights: Dict[str, float], target: int) -> Iterable[Tuple[str, int]]:
        total_weight = sum(max(weight, 0) for weight in weights.values()) or 1.0
        for platform, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
            yield platform, max(1, int(target * max(weight, 0) / total_weight))

    def _attach_initial_comments(self, adapter: DiscoveryAdapter, video: CollectedVideo) -> None:
        if video.snapshot_0.comments or not adapter.capabilities.comments:
            return
        try:
            if int(video.snapshot_0.commentCount or 0) <= 0:
                return
        except (TypeError, ValueError):
            pass
        collect_comments = getattr(adapter, "collect_comments", None)
        if collect_comments is None:
            return
        try:
            video.snapshot_0.comments = collect_comments(
                video.video_id,
                comments_limit=self.config.comments_per_snapshot,
            )
        except Exception as exc:
            # Комментарии не должны блокировать discovery, но причина нужна для диагностики.
            video.snapshot_0.raw["comments_error"] = str(exc)[:500]
            video.snapshot_0.comments = []

    @staticmethod
    def _reject(video: CollectedVideo, reason: str) -> RejectedRecord:
        return RejectedRecord(
            platform=video.platform,
            video_id=video.video_id,
            category=video.category,
            query=video.query,
            reason=reason,
            record=jsonable(video.dict()),
            rejected_at=utcnow(),
        )
