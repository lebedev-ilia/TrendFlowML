from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from fetcher.dataset_collector.collector import DatasetCollector
from fetcher.dataset_collector.age_buckets import allocate_counts, bucket_from_config
from fetcher.dataset_collector.balancer import DatasetBalancer
from fetcher.dataset_collector.config import default_campaign_config
from fetcher.dataset_collector.cookies import CookieRotator, apply_cookiefile
from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.export import export_legacy_json, validate_export
from fetcher.dataset_collector.hf_queues import run_hf_shard_upload_queue
from fetcher.dataset_collector.training_format import (
    compact_training_metadata,
    format_training_shard,
    merge_ytdlp_into_training_metadata,
    metadata_captions_are_bloated,
    slim_caption_tracks,
    training_entry_needs_ytdlp_enrichment,
)
from fetcher.dataset_collector.filters import VideoFilter
from fetcher.dataset_collector.proxy import (
    ProxyRotator,
    is_proxy_transport_error,
    load_proxy_file,
    normalize_proxy_url,
)
from fetcher.dataset_collector.schemas import (
    BalancerConfig,
    BalancerFieldConfig,
    CollectedVideo,
    ScheduleEntry,
    Snapshot,
)
from fetcher.dataset_collector.snapshots import SnapshotRunner, build_schedule_entry
from fetcher.dataset_collector.state import DatasetState, format_time_get, utcnow


class FakeAdapter:
    platform = "youtube"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=True,
        downloads=False,
    )

    def __init__(self, videos):
        self.videos = videos

    def discover(
        self,
        *,
        category: str,
        query: str,
        limit: int,
        published_after=None,
        published_before=None,
        time_interval=None,
    ):
        videos = []
        for video in self.videos[:limit]:
            videos.append(video.copy(update={"time_interval": time_interval}))
        return videos

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int):
        now = utcnow()
        return Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount="20",
            likeCount="2",
            commentCount="1",
        )


def make_video(video_id: str, *, views: int = 10) -> CollectedVideo:
    now = utcnow()
    return CollectedVideo(
        platform="youtube",
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        category="sports",
        query="sports",
        metadata={
            "title": f"Video {video_id}",
            "duration_seconds": 120,
            "view_count": views,
            "channel_id": "channel-1",
        },
        snapshot_0=Snapshot(
            snapshot_index=0,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str(views),
            likeCount="1",
            commentCount="0",
        ),
        discovered_at=now,
    )


@pytest.mark.unit
def test_dataset_balancer_rejects_overfilled_bucket(tmp_path):
    balancer = DatasetBalancer(
        BalancerConfig(
            enabled=True,
            mode="hard",
            min_accept_score=0.35,
            fields={
                "duration_seconds": BalancerFieldConfig(
                    coefficient=1.0,
                    buckets=[[0, 59], [60, 300]],
                    targets="uniform",
                )
            },
        ),
        state_root=tmp_path,
    )
    first = make_video("a").copy(update={"metadata": {"duration_seconds": 30}})
    second = make_video("b").copy(update={"metadata": {"duration_seconds": 35}})

    first_decision = balancer.decide(first)
    assert first_decision.accepted
    balancer.observe_accept(first)

    second_decision = balancer.decide(second)
    assert not second_decision.accepted
    assert second_decision.reason == "balancer_duration_seconds"


@pytest.mark.unit
def test_dataset_balancer_respects_unknown_cap(tmp_path):
    balancer = DatasetBalancer(
        BalancerConfig(
            enabled=True,
            mode="hard",
            fields={
                "language": BalancerFieldConfig(
                    coefficient=1.0,
                    unknown_policy="separate_cap",
                    unknown_max_share=0.1,
                )
            },
        ),
        state_root=tmp_path,
    )
    first = make_video("a").copy(update={"metadata": {"duration_seconds": 120}})
    second = make_video("b").copy(update={"metadata": {"duration_seconds": 120}})

    assert balancer.decide(first).accepted
    balancer.observe_accept(first)

    decision = balancer.decide(second)
    assert not decision.accepted
    assert decision.reason == "balancer_language"


@pytest.mark.unit
def test_dataset_balancer_coefficient_zero_disables_field(tmp_path):
    balancer = DatasetBalancer(
        BalancerConfig(
            enabled=True,
            mode="hard",
            fields={
                "language": BalancerFieldConfig(
                    coefficient=0.0,
                    unknown_policy="separate_cap",
                    unknown_max_share=0.0,
                )
            },
        ),
        state_root=tmp_path,
    )

    assert balancer.decide(make_video("a")).accepted
    assert balancer.enabled_field_names() == []


@pytest.mark.unit
def test_collector_deduplicates_and_writes_rejected(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.time_interval_buckets = []
    config.categories[0].keywords = ["sports"]
    config.categories[0].collect_count = 3
    config.categories[0].platform_weights = {"youtube": 1.0}
    state = DatasetState(config)
    state.initialize()

    collector = DatasetCollector(
        config,
        state,
        {"youtube": FakeAdapter([make_video("a"), make_video("a"), make_video("b", views=200_000_000)])},
    )
    result = collector.discover_category("sports", limit=3)

    assert result == {"accepted": 1, "rejected": 2}
    assert state.is_seen("youtube:a")
    assert not state.is_seen("youtube:b")
    assert state.load_manifest().counters["accepted"] == 1
    assert state.load_manifest().counters["rejected"] == 2


@pytest.mark.unit
def test_collector_applies_dataset_balancer_before_accept(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.time_interval_buckets = []
    config.categories[0].keywords = ["sports"]
    config.categories[0].collect_count = 3
    config.categories[0].platform_weights = {"youtube": 1.0}
    config.balancer_config = BalancerConfig(
        enabled=True,
        mode="hard",
        fields={
            "duration_seconds": BalancerFieldConfig(
                coefficient=1.0,
                buckets=[[0, 59], [60, 300]],
                targets="uniform",
            )
        },
    )
    state = DatasetState(config)
    state.initialize()
    videos = [
        make_video("a").copy(update={"metadata": {"duration_seconds": 30}}),
        make_video("b").copy(update={"metadata": {"duration_seconds": 35}}),
        make_video("c").copy(update={"metadata": {"duration_seconds": 120}}),
    ]

    collector = DatasetCollector(config, state, {"youtube": FakeAdapter(videos)})
    result = collector.discover_category("sports", limit=3)

    assert result == {"accepted": 2, "rejected": 1}
    assert state.is_seen("youtube:a")
    assert not state.is_seen("youtube:b")
    rejected = json.loads(next(state.rejected_dir.glob("part_*.json")).read_text(encoding="utf-8"))
    assert rejected[0]["reason"] == "balancer_duration_seconds"


@pytest.mark.unit
def test_snapshot_runner_collects_due_once(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    state = DatasetState(config)
    state.initialize()
    video = make_video("a")
    state.append_schedule(
        ScheduleEntry(
            platform="youtube",
            video_id="a",
            category="sports",
            url=video.url,
            baseline_collected_at=utcnow() - timedelta(days=8),
            due_at={"1": utcnow() - timedelta(days=1)},
        )
    )

    runner = SnapshotRunner(state, {"youtube": FakeAdapter([video])}, comments_limit=10)
    first = runner.collect_due(snapshot_index=1)
    second = runner.collect_due(snapshot_index=1)

    assert list(first) == ["youtube:a"]
    assert second == {}


@pytest.mark.unit
def test_merge_ytdlp_into_training_metadata():
    metadata = {"title": "t", "formats": [], "thumbnails_ytdlp": []}
    info = {
        "tags": ["a", "b"],
        "duration": 120,
        "formats": [{"fps": 30, "resolution": "1920x1080", "vcodec": "avc1"}],
        "thumbnails": [{"url": "https://example.com/t.jpg", "width": 1280, "height": 720}],
        "subtitles": {"en": [{"ext": "vtt"}]},
        "automatic_captions": {"ru": [{"ext": "vtt"}]},
        "_caption_texts_manual": {"en": {"language": "en", "ext": "vtt", "text": "hello"}},
        "_caption_texts_auto": {"ru": {"language": "ru", "ext": "vtt", "text": "привет"}},
    }
    merged = merge_ytdlp_into_training_metadata(metadata, info)
    assert merged["tags"] == ["a", "b"]
    assert merged["duration_seconds"] == 120
    assert len(merged["formats"]) == 1
    assert len(merged["thumbnails_ytdlp"]) == 1
    assert merged["subtitles"] == {"en": {"language": "en", "ext": "vtt", "text": "hello"}}
    assert merged["automatic_captions"] == {"ru": {"language": "ru", "ext": "vtt", "text": "привет"}}
    assert "url" not in json.dumps(merged["automatic_captions"])


@pytest.mark.unit
def test_training_entry_needs_ytdlp_enrichment():
    assert training_entry_needs_ytdlp_enrichment({"metadata": {"formats": [], "thumbnails_ytdlp": []}})
    assert not training_entry_needs_ytdlp_enrichment(
        {
            "metadata": {
                "formats": [{"fps": 30}],
                "subtitles": {"en": {"ext": "vtt", "text": "hi"}},
            },
            "_enriched": {"source": "yt_dlp"},
        }
    )
    assert training_entry_needs_ytdlp_enrichment(
        {
            "metadata": {"formats": [{"fps": 30}], "subtitles": {"en": [{"ext": "vtt"}]}},
            "_enriched": {"source": "yt_dlp"},
        }
    )
    assert not training_entry_needs_ytdlp_enrichment(
        {"metadata": {"formats": []}, "_enriched": {"source": "yt_dlp"}}
    )


@pytest.mark.unit
def test_metadata_shard_matches_training_json_shape(tmp_path):
    video = make_video("abc123", views=500)
    video.snapshot_0.comments = [
        {
            "text": "hi",
            "likeCount": 1,
            "repliesCount": 0,
            "publishedAt": "2025-01-01T00:00:00Z",
            "authorName": "@user",
        }
    ]
    shard = format_training_shard([video])
    assert list(shard.keys()) == ["abc123"]
    entry = shard["abc123"]
    assert set(entry.keys()) == {"query", "time_interval", "metadata", "snapshot_0"}
    assert entry["query"] == "sports"
    assert "platform" not in entry
    assert set(entry["metadata"].keys()) >= {
        "title",
        "description",
        "tags",
        "duration_seconds",
        "thumbnails",
    }
    assert "raw" not in entry["metadata"]
    assert entry["snapshot_0"]["comments"][0]["repliesCount"] == 0
    assert "snapshot_index" not in entry["snapshot_0"]


@pytest.mark.unit
def test_write_metadata_shard_enqueues_hf_shard_upload(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    state = DatasetState(config)
    state.initialize()
    state.write_metadata_shard("sports", [make_video("vid1")])
    lines = state.hf_shard_upload_queue_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "shards/metadata/" in row["shard"]
    assert row["category"] == "sports"


@pytest.mark.unit
def test_write_metadata_shard_enqueues_enrichment(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    state = DatasetState(config)
    state.initialize()
    state.write_metadata_shard("sports", [make_video("vid1")])
    lines = state.metadata_enrich_queue_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["video_id"] == "vid1"
    assert row["shard"].startswith("shards/metadata/")


@pytest.mark.unit
def test_export_legacy_json_splits_records(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path / "run"), categories=["sports"])
    state = DatasetState(config)
    state.initialize()
    state.write_metadata_shard("sports", [make_video("a"), make_video("b")])

    export_dir = tmp_path / "export"
    result = export_legacy_json(config.output_dir, export_dir, split_count=2)

    assert result == {"records": 2, "files": 2}
    assert (export_dir / "data_00.json").exists()
    assert validate_export(config.output_dir)["complete"] == 2


@pytest.mark.unit
def test_time_interval_buckets_allocate_recent_weight():
    config = default_campaign_config(output_dir="/tmp/unused", categories=["sports"])
    buckets = [bucket_from_config(raw) for raw in config.time_interval_buckets]
    allocation = allocate_counts(buckets, 100)

    assert allocation["lt_1d"] == 20
    assert allocation["1d_1w"] == 20
    assert allocation["gt_3y"] == 4


@pytest.mark.unit
def test_collector_sets_time_interval_from_bucket(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.categories[0].keywords = ["sports"]
    config.categories[0].collect_count = 1
    config.categories[0].platform_weights = {"youtube": 1.0}
    config.time_interval_buckets = [
        {"name": "lt_1d", "min_age_days": 0, "max_age_days": 1, "weight": 1.0}
    ]
    state = DatasetState(config)
    state.initialize()

    collector = DatasetCollector(config, state, {"youtube": FakeAdapter([make_video("fresh")])})
    result = collector.discover_category("sports", limit=1)

    assert result["accepted"] == 1
    shard = next((tmp_path / "shards" / "metadata").glob("**/*.json"))
    text = shard.read_text(encoding="utf-8")
    data = json.loads(text)
    assert isinstance(data, dict)
    assert len(data) == 1
    entry = next(iter(data.values()))
    assert entry["time_interval"] == "less-1day"
    assert "platform" not in entry
    assert "metadata" in entry and "snapshot_0" in entry


@pytest.mark.unit
def test_cookie_rotator_applies_cookiefile(tmp_path):
    first = tmp_path / "cookies_a.txt"
    second = tmp_path / "cookies_b.txt"
    first.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    second.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    rotator = CookieRotator([first, second], rotate_after_successes=2)
    opts = apply_cookiefile({"quiet": True}, rotator)
    next_opts = apply_cookiefile({"quiet": True}, rotator)

    assert opts["cookiefile"] == str(first)
    assert next_opts["cookiefile"] == str(first)

    rotator.record_success()
    assert apply_cookiefile({"quiet": True}, rotator)["cookiefile"] == str(first)
    rotator.record_success()
    assert apply_cookiefile({"quiet": True}, rotator)["cookiefile"] == str(second)


@pytest.mark.unit
def test_incremental_shard_flush_writes_before_category_end(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.categories[0].keywords = ["sports"]
    config.categories[0].collect_count = 5
    config.shard_size = 2
    config.time_interval_buckets = [
        {"name": "lt_1d", "min_age_days": 0, "max_age_days": 1, "weight": 1.0}
    ]
    state = DatasetState(config)
    state.initialize()

    videos = [make_video(f"v{i}") for i in range(5)]
    collector = DatasetCollector(config, state, {"youtube": FakeAdapter(videos)})
    result = collector.discover_category("sports", limit=5)

    assert result["accepted"] == 5
    shard_files = list((tmp_path / "shards" / "metadata").glob("**/*.json"))
    assert len(shard_files) == 3
    assert state.load_manifest().counters["accepted"] == 5


@pytest.mark.unit
def test_checkpoint_resume_starts_from_saved_keyword(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.categories[0].keywords = ["kw0", "kw1", "kw2"]
    config.categories[0].collect_count = 10
    config.shard_size = 100
    state = DatasetState(config)
    state.initialize()

    from fetcher.dataset_collector.checkpoint import DiscoveryCheckpoint

    state.save_checkpoint(
        DiscoveryCheckpoint(
            category="sports",
            bucket_name="lt_1d",
            platform="youtube",
            keyword_index=1,
            keyword="kw1",
        )
    )
    state.mark_seen("youtube:from_kw0", category="sports")

    calls = []

    class TrackingAdapter(FakeAdapter):
        def discover(self, **kwargs):
            calls.append(kwargs["query"])
            return super().discover(**kwargs)

    collector = DatasetCollector(
        config,
        state,
        {"youtube": TrackingAdapter([make_video("fresh")])},
    )
    collector.discover_category("sports", limit=1)

    assert calls == ["kw1"]


@pytest.mark.unit
def test_import_seen_updates_manifest_baseline(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.baseline_accepted = 100_000
    state = DatasetState(config)
    state.initialize()

    source = tmp_path / "ids.json"
    source.write_text(json.dumps({"abc123": {}, "def456": {}}), encoding="utf-8")

    from fetcher.dataset_collector.legacy_import import import_seen_ids

    imported = import_seen_ids(state, source)
    manifest = state.load_manifest()

    assert imported == 2
    assert manifest.legacy_seen_imported == 2
    assert manifest.baseline_accepted == 100_000


@pytest.mark.unit
def test_use_proxies_for_discovery_false_skips_api_pool(tmp_path):
    from fetcher.dataset_collector.config import default_campaign_config
    from fetcher.dataset_collector.proxy import configured_proxies, load_proxy_file

    proxy_file = tmp_path / "proxies.txt"
    proxy_file.write_text(
        "1.2.3.4:8080\n127.0.0.1:8881 download_only\n",
        encoding="utf-8",
    )
    config = default_campaign_config(output_dir=str(tmp_path))
    config.proxies_file = str(proxy_file)
    config.use_proxies_for_discovery = False

    assert configured_proxies(config=config, download_only=False) == []
    assert configured_proxies(config=config, download_only=True) == ["http://127.0.0.1:8881"]


@pytest.mark.unit
def test_proxy_file_excludes_local_for_discovery(tmp_path):
    proxy_file = tmp_path / "proxies.txt"
    proxy_file.write_text(
        "1.2.3.4:8080\n127.0.0.1:8084 download_only\n127.0.0.1:8881 nodpi\n",
        encoding="utf-8",
    )

    discovery_proxies = load_proxy_file(proxy_file, include_local=False, download_only=False)
    download_proxies = load_proxy_file(proxy_file, download_only=True)
    with_local = load_proxy_file(proxy_file, include_local=True, download_only=False)

    assert discovery_proxies == ["http://1.2.3.4:8080"]
    assert download_proxies == ["http://127.0.0.1:8084", "http://127.0.0.1:8881"]
    assert with_local == ["http://1.2.3.4:8080"]
    assert normalize_proxy_url("5.6.7.8:80") == "http://5.6.7.8:80"


@pytest.mark.unit
def test_pytubefix_proxy_dict():
    from fetcher.dataset_collector.proxy import pytubefix_proxy_dict

    assert pytubefix_proxy_dict("197.248.16.109:8080") == {
        "http": "http://197.248.16.109:8080",
        "https": "http://197.248.16.109:8080",
    }
    assert pytubefix_proxy_dict(None) is None


@pytest.mark.unit
def test_proxy_rotator_blacklists_transport_failures():
    import httpx

    rotator = ProxyRotator(proxies=["http://bad:80", "http://good:8080"])
    rotator.record_failure("http://bad:80", error=httpx.ProxyError("connection refused"))
    assert "http://bad:80" in rotator.blacklisted()
    assert rotator.next() == "http://good:8080"


@pytest.mark.unit
def test_proxy_rotator_prefers_last_good_proxy():
    rotator = ProxyRotator(proxies=["http://a:1", "http://b:2"])
    rotator.record_success("http://b:2")
    assert rotator.next() == "http://b:2"
    assert rotator.next() == "http://b:2"


@pytest.mark.unit
def test_proxy_rotator_ignores_api_errors_for_blacklist():
    rotator = ProxyRotator(proxies=["http://good:8080"])
    rotator.record_failure("http://good:8080", error=RuntimeError("403 Forbidden"))
    assert rotator.blacklisted() == set()
    assert rotator.next() == "http://good:8080"


@pytest.mark.unit
def test_is_comments_disabled_error():
    from fetcher.services.youtube_data_client import is_comments_disabled_error

    assert is_comments_disabled_error("videoId parameter has disabled comments")
    assert not is_comments_disabled_error("quotaExceeded")


@pytest.mark.unit
def test_collector_stops_keyword_after_min_unique(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.min_videos_per_keyword = 20
    config.keyword_search_multiplier = 10
    config.time_interval_buckets = []
    config.categories[0].keywords = ["kw-one", "kw-two"]
    config.categories[0].collect_count = 10_000
    state = DatasetState(config)
    state.initialize()

    class PerQueryAdapter(FakeAdapter):
        def discover(self, *, category, query, limit, published_after=None, published_before=None, time_interval=None):
            prefix = query.replace("-", "_")
            self.videos = [make_video(f"{prefix}_{i}") for i in range(limit)]
            return super().discover(
                category=category,
                query=query,
                limit=limit,
                published_after=published_after,
                published_before=published_before,
                time_interval=time_interval,
            )

    collector = DatasetCollector(
        config,
        state,
        {"youtube": PerQueryAdapter([])},
    )
    result = collector.discover_category("sports")
    assert result["accepted"] == 40
    progress = list(state.keyword_progress_path.read_text(encoding="utf-8").strip().splitlines())
    assert len(progress) == 2
    first = json.loads(progress[0])
    assert first["status"] == "done"
    assert first["accepted"] == 20


@pytest.mark.unit
def test_collector_skips_keywords_marked_done_in_progress(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.min_videos_per_keyword = 20
    config.time_interval_buckets = []
    config.categories[0].keywords = ["done-kw", "fresh-kw"]
    config.categories[0].collect_count = 10_000
    state = DatasetState(config)
    state.initialize()
    from fetcher.dataset_collector.keyword_progress import KeywordProgressEntry

    state.append_keyword_progress(
        KeywordProgressEntry(
            category="sports",
            bucket_name=None,
            platform="youtube",
            keyword_index=0,
            keyword="done-kw",
            accepted=25,
            min_required=20,
            status="done",
        )
    )

    class PerQueryAdapter(FakeAdapter):
        def discover(self, *, category, query, limit, published_after=None, published_before=None, time_interval=None):
            prefix = query.replace("-", "_")
            self.videos = [make_video(f"{prefix}_{i}") for i in range(limit)]
            return super().discover(
                category=category,
                query=query,
                limit=limit,
                published_after=published_after,
                published_before=published_before,
                time_interval=time_interval,
            )

    collector = DatasetCollector(config, state, {"youtube": PerQueryAdapter([])})
    result = collector.discover_category("sports")
    assert result["accepted"] == 20
    lines = state.keyword_progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["keyword"] == "fresh-kw"


@pytest.mark.unit
def test_hf_remote_paths():
    config = default_campaign_config(output_dir="/tmp/unused", categories=["sports"])
    config.hf_shards_path_prefix = "data/shards"
    config.hf_videos_path_prefix = "raw/videos"
    from fetcher.dataset_collector.hf_upload import remote_shard_path, remote_video_path

    assert remote_shard_path(config, "shards/metadata/category=Sport/part_000000.json").startswith(
        "data/shards/"
    )
    assert remote_video_path(config, category="Sport", video_id="abc") == "raw/videos/Sport/abc.mp4"


@pytest.mark.unit
def test_hf_commit_limits_single_colab_defaults():
    from fetcher.dataset_collector.hf_commit_budget import resolve_hf_commit_limits
    from fetcher.dataset_collector.schemas import CampaignConfig

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": "out",
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
        }
    )
    limits = resolve_hf_commit_limits(cfg)
    assert limits.parallel_colab_count == 1
    assert limits.hourly_limit_per_colab == 100
    assert limits.min_interval_seconds == 37


@pytest.mark.unit
def test_hf_commit_limits_split_across_parallel_colabs():
    from fetcher.dataset_collector.hf_commit_budget import resolve_hf_commit_limits
    from fetcher.dataset_collector.schemas import CampaignConfig

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": "out",
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "hf_parallel_colab_count": 3,
            "hf_repo_hourly_commit_limit": 128,
            "hf_commit_budget_reserve": 0.9,
        }
    )
    limits = resolve_hf_commit_limits(cfg)
    assert limits.parallel_colab_count == 3
    assert limits.hourly_limit_per_colab == 38
    assert limits.min_interval_seconds >= 95
    assert limits.coord_upload_min_interval_seconds >= 120


@pytest.mark.unit
def test_coord_flush_batches_hf_uploads(tmp_path, monkeypatch):
    from fetcher.dataset_collector.hf_coordination import WorkerCoordination
    from fetcher.dataset_collector.schemas import CampaignConfig
    from fetcher.dataset_collector.state import DatasetState

    uploads: list[str] = []

    def fake_upload(local_path, remote):
        uploads.append(remote)

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": str(tmp_path),
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "hf_coord_enabled": True,
            "hf_upload_enabled": True,
            "hf_shards_repo_id": "org/repo",
            "worker_id": "w1",
            "hf_parallel_colab_count": 3,
        }
    )
    state = DatasetState(cfg)
    state.initialize()
    coord = WorkerCoordination(state, cfg)
    monkeypatch.setattr(coord, "_upload_coord_file", fake_upload)
    key = "youtube:Sport:v1"
    assert coord.try_claim("download", key)
    assert uploads == []
    coord.mark_done("download", key, video_id="v1")
    assert uploads == []
    coord.flush_coord_uploads("download", force=True)
    assert len(uploads) == 2


@pytest.mark.unit
def test_stable_shard_slot_is_deterministic():
    from fetcher.dataset_collector.hf_coordination import stable_shard_slot
    from fetcher.dataset_collector.schemas import CampaignConfig

    key = "youtube:Sport:abc123"
    assert stable_shard_slot(key, 3) == stable_shard_slot(key, 3)
    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": "out",
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "worker_shard_index": stable_shard_slot(key, 3),
            "worker_shard_count": 3,
        }
    )
    from fetcher.dataset_collector.hf_coordination import key_in_worker_shard

    assert key_in_worker_shard(key, cfg)


@pytest.mark.unit
def test_merge_claims_expires_stale(tmp_path):
    from fetcher.dataset_collector.hf_coordination import WorkerCoordination
    from fetcher.dataset_collector.schemas import CampaignConfig
    from fetcher.dataset_collector.state import DatasetState

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": str(tmp_path),
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "hf_coord_enabled": True,
            "hf_upload_enabled": True,
            "hf_shards_repo_id": "org/repo",
            "worker_id": "w1",
        }
    )
    state = DatasetState(cfg)
    state.initialize()
    coord = WorkerCoordination(state, cfg)
    rows = [
        {
            "key": "youtube:Sport:v1",
            "owner": "other",
            "status": "active",
            "claimed_at": "2020-01-01T00:00:00+00:00",
            "expires_at": "2020-01-02T00:00:00+00:00",
        }
    ]
    merged = coord._merge_claim_rows(rows, ttl_seconds=7200)
    assert "youtube:Sport:v1" not in merged


@pytest.mark.unit
def test_should_suppress_huge_botguard_stderr():
    from fetcher.dataset_collector.worker_logging import _should_suppress_stderr_line

    assert _should_suppress_stderr_line("/path/botGuard.js:1\n!function(e,t){..." + "x" * 500)
    assert not _should_suppress_stderr_line("[download] bot_detection vid123\n")


@pytest.mark.unit
def test_pytubefix_sticky_client_stays_after_web_success(tmp_path, monkeypatch):
    import fetcher.dataset_collector.downloads as downloads_mod
    from fetcher.dataset_collector.downloads import BotDetectionDownloadError, download_video_local
    from fetcher.dataset_collector.schemas import CampaignConfig
    from fetcher.dataset_collector.state import DatasetState

    downloads_mod._pytubefix_sticky_client_index = 0
    clients_used: list[str] = []

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": str(tmp_path),
            "categories": [{"name": "Avto", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "download_pytubefix_clients": ["ANDROID_VR", "WEB"],
        }
    )
    state = DatasetState(cfg)
    state.initialize()

    def fake_attempt(*_args, client_name: str, target: Path, **_kwargs):
        clients_used.append(client_name)
        if client_name == "ANDROID_VR":
            raise BotDetectionDownloadError("bot")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 1024)
        return target

    monkeypatch.setattr(downloads_mod, "_pytubefix_client_sequence", lambda _cfg: ["ANDROID_VR", "WEB"])
    monkeypatch.setattr(downloads_mod, "_download_video_local_pytubefix_attempt", fake_attempt)
    monkeypatch.setattr(downloads_mod, "_download_video_local_ytdlp", lambda *_a, **_k: None)

    download_video_local(
        state,
        cfg,
        platform="youtube",
        video_id="vid1",
        url="https://www.youtube.com/watch?v=vid1",
        category="Avto",
        proxy_rotator=None,
        cookie_rotator=None,
    )
    assert "WEB" in clients_used
    assert downloads_mod._pytubefix_sticky_client_index == 1

    clients_used.clear()
    download_video_local(
        state,
        cfg,
        platform="youtube",
        video_id="vid2",
        url="https://www.youtube.com/watch?v=vid2",
        category="Avto",
        proxy_rotator=None,
        cookie_rotator=None,
    )
    assert clients_used[0] == "WEB"
    assert "ANDROID_VR" not in clients_used


@pytest.mark.unit
def test_download_pacing_bot_backoff_escalates():
    from fetcher.dataset_collector.download_pacing import (
        compute_download_pause_seconds,
        reset_download_pacing,
    )
    from fetcher.dataset_collector.schemas import CampaignConfig

    reset_download_pacing()
    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": "out",
            "categories": [{"name": "Sport", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "download_pause_after_bot_seconds": 60,
            "download_pause_after_bot_max_seconds": 200,
            "download_pause_bot_backoff_multiplier": 2.0,
        }
    )
    assert compute_download_pause_seconds(cfg, "bot") == 60
    assert compute_download_pause_seconds(cfg, "bot") == 120
    assert compute_download_pause_seconds(cfg, "bot") == 200
    compute_download_pause_seconds(cfg, "success")
    assert compute_download_pause_seconds(cfg, "bot") == 60


@pytest.mark.unit
def test_pytubefix_sticky_advances_after_all_cookies_bot_without_success(tmp_path, monkeypatch):
    import fetcher.dataset_collector.downloads as downloads_mod
    from fetcher.dataset_collector.downloads import BotDetectionDownloadError, download_video_local
    from fetcher.dataset_collector.schemas import CampaignConfig
    from fetcher.dataset_collector.state import DatasetState

    downloads_mod._pytubefix_sticky_client_index = 0

    cfg = CampaignConfig.parse_obj(
        {
            "name": "t",
            "output_dir": str(tmp_path),
            "categories": [{"name": "Avto", "keywords": ["x"], "target_count": 1, "collect_count": 1}],
            "download_pytubefix_clients": ["ANDROID_VR", "WEB"],
        }
    )
    state = DatasetState(cfg)
    state.initialize()

    def fake_attempt(*_args, client_name: str, **_kwargs):
        raise BotDetectionDownloadError("bot")

    monkeypatch.setattr(downloads_mod, "_pytubefix_client_sequence", lambda _cfg: ["ANDROID_VR", "WEB"])
    monkeypatch.setattr(downloads_mod, "_download_video_local_pytubefix_attempt", fake_attempt)
    monkeypatch.setattr(downloads_mod, "_download_video_local_ytdlp", lambda *_a, **_k: None)

    download_video_local(
        state,
        cfg,
        platform="youtube",
        video_id="vid1",
        url="https://www.youtube.com/watch?v=vid1",
        category="Avto",
        proxy_rotator=None,
        cookie_rotator=None,
    )
    assert downloads_mod._pytubefix_sticky_client_index == 1

    download_video_local(
        state,
        cfg,
        platform="youtube",
        video_id="vid2",
        url="https://www.youtube.com/watch?v=vid2",
        category="Avto",
        proxy_rotator=None,
        cookie_rotator=None,
    )
    assert downloads_mod._pytubefix_sticky_client_index == 1


@pytest.mark.unit
def test_is_pytubefix_client_error_detects_innertube_parse_failures():
    from fetcher.dataset_collector.downloads import is_pytubefix_client_error

    assert is_pytubefix_client_error(IndexError("list index out of range"))
    assert is_pytubefix_client_error(KeyError("visitorData"))
    assert not is_pytubefix_client_error(RuntimeError("disk full"))


@pytest.mark.unit
def test_resolve_hf_token_fallback_env(monkeypatch):
    from fetcher.dataset_collector.hf_upload import HuggingFaceUploadError, resolve_hf_token

    config = default_campaign_config()
    config.hf_token_env = "HF_TOKEN"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_from_hub")
    assert resolve_hf_token(config) == "hf_from_hub"


@pytest.mark.unit
def test_resolve_hf_token_rejects_token_in_hf_token_env():
    from fetcher.dataset_collector.hf_upload import HuggingFaceUploadError, resolve_hf_token

    config = default_campaign_config()
    # Looks like a token but is a deliberate misconfiguration fixture (not a real secret).
    config.hf_token_env = "hf_PASTED_TOKEN_VALUE_NOT_ENV_VAR_NAME"
    with pytest.raises(HuggingFaceUploadError, match="hf_token_env"):
        resolve_hf_token(config)


@pytest.mark.unit
def test_run_hf_shard_upload_queue(tmp_path, monkeypatch):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.hf_repo_id = "org/test-dataset"
    state = DatasetState(config)
    state.initialize()
    shard_path = state.write_metadata_shard("sports", [make_video("v1")])
    rel = str(shard_path.relative_to(state.root))
    state.enqueue_hf_shard_upload(shard_relpath=rel, category="sports")

    uploaded: list[str] = []

    def fake_upload(cfg, files, *, repo_id, commit_message, state_dir=None):
        uploaded.extend(path_in_repo for _, path_in_repo in files)

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "fetcher.dataset_collector.hf_queues.upload_local_files_commit",
        fake_upload,
    )

    result = run_hf_shard_upload_queue(state, config, limit=1)
    assert result["uploaded"] == 1
    assert uploaded


@pytest.mark.unit
def test_collector_keyword_search_budget(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
    config.min_videos_per_keyword = 20
    config.keyword_search_multiplier = 10
    config.categories[0].keywords = ["sports"]
    config.categories[0].collect_count = 100
    state = DatasetState(config)
    state.initialize()

    class LimitAdapter(FakeAdapter):
        last_limit = 0

        def discover(self, *, category, query, limit, published_after=None, published_before=None, time_interval=None):
            LimitAdapter.last_limit = limit
            return super().discover(
                category=category,
                query=query,
                limit=limit,
                published_after=published_after,
                published_before=published_before,
                time_interval=time_interval,
            )

    videos = [make_video(f"v{i}") for i in range(25)]
    collector = DatasetCollector(
        config,
        state,
        {"youtube": LimitAdapter(videos)},
    )
    collector.discover_category("sports", limit=5)
    assert LimitAdapter.last_limit >= 200


@pytest.mark.unit
def test_is_proxy_transport_error():
    import httpx

    assert is_proxy_transport_error(httpx.ProxyError("x"))
    assert not is_proxy_transport_error(RuntimeError("quota"))


@pytest.mark.unit
def test_inventory_register_shard_and_stats(tmp_path):
    from fetcher.dataset_collector.inventory import (
        compute_inventory_stats,
        list_shard_records,
        rebuild_inventory_from_disk,
        shards_index_path,
    )

    config = default_campaign_config(output_dir=str(tmp_path), categories=["Sport"])
    state = DatasetState(config)
    state.initialize()
    v1 = make_video("vid-a").copy(update={"category": "Sport"})
    v2 = make_video("vid-b").copy(update={"category": "Sport"})
    state.write_metadata_shard("Sport", [v1, v2])
    rows = list_shard_records(state, category="Sport")
    assert len(rows) == 1
    assert rows[0]["count"] == 2
    assert "vid-a" in rows[0]["video_ids"]
    assert shards_index_path(state).exists()

    state.enqueue_download(make_video("vid-q").copy(update={"video_id": "vid-q", "category": "Sport"}))
    stats = compute_inventory_stats(state, category="Sport")
    assert stats["videos"]["in_shards"] == 2
    assert stats["shards"]["total"] == 1

    rebuild_inventory_from_disk(state, category="Sport")
    assert len(list_shard_records(state, category="Sport")) == 1


@pytest.mark.unit
def test_update_inventory_gauges():
    from fetcher.dataset_collector.metrics import (
        dataset_collector_download_queue_pending,
        update_inventory_gauges,
    )

    summary = {
        "totals": {
            "shards": {"total": 3, "on_hf": 1, "pending_hf_upload": 2},
            "videos": {
                "in_shards": 100,
                "downloaded_local_files": 10,
                "on_hf": 5,
                "pending_download": 80,
                "pending_hf_upload": 5,
                "pending_enrich": 90,
                "enriched": 10,
            },
        },
        "by_category": {
            "Sport": {
                "shards": {"total": 3, "on_hf": 1, "pending_hf_upload": 2},
                "videos": {
                    "in_shards": 100,
                    "downloaded_local_files": 10,
                    "on_hf": 5,
                    "pending_download": 80,
                    "pending_hf_upload": 5,
                    "pending_enrich": 90,
                    "enriched": 10,
                },
            }
        },
    }
    update_inventory_gauges(summary)
    sample = dataset_collector_download_queue_pending.collect()[0]
    assert sample.samples[0].value == 80


@pytest.mark.unit
def test_parse_resolution_height():
    from fetcher.dataset_collector.downloads import _parse_resolution_height

    assert _parse_resolution_height("1080p") == 1080
    assert _parse_resolution_height("360p") == 360
    assert _parse_resolution_height(None) == 0


@pytest.mark.unit
def test_ytdlp_enrich_logger_suppresses_subtitle_403():
    from fetcher.dataset_collector.ytdlp_logging import YtdlpEnrichLogger

    logger = YtdlpEnrichLogger()
    # should not raise / print
    logger.warning("HTTP Error 403: Forbidden for url: timedtext")
    logger.error("Unable to download auto subtitles for xyz")


@pytest.mark.unit
def test_download_height_tiers():
    from fetcher.dataset_collector.downloads import DOWNLOAD_HEIGHT_TIERS, MAX_DOWNLOAD_HEIGHT

    assert MAX_DOWNLOAD_HEIGHT == 1080
    assert DOWNLOAD_HEIGHT_TIERS[0] == 1080
    assert 720 in DOWNLOAD_HEIGHT_TIERS


@pytest.mark.unit
def test_best_enrich_formats_and_thumbnails():
    from fetcher.dataset_collector.training_format import (
        extract_best_enrich_formats,
        extract_best_ytdlp_thumbnails,
    )

    info = {
        "formats": [
            {"fps": 30, "resolution": "1280x720", "vcodec": "avc1"},
            {"fps": 30, "resolution": "1920x1080", "vcodec": "avc1"},
            {"fps": 30, "resolution": "3840x2160", "vcodec": "vp9"},
        ],
        "thumbnails": [
            {"url": "low", "preference": -1, "height": 90},
            {"url": "best", "preference": 10, "height": 1080},
            {"url": "second", "preference": 5, "height": 720},
        ],
    }
    assert extract_best_enrich_formats(info) == [
        {"fps": 30, "resolution": "3840x2160"},
        {"fps": 30, "resolution": "1920x1080"},
    ]
    assert [thumb["url"] for thumb in extract_best_ytdlp_thumbnails(info)] == ["best", "second"]


@pytest.mark.unit
def test_install_pytubefix_session(tmp_path):
    from fetcher.dataset_collector.cookies import install_pytubefix_session

    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tPREF\ttest\n",
        encoding="utf-8",
    )
    install_pytubefix_session(
        proxies={"http": "http://127.0.0.1:8881", "https": "http://127.0.0.1:8881"},
        cookie_file=cookie_file,
    )


@pytest.mark.unit
def test_pass_had_work():
    from fetcher.dataset_collector.run_workers import _pass_had_work

    assert _pass_had_work({"attempted": 1, "downloaded": 0, "failed": 1})
    assert not _pass_had_work({"attempted": 0, "downloaded": 0, "skipped": 5})
    assert not _pass_had_work({})


@pytest.mark.unit
def test_slim_caption_tracks_drops_urls_and_foreign_langs():
    bloated = {
        "ru": [{"ext": "vtt", "url": "https://example.com/ru.vtt"}],
        "en": [{"ext": "srv1", "url": "https://example.com/en.srv1"}],
        "ab": [{"ext": "vtt", "url": "https://example.com/ab.vtt"}],
    }
    slim = slim_caption_tracks(bloated)
    assert set(slim.keys()) == {"ru", "en"}
    assert slim["ru"] == [{"ext": "vtt"}]
    assert metadata_captions_are_bloated({"automatic_captions": bloated})
    assert not metadata_captions_are_bloated(compact_training_metadata({"automatic_captions": bloated}))


@pytest.mark.unit
def test_slim_caption_tracks_merges_en_variants_and_empty():
    merged = slim_caption_tracks(
        {
            "en-US": [{"ext": "vtt", "url": "https://example.com/en.vtt"}],
            "en-GB": [{"ext": "srv3"}],
            "de": [{"ext": "vtt"}],
        }
    )
    assert merged == {"en": [{"ext": "vtt"}, {"ext": "srv3"}]}
    assert slim_caption_tracks({}) == {}
    assert slim_caption_tracks(None) == {}


@pytest.mark.unit
def test_parse_vtt_subtitle_text():
    from fetcher.dataset_collector.caption_text import parse_subtitle_content, parse_subtitle_payload

    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n\n00:00:04.000 --> 00:00:06.000\nSecond line\n"
    assert parse_subtitle_content(vtt, "vtt") == "Hello world\nSecond line"
    payload = parse_subtitle_payload(vtt, "vtt")
    assert payload["cues"][0]["start"] == "00:00:01.000"


@pytest.mark.unit
def test_merge_ytdlp_always_sets_caption_fields():
    metadata = {"title": "t"}
    info = {"formats": [], "subtitles": {}, "automatic_captions": {}}
    merged = merge_ytdlp_into_training_metadata(metadata, info)
    assert merged["subtitles"] == {}
    assert merged["automatic_captions"] == {}


@pytest.mark.unit
def test_video_filter_rejects_missing_duration():
    rules = {"duration_min_seconds": 4, "duration_max_seconds": 1500}
    filt = VideoFilter(rules)
    assert not filt.decide({"metadata": {"duration_seconds": None}}).accepted
    assert not filt.decide({"metadata": {"duration_seconds": 0}}).accepted
    assert filt.decide({"metadata": {"duration_seconds": 30}}).accepted


@pytest.mark.unit
def test_should_permanent_delete_on_drive_for_colab_paths():
    from fetcher.dataset_collector.local_delete import (
        is_ephemeral_download_artifact,
        is_google_drive_path,
        should_permanent_delete_on_drive,
    )

    path = Path("/content/drive/MyDrive/dataset_runs/20k-test/videos/cat/v.mp4")
    assert is_google_drive_path(path)
    assert should_permanent_delete_on_drive(path)
    assert should_permanent_delete_on_drive(
        path,
        output_dir="/content/drive/MyDrive/dataset_runs/20k-test",
    )
    assert not should_permanent_delete_on_drive(path, enabled=False)

    tmp = Path(
        "/content/drive/MyDrive/dataset_runs/20k-test/downloads/videos/cat/vid.video.tmp"
    )
    assert is_ephemeral_download_artifact(tmp)
    assert not should_permanent_delete_on_drive(tmp)


@pytest.mark.unit
def test_delete_local_file_uses_unlink_off_drive(tmp_path):
    from fetcher.dataset_collector.local_delete import delete_local_file

    target = tmp_path / "clip.mp4"
    target.write_bytes(b"abc")
    assert delete_local_file(target)
    assert not target.exists()


@pytest.mark.unit
def test_video_filter_rejects_live_after_enrich():
    rules = {"duration_min_seconds": 4, "duration_max_seconds": 1500}
    filt = VideoFilter(rules)
    decision = filt.decide_post_enrich(
        info={"is_live": True, "duration": 0},
        metadata={"duration_seconds": 0, "title": "LIVE match"},
    )
    assert not decision.accepted
    assert decision.reason == "live_stream"


@pytest.mark.unit
def test_build_schedule_entry_hours():
    video = make_video("v1")
    entry = build_schedule_entry(video, schedule_hours=[0, 1, 2, 3])
    assert entry.due_at["1"] == video.snapshot_0.collected_at + timedelta(hours=1)
    assert entry.due_at["2"] == video.snapshot_0.collected_at + timedelta(hours=2)


@pytest.mark.unit
def test_build_schedule_entry_minutes():
    video = make_video("v1")
    entry = build_schedule_entry(video, schedule_minutes=[0, 15, 30, 45])
    assert entry.due_at["1"] == video.snapshot_0.collected_at + timedelta(minutes=15)
    assert entry.due_at["3"] == video.snapshot_0.collected_at + timedelta(minutes=45)


@pytest.mark.unit
def test_build_schedule_entry_per_video_seconds():
    video = make_video("v1")
    entry = build_schedule_entry(
        video,
        snapshot_sleep_seconds=240,
        snapshot_follow_up_count=3,
    )
    base = video.snapshot_0.collected_at
    assert entry.due_at["1"] == base + timedelta(seconds=240)
    assert entry.due_at["2"] == base + timedelta(seconds=480)
    assert entry.due_at["3"] == base + timedelta(seconds=720)


@pytest.mark.unit
def test_snapshot_poll_report_format(tmp_path):
    from fetcher.dataset_collector.snapshots import format_snapshot_poll_report, snapshot_poll_report

    config = default_campaign_config(output_dir=str(tmp_path))
    config.snapshot_sleep_seconds = 240
    config.snapshot_follow_up_count = 2
    state = DatasetState(config)
    state.initialize()
    state.append_schedule(
        build_schedule_entry(
            make_video("v1"),
            snapshot_sleep_seconds=240,
            snapshot_follow_up_count=2,
        )
    )
    text = format_snapshot_poll_report(snapshot_poll_report(state, config))
    assert "[snapshot-poll]" in text
    assert "index 1" in text
    assert "pending=1" in text


@pytest.mark.unit
def test_seconds_until_next_snapshot_due(tmp_path):
    from fetcher.dataset_collector.snapshots import seconds_until_next_snapshot_due

    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports", "music"])
    config.snapshot_sleep_seconds = 240
    config.snapshot_follow_up_count = 2
    state = DatasetState(config)
    state.initialize()
    video = make_video("due1")
    state.append_schedule(
        build_schedule_entry(
            video,
            snapshot_sleep_seconds=240,
            snapshot_follow_up_count=2,
        )
    )
    future = video.snapshot_0.collected_at + timedelta(seconds=240)
    wait = seconds_until_next_snapshot_due(state, [1, 2], now=video.snapshot_0.collected_at)
    assert wait is not None
    assert 239 <= wait <= 241


@pytest.mark.unit
def test_discover_campaign_global_limit(tmp_path):
    videos = [make_video(f"v{i}") for i in range(20)]
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports", "music"])
    config.categories[0].keywords = ["kw"]
    config.categories[0].collect_count = 50
    config.categories[1].keywords = ["kw"]
    config.categories[1].collect_count = 50
    state = DatasetState(config)
    state.initialize()
    adapters = {"youtube": FakeAdapter(videos)}
    collector = DatasetCollector(config, state, adapters)
    total = collector.discover_campaign(["sports", "music"], limit=7)
    assert total["accepted"] == 7


@pytest.mark.unit
def test_is_allowed_metadata_shard_relpath():
    from fetcher.dataset_collector.hf_upload import is_allowed_metadata_shard_relpath

    assert is_allowed_metadata_shard_relpath("shards/metadata/category=Sport/part_000000.json")
    assert not is_allowed_metadata_shard_relpath("state/coordination/done/download/x.jsonl")


@pytest.mark.unit
def test_is_video_unavailable_error_detected():
    from fetcher.dataset_collector.downloads import is_video_unavailable_error

    assert is_video_unavailable_error("VideoUnavailable: abc is unavailable")
    assert not is_video_unavailable_error("bot_detection detected")


@pytest.mark.unit
def test_orphan_mp4_removed_when_already_on_hf(tmp_path, monkeypatch):
    from fetcher.dataset_collector.downloads import run_download_queue

    config = default_campaign_config(output_dir=str(tmp_path), categories=["Sport"])
    config.hf_repo_id = "org/test-dataset"
    config.hf_coord_enabled = False
    state = DatasetState(config)
    state.initialize()
    video = make_video("orphan1")
    state.buffer_accepted("sports", video)
    state.flush_pending("sports", shard_size=config.shard_size)
    state.enqueue_download(video)
    local = state.download_dir / "videos" / "sports" / "orphan1.mp4"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"fake-mp4")
    key = "youtube:sports:orphan1"
    state.mark_hf_video_upload_done(key, video_id="orphan1", category="sports", local_path=str(local.relative_to(state.root)))

    deleted: list[Path] = []

    def fake_delete(path, **kwargs):
        deleted.append(Path(path))
        Path(path).unlink(missing_ok=True)
        return True

    monkeypatch.setattr("fetcher.dataset_collector.downloads.delete_local_file", fake_delete)
    result = run_download_queue(state, config, state.download_dir / "queue.jsonl", limit=5)
    assert result["skipped"] >= 1
    assert deleted and deleted[0].name == "orphan1.mp4"
