from __future__ import annotations

from datetime import timedelta

import pytest

from fetcher.dataset_collector.collector import DatasetCollector
from fetcher.dataset_collector.age_buckets import allocate_counts, bucket_from_config
from fetcher.dataset_collector.config import default_campaign_config
from fetcher.dataset_collector.cookies import CookieRotator, apply_cookiefile
from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.export import export_legacy_json, validate_export
from fetcher.dataset_collector.proxy import load_proxy_file, normalize_proxy_url
from fetcher.dataset_collector.schemas import CollectedVideo, ScheduleEntry, Snapshot
from fetcher.dataset_collector.snapshots import SnapshotRunner
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
def test_collector_deduplicates_and_writes_rejected(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path), categories=["sports"])
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
    assert '"time_interval": "lt_1d"' in shard.read_text(encoding="utf-8")


@pytest.mark.unit
def test_cookie_rotator_applies_cookiefile(tmp_path):
    first = tmp_path / "cookies_a.txt"
    second = tmp_path / "cookies_b.txt"
    first.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    second.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    rotator = CookieRotator([first, second])
    opts = apply_cookiefile({"quiet": True}, rotator)
    next_opts = apply_cookiefile({"quiet": True}, rotator)

    assert opts["cookiefile"] == str(first)
    assert next_opts["cookiefile"] == str(second)


@pytest.mark.unit
def test_proxy_file_excludes_local_for_discovery(tmp_path):
    proxy_file = tmp_path / "proxies.txt"
    proxy_file.write_text("1.2.3.4:8080\n127.0.0.1:8084\n", encoding="utf-8")

    discovery_proxies = load_proxy_file(proxy_file, include_local=False)
    download_proxies = load_proxy_file(proxy_file, include_local=True)

    assert discovery_proxies == ["http://1.2.3.4:8080"]
    assert download_proxies == ["http://1.2.3.4:8080", "http://127.0.0.1:8084"]
    assert normalize_proxy_url("5.6.7.8:80") == "http://5.6.7.8:80"
