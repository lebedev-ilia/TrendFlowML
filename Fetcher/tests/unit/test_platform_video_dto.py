from __future__ import annotations

from fetcher.schemas.platform_video import PlatformVideoDto, from_twitch_helix, from_ytdlp


def test_from_ytdlp_maps_fields():
    dto = from_ytdlp(
        {
            "id": "abc",
            "title": "T",
            "view_count": 10,
            "like_count": 2,
            "comment_count": 1,
            "upload_date": "20240101",
            "channel_id": "ch",
            "channel": "Channel",
        }
    )
    assert dto.video_id == "abc"
    assert dto.view_count == 10
    assert dto.published_at is not None


def test_merge_from_fills_gaps():
    base = PlatformVideoDto(video_id="1", title="api", source_provider="api")
    other = PlatformVideoDto(video_id="1", thumbnail_url="http://x", source_provider="sdk")
    merged = base.merge_from(other)
    assert merged.title == "api"
    assert merged.thumbnail_url == "http://x"
    assert merged.source_provider == "merged"


def test_from_twitch_helix():
    dto = from_twitch_helix({"id": "99", "title": "Stream", "view_count": 5, "user_name": "u"})
    assert dto.video_id == "99"
    assert dto.channel_title == "u"
