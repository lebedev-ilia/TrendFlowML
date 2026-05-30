import pytest
from unittest.mock import MagicMock, patch

from fetcher.services.youtube_data_client import (
    YouTubeDataClient,
    VideoMetadataDto,
    CommentDto,
    YouTubeSearchResult,
    ChannelMetadataDto,
    QuotaExceededError,
)


@pytest.mark.unit
class TestYouTubeDataClient:
    @pytest.fixture
    def client(self):
        with patch("fetcher.services.youtube_data_client.settings") as mock_settings:
            mock_settings.youtube_data_api_key = "test-key"
            mock_settings.youtube_rate_limit_rps = 10
            mock_settings.youtube_daily_quota_limit = 10_000
            yield YouTubeDataClient(api_key="test-key", rate_limit_rps=10, daily_quota_limit=100)

    @pytest.fixture
    def mock_http_client(self, client):
        with patch.object(client, "client") as mock_client:
            yield mock_client

    def test_get_video_metadata_parses_response(self, client, mock_http_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "items": [
                {
                    "id": "video123",
                    "snippet": {
                        "title": "Test video",
                        "description": "Desc",
                        "channelId": "chan1",
                        "channelTitle": "Channel",
                        "publishedAt": "2024-01-01T00:00:00Z",
                    },
                    "contentDetails": {"duration": "PT1M30S"},
                    "statistics": {"viewCount": "10", "likeCount": "2", "commentCount": "3"},
                }
            ]
        }
        mock_http_client.get.return_value = response

        dto = client.get_video_metadata("video123")
        assert isinstance(dto, VideoMetadataDto)
        assert dto.video_id == "video123"
        assert dto.duration_seconds == 90
        assert dto.view_count == 10
        assert dto.like_count == 2
        assert dto.comment_count == 3

    def test_iter_comments_yields_dtos_and_stops_at_max(self, client, mock_http_client):
        response1 = MagicMock()
        response1.status_code = 200
        response1.json.return_value = {
            "items": [
                {
                    "id": "thread1",
                    "snippet": {
                        "topLevelComment": {
                            "id": "c1",
                            "snippet": {
                                "authorDisplayName": "User",
                                "textOriginal": "Hi",
                                "likeCount": 1,
                                "publishedAt": "2024-01-01T00:00:00Z",
                            },
                        }
                    },
                }
            ],
            "nextPageToken": None,
        }
        mock_http_client.get.return_value = response1

        comments = list(client.iter_comments("video123", max_count=1))
        assert len(comments) == 1
        assert isinstance(comments[0], CommentDto)
        assert comments[0].author_display_name == "User"
        assert comments[0].text_original == "Hi"

    def test_quota_tracker_raises_when_exceeded(self):
        with patch("fetcher.services.youtube_data_client.settings") as mock_settings:
            mock_settings.youtube_data_api_key = "test-key"
            mock_settings.youtube_rate_limit_rps = 10
            mock_settings.youtube_daily_quota_limit = 1
            client = YouTubeDataClient(api_key="test-key", rate_limit_rps=10, daily_quota_limit=1)

        with pytest.raises(QuotaExceededError):
            client.quota_tracker.consume(2)

    def test_search_videos_parses_response(self, client, mock_http_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "items": [
                {
                    "id": {"videoId": "video123"},
                    "snippet": {
                        "title": "Found",
                        "channelId": "chan1",
                        "channelTitle": "Channel",
                        "publishedAt": "2024-01-01T00:00:00Z",
                    },
                }
            ],
            "nextPageToken": "next",
        }
        mock_http_client.get.return_value = response

        result = client.search_videos("sports")

        assert isinstance(result, YouTubeSearchResult)
        assert result.items[0].video_id == "video123"
        assert result.next_page_token == "next"

    def test_get_channels_metadata_batch_parses_response(self, client, mock_http_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "items": [
                {
                    "id": "chan1",
                    "snippet": {"title": "Channel"},
                    "statistics": {
                        "subscriberCount": "10",
                        "videoCount": "2",
                        "viewCount": "100",
                    },
                }
            ]
        }
        mock_http_client.get.return_value = response

        channels = client.get_channels_metadata_batch(["chan1"])

        assert isinstance(channels[0], ChannelMetadataDto)
        assert channels[0].subscriber_count == 10
        assert channels[0].video_count == 2

    def test_comments_403_disabled_fails_fast(self, client, mock_http_client):
        response = MagicMock()
        response.status_code = 403
        response.text = '{"error":{"message":"The video identified by the videoId parameter has disabled comments."}}'
        response.json.return_value = {
            "error": {"message": "The video identified by the videoId parameter has disabled comments."}
        }
        mock_http_client.get.return_value = response

        comments = list(client.iter_comments("vid1", max_count=10))

        assert comments == []
        assert mock_http_client.get.call_count == 1

