"""Unit tests for TikTokAdapter (MVP)."""

from unittest.mock import MagicMock, patch

import pytest

from fetcher.platforms.tiktok.adapter import TikTokAdapter


@pytest.mark.unit
class TestTikTokAdapter:
    @patch("fetcher.platforms.tiktok.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.tiktok.adapter.acquire_token")
    @patch("fetcher.platforms.tiktok.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.tiktok.adapter.session_scope")
    @patch("fetcher.platforms.tiktok.adapter.storage_client")
    def test_fetch_comments_writes_empty_comments_artifact(
        self,
        mock_storage,
        mock_session_scope,
        mock_ydl,
        mock_acquire_token,
        mock_breaker,
    ):
        adapter = TikTokAdapter()

        breaker = MagicMock()
        breaker.is_open.return_value = False
        mock_breaker.return_value = breaker
        mock_acquire_token.return_value = True

        ydl_instance = MagicMock()
        ydl_instance.extract_info.return_value = {"id": "7351234567890123456"}
        mock_ydl.return_value.__enter__.return_value = ydl_instance

        video_row = MagicMock()
        video_row.id = "video-uuid"

        q_video = MagicMock()
        q_artifact = MagicMock()

        # First DB block: Video.one_or_none() -> None, then flush for insert.
        q_video.filter.return_value.one_or_none.return_value = None
        # Second/third blocks: Video.one() -> existing video
        q_video.filter.return_value.one.return_value = video_row

        q_artifact.filter.return_value.order_by.return_value.first.return_value = None

        def query_side_effect(model):
            name = getattr(model, "__name__", str(model))
            if name == "Video":
                return q_video
            return q_artifact

        db = MagicMock()
        db.query.side_effect = query_side_effect

        ctx = MagicMock()
        ctx.__enter__.return_value = db
        ctx.__exit__.return_value = None
        mock_session_scope.return_value = ctx

        mock_storage.upload_file.return_value = None

        adapter.fetch_comments("https://www.tiktok.com/@u/video/7351234567890123456", run_id="rid", limit=10)

        assert mock_storage.upload_file.called
        assert db.add.called  # should add Video and Artifact rows

