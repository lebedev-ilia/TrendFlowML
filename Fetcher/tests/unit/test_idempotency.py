"""Unit тесты для идемпотентности Fetcher."""

import pytest
from unittest.mock import Mock, patch
from fetcher.idempotency import (
    check_video_exists,
    check_artifact_exists,
    is_stage_idempotent,
)


@pytest.mark.unit
class TestIdempotency:
    """Тесты для идемпотентности."""

    @patch("fetcher.idempotency.session_scope")
    def test_check_video_exists_found(self, mock_session):
        """Тест проверки существования видео (найдено)."""
        mock_db = Mock()
        mock_video = Mock()
        mock_video.id = "test-video-uuid"
        mock_video.platform = "youtube"
        mock_video.platform_video_id = "dQw4w9WgXcQ"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        result = check_video_exists("youtube", "dQw4w9WgXcQ")
        assert result is not None
        assert result == "test-video-uuid"

    @patch("fetcher.idempotency.session_scope")
    def test_check_video_exists_not_found(self, mock_session):
        """Тест проверки существования видео (не найдено)."""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        result = check_video_exists("youtube", "nonexistent")
        assert result is None

    @patch("fetcher.idempotency.check_artifact_in_storage")
    @patch("fetcher.idempotency.check_artifact_exists")
    @patch("fetcher.idempotency.check_video_exists")
    def test_is_stage_idempotent_true(self, mock_check_video, mock_check_artifact, mock_check_storage):
        """Тест проверки идемпотентности stage (можно пропустить)."""
        mock_check_video.return_value = "video-id"
        mock_check_artifact.return_value = ("storage/path/meta.json", "sha256:abc")
        mock_check_storage.return_value = True

        can_skip, reason = is_stage_idempotent("youtube", "dQw4w9WgXcQ", "metadata")
        assert can_skip is True
        assert "already completed" in reason.lower()

    @patch("fetcher.idempotency.check_video_exists")
    def test_is_stage_idempotent_false_no_video(self, mock_check_video):
        """Тест проверки идемпотентности stage (видео не найдено)."""
        mock_check_video.return_value = None

        can_skip, reason = is_stage_idempotent("youtube", "nonexistent", "metadata")
        assert can_skip is False
        assert "not found" in reason.lower()

