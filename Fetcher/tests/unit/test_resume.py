"""Unit тесты для resume после сбоя Fetcher."""

import pytest
from unittest.mock import Mock, patch
from fetcher.resume import (
    get_incomplete_runs,
    get_missing_artifacts_for_run,
    determine_next_stage,
)


@pytest.mark.unit
class TestResume:
    """Тесты для resume после сбоя."""

    @patch("fetcher.resume.session_scope")
    def test_get_incomplete_runs(self, mock_session):
        """Тест получения незавершённых run'ов."""
        mock_db = Mock()
        mock_run = Mock()
        mock_run.status = "FETCHING_METADATA"
        mock_run.finished_at = None
        mock_db.query.return_value.filter.return_value.filter.return_value.all.return_value = [mock_run]
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        runs = get_incomplete_runs()
        assert len(runs) == 1
        assert runs[0].status == "FETCHING_METADATA"

    @patch("fetcher.resume.session_scope")
    def test_get_missing_artifacts_for_run(self, mock_session):
        """Тест получения отсутствующих артефактов."""
        mock_db = Mock()
        mock_video_source = Mock()
        mock_video_source.normalized_video_id = "dQw4w9WgXcQ"
        mock_video_source.platform = "youtube"

        mock_video = Mock()
        mock_video.id = "video-id"

        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_video_source
        mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        mock_db.query.return_value.filter.return_value.all.return_value = []  # Нет артефактов

        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        missing = get_missing_artifacts_for_run("run-id")
        assert "video_file" in missing
        assert "metadata_file" in missing
        assert "comments_file" in missing

    @patch("fetcher.resume.get_missing_artifacts_for_run")
    def test_determine_next_stage_metadata(self, mock_get_missing):
        """Тест определения следующей stage (metadata)."""
        mock_get_missing.return_value = ["metadata_file", "video_file", "comments_file"]

        next_stage = determine_next_stage("run-id")
        assert next_stage == "metadata"

    @patch("fetcher.resume.get_missing_artifacts_for_run")
    def test_determine_next_stage_finalize(self, mock_get_missing):
        """Тест определения следующей stage (finalize, все готово)."""
        mock_get_missing.return_value = []

        next_stage = determine_next_stage("run-id")
        assert next_stage == "finalize"

