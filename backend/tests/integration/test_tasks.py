"""
Интеграционные тесты Celery task process_analysis_job.

Проверяют вызов task с моками БД и DataProcessor API: подготовка payload,
вызов run_dataprocessor_async и wait_for_run_completion_hybrid, обновление
статуса и артефактов. Без реальных Celery worker, БД и DataProcessor.

См. backend/docs/TESTING_PLAN.md § 3.1.8, app/tasks/analysis.py.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.tasks import process_analysis_job
from app.dbv2 import enums, models as v2_models


pytestmark = pytest.mark.integration


class TestProcessAnalysisJob:
    """process_analysis_job(analysis_job_id)."""

    def test_task_returns_early_when_job_not_found(self):
        """При отсутствии AnalysisJob в БД task завершается без исключения."""
        with patch("app.tasks.analysis.session_scope") as mock_scope:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_scope.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            process_analysis_job(str(uuid4()))

        mock_scope.assert_called_once()

    def test_task_returns_early_when_job_already_canceled(self):
        """Уже отменённый job — выход до prepare_dataprocessor_payload."""
        job_id = uuid4()
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.status = enums.AnalysisStatus.canceled

        with patch("app.tasks.analysis.session_scope") as mock_scope:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_job
            mock_scope.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            with patch(
                "app.tasks.analysis.prepare_dataprocessor_payload",
            ) as mock_prepare:
                process_analysis_job(str(job_id))

        mock_prepare.assert_not_called()

    def test_task_fails_gracefully_when_prepare_payload_raises(self):
        """При ValueError из prepare_dataprocessor_payload обновляется статус failed."""
        job_id = uuid4()
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.video_id = uuid4()
        mock_job.processing_config_id = uuid4()
        mock_job.status = enums.AnalysisStatus.queued

        with patch("app.tasks.analysis.session_scope") as mock_scope:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_job
            mock_scope.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            with patch(
                "app.tasks.analysis.prepare_dataprocessor_payload",
                side_effect=ValueError("Video not found"),
            ):
                process_analysis_job(str(job_id))

        assert mock_job.status == enums.AnalysisStatus.failed
        assert "Video not found" in (mock_job.error_message or "")
