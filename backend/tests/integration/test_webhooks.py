"""
Интеграционные тесты webhook DataProcessor: POST /api/webhooks/dataprocessor.

Проверяют приём payload, валидацию подписи X-Webhook-Signature, обновление AnalysisJob
и ответ 200. Используются моки session_scope и publish_run_event.

См. backend/docs/TESTING_PLAN.md § 3.1.7, app/routers/webhooks.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dbv2 import enums


pytestmark = pytest.mark.integration


def _make_mock_analysis_job(job_id: str, status=enums.AnalysisStatus.processing):
    """Мок AnalysisJob с атрибутами, которые меняет webhook."""
    job = MagicMock()
    job.id = job_id
    job.status = status
    job.error_message = None
    job.completed_at = None
    return job


@contextmanager
def _mock_session_scope(mock_job):
    """Контекстный менеджер, подменяющий session_scope и возвращающий сессию с mock job."""
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = mock_job
    mock_db.query.return_value.filter.return_value.first.return_value = mock_job
    mock_db.flush = MagicMock()

    @contextmanager
    def fake_scope():
        yield mock_db

    with patch("app.routers.webhooks.session_scope", fake_scope):
        yield mock_db


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def webhook_payload():
    run_id = str(uuid4())
    return {
        "run_id": run_id,
        "status": "success",
        "progress": {"overall": 1.0},
        "error": None,
        "error_code": None,
        "timestamp": "2024-01-01T12:00:00Z",
    }


class TestDataprocessorWebhook:
    """POST /api/webhooks/dataprocessor."""

    def test_webhook_success_updates_job_and_returns_200(
        self, client, webhook_payload
    ):
        """При валидной подписи и найденном AnalysisJob возвращается 200, статус обновляется."""
        run_id = webhook_payload["run_id"]
        mock_job = _make_mock_analysis_job(run_id)

        with _mock_session_scope(mock_job):
            with patch(
                "app.routers.webhooks.publish_run_event",
                new_callable=AsyncMock,
            ) as mock_publish:
                with patch("app.routers.webhooks.settings") as mock_settings:
                    mock_settings.dataprocessor_api_key = "secret-key"

                    response = client.post(
                        "/api/webhooks/dataprocessor",
                        json=webhook_payload,
                        headers={"X-Webhook-Signature": "secret-key"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert data.get("run_id") == run_id
        assert mock_job.status == enums.AnalysisStatus.completed
        assert mock_job.completed_at is not None
        mock_publish.assert_called_once()

    def test_webhook_401_without_valid_signature(self, client, webhook_payload):
        """Без подписи или с неверной подписью при настроенном API key — 401."""
        with patch("app.routers.webhooks.settings") as mock_settings:
            mock_settings.dataprocessor_api_key = "required-key"

            response = client.post(
                "/api/webhooks/dataprocessor",
                json=webhook_payload,
            )
        assert response.status_code == 401
        assert "Invalid" in response.json().get("detail", "")

    def test_webhook_404_when_job_not_found(self, client, webhook_payload):
        """Когда AnalysisJob не найден — 404."""
        with _mock_session_scope(None):  # first() вернёт None
            with patch("app.routers.webhooks.settings") as mock_settings:
                mock_settings.dataprocessor_api_key = None  # dev mode, без проверки подписи

                response = client.post(
                    "/api/webhooks/dataprocessor",
                    json=webhook_payload,
                )

        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_webhook_error_status_sets_failed(self, client, webhook_payload):
        """status=error обновляет job в failed и error_message."""
        webhook_payload["status"] = "error"
        webhook_payload["error"] = "Processing failed"
        run_id = webhook_payload["run_id"]
        mock_job = _make_mock_analysis_job(run_id)

        with _mock_session_scope(mock_job):
            with patch("app.routers.webhooks.publish_run_event", new_callable=AsyncMock):
                with patch("app.routers.webhooks.settings") as mock_settings:
                    mock_settings.dataprocessor_api_key = None

                    response = client.post(
                        "/api/webhooks/dataprocessor",
                        json=webhook_payload,
                    )

        assert response.status_code == 200
        assert mock_job.status == enums.AnalysisStatus.failed
        assert mock_job.error_message == "Processing failed"

    def test_webhook_cancelled_sets_canceled_status(self, client, webhook_payload):
        """status=cancelled → AnalysisJob.canceled (ветка отмены из GAPS)."""
        webhook_payload["status"] = "cancelled"
        run_id = webhook_payload["run_id"]
        mock_job = _make_mock_analysis_job(run_id)

        with _mock_session_scope(mock_job):
            with patch("app.routers.webhooks.publish_run_event", new_callable=AsyncMock):
                with patch("app.routers.webhooks.settings") as mock_settings:
                    mock_settings.dataprocessor_api_key = None

                    response = client.post(
                        "/api/webhooks/dataprocessor",
                        json=webhook_payload,
                    )

        assert response.status_code == 200
        assert mock_job.status == enums.AnalysisStatus.canceled
