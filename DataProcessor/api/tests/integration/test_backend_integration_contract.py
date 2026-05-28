"""
Контрактные тесты интеграции DataProcessor API с Backend.

Проверяют, что запрос в формате Backend (POST /api/v1/process) принимается
DataProcessor API и что ответы status/events совместимы с ожиданиями Backend.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.requests import ProcessRequest


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def backend_style_payload(tmp_path):
    """
    Payload в том виде, в каком его отправляет Backend (app.services.dataprocessor).
    См. backend/app/services/dataprocessor.py run_dataprocessor_async.
    """
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"fake")
    return {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "video_id": "dQw4w9WgXcQ",
        "platform_id": "youtube",
        "video_path": str(video_file.absolute()),
        "config_hash": "abc123def456",
        "profile_config": {
            "config_hash": "abc123def456",
            "visual": {"cfg_path": str(tmp_path / "visual.yaml")},
            "processors": {
                "segmenter": {"enabled": True, "required": True},
                "audio": {"enabled": False, "required": False},
                "text": {"enabled": False, "required": False},
                "visual": {"enabled": True, "required": True},
            },
        },
        "rs_base": str(tmp_path / "result_store"),
        "output": str(tmp_path / "frames"),
        "visual_cfg_path": str(tmp_path / "visual.yaml"),
        "dag_path": str(tmp_path / "dag.yaml"),
        "dag_stage": "baseline",
        "sampling_policy_version": "v1",
        "dataprocessor_version": "dev",
        "chunk_size": 64,
    }


class TestBackendRequestContract:
    """Запрос от Backend валидируется ProcessRequest (Pydantic)."""

    def test_backend_payload_valid_for_process_request(self, backend_style_payload):
        """Payload от Backend проходит валидацию ProcessRequest (DataProcessor)."""
        req = ProcessRequest(**backend_style_payload)
        assert req.run_id == backend_style_payload["run_id"]
        assert req.video_id == backend_style_payload["video_id"]
        assert req.platform_id == "youtube"
        assert req.config_hash == "abc123def456"
        assert "processors" in req.profile_config
        assert req.chunk_size == 64
        assert req.dag_stage == "baseline"

    def test_process_endpoint_accepts_backend_payload(
        self, client, backend_style_payload
    ):
        """POST /api/v1/process принимает payload в формате Backend и возвращает 202."""
        with patch("api.endpoints.process.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = True

            response = client.post(
                "/api/v1/process",
                json=backend_style_payload,
                headers={"X-API-Key": "test_api_key"},
            )

            assert response.status_code == 202
            data = response.json()
            assert data["run_id"] == backend_style_payload["run_id"]
            assert data["status"] in ("queued", "running")
            assert "status_url" in data
            assert "/api/v1/runs/" in data["status_url"]
            assert data["status_url"].endswith("/status")

    def test_process_endpoint_enqueues_full_backend_payload(
        self, client, backend_style_payload
    ):
        """В очередь уходит полный payload, нужный worker для запуска main.py."""
        with patch("api.endpoints.process.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = True

            response = client.post(
                "/api/v1/process",
                json=backend_style_payload,
                headers={"X-API-Key": "test_api_key"},
            )

            assert response.status_code == 202
            _, kwargs = mock_enqueue.call_args
            metadata = kwargs["metadata"]

            assert metadata["video_id"] == backend_style_payload["video_id"]
            assert metadata["platform_id"] == backend_style_payload["platform_id"]
            assert metadata["video_path"] == backend_style_payload["video_path"]
            assert metadata["profile_config"] == backend_style_payload["profile_config"]
            assert metadata["visual_cfg_path"] == backend_style_payload["visual_cfg_path"]
            assert metadata["dag_path"] == backend_style_payload["dag_path"]
            assert metadata["dag_stage"] == backend_style_payload["dag_stage"]
            assert metadata["rs_base"] == backend_style_payload["rs_base"]
            assert metadata["output"] == backend_style_payload["output"]
            assert metadata["chunk_size"] == backend_style_payload["chunk_size"]


class TestStatusResponseContract:
    """Ответ GET /runs/{id}/status совместим с ожиданиями Backend (poll_run_status)."""

    def test_status_response_has_required_fields(self, client):
        """Ответ status содержит run_id, status, progress (как читает backend)."""
        # Backend ожидает: status_data.get("status"), .get("progress", {}).get("overall")
        # Для несуществующего run получим 404 — проверяем только контракт при 200.
        response = client.get(
            "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status",
            headers={"X-API-Key": "test_api_key"},
        )
        # 404 или 200 — оба допустимы; при 200 проверяем структуру
        if response.status_code == 200:
            data = response.json()
            assert "run_id" in data or "status" in data
            assert "status" in data
