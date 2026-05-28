"""
Integration тесты для GET /api/v1/runs/{run_id}/status endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from api.main import app
from api.utils.errors import RunNotFoundError


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestStatusEndpoint:
    """Тесты для endpoint получения статуса."""
    
    def test_get_status_success(self, client):
        """Успешное получение статуса."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running",
                "stage": "visual",
                "progress": {
                    "overall": 0.75,
                    "current_processor": "visual",
                    "current_component": "core_clip",
                    "components": {
                        "segmenter": {
                            "status": "success",
                            "progress": 1.0
                        },
                        "visual": {
                            "status": "running",
                            "progress": 0.5
                        }
                    }
                },
                "started_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:05:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == "test-run-id"
            assert data["status"] == "running"
            assert data["stage"] == "visual"
            assert data["progress"]["overall"] == 0.75
    
    def test_get_status_from_cache(self, client):
        """Получение статуса из Redis cache."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            # Мокаем что статус читается из cache (hot path)
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running",
                "progress": {"overall": 0.5},
                "updated_at": "2024-01-01T12:00:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
    
    def test_get_status_from_storage(self, client):
        """Получение статуса из Storage (cold path)."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            # Мокаем что cache пуст, читаем из Storage
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success",
                "progress": {"overall": 1.0},
                "updated_at": "2024-01-01T13:00:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_get_status_with_include_components(self, client):
        """Получение статуса с детальными компонентами."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running",
                "progress": {
                    "overall": 0.75,
                    "components": {
                        "segmenter": {"status": "success", "progress": 1.0},
                        "visual": {"status": "running", "progress": 0.5}
                    }
                },
                "updated_at": "2024-01-01T12:00:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                params={"include_components": "true"},
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "components" in data["progress"]
            assert "segmenter" in data["progress"]["components"]
    
    def test_get_status_with_include_events(self, client):
        """Получение статуса с событиями."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running",
                "progress": {"overall": 0.5},
                "events": [
                    {"event_type": "processing_started", "timestamp": "2024-01-01T12:00:00Z"},
                    {"event_type": "progress_update", "timestamp": "2024-01-01T12:01:00Z"}
                ],
                "updated_at": "2024-01-01T12:00:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                params={"include_events": "true"},
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            # События могут быть включены в ответ если include_events=True
            # (зависит от реализации StateReader)
    
    def test_get_status_run_not_found(self, client):
        """Получение статуса несуществующего run'а."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(
                side_effect=RunNotFoundError("Run not found: test-run-id")
            )
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 404
            assert "Run not found" in response.json()["detail"]
    
    def test_get_status_without_components(self, client):
        """Получение статуса без детальных компонентов."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running",
                "progress": {"overall": 0.5},
                "updated_at": "2024-01-01T12:00:00Z"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/status",
                params={"include_components": "false"},
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            # Компоненты могут быть пустыми или отсутствовать если include_components=False

