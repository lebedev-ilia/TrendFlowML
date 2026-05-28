"""
Integration тесты для POST /api/v1/process endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from api.main import app


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestProcessEndpoint:
    """Тесты для endpoint обработки видео."""
    
    def test_process_video_success(self, client):
        """Успешный запрос на обработку видео."""
        with patch("api.endpoints.process.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = True
            
            payload = {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "video_id": "test_video",
                "platform_id": "youtube",
                "video_path": "/tmp/test_video.mp4",
                "config_hash": "test_hash",
                "profile_config": {
                    "processors": {
                        "segmenter": {"enabled": True, "required": True},
                        "visual": {"enabled": True, "required": True}
                    }
                }
            }
            
            response = client.post(
                "/api/v1/process",
                json=payload,
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["run_id"] == payload["run_id"]
            assert data["status"] in ["queued", "running"]
            assert "status_url" in data
    
    def test_process_video_missing_api_key(self, client):
        """Запрос без API ключа."""
        payload = {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {"processors": {}}
        }
        
        response = client.post("/api/v1/process", json=payload)
        
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]
    
    def test_process_video_invalid_api_key(self, client):
        """Запрос с невалидным API ключом."""
        payload = {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {"processors": {}}
        }
        
        response = client.post(
            "/api/v1/process",
            json=payload,
            headers={"X-API-Key": "invalid_key"}
        )
        
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]
    
    def test_process_video_invalid_payload(self, client):
        """Запрос с невалидным payload."""
        payload = {
            "run_id": "invalid-uuid",
            "video_id": "",
            "platform_id": "invalid_platform",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {}
        }
        
        response = client.post(
            "/api/v1/process",
            json=payload,
            headers={"X-API-Key": "test_api_key"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_video_duplicate_run_id(self, client):
        """Запрос с дубликатом run_id."""
        with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock) as mock_lock:
            from api.utils.errors import RunAlreadyExistsError
            mock_lock.side_effect = RunAlreadyExistsError("test-run-id")
            
            payload = {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "video_id": "test_video",
                "platform_id": "youtube",
                "video_path": "/tmp/test_video.mp4",
                "config_hash": "test_hash",
                "profile_config": {"processors": {}}
            }
            
            response = client.post(
                "/api/v1/process",
                json=payload,
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 409
            assert "Run already exists" in response.json()["error"]

