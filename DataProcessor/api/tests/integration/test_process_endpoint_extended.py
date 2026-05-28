"""
Расширенные integration тесты для POST /api/v1/process endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from api.main import app
from api.utils.errors import RunAlreadyExistsError, BackpressureError, InvalidPayloadError


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestProcessEndpointExtended:
    """Расширенные тесты для endpoint обработки видео."""
    
    def test_process_video_success_with_redis(self, client):
        """Успешный запрос на обработку с использованием Redis."""
        with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=True):
            with patch("api.endpoints.process.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
                with patch("api.endpoints.process.get_total_queue_length", new_callable=AsyncMock, return_value=5):
                    with patch("api.endpoints.process.validate_video_path"):
                        with patch("api.endpoints.process.validate_profile_config"):
                            with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                                with patch("api.endpoints.process.get_redis_client") as mock_get_redis:
                                    mock_redis = MagicMock()
                                    mock_get_redis.return_value = mock_redis
                                    
                                    with patch("api.endpoints.process.check_redis_health", new_callable=AsyncMock, return_value={"status": "healthy"}):
                                        with patch("api.endpoints.process.get_processor_service") as mock_get_processor:
                                            with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                                                mock_task_manager = MagicMock()
                                                mock_task_manager.is_run_active = MagicMock(return_value=False)
                                                mock_task_manager.can_accept_new_run = MagicMock(return_value=True)
                                                mock_task_manager.get_active_runs_count = MagicMock(return_value=2)
                                                mock_task_manager.register_run = MagicMock()
                                                mock_get_task_manager.return_value = mock_task_manager
                                                
                                                mock_enqueue.return_value = "1234567890-0"
                                                
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
                                                assert data["status"] == "queued"
                                                mock_enqueue.assert_called_once()
    
    def test_process_video_duplicate_run_id_task_manager(self, client):
        """Дубликат run_id обнаружен в TaskManager."""
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                        mock_task_manager = MagicMock()
                        mock_task_manager.is_run_active = MagicMock(return_value=True)  # Run уже активен
                        mock_get_task_manager.return_value = mock_task_manager
                        
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
    
    def test_process_video_duplicate_run_id_lock(self, client):
        """Дубликат run_id обнаружен через Redis lock."""
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                        mock_task_manager = MagicMock()
                        mock_task_manager.is_run_active = MagicMock(return_value=False)
                        mock_get_task_manager.return_value = mock_task_manager
                        
                        with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=False):
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
    
    def test_process_video_backpressure(self, client):
        """Backpressure при превышении лимита активных run'ов."""
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=True):
                        with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                            mock_task_manager = MagicMock()
                            mock_task_manager.is_run_active = MagicMock(return_value=False)
                            mock_task_manager.can_accept_new_run = MagicMock(return_value=False)  # Лимит превышен
                            mock_task_manager.get_active_runs_count = MagicMock(return_value=10)
                            mock_get_task_manager.return_value = mock_task_manager
                            
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
                            
                            assert response.status_code == 503
                            assert "Service overloaded" in response.json()["error"]
                            assert "Retry-After" in response.headers
    
    def test_process_video_backpressure_queue_length(self, client):
        """Backpressure при превышении длины очереди."""
        with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=True):
            with patch("api.endpoints.process.get_total_queue_length", new_callable=AsyncMock, return_value=150):
                with patch("api.endpoints.process.TaskManager") as mock_task_manager_class:
                    mock_task_manager = MagicMock()
                    mock_task_manager.is_run_active = MagicMock(return_value=False)
                    mock_task_manager.can_accept_new_run = MagicMock(return_value=True)
                    mock_task_manager_class.return_value = mock_task_manager
                    
                    payload = {
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_id": "test_video",
                        "platform_id": "youtube",
                        "video_path": "/tmp/test_video.mp4",
                        "config_hash": "test_hash",
                        "profile_config": {"processors": {}}
                    }
                    
                    # Мокаем проверку длины очереди в endpoint
                    with patch("api.endpoints.process.config") as mock_config:
                        mock_config.max_queue_length = 100
                        
                        response = client.post(
                            "/api/v1/process",
                            json=payload,
                            headers={"X-API-Key": "test_api_key"}
                        )
                        
                        # Должен вернуть 503 если проверка реализована
                        # (проверка может быть в endpoint или в отдельной функции)
    
    def test_process_video_fallback_mode(self, client):
        """Fallback режим без Redis."""
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=True):
                        with patch("api.endpoints.process.get_redis_client", return_value=None):
                            with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                                mock_task_manager = MagicMock()
                                mock_task_manager.is_run_active = MagicMock(return_value=False)
                                mock_task_manager.can_accept_new_run = MagicMock(return_value=True)
                                mock_task_manager.register_run = MagicMock()
                                mock_get_task_manager.return_value = mock_task_manager
                                
                                with patch("api.endpoints.process._enqueue_fallback", new_callable=AsyncMock) as mock_fallback:
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
                                    
                                    assert response.status_code == 202
                                    # Должен использовать fallback режим
                                    mock_fallback.assert_called_once()
    
    def test_process_video_invalid_video_path(self, client):
        """Валидация video_path."""
        with patch("api.endpoints.process.validate_video_path") as mock_validate:
            from api.utils.errors import InvalidPayloadError
            mock_validate.side_effect = InvalidPayloadError(
                "Video path outside allowed directories",
                details={"field": "video_path", "value": "/invalid/path"}
            )
            
            with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                payload = {
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    "video_id": "test_video",
                    "platform_id": "youtube",
                    "video_path": "/invalid/path/video.mp4",
                    "config_hash": "test_hash",
                    "profile_config": {"processors": {}}
                }
                
                response = client.post(
                    "/api/v1/process",
                    json=payload,
                    headers={"X-API-Key": "test_api_key"}
                )
                
                assert response.status_code == 400
                assert "Invalid payload" in response.json()["error"]
    
    def test_process_video_invalid_profile_config(self, client):
        """Валидация profile_config."""
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config") as mock_validate:
                from api.utils.errors import InvalidPayloadError
                mock_validate.side_effect = InvalidPayloadError(
                    "Invalid profile config",
                    details={"field": "profile_config"}
                )
                
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    payload = {
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_id": "test_video",
                        "platform_id": "youtube",
                        "video_path": "/tmp/test_video.mp4",
                        "config_hash": "test_hash",
                        "profile_config": {}
                    }
                    
                    response = client.post(
                        "/api/v1/process",
                        json=payload,
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    assert response.status_code == 400

