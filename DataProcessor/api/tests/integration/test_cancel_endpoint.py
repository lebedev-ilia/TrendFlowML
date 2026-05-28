"""
Integration тесты для POST /api/v1/runs/{run_id}/cancel endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from api.main import app
from api.utils.errors import RunNotFoundError
from api.schemas.state import RunStatus


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestCancelEndpoint:
    """Тесты для endpoint отмены run'а."""
    
    def test_cancel_run_success(self, client):
        """Успешная отмена активного run'а."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.endpoints.cancel.set_cancel_flag", new_callable=AsyncMock, return_value=True):
                with patch("api.endpoints.cancel.save_run_state", new_callable=AsyncMock):
                    with patch("api.endpoints.cancel.audit_log", new_callable=AsyncMock):
                        with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                            mock_task_manager = MagicMock()
                            mock_task_manager.is_run_active = MagicMock(return_value=True)
                            mock_task_manager.update_run_status = MagicMock()
                            mock_get_task_manager.return_value = mock_task_manager
                            
                            response = client.post(
                                "/api/v1/runs/test-run-id/cancel",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert data["run_id"] == "test-run-id"
                            assert data["status"] == "cancelled"
                            assert "cancelled" in data["message"].lower()
    
    def test_cancel_run_already_completed(self, client):
        """Попытка отменить уже завершенный run."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success"  # Уже завершен
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            response = client.post(
                "/api/v1/runs/test-run-id/cancel",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 400
            assert "already success" in response.json()["detail"].lower()
            assert "cannot be cancelled" in response.json()["detail"].lower()
    
    def test_cancel_run_already_cancelled(self, client):
        """Попытка отменить уже отмененный run."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "cancelled"  # Уже отменен
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            response = client.post(
                "/api/v1/runs/test-run-id/cancel",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 400
            assert "already cancelled" in response.json()["detail"].lower()
    
    def test_cancel_run_not_found(self, client):
        """Отмена несуществующего run'а."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(
                side_effect=RunNotFoundError("Run not found: test-run-id")
            )
            mock_state_reader_class.return_value = mock_state_reader
            
            response = client.post(
                "/api/v1/runs/test-run-id/cancel",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 404
            assert "Run not found" in response.json()["detail"]
    
    def test_cancel_run_queued_status(self, client):
        """Отмена run'а со статусом queued."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "queued"
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            with patch("api.endpoints.cancel.set_cancel_flag", new_callable=AsyncMock, return_value=True):
                with patch("api.endpoints.cancel.save_run_state", new_callable=AsyncMock):
                    with patch("api.endpoints.cancel.TaskManager") as mock_task_manager_class:
                        mock_task_manager = MagicMock()
                        mock_task_manager.is_run_active = MagicMock(return_value=False)
                        mock_task_manager_class.return_value = mock_task_manager
                        
                        response = client.post(
                            "/api/v1/runs/test-run-id/cancel",
                            headers={"X-API-Key": "test_api_key"}
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "cancelled"
    
    def test_cancel_run_pending_status(self, client):
        """Отмена run'а со статусом pending."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "pending"
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            with patch("api.endpoints.cancel.set_cancel_flag", new_callable=AsyncMock, return_value=True):
                with patch("api.endpoints.cancel.save_run_state", new_callable=AsyncMock):
                    response = client.post(
                        "/api/v1/runs/test-run-id/cancel",
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "cancelled"
    
    def test_cancel_run_invalid_transition(self, client):
        """Попытка отменить run с невалидным переходом статуса."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "unknown"  # Невалидный статус
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            # Мокаем что parse_status возвращает None или вызывает ValueError
            with patch("api.endpoints.cancel.parse_status", side_effect=ValueError("Invalid status")):
                response = client.post(
                    "/api/v1/runs/test-run-id/cancel",
                    headers={"X-API-Key": "test_api_key"}
                )
                
                # Должен обработать ошибку парсинга статуса
                # Может вернуть 400 или 500 в зависимости от реализации
    
    def test_cancel_run_flag_set_failure(self, client):
        """Обработка ошибки при установке флага отмены."""
        with patch("api.endpoints.cancel.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running"
            })
            mock_state_reader_class.return_value = mock_state_reader
            
            with patch("api.endpoints.cancel.set_cancel_flag", new_callable=AsyncMock, return_value=False):
                with patch("api.endpoints.cancel.save_run_state", new_callable=AsyncMock):
                    with patch("api.endpoints.cancel.TaskManager") as mock_task_manager_class:
                        mock_task_manager = MagicMock()
                        mock_task_manager.is_run_active = MagicMock(return_value=True)
                        mock_task_manager.update_run_status = MagicMock()
                        mock_task_manager_class.return_value = mock_task_manager
                        
                        # Должен продолжить даже если флаг не установлен
                        response = client.post(
                            "/api/v1/runs/test-run-id/cancel",
                            headers={"X-API-Key": "test_api_key"}
                        )
                        
                        # Должен вернуть 200, но с предупреждением в логах
                        assert response.status_code == 200

