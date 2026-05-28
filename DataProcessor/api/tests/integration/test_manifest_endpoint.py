"""
Integration тесты для GET /api/v1/runs/{run_id}/manifest endpoint
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from api.main import app
from api.utils.errors import RunNotFoundError


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestManifestEndpoint:
    """Тесты для endpoint получения manifest."""
    
    def test_get_manifest_success(self, client, mock_storage):
        """Успешное получение manifest."""
        manifest_data = {
            "schema_version": "v1",
            "run": {
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "config_hash": "test_hash",
                "created_at": "2024-01-01T12:00:00Z",
                "finished_at": "2024-01-01T13:00:00Z"
            },
            "components": [
                {
                    "name": "segmenter",
                    "kind": "processor",
                    "status": "success",
                    "artifacts": [
                        {
                            "path": "segmenter/features.npz",
                            "size_bytes": 1024000
                        }
                    ]
                },
                {
                    "name": "visual",
                    "kind": "processor",
                    "status": "success",
                    "artifacts": [
                        {
                            "path": "visual/features.npz",
                            "size_bytes": 2048000
                        }
                    ]
                }
            ]
        }
        
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock, return_value=True):
                        with patch.object(mock_storage, 'read_bytes', new_callable=AsyncMock) as mock_read:
                            mock_read.return_value = json.dumps(manifest_data).encode("utf-8")
                            
                            response = client.get(
                                "/api/v1/runs/test-run-id/manifest",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert data["run_id"] == "test-run-id"
                            assert data["video_id"] == "test_video"
                            assert "components" in data
                            assert "segmenter" in data["components"]
                            assert "visual" in data["components"]
    
    def test_get_manifest_run_not_found(self, client):
        """Получение manifest несуществующего run'а."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(
                side_effect=RunNotFoundError("Run not found: test-run-id")
            )
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/manifest",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 404
    
    def test_get_manifest_not_found(self, client, mock_storage):
        """Manifest.json не найден."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "running"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock, return_value=False):
                        response = client.get(
                            "/api/v1/runs/test-run-id/manifest",
                            headers={"X-API-Key": "test_api_key"}
                        )
                        
                        assert response.status_code == 404
    
    def test_get_manifest_completed_run_gone(self, client, mock_storage):
        """Manifest для завершенного run'а больше не доступен (410 Gone)."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success"  # Завершенный run
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock, return_value=False):
                        response = client.get(
                            "/api/v1/runs/test-run-id/manifest",
                            headers={"X-API-Key": "test_api_key"}
                        )
                        
                        # Должен вернуть 410 Gone для завершенного run'а
                        assert response.status_code == 410
    
    def test_get_manifest_invalid_json(self, client, mock_storage):
        """Обработка невалидного JSON в manifest.json."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock, return_value=True):
                        with patch.object(mock_storage, 'read_bytes', new_callable=AsyncMock) as mock_read:
                            mock_read.return_value = b"Invalid JSON {"
                            
                            response = client.get(
                                "/api/v1/runs/test-run-id/manifest",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 500

