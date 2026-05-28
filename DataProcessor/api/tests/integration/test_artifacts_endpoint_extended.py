"""
Расширенные integration тесты для GET /api/v1/runs/{run_id}/artifacts/{component} endpoint
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


class TestArtifactsEndpointExtended:
    """Расширенные тесты для endpoint получения артефактов."""
    
    def test_get_artifact_raw_format(self, client, mock_storage):
        """Получение артефакта в формате raw (NPZ)."""
        manifest_data = {
            "components": [
                {
                    "name": "visual",
                    "artifacts": [
                        {
                            "path": "visual/features.npz",
                            "size_bytes": 1024000
                        }
                    ]
                }
            ]
        }
        
        artifact_bytes = b"NPZ file content"
        
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
                            # Первый вызов - manifest.json, второй - артефакт
                            mock_read.side_effect = [
                                json.dumps(manifest_data).encode("utf-8"),
                                artifact_bytes
                            ]
                            
                            response = client.get(
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                params={"format": "raw"},
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 200
                            assert response.headers["content-type"] == "application/octet-stream"
                            assert response.content == artifact_bytes
                            assert "Content-Disposition" in response.headers
    
    def test_get_artifact_info_format(self, client, mock_storage):
        """Получение метаданных артефакта в формате info."""
        manifest_data = {
            "components": [
                {
                    "name": "visual",
                    "status": "success",
                    "started_at": "2024-01-01T12:00:00Z",
                    "finished_at": "2024-01-01T12:05:00Z",
                    "schema_version": "v1",
                    "producer_version": "1.0.0",
                    "artifacts": [
                        {
                            "path": "visual/features.npz",
                            "size_bytes": 1024000,
                            "schema_version": "v1"
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
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                params={"format": "info"},
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert data["component"] == "visual"
                            assert data["artifact_path"] == "visual/features.npz"
                            assert data["size_bytes"] == 1024000
                            assert data["schema_version"] == "v1"
    
    def test_get_artifact_component_not_found(self, client, mock_storage):
        """Компонент не найден в manifest."""
        manifest_data = {
            "components": [
                {
                    "name": "segmenter",
                    "artifacts": []
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
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 404
                            assert "Component 'visual' not found" in response.json()["detail"]
    
    def test_get_artifact_no_artifacts(self, client, mock_storage):
        """Компонент не имеет артефактов."""
        manifest_data = {
            "components": [
                {
                    "name": "visual",
                    "artifacts": []
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
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 404
                            assert "No artifacts found" in response.json()["detail"]
    
    def test_get_artifact_file_not_found(self, client, mock_storage):
        """Артефакт файл не найден."""
        manifest_data = {
            "components": [
                {
                    "name": "visual",
                    "artifacts": [
                        {
                            "path": "visual/features.npz",
                            "size_bytes": 1024000
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
                "status": "running"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock) as mock_exists:
                        # Первый вызов - manifest существует, второй - артефакт не существует
                        mock_exists.side_effect = [True, False]
                        
                        with patch.object(mock_storage, 'read_bytes', new_callable=AsyncMock) as mock_read:
                            mock_read.return_value = json.dumps(manifest_data).encode("utf-8")
                            
                            response = client.get(
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            assert response.status_code == 404
    
    def test_get_artifact_completed_run_gone(self, client, mock_storage):
        """Артефакт для завершенного run'а больше не доступен (410 Gone)."""
        manifest_data = {
            "components": [
                {
                    "name": "visual",
                    "artifacts": [
                        {
                            "path": "visual/features.npz",
                            "size_bytes": 1024000
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
                "status": "success"  # Завершенный run
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            with patch("api.dependencies.get_storage", return_value=mock_storage):
                with patch("api.dependencies.get_key_layout") as mock_get_key_layout:
                    mock_key_layout = MagicMock()
                    mock_key_layout.result_store_run_prefix = MagicMock(return_value="test/prefix")
                    mock_get_key_layout.return_value = mock_key_layout
                    
                    with patch.object(mock_storage, 'exists', new_callable=AsyncMock) as mock_exists:
                        # Manifest существует, но артефакт не существует
                        mock_exists.side_effect = [True, False]
                        
                        with patch.object(mock_storage, 'read_bytes', new_callable=AsyncMock) as mock_read:
                            mock_read.return_value = json.dumps(manifest_data).encode("utf-8")
                            
                            response = client.get(
                                "/api/v1/runs/test-run-id/artifacts/visual",
                                headers={"X-API-Key": "test_api_key"}
                            )
                            
                            # Должен вернуть 410 Gone для завершенного run'а
                            assert response.status_code == 410
    
    def test_get_artifact_invalid_format(self, client):
        """Невалидный формат запроса."""
        with patch("api.dependencies.get_state_reader") as mock_get_state_reader:
            mock_state_reader = MagicMock()
            mock_state_reader.get_run_status = AsyncMock(return_value={
                "run_id": "test-run-id",
                "video_id": "test_video",
                "platform_id": "youtube",
                "status": "success"
            })
            mock_get_state_reader.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/artifacts/visual",
                params={"format": "invalid"},
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 400
            assert "Invalid format" in response.json()["detail"]

