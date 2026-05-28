"""
Integration тесты для artifacts endpoint

Тестирует:
- Получение артефактов в формате raw
- Получение артефактов в формате info
- Получение presigned URL для артефактов (S3)
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from storage.base import Storage


@pytest.fixture
def mock_storage_with_presigned():
    """Фикстура для мок Storage с поддержкой presigned URL."""
    storage = Mock(spec=Storage)
    storage.exists = AsyncMock(return_value=True)
    storage.read_bytes = AsyncMock(return_value=b"fake npz content")
    storage.generate_presigned_url = Mock(
        return_value="https://s3.amazonaws.com/bucket/key?X-Amz-Algorithm=..."
    )
    return storage


@pytest.fixture
def mock_storage_without_presigned():
    """Фикстура для мок Storage без поддержки presigned URL."""
    storage = Mock(spec=Storage)
    storage.exists = AsyncMock(return_value=True)
    storage.read_bytes = AsyncMock(return_value=b"fake npz content")
    # Нет метода generate_presigned_url
    return storage


@pytest.mark.integration
class TestArtifactsEndpoint:
    """Тесты для artifacts endpoint."""
    
    @patch("api.endpoints.artifacts.get_storage")
    @patch("api.endpoints.artifacts.get_key_layout")
    @patch("api.endpoints.artifacts.get_state_reader")
    def test_get_artifact_raw_format(
        self,
        mock_get_state_reader,
        mock_get_key_layout,
        mock_get_storage,
        mock_storage_with_presigned,
        client
    ):
        """Тест получения артефакта в формате raw."""
        # Настроить моки
        mock_get_storage.return_value = mock_storage_with_presigned
        
        mock_key_layout = Mock()
        mock_key_layout.result_store_run_prefix.return_value = "test/prefix"
        mock_get_key_layout.return_value = mock_key_layout
        
        mock_state_reader = Mock()
        mock_state_reader.get_run_status = AsyncMock(return_value={
            "run_id": "test-run-id",
            "status": "success",
            "components": {
                "test_component": {
                    "status": "success",
                    "artifact_path": "test/artifact.npz"
                }
            }
        })
        mock_get_state_reader.return_value = mock_state_reader
        
        # Выполнить запрос
        response = client.get(
            "/api/v1/runs/test-run-id/artifacts/test_component?format=raw",
            headers={"X-API-Key": "test_api_key"}
        )
        
        # Проверить результат
        assert response.status_code == 200
        assert response.content == b"fake npz content"
        assert response.headers["Content-Type"] == "application/octet-stream"
    
    @patch("api.endpoints.artifacts.get_storage")
    @patch("api.endpoints.artifacts.get_key_layout")
    @patch("api.endpoints.artifacts.get_state_reader")
    def test_get_artifact_info_format(
        self,
        mock_get_state_reader,
        mock_get_key_layout,
        mock_get_storage,
        mock_storage_with_presigned,
        client
    ):
        """Тест получения артефакта в формате info."""
        # Настроить моки
        mock_get_storage.return_value = mock_storage_with_presigned
        
        mock_key_layout = Mock()
        mock_key_layout.result_store_run_prefix.return_value = "test/prefix"
        mock_get_key_layout.return_value = mock_key_layout
        
        mock_state_reader = Mock()
        mock_state_reader.get_run_status = AsyncMock(return_value={
            "run_id": "test-run-id",
            "status": "success",
            "components": {
                "test_component": {
                    "status": "success",
                    "artifact_path": "test/artifact.npz",
                    "started_at": "2024-01-01T00:00:00Z"
                }
            }
        })
        mock_get_state_reader.return_value = mock_state_reader
        
        # Выполнить запрос
        response = client.get(
            "/api/v1/runs/test-run-id/artifacts/test_component?format=info",
            headers={"X-API-Key": "test_api_key"}
        )
        
        # Проверить результат
        assert response.status_code == 200
        data = response.json()
        assert data["component"] == "test_component"
        assert data["artifact_path"] == "test/artifact.npz"
    
    @patch("api.endpoints.artifacts.get_storage")
    @patch("api.endpoints.artifacts.get_key_layout")
    @patch("api.endpoints.artifacts.get_state_reader")
    def test_get_artifact_presigned_url_s3(
        self,
        mock_get_state_reader,
        mock_get_key_layout,
        mock_get_storage,
        mock_storage_with_presigned,
        client
    ):
        """Тест получения presigned URL для артефакта (S3)."""
        # Настроить моки
        mock_get_storage.return_value = mock_storage_with_presigned
        
        mock_key_layout = Mock()
        mock_key_layout.result_store_run_prefix.return_value = "test/prefix"
        mock_get_key_layout.return_value = mock_key_layout
        
        mock_state_reader = Mock()
        mock_state_reader.get_run_status = AsyncMock(return_value={
            "run_id": "test-run-id",
            "status": "success",
            "components": {
                "test_component": {
                    "status": "success",
                    "artifact_path": "test/artifact.npz"
                }
            }
        })
        mock_get_state_reader.return_value = mock_state_reader
        
        # Выполнить запрос
        response = client.get(
            "/api/v1/runs/test-run-id/artifacts/test_component?format=url",
            headers={"X-API-Key": "test_api_key"}
        )
        
        # Проверить результат
        assert response.status_code == 200
        data = response.json()
        assert data["component"] == "test_component"
        assert "url" in data
        assert data["url"] == "https://s3.amazonaws.com/bucket/key?X-Amz-Algorithm=..."
        assert data["expires_in"] == 3600
        
        # Проверить вызов generate_presigned_url
        mock_storage_with_presigned.generate_presigned_url.assert_called_once_with(
            key="test/prefix/test/artifact.npz",
            expiration=3600,
            http_method="GET"
        )
    
    @patch("api.endpoints.artifacts.get_storage")
    @patch("api.endpoints.artifacts.get_key_layout")
    @patch("api.endpoints.artifacts.get_state_reader")
    def test_get_artifact_presigned_url_fs(
        self,
        mock_get_state_reader,
        mock_get_key_layout,
        mock_get_storage,
        mock_storage_without_presigned,
        client
    ):
        """Тест получения presigned URL для артефакта (FileSystemStorage)."""
        # Настроить моки
        mock_get_storage.return_value = mock_storage_without_presigned
        
        mock_key_layout = Mock()
        mock_key_layout.result_store_run_prefix.return_value = "test/prefix"
        mock_get_key_layout.return_value = mock_key_layout
        
        mock_state_reader = Mock()
        mock_state_reader.get_run_status = AsyncMock(return_value={
            "run_id": "test-run-id",
            "status": "success",
            "components": {
                "test_component": {
                    "status": "success",
                    "artifact_path": "test/artifact.npz"
                }
            }
        })
        mock_get_state_reader.return_value = mock_state_reader
        
        # Выполнить запрос
        response = client.get(
            "/api/v1/runs/test-run-id/artifacts/test_component?format=url",
            headers={"X-API-Key": "test_api_key"}
        )
        
        # Проверить результат
        assert response.status_code == 200
        data = response.json()
        assert data["component"] == "test_component"
        assert "url" in data
        # FileSystemStorage возвращает относительный путь
        assert data["url"] == "/api/v1/runs/test-run-id/artifacts/test_component?format=raw"
        assert data["expires_in"] is None
        assert "note" in data

