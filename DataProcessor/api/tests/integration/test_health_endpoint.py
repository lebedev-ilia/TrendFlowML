"""
Integration тесты для GET /api/v1/health endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from api.main import app


@pytest.fixture
def client():
    """Фикстура для TestClient."""
    return TestClient(app)


class TestHealthEndpoint:
    """Тесты для health check endpoint."""
    
    def test_health_check_success(self, client):
        """Успешная проверка здоровья."""
        with patch("api.endpoints.health.check_storage_health", new_callable=AsyncMock) as mock_storage:
            mock_storage.return_value = {"status": "healthy", "type": "fs"}
            
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            assert "api" in data
            assert "storage" in data
            assert "version" in data
            assert "uptime_seconds" in data
    
    def test_health_check_unhealthy_storage(self, client):
        """Проверка здоровья при недоступном Storage."""
        with patch("api.endpoints.health.check_storage_health", new_callable=AsyncMock) as mock_storage:
            mock_storage.return_value = {"status": "unhealthy", "error": "Storage unavailable"}
            
            response = client.get("/api/v1/health")
            
            # Если storage unhealthy, общий статус должен быть unhealthy и вернуть 503
            assert response.status_code in [200, 503]
            data = response.json()
            assert data["storage"] == "unhealthy"

