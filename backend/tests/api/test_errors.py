"""
Тесты кодов ошибок API: 401 (нет/невалидный токен), 403 (доступ запрещён), 404 (не найдено).

Проверяют, что защищённые endpoints без Authorization возвращают 401;
конкретные 403/404 покрыты в test_workspaces, test_channels, test_videos, test_analysis.

См. backend/docs/TESTING_PLAN.md § 3.2.8.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def client():
    return TestClient(app)


class Test401Unauthorized:
    """Эндпоинты, требующие JWT, возвращают 401 без токена."""

    def test_me_without_token(self, client: TestClient):
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_workspaces_without_token(self, client: TestClient):
        response = client.get("/api/workspaces")
        assert response.status_code == 401

    def test_channels_without_token(self, client: TestClient):
        response = client.get(
            f"/api/workspaces/{uuid4()}/channels"
        )
        assert response.status_code == 401

    def test_videos_without_token(self, client: TestClient):
        response = client.get(
            f"/api/channels/{uuid4()}/videos"
        )
        assert response.status_code == 401

    def test_analysis_list_without_token(self, client: TestClient):
        response = client.get(
            f"/api/workspaces/{uuid4()}/analysis"
        )
        assert response.status_code == 401

    def test_predictions_without_token(self, client: TestClient):
        response = client.get(
            f"/api/analysis/{uuid4()}/predictions"
        )
        assert response.status_code == 401

    def test_runs_list_without_token(self, client: TestClient):
        response = client.get("/api/runs")
        assert response.status_code == 401

    def test_runs_get_without_token(self, client: TestClient):
        response = client.get(f"/api/runs/{uuid4()}")
        assert response.status_code == 401


class TestInvalidToken:
    """Невалидный Bearer токен — 401."""

    def test_me_with_bad_token(self, client: TestClient):
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert response.status_code == 401
