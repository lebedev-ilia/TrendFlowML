"""
Тесты API workspaces: создание, список, получение по id, права доступа.

Используют dependency overrides (мок пользователя и сессии) для проверки
успешных ответов и структуры данных без реальной БД.

См. backend/docs/TESTING_PLAN.md § 3.2.2.
"""

from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, get_current_user
from app.dbv2.models import User as CoreUser, Workspace


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "owner@example.com"
    u.email_verified = False
    u.created_at = None
    u.updated_at = None
    return u


@pytest.fixture
def client_unauth():
    """Клиент без авторизации — для проверки 401."""
    return TestClient(app)


@pytest.fixture
def client_with_user(mock_user):
    """Клиент с мок-пользователем и мок-сессией (пустой список workspaces)."""
    def override_get_current_user():
        return mock_user

    stored_workspaces = []

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Workspace:
                q.filter.return_value.all.return_value = stored_workspaces.copy()
                q.filter.return_value.first.return_value = None
                joined = MagicMock()
                joined.filter.return_value.first.return_value = None
                joined.filter.return_value.order_by.return_value.all.return_value = (
                    stored_workspaces.copy()
                )
                q.join.return_value = joined

                def add(obj):
                    stored_workspaces.append(obj)

                session.add.side_effect = add
            return q
        session.query.side_effect = query
        session.flush = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestWorkspacesAuth:
    """Проверка обязательности авторизации."""

    def test_list_workspaces_without_token_returns_401(self, client_unauth: TestClient):
        """GET /api/workspaces без токена — 401."""
        response = client_unauth.get("/api/workspaces")
        assert response.status_code == 401


class TestWorkspacesWithMockUser:
    """Эндпоинты с подставленным пользователем и мок-сессией."""

    def test_list_workspaces_returns_200_and_list(
        self, client_with_user: TestClient
    ):
        """GET /api/workspaces с валидным пользователем возвращает 200 и массив."""
        response = client_with_user.get("/api/workspaces")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_workspace_by_id_not_found_returns_404(
        self, client_with_user: TestClient
    ):
        """GET /api/workspaces/{id} при отсутствии workspace — 404."""
        response = client_with_user.get(
            f"/api/workspaces/{uuid4()}"
        )
        assert response.status_code == 404
