"""
Тесты API channels: создание канала, список каналов по workspace.

Проверяют 401 без токена, 403 при отсутствии доступа к workspace, 404 при
несуществующем workspace, 200 и структуру ответа при успехе (с мок-сессией).

См. backend/docs/TESTING_PLAN.md § 3.2.3.
"""

from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, get_current_user
from app.dbv2.models import User as CoreUser, Workspace, Channel, WorkspaceMember


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "user@example.com"
    u.email_verified = False
    u.created_at = None
    u.updated_at = None
    return u


@pytest.fixture
def workspace_id():
    return uuid4()


@pytest.fixture
def client_unauth():
    return TestClient(app)


@pytest.fixture
def client_member(mock_user, workspace_id):
    """Клиент с пользователем — членом workspace (список каналов пустой)."""
    mock_ws = MagicMock()
    mock_ws.id = workspace_id
    mock_member = MagicMock()
    mock_member.workspace_id = workspace_id
    mock_member.user_id = mock_user.id

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Workspace:
                q.filter.return_value.first.return_value = mock_ws
            elif model == WorkspaceMember:
                q.filter.return_value.first.return_value = mock_member
            elif model == Channel:
                q.filter.return_value.order_by.return_value.all.return_value = []
            return q
        session.query.side_effect = query
        session.add = MagicMock()
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


@pytest.fixture
def client_not_member(mock_user, workspace_id):
    """Клиент с пользователем, не являющимся членом workspace (403)."""
    mock_ws = MagicMock()
    mock_ws.id = workspace_id

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Workspace:
                q.filter.return_value.first.return_value = mock_ws
            elif model == WorkspaceMember:
                q.filter.return_value.first.return_value = None  # не член
            return q
        session.query.side_effect = query
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_workspace_not_found(mock_user, workspace_id):
    """Workspace не найден — 404."""
    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Workspace:
                q.filter.return_value.first.return_value = None
            return q
        session.query.side_effect = query
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestChannelsAuth:
    def test_list_channels_without_token_returns_401(self, client_unauth, workspace_id):
        """GET /api/workspaces/{id}/channels без токена — 401."""
        response = client_unauth.get(f"/api/workspaces/{workspace_id}/channels")
        assert response.status_code == 401


class TestChannelsAccess:
    def test_list_channels_returns_200_and_list(
        self, client_member, workspace_id
    ):
        """GET /api/workspaces/{id}/channels с членством — 200 и массив."""
        response = client_member.get(f"/api/workspaces/{workspace_id}/channels")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_channels_not_member_returns_403(
        self, client_not_member, workspace_id
    ):
        """GET при отсутствии членства в workspace — 403."""
        response = client_not_member.get(
            f"/api/workspaces/{workspace_id}/channels"
        )
        assert response.status_code == 403
        assert "access denied" in response.json().get("detail", "").lower()

    def test_list_channels_workspace_not_found_returns_404(
        self, client_workspace_not_found, workspace_id
    ):
        """GET при несуществующем workspace — 404."""
        response = client_workspace_not_found.get(
            f"/api/workspaces/{workspace_id}/channels"
        )
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()
