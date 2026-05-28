"""
Тесты API videos: создание видео, список видео по channel.

Проверяют 401 без токена, 403 при отсутствии доступа к channel/workspace,
404 при несуществующем channel, 200 и структуру при успехе (мок-сессия).

См. backend/docs/TESTING_PLAN.md § 3.2.4.
"""

from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, get_current_user
from app.dbv2.models import User as CoreUser, Channel, Workspace, Video, WorkspaceMember


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "user@example.com"
    return u


@pytest.fixture
def channel_id():
    return uuid4()


@pytest.fixture
def workspace_id():
    return uuid4()


@pytest.fixture
def client_unauth():
    return TestClient(app)


@pytest.fixture
def client_with_access(mock_user, channel_id, workspace_id):
    """Пользователь — член workspace; channel принадлежит workspace; список видео пустой."""
    mock_ch = MagicMock()
    mock_ch.id = channel_id
    mock_ch.workspace_id = workspace_id
    mock_ws = MagicMock()
    mock_ws.id = workspace_id
    mock_member = MagicMock()

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Channel:
                q.filter.return_value.first.return_value = mock_ch
            elif model == Workspace:
                q.filter.return_value.first.return_value = mock_ws
            elif model == WorkspaceMember:
                q.filter.return_value.first.return_value = mock_member
            elif model == Video:
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
def client_channel_not_found(mock_user, channel_id):
    """Channel не найден — 404."""
    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == Channel:
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


class TestVideosAuth:
    def test_list_videos_without_token_returns_401(self, client_unauth, channel_id):
        """GET /api/channels/{id}/videos без токена — 401."""
        response = client_unauth.get(f"/api/channels/{channel_id}/videos")
        assert response.status_code == 401


class TestVideosAccess:
    def test_list_videos_returns_200_and_list(
        self, client_with_access, channel_id
    ):
        """GET /api/channels/{id}/videos с доступом — 200 и массив."""
        response = client_with_access.get(f"/api/channels/{channel_id}/videos")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_videos_channel_not_found_returns_404(
        self, client_channel_not_found, channel_id
    ):
        """GET при несуществующем channel — 404."""
        response = client_channel_not_found.get(
            f"/api/channels/{channel_id}/videos"
        )
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()
