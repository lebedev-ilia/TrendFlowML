"""
API-тесты проверки доступа по admin_emails: GET /api/auth/admin-check.

Проверяют, что require_admin_user возвращает 403, если email пользователя
не в TF_BACKEND_ADMIN_EMAILS, и 200 — если в списке.

См. backend/docs/TESTING_PLAN.md § 3.6.4.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_current_user, get_settings


@pytest.fixture
def mock_admin_settings():
    """Настройки с одним админ-email."""
    s = MagicMock()
    s.admin_email_set.return_value = {"admin@example.com"}
    return s


@pytest.fixture
def mock_non_admin_settings():
    """Настройки с другим админ-списком (текущий пользователь не админ)."""
    s = MagicMock()
    s.admin_email_set.return_value = {"root@example.com"}
    return s


@pytest.fixture
def mock_empty_admin_settings():
    """Настройки с пустым списком админов (admin не настроен)."""
    s = MagicMock()
    s.admin_email_set.return_value = set()
    return s


@pytest.fixture
def client_admin_user(mock_user, mock_admin_settings):
    """Client с пользователем admin@example.com и настройками, где он в списке админов."""
    mock_user.email = "admin@example.com"

    def override_get_current_user():
        return mock_user

    def override_get_settings():
        return mock_admin_settings

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_settings] = override_get_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_non_admin_user(mock_user, mock_non_admin_settings):
    """Client с пользователем other@example.com, не в списке админов."""
    mock_user.email = "other@example.com"

    def override_get_current_user():
        return mock_user

    def override_get_settings():
        return mock_non_admin_settings

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_settings] = override_get_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_empty_admin_list(mock_user, mock_empty_admin_settings):
    """Client при пустом admin_emails (админ не настроен)."""
    mock_user.email = "any@example.com"

    def override_get_current_user():
        return mock_user

    def override_get_settings():
        return mock_empty_admin_settings

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_settings] = override_get_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestAdminCheckEndpoint:
    """GET /api/auth/admin-check."""

    def test_admin_check_200_when_user_in_admin_list(self, client_admin_user):
        """Когда email пользователя в admin_emails — 200 и {"admin": true}."""
        r = client_admin_user.get("/api/auth/admin-check")
        assert r.status_code == 200
        assert r.json() == {"admin": True}

    def test_admin_check_403_when_user_not_in_admin_list(self, client_non_admin_user):
        """Когда email пользователя не в admin_emails — 403."""
        r = client_non_admin_user.get("/api/auth/admin-check")
        assert r.status_code == 403
        assert "admin" in r.json().get("detail", "").lower() or "Admin" in r.json().get("detail", "")

    def test_admin_check_403_when_admin_list_empty(self, client_empty_admin_list):
        """Когда admin_emails пустой — 403 (Admin access not configured)."""
        r = client_empty_admin_list.get("/api/auth/admin-check")
        assert r.status_code == 403
        assert "admin" in r.json().get("detail", "").lower() or "configured" in r.json().get("detail", "").lower()
