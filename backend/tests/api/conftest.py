"""
Фикстуры для API-тестов: TestClient, мок пользователя, переопределение get_db/get_current_user.

Позволяет тестировать защищённые endpoints без реальной БД и JWT.
См. backend/docs/TESTING_PLAN.md § 3.2.
"""

from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dbv2.models import User as CoreUser


@pytest.fixture
def mock_user():
    """Пользователь для подстановки в get_current_user."""
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "test@example.com"
    u.email_verified = False
    u.created_at = None
    u.updated_at = None
    return u


@pytest.fixture
def client():
    """TestClient без переопределения зависимостей (для проверки 401)."""
    return TestClient(app)


@pytest.fixture
def client_with_user(mock_user):
    """TestClient с переопределением get_current_user и get_db (мок сессии)."""
    from app.deps import get_db, get_current_user

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        # Для списка workspaces: query(Workspace).filter().all() -> []
        session.query.return_value.filter.return_value.all.return_value = []
        session.query.return_value.filter.return_value.first.return_value = None
        session.add = MagicMock()
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
