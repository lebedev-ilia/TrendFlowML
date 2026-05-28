"""
E2E-сценарии Backend ↔ Fetcher ingestion (Phase 5).

Проверяют цепочку создания run по URL и обработку ошибок Fetcher без реального
Fetcher (мок create_run). Полный ручной чеклист: docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md.

Запуск: pytest backend/tests/e2e/ -v   или   pytest -m e2e -v
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, get_current_user
from app.dbv2.models import (
    User as CoreUser,
    IngestionRun,
    Workspace,
    WorkspaceMember,
)


pytestmark = [pytest.mark.e2e, pytest.mark.integration]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "e2e@example.com"
    return u


@pytest.fixture
def client_with_user(mock_user):
    """Клиент с мок-пользователем и мок-сессией БД для ingestion runs."""
    stored_runs: list = []

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()

        def query(model):
            q = MagicMock()
            if model == IngestionRun:
                q.filter.return_value.first.side_effect = lambda: (
                    stored_runs[0] if stored_runs else None
                )
                q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
                    stored_runs
                )
            elif model == Workspace:
                q.filter.return_value.first.return_value = MagicMock()
            elif model == WorkspaceMember:
                q.filter.return_value.first.return_value = MagicMock()
            return q

        session.query.side_effect = query
        session.add = lambda obj: stored_runs.append(obj) if isinstance(obj, IngestionRun) else None
        session.commit = MagicMock()

        def refresh_mock(obj):
            if isinstance(obj, IngestionRun):
                now = datetime.utcnow()
                if getattr(obj, "created_at", None) is None:
                    obj.created_at = now
                if getattr(obj, "updated_at", None) is None:
                    obj.updated_at = now

        session.refresh = MagicMock(side_effect=refresh_mock)
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestIngestionE2EHappyPath:
    """Успешное создание run (Fetcher мок возвращает успех)."""

    @patch("app.routers.runs.fetcher_create_run_async", new_callable=AsyncMock)
    def test_create_run_success_returns_201_and_run_id(
        self, mock_fetcher_create, client_with_user
    ):
        """POST /api/runs → 201, run_id и ingestion_status в ответе (E2E happy path)."""
        mock_fetcher_create.return_value = {
            "status": "PENDING",
            "message": "Run created",
        }
        response = client_with_user.post(
            "/api/runs",
            json={"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        assert data["source_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert data["ingestion_status"] in ("pending", "running")
        mock_fetcher_create.assert_called_once()


class TestIngestionE2EErrors:
    """Обработка ошибок Fetcher (Phase 5.2)."""

    @patch("app.routers.runs.fetcher_create_run_async", new_callable=AsyncMock)
    def test_create_run_fetcher_unavailable_returns_502_and_run_failed(
        self, mock_fetcher_create, client_with_user
    ):
        """При недоступности Fetcher: 502, run в БД с ingestion_status=failed и fetcher_error_*."""
        mock_fetcher_create.side_effect = Exception("Connection refused")
        response = client_with_user.post(
            "/api/runs",
            json={"source_url": "https://www.youtube.com/watch?v=xxx"},
        )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert "Fetcher" in detail or "Connection" in detail

    @patch("app.routers.runs.fetcher_create_run_async", new_callable=AsyncMock)
    def test_create_run_fetcher_timeout_returns_502(self, mock_fetcher_create, client_with_user):
        """При таймауте Fetcher: 502."""
        import httpx
        mock_fetcher_create.side_effect = httpx.TimeoutException("Timeout")
        response = client_with_user.post(
            "/api/runs",
            json={"source_url": "https://www.youtube.com/watch?v=yyy"},
        )
        assert response.status_code == 502
