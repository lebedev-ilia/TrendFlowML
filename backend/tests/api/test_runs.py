"""
Тесты API runs (ингестиция по URL): создание run, список, получение по id, trigger-processing.

Эндпоинты: POST /api/runs, GET /api/runs, GET /api/runs/{run_id},
POST /api/runs/{run_id}/trigger-processing. Мок Fetcher API и БД.

См. backend/docs/TESTING_PLAN.md, backend/docs/FETCHER_INTEGRATION.md,
app/routers/runs.py.
"""

from __future__ import annotations

from uuid import uuid4
from datetime import datetime
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


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "user@example.com"
    return u


@pytest.fixture
def client_unauth():
    return TestClient(app)


@pytest.fixture
def client_with_user(mock_user):
    """Клиент с мок-пользователем и мок-сессией (пустой список runs)."""
    stored_runs: list = []

    def override_get_current_user():
        return mock_user

    def override_get_db():
        session = MagicMock()
        def query(model):
            q = MagicMock()
            if model == IngestionRun:
                q.filter.return_value.first.return_value = None
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


class TestRunsAuth:
    def test_list_runs_without_token_returns_401(self, client_unauth):
        """GET /api/runs без токена — 401."""
        response = client_unauth.get("/api/runs")
        assert response.status_code == 401

    def test_create_run_without_token_returns_401(self, client_unauth):
        """POST /api/runs без токена — 401."""
        response = client_unauth.post(
            "/api/runs",
            json={"source_url": "https://youtube.com/watch?v=xxx"},
        )
        assert response.status_code == 401


class TestRunsCreateAndList:
    @patch("app.routers.runs.fetcher_create_run_async", new_callable=AsyncMock)
    def test_create_run_returns_201_and_calls_fetcher(
        self, mock_fetcher_create, client_with_user
    ):
        """POST /api/runs с валидным source_url — 201, вызов Fetcher API замокан."""
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
        assert data["ingestion_status"] in ("pending", "running", "PENDING".lower())
        mock_fetcher_create.assert_called_once()

    def test_list_runs_returns_200_and_list(self, client_with_user):
        """GET /api/runs с токеном — 200 и массив."""
        response = client_with_user.get("/api/runs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestRunsCreateIdempotency:
    """POST /api/runs: заголовок Idempotency-Key и существующий run (GAPS / портфолио 5.2)."""

    @patch("app.routers.runs.fetcher_create_run_async", new_callable=AsyncMock)
    def test_create_run_idempotency_returns_existing_without_fetcher(
        self, mock_fetcher, mock_user
    ):
        """Повтор с тем же ключом — 201, тот же run_id, Fetcher не вызывается."""
        existing_rid = uuid4()
        existing = MagicMock()
        existing.run_id = existing_rid
        existing.source_url = "https://www.youtube.com/watch?v=existing"
        existing.workspace_id = None
        existing.ingestion_status = "running"
        existing.created_at = datetime.utcnow()
        existing.updated_at = datetime.utcnow()
        existing.fetcher_stage = None
        existing.fetcher_error_code = None
        existing.fetcher_error_message = None

        def override_get_current_user():
            return mock_user

        def override_get_db():
            session = MagicMock()

            def query(model):
                q = MagicMock()
                if model == IngestionRun:
                    q.filter.return_value.first.return_value = existing
                return q

            session.query.side_effect = query
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                response = c.post(
                    "/api/runs",
                    json={"source_url": "https://www.youtube.com/watch?v=other"},
                    headers={"Idempotency-Key": "portfolio-idem-1"},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 201
        data = response.json()
        assert data["run_id"] == str(existing_rid)
        assert data.get("message") == "Run already exists (idempotency key matched)"
        mock_fetcher.assert_not_called()


class TestRunsGet:
    def test_get_run_not_found_returns_404(self, client_with_user):
        """GET /api/runs/{run_id} при отсутствии run у пользователя — 404."""
        response = client_with_user.get(f"/api/runs/{uuid4()}")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()


class TestRunsTriggerProcessing:
    def test_trigger_processing_run_not_found_returns_404(self):
        """POST /api/runs/{run_id}/trigger-processing при отсутствии run в БД — 404."""
        def override_get_db():
            session = MagicMock()
            session.query.return_value.filter.return_value.first.return_value = None
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        with patch("app.routers.runs.Settings") as MockSettings:
            MockSettings.return_value.run_trigger_api_key = None
            try:
                with TestClient(app) as c:
                    response = c.post(
                        f"/api/runs/{uuid4()}/trigger-processing",
                    )
            finally:
                app.dependency_overrides.pop(get_db, None)
        assert response.status_code == 404

    @patch("app.routers.runs.process_ingestion_run")
    def test_trigger_processing_run_exists_returns_202(self, mock_process):
        """POST trigger-processing при найденном run — 202, Celery task ставится в очередь."""
        run_id = uuid4()
        mock_run = MagicMock()
        mock_run.run_id = run_id

        def override_get_db():
            session = MagicMock()
            session.query.return_value.filter.return_value.first.return_value = mock_run
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        with patch("app.routers.runs.Settings") as MockSettings:
            MockSettings.return_value.run_trigger_api_key = None
            try:
                with TestClient(app) as c:
                    response = c.post(
                        f"/api/runs/{run_id}/trigger-processing",
                    )
            finally:
                app.dependency_overrides.pop(get_db, None)
        assert response.status_code == 202
        data = response.json()
        assert data.get("status") == "accepted"
        assert data.get("run_id") == str(run_id)
        mock_process.delay.assert_called_once_with(str(run_id))

    @patch("app.routers.runs.process_ingestion_run")
    def test_trigger_processing_idempotent_when_already_processing(self, mock_process):
        """Phase 5.4: при ingestion_status=processing повторный trigger — 202, task не ставится."""
        run_id = uuid4()
        mock_run = MagicMock()
        mock_run.run_id = run_id
        mock_run.ingestion_status = "processing"

        def override_get_db():
            session = MagicMock()
            session.query.return_value.filter.return_value.first.return_value = mock_run
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        with patch("app.routers.runs.Settings") as MockSettings:
            MockSettings.return_value.run_trigger_api_key = None
            try:
                with TestClient(app) as c:
                    response = c.post(
                        f"/api/runs/{run_id}/trigger-processing",
                    )
            finally:
                app.dependency_overrides.pop(get_db, None)
        assert response.status_code == 202
        data = response.json()
        assert data.get("message") == "Processing already triggered"
        mock_process.delay.assert_not_called()
