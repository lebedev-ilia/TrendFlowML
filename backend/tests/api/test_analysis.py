"""
Тесты API analysis: создание analysis job, список jobs по workspace,
создание и список predictions по analysis_job_id.

Проверяют 401 без токена, 403/404 при отсутствии доступа или сущности,
200/201 и структуру ответа, **POST /api/analysis/{id}/cancel** (очередь, processing, noop).
Celery process_analysis_job.delay замокан где нужно.

См. backend/docs/TESTING_PLAN.md § 3.2.5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deps import get_db, get_current_user
from app.dbv2.models import (
    User as CoreUser,
    Workspace,
    WorkspaceMember,
    Video,
    AnalysisJob,
    Prediction,
)
from app.dbv2.enums import AnalysisStatus


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def mock_user():
    u = MagicMock(spec=CoreUser)
    u.id = uuid4()
    u.email = "user@example.com"
    return u


@pytest.fixture
def workspace_id():
    return uuid4()


@pytest.fixture
def video_id():
    return uuid4()


@pytest.fixture
def analysis_job_id():
    return uuid4()


@pytest.fixture
def client_unauth():
    return TestClient(app)


@pytest.fixture
def client_with_access(mock_user, workspace_id, video_id, analysis_job_id):
    """Пользователь — член workspace; video в workspace; список jobs и predictions пустой."""
    mock_ws = MagicMock()
    mock_ws.id = workspace_id
    mock_member = MagicMock()
    mock_video = MagicMock()
    mock_video.id = video_id
    mock_video.channel_id = uuid4()
    mock_job = MagicMock()
    mock_job.id = analysis_job_id
    mock_job.workspace_id = workspace_id

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
            elif model == Video:
                q.filter.return_value.first.return_value = mock_video
            elif model == AnalysisJob:
                q.filter.return_value.first.return_value = mock_job
                q.filter.return_value.order_by.return_value.all.return_value = []
            elif model == Prediction:
                q.filter.return_value.order_by.return_value.all.return_value = []
            return q
        session.query.side_effect = query
        session.add = MagicMock()
        session.commit = MagicMock()

        def refresh(obj):
            if isinstance(obj, AnalysisJob):
                if getattr(obj, "id", None) is None:
                    obj.id = analysis_job_id
                now = datetime.now(timezone.utc)
                if getattr(obj, "created_at", None) is None:
                    obj.created_at = now
                if getattr(obj, "updated_at", None) is None:
                    obj.updated_at = now

        session.refresh = MagicMock(side_effect=refresh)
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
def client_video_not_found(mock_user, workspace_id, video_id):
    """Video не найден — 404 при создании job."""
    mock_ws = MagicMock()
    mock_ws.id = workspace_id
    mock_member = MagicMock()

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
            elif model == Video:
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


class TestAnalysisAuth:
    def test_list_analysis_without_token_returns_401(
        self, client_unauth, workspace_id
    ):
        """GET /api/workspaces/{id}/analysis без токена — 401."""
        response = client_unauth.get(
            f"/api/workspaces/{workspace_id}/analysis"
        )
        assert response.status_code == 401


class TestAnalysisJobs:
    @patch("app.routers.analysis.process_analysis_job")
    def test_create_analysis_job_returns_201_and_calls_celery(
        self, mock_delay, client_with_access, workspace_id, video_id
    ):
        """POST .../videos/{video_id}/analysis — 201, Celery task ставится в очередь."""
        mock_delay.delay = MagicMock(return_value=None)
        response = client_with_access.post(
            f"/api/workspaces/{workspace_id}/videos/{video_id}/analysis",
            json={
                "processing_config_id": str(uuid4()),
                "model_version_id": "v1",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data.get("status") == AnalysisStatus.queued.value
        mock_delay.delay.assert_called_once()

    def test_list_analysis_jobs_returns_200(
        self, client_with_access, workspace_id
    ):
        """GET /api/workspaces/{id}/analysis — 200 и массив."""
        response = client_with_access.get(
            f"/api/workspaces/{workspace_id}/analysis"
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_analysis_job_video_not_found_returns_404(
        self, client_video_not_found, workspace_id, video_id
    ):
        """POST при несуществующем video — 404."""
        response = client_video_not_found.post(
            f"/api/workspaces/{workspace_id}/videos/{video_id}/analysis",
            json={
                "processing_config_id": str(uuid4()),
                "model_version_id": "v1",
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()


class TestAnalysisCancel:
    """POST /api/analysis/{id}/cancel — queued, processing, noop."""

    def test_cancel_queued_marks_canceled(self, mock_user, workspace_id, analysis_job_id):
        fake_job = SimpleNamespace(
            id=analysis_job_id,
            workspace_id=workspace_id,
            status=AnalysisStatus.queued,
            completed_at=None,
        )
        mock_ws = MagicMock()
        mock_ws.id = workspace_id
        mock_member = MagicMock()

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
                elif model == AnalysisJob:
                    q.filter.return_value.first.return_value = fake_job
                return q

            session.query.side_effect = query
            session.commit = MagicMock()
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                r = c.post(f"/api/analysis/{analysis_job_id}/cancel")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200
        assert r.json()["status"] == "canceled"
        assert fake_job.status == AnalysisStatus.canceled
        assert fake_job.completed_at is not None

    def test_cancel_completed_is_noop(self, mock_user, workspace_id, analysis_job_id):
        fake_job = MagicMock(spec=AnalysisJob)
        fake_job.id = analysis_job_id
        fake_job.workspace_id = workspace_id
        fake_job.status = AnalysisStatus.completed
        mock_ws = MagicMock()
        mock_ws.id = workspace_id
        mock_member = MagicMock()

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
                elif model == AnalysisJob:
                    q.filter.return_value.first.return_value = fake_job
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
                r = c.post(f"/api/analysis/{analysis_job_id}/cancel")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200
        assert r.json()["status"] == "noop"

    @patch("app.routers.analysis.request_dataprocessor_cancel", return_value=True)
    def test_cancel_processing_calls_dataprocessor(
        self, mock_dp_cancel, mock_user, workspace_id, analysis_job_id
    ):
        fake_job = MagicMock(spec=AnalysisJob)
        fake_job.id = analysis_job_id
        fake_job.workspace_id = workspace_id
        fake_job.status = AnalysisStatus.processing
        mock_ws = MagicMock()
        mock_ws.id = workspace_id
        mock_member = MagicMock()

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
                elif model == AnalysisJob:
                    q.filter.return_value.first.return_value = fake_job
                return q

            session.query.side_effect = query
            session.commit = MagicMock()
            try:
                yield session
            finally:
                pass

        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                r = c.post(f"/api/analysis/{analysis_job_id}/cancel")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200
        assert r.json()["status"] == "cancel_requested"
        assert r.json()["dataprocessor_notified"] is True
        mock_dp_cancel.assert_called_once_with(str(analysis_job_id))


class TestPredictions:
    def test_list_predictions_returns_200(
        self, client_with_access, analysis_job_id
    ):
        """GET /api/analysis/{job_id}/predictions — 200 и массив."""
        response = client_with_access.get(
            f"/api/analysis/{analysis_job_id}/predictions"
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_predictions_job_not_found_returns_404(
        self, client_with_access
    ):
        """GET при несуществующем analysis_job_id — 404."""
        fake_job_id = uuid4()
        # Нужен клиент, где query(AnalysisJob).filter().first() вернёт None
        mock_user = MagicMock(spec=CoreUser)
        mock_user.id = uuid4()
        mock_ws = MagicMock()
        mock_member = MagicMock()
        mock_ws.id = uuid4()

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
                elif model == AnalysisJob:
                    q.filter.return_value.first.return_value = None
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
                response = c.get(f"/api/analysis/{fake_job_id}/predictions")
        finally:
            app.dependency_overrides.clear()
        assert response.status_code == 404
