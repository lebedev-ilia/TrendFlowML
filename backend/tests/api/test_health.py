"""Тесты health endpoints (без JWT)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthLive:
    def test_health_root_returns_live(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "live"}

    def test_health_live_returns_live(self, client: TestClient):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "live"}


class TestHealthReady:
    def test_health_ready_ok_with_mocks(self, client: TestClient, monkeypatch):
        class _Conn:
            def execute(self, *_a, **_kw):
                return None

        class _ConnCtx:
            def __enter__(self):
                return _Conn()

            def __exit__(self, *_a):
                return False

        monkeypatch.setattr("app.routers.health.engine.connect", lambda: _ConnCtx())

        class _Redis:
            def ping(self):
                return True

            def close(self):
                pass

        monkeypatch.setattr("app.routers.health.redis.from_url", lambda _url: _Redis())

        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["database"] == "ok"
        assert body["redis"] == "ok"

    def test_health_ready_503_when_database_fails(self, client: TestClient, monkeypatch):
        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr("app.routers.health.engine.connect", _boom)

        class _Redis:
            def ping(self):
                return True

            def close(self):
                pass

        monkeypatch.setattr("app.routers.health.redis.from_url", lambda _url: _Redis())

        response = client.get("/health/ready")
        assert response.status_code == 503
        assert "checks_failed" in response.json()["detail"]
        assert "database" in response.json()["detail"]["checks_failed"]
