"""
Интеграционные тесты WebSocket-эндпоинта /api/runs/{id}/events.

Проверяют:
- отказ без или с неверным JWT (закрытие до accept);
- успешное подключение с ?token= (владелец run);
- получение событий из subscribe_run_events;
- обработку ошибок subscribe_run_events (код 1011).

См. backend/docs/TESTING_PLAN.md § 3.4.2–3.4.3, app/routers/runs.py, SECURITY.md.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.auth import create_access_token
from app.dbv2.models import IngestionRun
from app.dbv2.models import User as CoreUser
from app.deps import get_db
from app.main import app


pytestmark = pytest.mark.integration


@pytest.fixture
def ws_run_session_override():
    """Мок БД: пользователь и IngestionRun с совпадающим user_id (для прохождения WS auth)."""
    user_id = uuid4()
    run_id = uuid4()
    user = MagicMock(spec=CoreUser)
    user.id = user_id
    run = MagicMock(spec=IngestionRun)
    run.run_id = run_id
    run.user_id = user_id

    def query_side_effect(model):
        q = MagicMock()
        if model is CoreUser:
            q.filter.return_value.first.return_value = user
        elif model is IngestionRun:
            q.filter.return_value.first.return_value = run
        else:
            q.filter.return_value.first.return_value = None
        return q

    session = MagicMock()
    session.query.side_effect = query_side_effect

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    token = create_access_token(str(user_id))
    try:
        yield run_id, token
    finally:
        app.dependency_overrides.clear()


class TestWsRunEvents:
    """WebSocket /api/runs/{id}/events."""

    def test_ws_run_events_rejects_without_token(self):
        run_id = uuid4()
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(f"/api/runs/{run_id}/events"):
                    pass
        assert exc.value.code == 1008

    def test_ws_run_events_rejects_invalid_token(self, ws_run_session_override):
        run_id, _ = ws_run_session_override
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(
                    f"/api/runs/{run_id}/events?token=not-a-jwt",
                ):
                    pass
        assert exc.value.code == 1008

    def test_ws_run_events_streams_messages_from_subscribe(self, ws_run_session_override):
        """Клиент с валидным token получает события, которые возвращает subscribe_run_events."""
        run_id, token = ws_run_session_override
        messages = [
            {"type": "run.status_changed", "status": "running"},
            {"type": "run.status_changed", "status": "succeeded"},
        ]

        async def fake_subscribe(run_id_str: str):
            assert run_id_str == str(run_id)
            for payload in messages:
                yield payload

        with patch(
            "app.routers.runs.subscribe_run_events",
            side_effect=fake_subscribe,
        ):
            with TestClient(app) as client:
                url = f"/api/runs/{run_id}/events?token={token}"
                with client.websocket_connect(url) as ws:
                    first = json.loads(ws.receive_text())
                    second = json.loads(ws.receive_text())

        assert first == messages[0]
        assert second == messages[1]

    def test_ws_run_events_closes_on_internal_error(self, ws_run_session_override):
        """При исключении внутри subscribe_run_events соединение закрывается с ошибкой."""
        run_id, token = ws_run_session_override

        async def broken_subscribe(run_id_str: str):
            raise RuntimeError("boom")
            yield {}  # async generator for `async for`; unreachable

        with patch(
            "app.routers.runs.subscribe_run_events",
            side_effect=broken_subscribe,
        ):
            with TestClient(app) as client:
                with client.websocket_connect(
                    f"/api/runs/{run_id}/events?token={token}",
                ) as ws:
                    with pytest.raises(WebSocketDisconnect) as exc:
                        ws.receive_text()

        assert exc.value.code == 1011
