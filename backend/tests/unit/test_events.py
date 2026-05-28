"""
Unit-тесты сервиса событий (publish_run_event, run_channel, формат payload).

Используется мок Redis. См. backend/docs/TESTING_PLAN.md § 3.4.1, EVENTS_AND_LOGGING.md.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.events import run_channel, publish_run_event


pytestmark = pytest.mark.unit


class TestRunChannel:
    """Формат имени канала Redis для run."""

    def test_run_channel_format(self):
        """run_channel(run_id) возвращает 'run:<run_id>'."""
        assert run_channel("abc-123") == "run:abc-123"
        assert run_channel("550e8400-e29b-41d4-a716-446655440000") == (
            "run:550e8400-e29b-41d4-a716-446655440000"
        )


class TestPublishRunEvent:
    """Публикация события в Redis."""

    @pytest.mark.asyncio
    async def test_publish_run_event_calls_redis_publish(self):
        """publish_run_event публикует JSON payload в канал run:{run_id}."""
        mock_publish = AsyncMock(return_value=1)
        mock_close = AsyncMock(return_value=None)

        with patch("app.services.events.redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.publish = mock_publish
            mock_client.close = mock_close
            mock_redis.from_url.return_value = mock_client

            await publish_run_event(
                "run-1",
                {"type": "run.status_changed", "payload": {"status": "success"}},
            )

        mock_redis.from_url.assert_called_once()
        mock_publish.assert_called_once()
        channel = mock_publish.call_args[0][0]
        message = mock_publish.call_args[0][1]
        assert channel == "run:run-1"
        payload = json.loads(message)
        assert payload["type"] == "run.status_changed"
        assert payload["payload"]["status"] == "success"
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_run_event_serializes_unicode(self):
        """Payload сериализуется с ensure_ascii=False (кириллица сохраняется)."""
        mock_publish = AsyncMock(return_value=1)
        mock_close = AsyncMock(return_value=None)

        with patch("app.services.events.redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.publish = mock_publish
            mock_client.close = mock_close
            mock_redis.from_url.return_value = mock_client

            await publish_run_event("r1", {"message": "Ошибка обработки"})

        message = mock_publish.call_args[0][1]
        assert "Ошибка" in message
        assert json.loads(message)["message"] == "Ошибка обработки"
