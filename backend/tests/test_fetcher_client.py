"""
Unit-тесты HTTP-клиента Backend → Fetcher API (Phase 0).

Проверяют контракт запрос/ответ с замоканным httpx:
- create_run / create_run_async — POST /api/v1/runs, заголовки X-API-Key, Idempotency-Key
- get_run / get_run_async — GET /api/v1/runs/{run_id}
- get_run_manifest / get_run_manifest_async — GET /api/v1/runs/{run_id}/manifest
- get_run_artifacts / get_run_artifacts_async — GET /api/v1/runs/{run_id}/artifacts

Контракт: Fetcher/docs/BACKEND_CONTRACTS.md, Fetcher/fetcher/schemas/api.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
import pytest

from app.services.fetcher_client import (
    create_run,
    create_run_async,
    get_run,
    get_run_async,
    get_run_artifacts,
    get_run_artifacts_async,
    get_run_manifest,
    get_run_manifest_async,
)


RUN_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
SOURCE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def mock_fetcher_settings():
    """Настройки Backend с URL и API key Fetcher."""
    with patch("app.services.fetcher_client.Settings") as MockSettings:
        s = MagicMock()
        s.fetcher_api_url = "http://fetcher-test:8002"
        s.fetcher_api_key = "fetcher-test-key"
        s.fetcher_timeout_seconds = 30.0
        MockSettings.return_value = s
        yield s


class TestCreateRun:
    """Тесты POST /api/v1/runs (create_run)."""

    def test_sends_post_with_run_id_and_source_url(self, mock_fetcher_settings):
        """Клиент отправляет POST с run_id и source_url в теле."""
        request_payload = None
        request_headers = None

        def fake_post(url, json=None, headers=None, **kwargs):
            nonlocal request_payload, request_headers
            request_payload = json
            request_headers = headers or {}
            resp = MagicMock()
            resp.status_code = 201
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(
                return_value={
                    "run_id": str(RUN_ID),
                    "status": "PENDING",
                    "source_url": SOURCE_URL,
                    "platform": "youtube",
                    "created_at": "2026-03-10T12:00:00Z",
                    "message": "Run created",
                }
            )
            return resp

        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post = fake_post
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            result = create_run(RUN_ID, SOURCE_URL, settings=mock_fetcher_settings)

        assert result["run_id"] == str(RUN_ID)
        assert result["status"] == "PENDING"
        assert result["source_url"] == SOURCE_URL
        assert request_payload is not None
        assert request_payload["run_id"] == str(RUN_ID)
        assert request_payload["source_url"] == SOURCE_URL
        assert request_payload["priority"] == "normal"
        assert request_headers.get("X-API-Key") == "fetcher-test-key"
        assert request_headers.get("Content-Type") == "application/json"

    def test_sends_optional_idempotency_key(self, mock_fetcher_settings):
        """При передаче idempotency_key заголовок Idempotency-Key устанавливается."""
        request_headers = None

        def capture_post(url, json=None, headers=None, **kwargs):
            nonlocal request_headers
            request_headers = headers or {}
            resp = MagicMock()
            resp.status_code = 201
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"run_id": str(RUN_ID), "status": "PENDING", "message": "OK"})
            return resp

        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post = capture_post
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            create_run(
                RUN_ID,
                SOURCE_URL,
                idempotency_key="key-123",
                settings=mock_fetcher_settings,
            )

        assert request_headers.get("Idempotency-Key") == "key-123"

    def test_raises_on_http_error(self, mock_fetcher_settings):
        """При 4xx/5xx от Fetcher пробрасывается HTTPStatusError."""
        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 409
            mock_resp.text = "Conflict"
            mock_resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError("409", request=MagicMock(), response=mock_resp)
            )
            mock_client_instance.post = lambda *a, **k: mock_resp
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            with pytest.raises(httpx.HTTPStatusError):
                create_run(RUN_ID, SOURCE_URL, settings=mock_fetcher_settings)


class TestGetRun:
    """Тесты GET /api/v1/runs/{run_id} (get_run)."""

    def test_sends_get_with_run_id(self, mock_fetcher_settings):
        """Клиент отправляет GET на правильный URL с заголовком X-API-Key."""
        request_url = None
        request_headers = None

        def fake_get(url, headers=None, **kwargs):
            nonlocal request_url, request_headers
            request_url = url
            request_headers = headers or {}
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(
                return_value={
                    "run_id": str(RUN_ID),
                    "status": "COMPLETED",
                    "source_url": SOURCE_URL,
                    "platform": "youtube",
                    "platform_video_id": "dQw4w9WgXcQ",
                    "created_at": "2026-03-10T12:00:00Z",
                    "started_at": "2026-03-10T12:00:01Z",
                    "finished_at": "2026-03-10T12:05:00Z",
                }
            )
            return resp

        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = fake_get
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            result = get_run(RUN_ID, settings=mock_fetcher_settings)

        assert result["run_id"] == str(RUN_ID)
        assert result["status"] == "COMPLETED"
        assert "fetcher-test:8002" in request_url
        assert "/api/v1/runs/" in request_url
        assert str(RUN_ID) in request_url
        assert request_headers.get("X-API-Key") == "fetcher-test-key"


class TestGetRunManifest:
    """Тесты GET /api/v1/runs/{run_id}/manifest."""

    def test_returns_manifest_dict(self, mock_fetcher_settings):
        """Клиент возвращает словарь manifest (version, run_id, artifacts)."""
        manifest_data = {
            "manifest_version": "1.0",
            "run_id": str(RUN_ID),
            "video_id": "dQw4w9WgXcQ",
            "platform": "youtube",
            "duration_seconds": 212.0,
            "storage_layout_version": "1.0",
            "artifacts": {
                "video_file": {"path": "raw/youtube/2026/03/10/dQw4w9WgXcQ/video.mp4", "checksum": "sha256:abc"},
                "meta_file": {"path": "raw/youtube/2026/03/10/dQw4w9WgXcQ/meta.json"},
            },
        }

        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = lambda url, **kwargs: MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value=manifest_data),
            )
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            result = get_run_manifest(RUN_ID, settings=mock_fetcher_settings)

        assert result["manifest_version"] == "1.0"
        assert result["run_id"] == str(RUN_ID)
        assert result["platform"] == "youtube"
        assert "artifacts" in result
        assert "video_file" in result["artifacts"]


class TestGetRunArtifacts:
    """Тесты GET /api/v1/runs/{run_id}/artifacts."""

    def test_returns_artifacts_dict(self, mock_fetcher_settings):
        """Клиент возвращает словарь с артефактами (download_url и т.д.)."""
        artifacts_data = {
            "artifacts": [
                {
                    "artifact_type": "video_file",
                    "download_url": "https://storage.example/signed/video.mp4",
                    "download_url_expires_at": "2026-03-10T13:00:00Z",
                    "size_bytes": 12345678,
                    "artifact_status": "READY",
                }
            ],
        }

        with patch("app.services.fetcher_client.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = lambda url, **kwargs: MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value=artifacts_data),
            )
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client_instance

            result = get_run_artifacts(RUN_ID, settings=mock_fetcher_settings)

        assert "artifacts" in result
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["artifact_type"] == "video_file"
        assert "download_url" in result["artifacts"][0]


@pytest.mark.asyncio
class TestCreateRunAsync:
    """Тесты асинхронного create_run_async."""

    async def test_sends_post_and_returns_response(self, mock_fetcher_settings):
        """Async клиент отправляет POST и возвращает ответ Fetcher."""
        from unittest.mock import AsyncMock

        async def fake_post(url, json=None, headers=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 201
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(
                return_value={
                    "run_id": str(RUN_ID),
                    "status": "PENDING",
                    "source_url": SOURCE_URL,
                    "message": "Run created",
                }
            )
            return resp

        with patch("app.services.fetcher_client.httpx.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(side_effect=fake_post)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await create_run_async(RUN_ID, SOURCE_URL, settings=mock_fetcher_settings)

        assert result["run_id"] == str(RUN_ID)
        assert result["status"] == "PENDING"


@pytest.mark.asyncio
class TestGetRunAsync:
    """Тесты асинхронного get_run_async."""

    async def test_returns_run_status(self, mock_fetcher_settings):
        """Async get_run возвращает статус run."""
        from unittest.mock import AsyncMock

        run_data = {
            "run_id": str(RUN_ID),
            "status": "FETCHING_METADATA",
            "source_url": SOURCE_URL,
            "platform": "youtube",
        }

        with patch("app.services.fetcher_client.httpx.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    raise_for_status=MagicMock(),
                    json=MagicMock(return_value=run_data),
                )
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await get_run_async(RUN_ID, settings=mock_fetcher_settings)

        assert result["status"] == "FETCHING_METADATA"
        assert result["run_id"] == str(RUN_ID)


@pytest.mark.asyncio
class TestGetRunManifestAsync:
    """Тесты асинхронного get_run_manifest_async."""

    async def test_returns_manifest(self, mock_fetcher_settings):
        """Async get_run_manifest возвращает manifest."""
        from unittest.mock import AsyncMock

        manifest = {
            "manifest_version": "1.0",
            "run_id": str(RUN_ID),
            "video_id": "abc",
            "platform": "youtube",
            "duration_seconds": 100,
            "storage_layout_version": "1.0",
            "artifacts": {},
        }

        with patch("app.services.fetcher_client.httpx.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    raise_for_status=MagicMock(),
                    json=MagicMock(return_value=manifest),
                )
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await get_run_manifest_async(RUN_ID, settings=mock_fetcher_settings)

        assert result["manifest_version"] == "1.0"
        assert result["run_id"] == str(RUN_ID)


@pytest.mark.asyncio
class TestGetRunArtifactsAsync:
    """Тесты асинхронного get_run_artifacts_async."""

    async def test_returns_artifacts(self, mock_fetcher_settings):
        """Async get_run_artifacts возвращает артефакты."""
        from unittest.mock import AsyncMock

        with patch("app.services.fetcher_client.httpx.AsyncClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    raise_for_status=MagicMock(),
                    json=MagicMock(return_value={"artifacts": []}),
                )
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = await get_run_artifacts_async(RUN_ID, settings=mock_fetcher_settings)

        assert "artifacts" in result
        assert result["artifacts"] == []
