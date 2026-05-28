"""
Интеграционные тесты клиента Backend → DataProcessor API.

Проверяют вызовы HTTP к DataProcessor (POST /process, GET /status, GET /events)
с замоканным httpx.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Импорт после conftest (backend в path)
from app.services.dataprocessor import (
    run_dataprocessor_async,
    poll_run_status,
    stream_run_events_sse,
    resolve_run_paths,
    RunPaths,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestRunDataprocessorAsync:
    """Тесты асинхронного запуска обработки через DataProcessor API."""

    async def test_sends_post_to_process_endpoint(
        self, sample_run_id, sample_profile_config, tmp_path, mock_settings
    ):
        """Backend отправляет POST на /api/v1/process с корректным payload и заголовками."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"x")
        result_store = tmp_path / "rs"
        frames_dir = tmp_path / "frames"
        visual_cfg = tmp_path / "visual.yaml"
        visual_cfg.write_text("")

        request_payload = None
        request_headers = None

        async def fake_post(url, json=None, headers=None, **kwargs):
            nonlocal request_payload, request_headers
            request_payload = json
            request_headers = headers or {}
            resp = MagicMock()
            resp.status_code = 202
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(
                return_value={
                    "run_id": sample_run_id,
                    "status": "queued",
                    "message": "Processing started",
                }
            )
            return resp

        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=fake_post)
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.services.dataprocessor.Settings") as MockSettings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.dataprocessor_api_url = "http://dp:8001"
            mock_settings_instance.dataprocessor_api_key = "secret"
            mock_settings_instance.resolve_paths.return_value = MagicMock(
                dataproc_root=tmp_path
            )
            MockSettings.return_value = mock_settings_instance

            with patch(
                "app.services.dataprocessor.httpx.AsyncClient",
                return_value=fake_client,
            ):
                run_paths = await run_dataprocessor_async(
                    video_path=video_path,
                    platform_id="youtube",
                    video_id="vid123",
                    run_id=sample_run_id,
                    profile_config=sample_profile_config,
                    result_store_base=result_store,
                    frames_dir_base=frames_dir,
                    visual_cfg_default=visual_cfg,
                )

        assert run_paths is not None
        assert isinstance(run_paths, RunPaths)
        assert request_payload is not None
        assert request_payload["run_id"] == sample_run_id
        assert request_payload["video_id"] == "vid123"
        assert request_payload["platform_id"] == "youtube"
        assert request_payload["video_path"] == str(video_path.absolute())
        assert request_payload["config_hash"] == sample_profile_config.get("config_hash", "")
        assert "profile_config" in request_payload
        assert request_payload.get("chunk_size") == 64
        assert request_payload.get("dag_stage") == "baseline"
        assert request_headers.get("X-API-Key") == "secret"

    async def test_raises_on_http_error(self, sample_profile_config, tmp_path, mock_settings):
        """При 4xx/5xx от DataProcessor API пробрасывается исключение."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"x")
        import httpx

        with patch("app.services.dataprocessor.Settings") as MockSettings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.dataprocessor_api_url = "http://dp:8001"
            mock_settings_instance.dataprocessor_api_key = None
            mock_settings_instance.resolve_paths.return_value = MagicMock(
                dataproc_root=tmp_path
            )
            MockSettings.return_value = mock_settings_instance

            with patch(
                "app.services.dataprocessor.httpx.AsyncClient"
            ) as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 503
                mock_resp.text = "Service Unavailable"
                mock_resp.raise_for_status = MagicMock(
                    side_effect=httpx.HTTPStatusError(
                        "503", request=MagicMock(), response=mock_resp
                    )
                )
                mock_resp.json = MagicMock(return_value={})
                fake_client = MagicMock()
                fake_client.post = AsyncMock(return_value=mock_resp)
                fake_client.__aenter__ = AsyncMock(return_value=fake_client)
                fake_client.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = fake_client

                with pytest.raises(httpx.HTTPStatusError):
                    await run_dataprocessor_async(
                        video_path=video_path,
                        platform_id="upload",
                        video_id="v1",
                        run_id="550e8400-e29b-41d4-a716-446655440001",
                        profile_config=sample_profile_config,
                        result_store_base=tmp_path / "rs",
                        frames_dir_base=tmp_path / "frames",
                        visual_cfg_default=tmp_path / "v.yaml",
                    )


class TestPollRunStatus:
    """Тесты опроса статуса run через DataProcessor API."""

    async def test_poll_returns_final_status(self, sample_run_id, mock_settings):
        """GET /api/v1/runs/{run_id}/status возвращает финальный статус."""
        with patch("app.services.dataprocessor.Settings") as MockSettings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.dataprocessor_api_url = "http://dp:8001"
            mock_settings_instance.dataprocessor_api_key = "key"
            mock_settings_instance.dataprocessor_timeout_seconds = 300
            mock_settings_instance.dataprocessor_poll_interval = 1
            MockSettings.return_value = mock_settings_instance

            call_count = 0

            async def fake_get(url, headers=None, **kwargs):
                nonlocal call_count
                call_count += 1
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json = MagicMock(
                    return_value={
                        "run_id": sample_run_id,
                        "status": "success" if call_count >= 1 else "running",
                        "progress": {"overall": 1.0},
                    }
                )
                return resp

            fake_client = MagicMock()
            fake_client.get = AsyncMock(side_effect=fake_get)
            fake_client.__aenter__ = AsyncMock(return_value=fake_client)
            fake_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "app.services.dataprocessor.httpx.AsyncClient",
                return_value=fake_client,
            ):
                result = await poll_run_status(
                    sample_run_id,
                    timeout_seconds=10,
                    poll_interval=1,
                )

        assert result["status"] == "success"
        assert result["run_id"] == sample_run_id

    async def test_poll_raises_timeout(self, sample_run_id, mock_settings):
        """При истечении timeout выбрасывается TimeoutError."""
        with patch("app.services.dataprocessor.Settings") as MockSettings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.dataprocessor_api_url = "http://dp:8001"
            mock_settings_instance.dataprocessor_api_key = None
            mock_settings_instance.dataprocessor_timeout_seconds = 1
            mock_settings_instance.dataprocessor_poll_interval = 1
            MockSettings.return_value = mock_settings_instance

            async def always_running(*args, **kwargs):
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json = MagicMock(
                    return_value={
                        "run_id": sample_run_id,
                        "status": "running",
                        "progress": {"overall": 0.5},
                    }
                )
                return resp

            fake_client = MagicMock()
            fake_client.get = AsyncMock(side_effect=always_running)
            fake_client.__aenter__ = AsyncMock(return_value=fake_client)
            fake_client.__aexit__ = AsyncMock(return_value=None)

            with patch(
                "app.services.dataprocessor.httpx.AsyncClient",
                return_value=fake_client,
            ):
                with pytest.raises(TimeoutError):
                    await poll_run_status(
                        sample_run_id,
                        timeout_seconds=2,
                        poll_interval=1,
                    )


class TestStreamRunEventsSse:
    """Тесты SSE: GET /api/v1/runs/{run_id}/events (клиент)."""

    async def test_sse_yields_events_until_complete(
        self, sample_run_id, mock_settings
    ):
        """stream_run_events_sse отдаёт события и завершается при event complete."""
        lines = [
            "event: progress",
            'data: {"progress": {"overall": 0.5}}',
            "",
            "event: complete",
            'data: {"status": "success"}',
            "",
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.aiter_lines = fake_aiter_lines

        fake_stream_ctx = MagicMock()
        fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_response)
        fake_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=fake_stream_ctx)
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.services.dataprocessor.Settings") as MockSettings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.dataprocessor_api_url = "http://dp:8001"
            mock_settings_instance.dataprocessor_api_key = "key"
            mock_settings_instance.dataprocessor_timeout_seconds = 300
            MockSettings.return_value = mock_settings_instance

            with patch(
                "app.services.dataprocessor.httpx.AsyncClient",
                return_value=fake_client,
            ):
                events = []
                async for ev in stream_run_events_sse(
                    sample_run_id, timeout_seconds=60
                ):
                    events.append(ev)
                    if ev.get("type") == "complete":
                        break

        assert len(events) >= 1
        assert any(e.get("type") == "complete" for e in events)
        assert any("progress" in str(e) or e.get("data") for e in events)


class TestResolveRunPaths:
    """Тесты формирования путей результатов (контракт с DataProcessor)."""

    def test_paths_match_dataprocessor_layout(self, tmp_path):
        """Пути соответствуют структуре result_store DataProcessor."""
        result_store_base = tmp_path / "result_store"
        paths = resolve_run_paths(
            platform_id="youtube",
            video_id="vid123",
            run_id="550e8400-e29b-41d4-a716-446655440000",
            result_store_base=result_store_base,
        )
        assert paths.run_rs_path == result_store_base / "youtube" / "vid123" / "550e8400-e29b-41d4-a716-446655440000"
        assert paths.manifest_path == paths.run_rs_path / "manifest.json"
        runs_root = result_store_base.parent
        assert paths.state_events_path == (
            runs_root / "state" / "youtube" / "vid123" / "550e8400-e29b-41d4-a716-446655440000" / "state_events.jsonl"
        )
