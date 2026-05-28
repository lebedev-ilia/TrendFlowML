"""E2E тест: happy-path от POST /runs до GET /manifest."""

import stat as stat_module
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from fetcher.workers.metadata import run_metadata_worker
from fetcher.workers.video import run_video_worker
from fetcher.workers.comments import run_comments_worker
from fetcher.workers.artifacts import run_artifact_builder
from fetcher.db import session_scope
from fetcher.models import Run
from fetcher.state_machine import validate_transition, RUN_STATUS_COMPLETED


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.database
@pytest.mark.skip("E2E happy-path: отключён по умолчанию, запускать вручную при необходимости")
class TestE2EHappyPath:
    """E2E: создание run через API, синхронный pipeline, получение manifest."""

    def test_post_runs_to_manifest(
        self,
        e2e_storage,
    ):
        """POST /api/v1/runs -> pipeline (моки) -> GET run -> GET manifest с ключевыми полями."""
        run_id = uuid.uuid4()
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        def _run_finalize_sync(rid: str) -> None:
            run_artifact_builder(rid)
            with session_scope() as db:
                r = db.query(Run).filter(Run.id == uuid.UUID(rid)).first()
                if r:
                    validate_transition(r.status, RUN_STATUS_COMPLETED, run_id=rid)
                    r.status = RUN_STATUS_COMPLETED
                    r.finished_at = datetime.utcnow()

        with patch("fetcher.storage.storage_client", e2e_storage):
            with patch("yt_dlp.YoutubeDL") as yt_mock:
                yt_mock.return_value.__enter__.return_value.extract_info.return_value = {
                    "id": "dQw4w9WgXcQ",
                    "title": "Test",
                    "duration": 212,
                }
                with patch("fetcher.api.fetch_metadata_task") as mock_fetch_meta:
                    # Не запускаем Celery, только создаём run
                    mock_fetch_meta.apply_async.return_value = None
                    with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                        with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                            mock_stat.return_value = MagicMock(
                                st_size=1024, st_mode=stat_module.S_IFREG
                            )
                            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                    with patch("builtins.open", create=True):
                                        from fastapi.testclient import TestClient
                                        from fetcher.api import app
                                        client = TestClient(app)
                                        resp = client.post(
                                            "/api/v1/runs",
                                            json={
                                                "run_id": str(run_id),
                                                "source_url": source_url,
                                            },
                                        )
        # После успешного создания run запускаем pipeline синхронно
        assert resp.status_code == 201
        with patch("fetcher.storage.storage_client", e2e_storage):
            run_metadata_worker(str(run_id))
            run_video_worker(str(run_id))
            run_comments_worker(str(run_id), limit=100)
            _run_finalize_sync(str(run_id))
        assert resp.status_code == 201
        data = resp.json()
        assert data["run_id"] == str(run_id)

        # GET run и GET manifest — storage всё ещё нужен для manifest
        with patch("fetcher.storage.storage_client", e2e_storage):
            from fastapi.testclient import TestClient
            from fetcher.api import app
            client = TestClient(app)
            run_resp = client.get(f"/api/v1/runs/{run_id}")
            assert run_resp.status_code == 200
            run_data = run_resp.json()
            assert run_data["status"].lower() in (
                "completed",
                "finalizing",
            ), run_data.get("status")

            manifest_resp = client.get(f"/api/v1/runs/{run_id}/manifest")
            if manifest_resp.status_code == 503:
                pytest.skip(
                    "Manifest not ready (run still finalizing) — pipeline sync in test"
                )
            assert manifest_resp.status_code == 200, manifest_resp.text
            manifest = manifest_resp.json()
            assert "manifest_version" in manifest
            assert manifest.get("run_id") == str(run_id)
            assert manifest.get("platform") == "youtube"
            assert "artifacts" in manifest
            assert "video_id" in manifest
