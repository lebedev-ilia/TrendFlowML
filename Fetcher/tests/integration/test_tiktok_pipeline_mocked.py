"""Integration test: TikTok pipeline (mocked external calls).

Requires Postgres (skipped if unavailable by conftest).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from fetcher.db import session_scope
from fetcher.models import Run, VideoSource
from fetcher.orchestrator import fetch_video
from fetcher.workers.comments import run_comments_worker
from fetcher.workers.metadata import run_metadata_worker
from fetcher.workers.video import run_video_worker


@pytest.mark.integration
class TestTikTokPipelineMocked:
    @pytest.fixture
    def tiktok_run(self):
        import uuid

        run_id = uuid.uuid4()
        url = "https://www.tiktok.com/@u/video/7351234567890123456"
        with session_scope() as db:
            run = Run(
                id=run_id,
                source_type="tiktok",
                source_url=url,
                status="PENDING",
                started_at=datetime.now(timezone.utc),
            )
            db.add(run)
            db.flush()
            vs = VideoSource(
                run_id=run_id,
                platform="tiktok",
                url=url,
                normalized_video_id=None,  # normalize_source should fill
            )
            db.add(vs)
            db.commit()
        return type("RunRef", (), {"id": run_id})()

    @patch("fetcher.platforms.registry.settings")
    @patch("fetcher.platforms.tiktok.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.tiktok.adapter.acquire_token")
    @patch("fetcher.platforms.tiktok.adapter.acquire_video_lock")
    @patch("fetcher.platforms.tiktok.adapter.release_video_lock")
    @patch("fetcher.platforms.tiktok.adapter.get_next_proxy")
    @patch("fetcher.platforms.tiktok.adapter.storage_client")
    @patch("fetcher.platforms.tiktok.adapter.compute_sha256")
    @patch("fetcher.platforms.tiktok.adapter.yt_dlp.YoutubeDL")
    def test_full_pipeline_tiktok_success(
        self,
        mock_ydl,
        mock_checksum,
        mock_storage,
        mock_proxy,
        mock_release_lock,
        mock_lock,
        mock_rate_limit,
        mock_breaker,
        mock_registry_settings,
        tiktok_run,
    ):
        # Enable TikTok in registry settings for this test
        mock_registry_settings.enabled_platforms = ["youtube", "tiktok"]
        mock_registry_settings.tiktok_enabled = True

        breaker = MagicMock()
        breaker.is_open.return_value = False
        mock_breaker.return_value = breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_proxy.return_value = None
        mock_checksum.return_value = "abc123def456"
        mock_storage.upload_file.return_value = None

        ydl_instance = MagicMock()
        ydl_instance.extract_info.return_value = {
            "id": "7351234567890123456",
            "title": "Test TikTok",
            "description": "Desc",
            "duration": 12,
            "uploader": "Creator",
            "uploader_id": "creator123",
        }
        ydl_instance.prepare_filename.return_value = "/tmp/7351234567890123456.mp4"
        mock_ydl.return_value.__enter__.return_value = ydl_instance

        # Run orchestrator with tasks executed synchronously
        with patch("fetcher.orchestrator.finalize_task") as finalize_task:
            with patch("fetcher.orchestrator.fetch_comments_task") as fetch_comments_task:
                with patch("fetcher.orchestrator.download_video_task") as download_video_task:
                    with patch("fetcher.orchestrator.fetch_metadata_task") as fetch_metadata_task:
                        fetch_metadata_task.delay.side_effect = lambda run_id: run_metadata_worker(run_id)
                        download_video_task.delay.side_effect = lambda run_id: run_video_worker(run_id)
                        fetch_comments_task.delay.side_effect = lambda run_id: run_comments_worker(run_id, limit=100)
                        finalize_task.delay.side_effect = lambda run_id: None

                        with patch("pathlib.Path.write_text"):
                            with patch("pathlib.Path.stat") as mock_stat:
                                mock_stat.return_value.st_size = 1024
                                with patch("pathlib.Path.mkdir"):
                                    with patch("pathlib.Path.unlink"):
                                        fetch_video(str(tiktok_run.id))

        with session_scope() as db:
            run_after = db.query(Run).filter(Run.id == tiktok_run.id).one()
            assert run_after.status in ("FINALIZING", "COMPLETED")

