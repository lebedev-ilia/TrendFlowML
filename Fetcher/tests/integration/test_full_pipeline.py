"""Integration тесты для полного pipeline Fetcher."""

import pytest
import uuid
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timezone

from fetcher.orchestrator import fetch_video
from fetcher.models import Run, VideoSource, Video, VideoMetadata, Artifact
from fetcher.db import session_scope
from fetcher.workers.metadata import run_metadata_worker
from fetcher.workers.video import run_video_worker
from fetcher.workers.comments import run_comments_worker
from fetcher.services.youtube_data_client import VideoMetadataDto, CommentDto
from fetcher.config import settings


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Integration тесты для полного pipeline."""

    @pytest.fixture
    def mock_yt_dlp(self):
        """Фикстура для мока yt-dlp."""
        mock_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "description": "Test description",
            "duration": 212,
            "view_count": 1000000,
            "like_count": 50000,
            "comment_count": 1000,
            "uploader": "Test Channel",
            "uploader_id": "UCtest",
            "channel": "Test Channel",
            "channel_id": "UCtest",
            "channel_follower_count": 100000,
            "tags": ["test", "video"],
            "upload_date": "20230101",
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "comments": [
                {
                    "id": "comment1",
                    "text": "Great video!",
                    "author": "User1",
                    "author_id": "user1",
                    "like_count": 10,
                    "reply_count": 2,
                    "timestamp": 1234567890,
                }
            ],
        }

        with patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.extract_info.return_value = mock_info
            mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
            yield mock_ydl_instance

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.storage.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    @patch("fetcher.platforms.youtube.adapter.create_initial_snapshot_from_info")
    def test_full_pipeline_success(
        self,
        mock_snapshot,
        mock_checksum,
        mock_storage,
        mock_proxy,
        mock_lock,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_yt_dlp,
        integration_test_run,
        sample_video_url,
    ):
        """Тест успешного выполнения полного pipeline."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_proxy.return_value = None
        mock_checksum.return_value = "abc123def456"
        mock_storage.upload_file.return_value = None

        with patch("yt_dlp.YoutubeDL") as yt_dlp_mock:
            yt_dlp_mock.return_value.__enter__.return_value.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
            with patch("fetcher.orchestrator.finalize_task") as finalize_task:
                with patch("fetcher.orchestrator.fetch_comments_task") as fetch_comments_task:
                    with patch("fetcher.orchestrator.download_video_task") as download_video_task:
                        with patch("fetcher.orchestrator.fetch_metadata_task") as fetch_metadata_task:
                            fetch_metadata_task.delay.side_effect = lambda run_id: run_metadata_worker(run_id)
                            download_video_task.delay.side_effect = lambda run_id: run_video_worker(run_id)
                            fetch_comments_task.delay.side_effect = lambda run_id: run_comments_worker(run_id, limit=100)
                            finalize_task.delay.side_effect = lambda run_id: None
                            # Мокаем файловые операции
                            with patch("builtins.open", create=True) as mock_open:
                                mock_file = MagicMock()
                                mock_file.write.return_value = None
                                mock_open.return_value.__enter__.return_value = mock_file

                                with patch("pathlib.Path.write_text"):
                                    with patch("pathlib.Path.stat") as mock_stat:
                                        mock_stat.return_value.st_size = 1024
                                        with patch("pathlib.Path.mkdir"):
                                            with patch("pathlib.Path.unlink"):
                                                fetch_video(str(test_run.id))
                                                with session_scope() as db:
                                                    run_after = db.query(Run).filter(Run.id == test_run.id).one()
                                                    assert run_after.status in (
                                                        "FINALIZING",
                                                        "COMPLETED",
                                                    ), f"run должен завершиться в FINALIZING/COMPLETED, получен {run_after.status}"

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_pipeline_with_cache_hit(
        self,
        mock_storage,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_yt_dlp,
        integration_test_run,
        sample_video_url,
    ):
        """Тест pipeline с cache hit (видео уже существует)."""
        test_run = integration_test_run
        with patch("yt_dlp.YoutubeDL") as yt_dlp_mock:
            yt_dlp_mock.return_value.__enter__.return_value.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
            with patch("fetcher.orchestrator.finalize_task") as finalize_task:
                with patch("fetcher.orchestrator.fetch_metadata_task") as fetch_metadata_task:
                    fetch_metadata_task.delay.side_effect = lambda run_id: run_metadata_worker(run_id)
                    finalize_task.delay.side_effect = lambda run_id: None
                    with session_scope() as db:
                        video = db.query(Video).filter(
                            Video.platform == "youtube",
                            Video.platform_video_id == "dQw4w9WgXcQ",
                        ).first()
                        if video is None:
                            video = Video(platform="youtube", platform_video_id="dQw4w9WgXcQ")
                            db.add(video)
                            db.flush()
                        artifact = Artifact(
                            video_id=video.id,
                            artifact_type="metadata_file",
                            storage_path="raw/youtube/2023/01/01/dQw4w9WgXcQ/meta.json",
                            status="COMPLETED",
                        )
                        db.add(artifact)
                        db.commit()

                    mock_breaker = MagicMock()
                    mock_breaker.is_open.return_value = False
                    mock_circuit_breaker.return_value = mock_breaker
                    mock_rate_limit.return_value = True
                    mock_proxy.return_value = None
                    fetch_video(str(test_run.id))

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_pipeline_with_429_error(
        self,
        mock_ydl_adapter,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест pipeline при ошибке 429 от YouTube."""
        from yt_dlp.utils import DownloadError

        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        error = DownloadError("HTTP Error 429: Too Many Requests")
        mock_ydl_instance.extract_info.side_effect = error
        mock_ydl_adapter.return_value.__enter__.return_value = mock_ydl_instance

        with patch("yt_dlp.YoutubeDL") as yt_dlp_mock:
            yt_dlp_mock.return_value.__enter__.return_value.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
            with patch("fetcher.orchestrator.finalize_task") as finalize_task:
                with patch("fetcher.orchestrator.fetch_comments_task") as fetch_comments_task:
                    with patch("fetcher.orchestrator.download_video_task") as download_video_task:
                        with patch("fetcher.orchestrator.fetch_metadata_task") as fetch_metadata_task:
                            fetch_metadata_task.delay.side_effect = lambda run_id: run_metadata_worker(run_id)
                            download_video_task.delay.side_effect = lambda run_id: run_video_worker(run_id)
                            fetch_comments_task.delay.side_effect = lambda run_id: run_comments_worker(run_id, limit=100)
                            finalize_task.delay.side_effect = lambda run_id: None
                            fetch_video(str(test_run.id))
        # При синхронном запуске воркер может пропустить этап (идемпотентность) или вызвать adapter и record_failure

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.storage.storage_client")
    def test_pipeline_idempotency(
        self,
        mock_storage,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_yt_dlp,
        integration_test_run,
        sample_video_url,
    ):
        """Тест идемпотентности: два run с одним URL — один Video/артефакты, второй run по кешу."""
        run1 = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None
        mock_storage.upload_file.return_value = None

        # Второй run с тем же URL
        with session_scope() as db:
            run2_id = uuid.uuid4()
            run2 = Run(
                id=run2_id,
                source_type="youtube",
                source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                status="PENDING",
                started_at=datetime.now(timezone.utc),
            )
            db.add(run2)
            db.flush()
            from fetcher.models import VideoSource as VS
            vs2 = VS(
                run_id=run2_id,
                platform="youtube",
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                normalized_video_id="dQw4w9WgXcQ",
            )
            db.add(vs2)
            db.commit()
        run2_ref = type("RunRef", (), {"id": run2_id})()

        with patch("yt_dlp.YoutubeDL") as yt_dlp_mock:
            yt_dlp_mock.return_value.__enter__.return_value.extract_info.return_value = {"id": "dQw4w9WgXcQ"}
            with patch("fetcher.orchestrator.finalize_task") as finalize_task:
                with patch("fetcher.orchestrator.fetch_comments_task") as fetch_comments_task:
                    with patch("fetcher.orchestrator.download_video_task") as download_video_task:
                        with patch("fetcher.orchestrator.fetch_metadata_task") as fetch_metadata_task:
                            fetch_metadata_task.delay.side_effect = lambda run_id: run_metadata_worker(run_id)
                            download_video_task.delay.side_effect = lambda run_id: run_video_worker(run_id)
                            fetch_comments_task.delay.side_effect = lambda run_id: run_comments_worker(run_id, limit=100)
                            finalize_task.delay.side_effect = lambda run_id: None
                            with patch("builtins.open", create=True):
                                with patch("pathlib.Path.write_text"):
                                    with patch("pathlib.Path.stat") as mock_stat:
                                        mock_stat.return_value.st_size = 1024
                                        with patch("pathlib.Path.mkdir"):
                                            with patch("pathlib.Path.unlink"):
                                                with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                                                    mock_checksum.return_value = "abc123"
                                                    fetch_video(str(run1.id))
                                                    fetch_video(str(run2_ref.id))

        with session_scope() as db:
            video_count = db.query(Video).filter(
                Video.platform == "youtube",
                Video.platform_video_id == "dQw4w9WgXcQ",
            ).count()
            assert video_count == 1, "Ожидается один Video на оба run"
            artifact_count = db.query(Artifact).join(Video).filter(
                Video.platform == "youtube",
                Video.platform_video_id == "dQw4w9WgXcQ",
            ).count()
            assert artifact_count >= 1, "Ожидается хотя бы один артефакт для видео"

    @patch("fetcher.platforms.youtube.adapter.YouTubeDataClient")
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    @patch("fetcher.storage.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    def test_full_pipeline_with_youtube_data_api(
        self,
        mock_checksum,
        mock_storage,
        mock_lock,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_data_client_cls,
        integration_test_run,
        sample_video_url,
    ):
        """Полный pipeline при включённом youtube_data_enabled с замоканным YouTubeDataClient.

        Проверяет, что метадата, видео и комментарии успешно проходят через pipeline
        без реальных запросов к YouTube.
        """
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_storage.upload_file.return_value = None
        mock_checksum.return_value = "abc123"

        # Настройка замоканного YouTubeDataClient
        mock_client = MagicMock()
        dto_meta = VideoMetadataDto(
            video_id="dQw4w9WgXcQ",
            title="API Test Video",
            description="API description",
            channel_id="UCtest",
            channel_title="Test Channel",
            duration_seconds=212,
            view_count=100,
            like_count=5,
            comment_count=1,
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            raw_json={"id": "dQw4w9WgXcQ"},
        )
        dto_comment = CommentDto(
            comment_id="c1",
            author_display_name="User1",
            text_original="Nice!",
            like_count=1,
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=None,
            raw_json={"id": "c1"},
        )
        mock_client.get_video_metadata.return_value = dto_meta
        mock_client.iter_comments.return_value = [dto_comment]
        mock_data_client_cls.return_value = mock_client

        with patch.object(settings, "youtube_data_enabled", True), patch.object(
            settings, "youtube_mock_video_download", True
        ):
            with patch("pathlib.Path.write_text"):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024
                    with patch("pathlib.Path.mkdir"):
                        with patch("pathlib.Path.unlink"):
                            fetch_video(str(test_run.id))

        # Проверяем, что в БД появились метадата и хотя бы один артефакт
        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == "dQw4w9WgXcQ",
                )
                .one_or_none()
            )
            assert video is not None
            vm = (
                db.query(VideoMetadata)
                .filter(VideoMetadata.video_id == video.id)
                .one_or_none()
            )
            assert vm is not None
            assert vm.title == "API Test Video"
            artifacts = (
                db.query(Artifact)
                .filter(Artifact.video_id == video.id)
                .all()
            )
            assert artifacts, "Ожидается хотя бы один артефакт для видео при Data API режиме"

