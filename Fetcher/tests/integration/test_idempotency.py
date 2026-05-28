"""Integration тесты для идемпотентности Fetcher."""

import stat as stat_module
import pytest
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from fetcher.workers.metadata import run_metadata_worker
from fetcher.workers.video import run_video_worker
from fetcher.workers.comments import run_comments_worker
from fetcher.models import Run, VideoSource, Video, Artifact
from fetcher.db import session_scope


@pytest.mark.integration
@pytest.mark.idempotency
class TestIdempotencyIntegration:
    """Integration тесты для идемпотентности."""

    @pytest.mark.slow
    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    @patch("fetcher.platforms.youtube.adapter.create_initial_snapshot_from_info")
    def test_idempotent_metadata_worker(
        self,
        mock_snapshot,
        mock_checksum,
        mock_storage,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест идемпотентности metadata worker (повторный запуск не создаёт дубликаты)."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None
        mock_checksum.return_value = "abc123"
        mock_storage.upload_file.return_value = None

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
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    import stat as stat_module
                    # Все вызовы stat() должны возвращать объект с целым st_size (иначе БД не примет)
                    mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                    with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                        with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                            run_metadata_worker(str(test_run.id))

        with session_scope() as db:
            video = db.query(Video).filter(
                Video.platform == "youtube",
                Video.platform_video_id == "dQw4w9WgXcQ"
            ).first()
            assert video is not None
            video_count_before = db.query(Video).count()

        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                    with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                        with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                            run_metadata_worker(str(test_run.id))

        with session_scope() as db:
            video_count_after = db.query(Video).count()
        assert video_count_after == video_count_before

    @pytest.mark.slow
    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    def test_idempotent_video_worker(
        self,
        mock_checksum,
        mock_storage,
        mock_ydl,
        mock_proxy,
        mock_lock,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест идемпотентности video worker (повторный запуск не скачивает повторно)."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_proxy.return_value = None
        mock_checksum.return_value = "abc123"
        mock_storage.upload_file.return_value = None

        # Создаем или берём существующее видео и артефакт в реальной БД
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
                artifact_type="video_file",
                storage_path="raw/youtube/2023/01/01/dQw4w9WgXcQ/video.mp4",
                status="COMPLETED",
            )
            db.add(artifact)
            db.commit()

        mock_info = {"id": "dQw4w9WgXcQ", "title": "Test Video", "duration": 212}
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        with session_scope() as db:
            artifact_count_before = db.query(Artifact).filter(
                Artifact.artifact_type == "video_file"
            ).count()

        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                    run_video_worker(str(test_run.id))

        with session_scope() as db:
            artifact_count_after = db.query(Artifact).filter(
                Artifact.artifact_type == "video_file"
            ).count()
        assert artifact_count_after == artifact_count_before

    @pytest.mark.slow
    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    def test_idempotent_comments_worker(
        self,
        mock_checksum,
        mock_storage,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест идемпотентности comments worker (повторный запуск не загружает повторно)."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None
        mock_checksum.return_value = "abc123"
        mock_storage.upload_file.return_value = None

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
                artifact_type="comments_file",
                storage_path="raw/youtube/2023/01/01/dQw4w9WgXcQ/comments.json",
                status="COMPLETED",
            )
            db.add(artifact)
            db.commit()

        mock_info = {
            "id": "dQw4w9WgXcQ",
            "comments": [
                {"id": "comment1", "text": "Great video!", "author": "User1", "author_id": "user1", "like_count": 10, "reply_count": 2, "timestamp": 1234567890}
            ],
        }
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        with session_scope() as db:
            artifact_count_before = db.query(Artifact).filter(
                Artifact.artifact_type == "comments_file"
            ).count()

        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                    with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                        with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                            run_comments_worker(str(test_run.id), limit=100)

        with session_scope() as db:
            artifact_count_after = db.query(Artifact).filter(
                Artifact.artifact_type == "comments_file"
            ).count()
        assert artifact_count_after == artifact_count_before

