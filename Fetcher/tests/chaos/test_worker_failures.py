"""Chaos тесты для падений worker'ов."""

import stat as stat_module
import pytest
from unittest.mock import patch, MagicMock

from fetcher.workers.metadata import run_metadata_worker
from fetcher.workers.video import run_video_worker
from fetcher.workers.comments import run_comments_worker
from fetcher.models import Run
from fetcher.db import session_scope


@pytest.mark.chaos
@pytest.mark.slow
class TestWorkerFailures:
    """Тесты для проверки устойчивости к падениям worker'ов."""

    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_metadata_worker_crash_recovery(
        self,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест восстановления после падения metadata worker'а."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        # Симулируем падение worker'а (исключение в середине выполнения)
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = RuntimeError("Worker crashed")
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Первый запуск падает
        with pytest.raises(RuntimeError):
            run_metadata_worker(str(test_run.id))

        # Проверяем, что run есть в БД (реальная сессия)
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == test_run.id).first()
            assert run is not None

        # Восстанавливаем worker (убираем ошибку)
        mock_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "description": "Test description",
            "duration": 212,
        }
        mock_ydl_instance.extract_info.side_effect = None
        mock_ydl_instance.extract_info.return_value = mock_info

        # Второй запуск должен успешно завершиться
        with patch("fetcher.platforms.youtube.adapter.storage_client") as mock_storage:
            with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                mock_storage.upload_file.return_value = None
                mock_checksum.return_value = "abc123"
                with patch("builtins.open", create=True):
                    with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                        with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                            mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                    run_metadata_worker(str(test_run.id))

    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_video_worker_crash_recovery(
        self,
        mock_ydl,
        mock_proxy,
        mock_lock,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест восстановления после падения video worker'а."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_proxy.return_value = None

        # Симулируем падение worker'а
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = RuntimeError("Worker crashed")
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Первый запуск падает
        with pytest.raises(RuntimeError):
            run_video_worker(str(test_run.id))

        # Восстанавливаем worker
        mock_info = {"id": "dQw4w9WgXcQ", "title": "Test Video", "duration": 212}
        mock_ydl_instance.extract_info.side_effect = None
        mock_ydl_instance.extract_info.return_value = mock_info

        # Второй запуск должен успешно завершиться
        with patch("fetcher.platforms.youtube.adapter.storage_client") as mock_storage:
            with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                mock_storage.upload_file.return_value = None
                mock_checksum.return_value = "abc123"
                with patch("builtins.open", create=True):
                    with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                        with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                            mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                    run_video_worker(str(test_run.id))

    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_comments_worker_crash_recovery(
        self,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест восстановления после падения comments worker'а."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        # Симулируем падение worker'а
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = RuntimeError("Worker crashed")
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Первый запуск падает
        with pytest.raises(RuntimeError):
            run_comments_worker(str(test_run.id), limit=100)

        # Восстанавливаем worker
        mock_info = {
            "id": "dQw4w9WgXcQ",
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
        mock_ydl_instance.extract_info.side_effect = None
        mock_ydl_instance.extract_info.return_value = mock_info

        # Второй запуск должен успешно завершиться
        with patch("fetcher.platforms.youtube.adapter.storage_client") as mock_storage:
            with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                mock_storage.upload_file.return_value = None
                mock_checksum.return_value = "abc123"
                with patch("builtins.open", create=True):
                    with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                        with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                            mock_stat.return_value = MagicMock(st_size=1024, st_mode=stat_module.S_IFREG)
                            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                    run_comments_worker(str(test_run.id), limit=100)

    @pytest.mark.database
    def test_finalize_worker_crash_recovery(self, sample_video_url, sample_run_id):
        """Тест восстановления после падения finalize worker'а."""
        # TODO: Реализовать когда будет finalize worker
        pass

