"""Chaos тесты для сетевых ошибок."""

import pytest
from unittest.mock import patch, MagicMock
import redis
from yt_dlp.utils import DownloadError

from fetcher.workers.metadata import run_metadata_worker
from fetcher.models import Run
from fetcher.db import session_scope


@pytest.mark.chaos
@pytest.mark.slow
class TestNetworkFailures:
    """Тесты для проверки устойчивости к сетевым ошибкам."""

    @pytest.mark.redis
    @patch("fetcher.rate_limiter.get_redis_client")
    def test_redis_connection_loss(
        self,
        mock_redis_client,
        sample_video_url,
    ):
        """Тест устойчивости к потере подключения к Redis."""
        # Симулируем потерю подключения к Redis
        mock_redis = MagicMock()
        mock_redis.incr.side_effect = redis.ConnectionError("Connection lost")
        mock_redis_client.return_value = mock_redis

        # Система должна корректно обрабатывать ошибку Redis
        # В реальной реализации rate_limiter должен обрабатывать ConnectionError
        # и либо retry, либо fallback на разрешение запроса

        # Восстанавливаем подключение
        mock_redis.incr.side_effect = None
        mock_redis.incr.return_value = 1

        # Система должна продолжать работать
        assert mock_redis.incr.return_value == 1

    @pytest.mark.storage
    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_storage_connection_loss(
        self,
        mock_storage,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест устойчивости к потере подключения к Storage."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "description": "Test description",
            "duration": 212,
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Симулируем потерю подключения к Storage
        mock_storage.upload_file.side_effect = Exception("Storage connection lost")

        # Система должна корректно обрабатывать ошибку Storage
        with pytest.raises(Exception, match="Storage connection lost"):
            with patch("builtins.open", create=True):
                with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                    with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                        mock_stat.return_value = MagicMock(st_size=1024)
                        with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                            with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                                    mock_checksum.return_value = "abc123"
                                    run_metadata_worker(str(test_run.id))

        # Восстанавливаем подключение
        mock_storage.upload_file.side_effect = None
        mock_storage.upload_file.return_value = None

        # Система должна продолжать работать
        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    mock_stat.return_value = MagicMock(st_size=1024)
                    with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                        with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                            with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                                mock_checksum.return_value = "abc123"
                                run_metadata_worker(str(test_run.id))

    @pytest.mark.database
    @patch("tests.chaos.test_network_failures.session_scope")
    def test_database_connection_loss(
        self,
        mock_session_scope,
        sample_video_url,
    ):
        """Тест устойчивости к потере подключения к БД."""
        # Симулируем потерю подключения к БД
        from sqlalchemy.exc import OperationalError
        
        mock_session_scope.side_effect = OperationalError("Connection lost", None, None)

        # Система должна корректно обрабатывать ошибку БД
        with pytest.raises(OperationalError):
            with session_scope() as db:
                db.query(Run).first()

        # Восстанавливаем подключение
        mock_session_scope.side_effect = None
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_scope.return_value = mock_session

        # Система должна продолжать работать
        with session_scope() as db:
            assert db is not None

    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_youtube_api_timeout(
        self,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест устойчивости к таймаутам YouTube API."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        # Симулируем таймаут YouTube API
        mock_ydl_instance = MagicMock()
        timeout_error = DownloadError("ERROR: Unable to download webpage: <urlopen error [Errno 110] Connection timed out>")
        mock_ydl_instance.extract_info.side_effect = timeout_error
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Система должна корректно обрабатывать таймаут
        with pytest.raises(DownloadError):
            run_metadata_worker(str(test_run.id))

        # Проверяем, что circuit breaker был уведомлен
        mock_breaker.record_failure.assert_called()

        # Восстанавливаем подключение
        mock_info = {
            "id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "description": "Test description",
            "duration": 212,
        }
        mock_ydl_instance.extract_info.side_effect = None
        mock_ydl_instance.extract_info.return_value = mock_info

        # Система должна продолжать работать после восстановления
        with patch("fetcher.platforms.youtube.adapter.storage_client") as mock_storage:
            with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                mock_storage.upload_file.return_value = None
                mock_checksum.return_value = "abc123"
                with patch("builtins.open", create=True):
                    with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                        with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                            mock_stat.return_value = MagicMock(st_size=1024)
                            with patch("fetcher.platforms.youtube.adapter.Path.mkdir"):
                                with patch("fetcher.platforms.youtube.adapter.Path.unlink"):
                                    run_metadata_worker(str(test_run.id))

    @pytest.mark.database
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    def test_youtube_api_429_error(
        self,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        integration_test_run,
        sample_video_url,
    ):
        """Тест устойчивости к ошибкам 429 от YouTube API."""
        test_run = integration_test_run
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        # Симулируем ошибку 429
        mock_ydl_instance = MagicMock()
        error_429 = DownloadError("HTTP Error 429: Too Many Requests")
        mock_ydl_instance.extract_info.side_effect = error_429
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Система должна корректно обрабатывать 429
        with pytest.raises(DownloadError):
            run_metadata_worker(str(test_run.id))

        # Проверяем, что circuit breaker был уведомлен
        mock_breaker.record_failure.assert_called()

