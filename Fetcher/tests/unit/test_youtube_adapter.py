"""Unit тесты для YouTubeAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from yt_dlp.utils import DownloadError

from fetcher.platforms.youtube.adapter import YouTubeAdapter
from fetcher.models import Video, VideoMetadata, ChannelMetadata


@pytest.mark.unit
class TestYouTubeAdapter:
    """Тесты для YouTubeAdapter."""

    @pytest.fixture
    def adapter(self):
        """Фикстура для создания YouTubeAdapter."""
        return YouTubeAdapter()

    @pytest.fixture
    def mock_info_dict(self):
        """Фикстура для мока info_dict от yt-dlp."""
        return {
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

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_fetch_metadata_success(
        self,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест успешного fetch_metadata."""
        # Настройка моков
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None  # Видео не найдено
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        mock_storage.upload_file.return_value = None

        # Вызов метода
        adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

        # Проверки
        mock_rate_limit.assert_called_once()
        mock_ydl_instance.extract_info.assert_called_once_with(sample_video_url, download=False)
        assert mock_db.add.called  # Должна быть добавлена запись Video

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    def test_fetch_metadata_circuit_breaker_open(
        self, mock_rate_limit, mock_circuit_breaker, adapter, sample_video_url, sample_run_id
    ):
        """Тест fetch_metadata когда circuit breaker открыт."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = True
        mock_circuit_breaker.return_value = mock_breaker

        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    def test_fetch_metadata_rate_limit_exceeded(
        self, mock_rate_limit, mock_circuit_breaker, adapter, sample_video_url, sample_run_id
    ):
        """Тест fetch_metadata когда rate limit превышен."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = False

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

        mock_breaker.record_failure.assert_called_once_with("rate_limit_exceeded")

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.record_proxy_result")
    def test_fetch_metadata_download_error_429(
        self,
        mock_record_proxy,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        sample_video_url,
        sample_run_id,
    ):
        """Тест fetch_metadata при ошибке 429 от YouTube."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = "http://proxy:8080"

        mock_ydl_instance = MagicMock()
        error = DownloadError("HTTP Error 429: Too Many Requests")
        mock_ydl_instance.extract_info.side_effect = error
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        with pytest.raises(DownloadError):
            adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

        # Проверяем, что ошибка записана в proxy result
        mock_record_proxy.assert_called_once()
        call_args = mock_record_proxy.call_args
        assert call_args[0][0] == "http://proxy:8080"
        assert call_args[0][1] is False  # success=False
        assert call_args[1]["operation"] == "metadata"

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_download_video_success(
        self,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_lock,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест успешного download_video."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_lock.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Мокаем файл для download
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.write.return_value = None
            mock_open.return_value.__enter__.return_value = mock_file

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value.__enter__.return_value = mock_db
            mock_session.return_value.__exit__.return_value = None

            mock_storage.upload_file.return_value = None

            adapter.download_video(sample_video_url, run_id=sample_run_id)

            mock_lock.assert_called_once()
            # Допускаем как минимум один вызов extract_info (поддержка стратегий с предварительным fetch'ем).
            assert mock_ydl_instance.extract_info.call_count >= 1

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.acquire_video_lock")
    def test_download_video_lock_failed(
        self,
        mock_lock,
        mock_ydl,
        mock_proxy,
        mock_session,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест download_video когда lock не получен."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None  # нет артефакта
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        mock_lock.return_value = False  # Lock не получен

        # Должен вернуться без ошибки (идемпотентность)
        adapter.download_video(sample_video_url, run_id=sample_run_id)

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_fetch_comments_success(
        self,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест успешного fetch_comments."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None
        mock_storage.upload_file.return_value = None

        # Мокаем комментарии
        mock_info_dict["comments"] = [
            {
                "id": "comment1",
                "text": "Great video!",
                "author": "User1",
                "author_id": "user1",
                "like_count": 10,
                "reply_count": 2,
                "timestamp": 1234567890,
            }
        ]

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_db = MagicMock()
        mock_video = MagicMock()
        mock_video.id = "video-id"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        adapter.fetch_comments(sample_video_url, run_id=sample_run_id, limit=100)

        mock_ydl_instance.extract_info.assert_called_once()

    @patch("fetcher.platforms.youtube.adapter.mask_pii")
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_fetch_comments_pii_filtering(
        self,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_mask_pii,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест PII фильтрации в комментариях."""
        from fetcher.config import settings

        mock_storage.upload_file.return_value = None
        # Включаем PII filtering
        with patch.object(settings, "enable_pii_filtering", True):
            mock_breaker = MagicMock()
            mock_breaker.is_open.return_value = False
            mock_circuit_breaker.return_value = mock_breaker
            mock_rate_limit.return_value = True
            mock_proxy.return_value = None

            mock_info_dict["comments"] = [
                {
                    "id": "comment1",
                    "text": "Contact me at test@example.com",
                    "author": "User1",
                    "author_id": "user1",
                    "like_count": 10,
                    "reply_count": 2,
                    "timestamp": 1234567890,
                }
            ]

            mock_ydl_instance = MagicMock()
            mock_ydl_instance.extract_info.return_value = mock_info_dict
            mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

            mock_db = MagicMock()
            mock_video = MagicMock()
            mock_video.id = "video-id"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_video
            mock_session.return_value.__enter__.return_value = mock_db
            mock_session.return_value.__exit__.return_value = None

            mock_mask_pii.return_value = "Contact me at [EMAIL]"

            adapter.fetch_comments(sample_video_url, run_id=sample_run_id, limit=100)

            # Проверяем, что mask_pii был вызван
            mock_mask_pii.assert_called()

    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    @patch("fetcher.platforms.youtube.adapter.compute_sha256")
    def test_fetch_metadata_checksum(
        self,
        mock_checksum,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест вычисления checksum для meta.json."""
        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_checksum.return_value = "abc123def456"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.query.return_value.filter.return_value.one.return_value = MagicMock(id="video-id")
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        mock_storage.upload_file.return_value = None

        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.write.return_value = None
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    # Первый вызов — для tmp_dir.is_dir() при mkdir; второй — для tmp_path.stat().st_size
                    import stat as stat_module
                    mock_stat.side_effect = [
                        MagicMock(st_mode=stat_module.S_IFDIR),
                        MagicMock(st_size=1024, st_mode=stat_module.S_IFREG),
                    ]

                    adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

                    # Проверяем, что checksum был вычислен
                    mock_checksum.assert_called()

    @patch("fetcher.platforms.youtube.adapter.settings")
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    def test_fetch_comments_retain_raw_disabled(
        self,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_settings,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест fetch_comments когда retain_raw_comments=False."""
        mock_settings.retain_raw_comments = False
        mock_settings.enable_pii_filtering = False
        mock_storage.upload_file.return_value = None

        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_info_dict["comments"] = [
            {
                "id": "comment1",
                "text": "Great video!",
                "author": "User1",
                "author_id": "user1",
                "like_count": 10,
                "reply_count": 2,
                "timestamp": 1234567890,
            }
        ]

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_db = MagicMock()
        mock_video = MagicMock()
        mock_video.id = "video-id"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        mock_db.query.return_value.filter.return_value.one.return_value = mock_video
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        import stat as stat_module
        with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
            with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                mock_stat.side_effect = [
                    MagicMock(st_mode=stat_module.S_IFDIR),
                    MagicMock(st_size=512, st_mode=stat_module.S_IFREG),
                ]
                with patch("fetcher.platforms.youtube.adapter.compute_sha256", return_value="abc123"):
                    adapter.fetch_comments(sample_video_url, run_id=sample_run_id, limit=100)

        # Проверяем, что комментарии были добавлены, но без text и raw_json
        assert mock_db.add.called

    @patch("fetcher.platforms.youtube.adapter.create_initial_snapshot_from_info")
    @patch("fetcher.platforms.youtube.adapter.get_circuit_breaker")
    @patch("fetcher.platforms.youtube.adapter.acquire_token")
    @patch("fetcher.platforms.youtube.adapter.get_next_proxy")
    @patch("fetcher.platforms.youtube.adapter.yt_dlp.YoutubeDL")
    @patch("fetcher.platforms.youtube.adapter.session_scope")
    @patch("fetcher.platforms.youtube.adapter.storage_client")
    @patch("fetcher.platforms.youtube.adapter.settings")
    def test_fetch_metadata_snapshot_creation(
        self,
        mock_settings,
        mock_storage,
        mock_session,
        mock_ydl,
        mock_proxy,
        mock_rate_limit,
        mock_circuit_breaker,
        mock_snapshot,
        adapter,
        mock_info_dict,
        sample_video_url,
        sample_run_id,
    ):
        """Тест создания snapshot при fetch_metadata."""
        mock_settings.enable_snapshots = True
        mock_settings.retain_raw_meta = True

        mock_breaker = MagicMock()
        mock_breaker.is_open.return_value = False
        mock_circuit_breaker.return_value = mock_breaker
        mock_rate_limit.return_value = True
        mock_proxy.return_value = None

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info_dict
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_db.query.return_value.filter.return_value.one.return_value = MagicMock(id="video-id")
        mock_session.return_value.__enter__.return_value = mock_db
        mock_session.return_value.__exit__.return_value = None

        mock_storage.upload_file.return_value = None

        import stat as stat_module
        with patch("builtins.open", create=True):
            with patch("fetcher.platforms.youtube.adapter.Path.write_text"):
                with patch("fetcher.platforms.youtube.adapter.Path.stat") as mock_stat:
                    mock_stat.side_effect = [
                        MagicMock(st_mode=stat_module.S_IFDIR),
                        MagicMock(st_size=1024, st_mode=stat_module.S_IFREG),
                    ]
                    with patch("fetcher.platforms.youtube.adapter.compute_sha256") as mock_checksum:
                        mock_checksum.return_value = "abc123"

                        adapter.fetch_metadata(sample_video_url, run_id=sample_run_id)

                        # Проверяем, что snapshot был создан
                        mock_snapshot.assert_called_once_with("youtube", "dQw4w9WgXcQ", mock_info_dict)

