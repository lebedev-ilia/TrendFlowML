"""
Unit тесты для Recovery Service
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.services.recovery import (
    check_and_recover_run,
    recover_run,
    recover_all_crashed_runs
)
from api.schemas.state import RunStatus


class TestCheckAndRecoverRun:
    """Тесты для функции check_and_recover_run."""
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_no_heartbeat(self, mock_redis_client):
        """Обнаружение crashed run без heartbeat."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_state", new_callable=AsyncMock) as mock_get_state:
                with patch("api.services.recovery.get_run_heartbeat", new_callable=AsyncMock) as mock_get_heartbeat:
                    with patch("api.services.recovery.recover_run", new_callable=AsyncMock) as mock_recover:
                        mock_get_state.return_value = {
                            "status": "running",
                            "updated_at": 1234567890.0
                        }
                        mock_get_heartbeat.return_value = None  # Heartbeat отсутствует
                        mock_recover.return_value = True
                        
                        result = await check_and_recover_run("test-run-id")
                        
                        assert result is True
                        mock_recover.assert_called_once_with("test-run-id")
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_with_heartbeat(self, mock_redis_client):
        """Run с heartbeat не требует восстановления."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_state", new_callable=AsyncMock) as mock_get_state:
                with patch("api.services.recovery.get_run_heartbeat", new_callable=AsyncMock) as mock_get_heartbeat:
                    with patch("api.services.recovery.recover_run", new_callable=AsyncMock) as mock_recover:
                        mock_get_state.return_value = {
                            "status": "running",
                            "updated_at": 1234567890.0
                        }
                        mock_get_heartbeat.return_value = 1234567890.0  # Heartbeat есть
                        
                        result = await check_and_recover_run("test-run-id")
                        
                        assert result is False
                        mock_recover.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_not_running(self, mock_redis_client):
        """Run не в статусе running не проверяется."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_state", new_callable=AsyncMock) as mock_get_state:
                with patch("api.services.recovery.get_run_heartbeat", new_callable=AsyncMock) as mock_get_heartbeat:
                    with patch("api.services.recovery.recover_run", new_callable=AsyncMock) as mock_recover:
                        mock_get_state.return_value = {
                            "status": "success",
                            "updated_at": 1234567890.0
                        }
                        
                        result = await check_and_recover_run("test-run-id")
                        
                        assert result is False
                        mock_get_heartbeat.assert_not_called()
                        mock_recover.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_not_found(self, mock_redis_client):
        """Run не найден в Redis."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_state", new_callable=AsyncMock) as mock_get_state:
                mock_get_state.return_value = None
                
                result = await check_and_recover_run("test-run-id")
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_redis_unavailable(self):
        """Redis недоступен."""
        with patch("api.services.recovery.get_redis_client", return_value=None):
            result = await check_and_recover_run("test-run-id")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_and_recover_run_invalid_status(self, mock_redis_client):
        """Невалидный статус run'а."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_state", new_callable=AsyncMock) as mock_get_state:
                mock_get_state.return_value = {
                    "status": "invalid_status",
                    "updated_at": 1234567890.0
                }
                
                result = await check_and_recover_run("test-run-id")
                
                assert result is False


class TestRecoverRun:
    """Тесты для функции recover_run."""
    
    @pytest.mark.asyncio
    async def test_recover_run_success(self, mock_redis_client):
        """Успешное восстановление run'а."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_priority", new_callable=AsyncMock) as mock_get_priority:
                with patch("api.services.recovery.get_run_metadata", new_callable=AsyncMock) as mock_get_metadata:
                    with patch("api.services.recovery.save_run_state", new_callable=AsyncMock) as mock_save_state:
                        with patch("api.services.recovery.add_run_event", new_callable=AsyncMock) as mock_add_event:
                            with patch("api.services.recovery.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
                                mock_get_priority.return_value = "high"
                                mock_get_metadata.return_value = {
                                    "video_id": "test_video",
                                    "platform_id": "youtube"
                                }
                                mock_enqueue.return_value = "1234567890-0"
                                
                                result = await recover_run("test-run-id")
                                
                                assert result is True
                                mock_save_state.assert_called_once()
                                # Проверить что статус установлен на "recovering"
                                call_args = mock_save_state.call_args[0]
                                assert call_args[0] == "test-run-id"
                                assert call_args[1]["status"] == "recovering"
                                mock_add_event.assert_called_once()
                                mock_enqueue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recover_run_no_metadata(self, mock_redis_client):
        """Восстановление run'а без метаданных."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_priority", new_callable=AsyncMock) as mock_get_priority:
                with patch("api.services.recovery.get_run_metadata", new_callable=AsyncMock) as mock_get_metadata:
                    mock_get_priority.return_value = "normal"
                    mock_get_metadata.return_value = None  # Метаданные отсутствуют
                    
                    result = await recover_run("test-run-id")
                    
                    assert result is False
    
    @pytest.mark.asyncio
    async def test_recover_run_default_priority(self, mock_redis_client):
        """Восстановление run'а с приоритетом по умолчанию."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_priority", new_callable=AsyncMock) as mock_get_priority:
                with patch("api.services.recovery.get_run_metadata", new_callable=AsyncMock) as mock_get_metadata:
                    with patch("api.services.recovery.save_run_state", new_callable=AsyncMock):
                        with patch("api.services.recovery.add_run_event", new_callable=AsyncMock):
                            with patch("api.services.recovery.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
                                mock_get_priority.return_value = None  # Приоритет не установлен
                                mock_get_metadata.return_value = {"video_id": "test_video"}
                                mock_enqueue.return_value = "1234567890-0"
                                
                                result = await recover_run("test-run-id")
                                
                                assert result is True
                                # Проверить что использован приоритет "normal"
                                call_args = mock_enqueue.call_args
                                assert call_args[1]["priority"] == "normal"
    
    @pytest.mark.asyncio
    async def test_recover_run_redis_unavailable(self):
        """Redis недоступен при восстановлении."""
        with patch("api.services.recovery.get_redis_client", return_value=None):
            result = await recover_run("test-run-id")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_recover_run_enqueue_error(self, mock_redis_client):
        """Ошибка при добавлении в очередь."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.get_run_priority", new_callable=AsyncMock) as mock_get_priority:
                with patch("api.services.recovery.get_run_metadata", new_callable=AsyncMock) as mock_get_metadata:
                    with patch("api.services.recovery.save_run_state", new_callable=AsyncMock):
                        with patch("api.services.recovery.add_run_event", new_callable=AsyncMock):
                            with patch("api.services.recovery.enqueue_run", new_callable=AsyncMock) as mock_enqueue:
                                mock_get_priority.return_value = "normal"
                                mock_get_metadata.return_value = {"video_id": "test_video"}
                                mock_enqueue.side_effect = Exception("Enqueue error")
                                
                                result = await recover_run("test-run-id")
                                
                                assert result is False


class TestRecoverAllCrashedRuns:
    """Тесты для функции recover_all_crashed_runs."""
    
    @pytest.mark.asyncio
    async def test_recover_all_crashed_runs_success(self, mock_redis_client):
        """Успешное восстановление всех crashed run'ов."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.check_and_recover_run", new_callable=AsyncMock) as mock_check:
                # Мокаем scan_iter для поиска run:state:* ключей
                async def mock_scan_iter(match):
                    keys = [
                        b"run:state:run-1",
                        b"run:state:run-2",
                        b"run:state:run-3"
                    ]
                    for key in keys:
                        if match in key.decode():
                            yield key
                
                mock_redis_client.scan_iter = mock_scan_iter
                # Первый и третий run требуют восстановления
                mock_check.side_effect = [True, False, True]
                
                count = await recover_all_crashed_runs()
                
                assert count == 2
                assert mock_check.call_count == 3
    
    @pytest.mark.asyncio
    async def test_recover_all_crashed_runs_no_crashed(self, mock_redis_client):
        """Нет crashed run'ов для восстановления."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.recovery.check_and_recover_run", new_callable=AsyncMock) as mock_check:
                async def mock_scan_iter(match):
                    keys = [
                        b"run:state:run-1",
                        b"run:state:run-2"
                    ]
                    for key in keys:
                        if match in key.decode():
                            yield key
                
                mock_redis_client.scan_iter = mock_scan_iter
                mock_check.return_value = False  # Все run'ы в порядке
                
                count = await recover_all_crashed_runs()
                
                assert count == 0
                assert mock_check.call_count == 2
    
    @pytest.mark.asyncio
    async def test_recover_all_crashed_runs_no_runs(self, mock_redis_client):
        """Нет run'ов для проверки."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            async def mock_scan_iter(match):
                # Пустой итератор
                return
                yield  # Для async generator
            
            mock_redis_client.scan_iter = mock_scan_iter
            
            count = await recover_all_crashed_runs()
            
            assert count == 0
    
    @pytest.mark.asyncio
    async def test_recover_all_crashed_runs_redis_unavailable(self):
        """Redis недоступен."""
        with patch("api.services.recovery.get_redis_client", return_value=None):
            count = await recover_all_crashed_runs()
            assert count == 0
    
    @pytest.mark.asyncio
    async def test_recover_all_crashed_runs_error(self, mock_redis_client):
        """Ошибка при восстановлении."""
        with patch("api.services.recovery.get_redis_client", return_value=mock_redis_client):
            async def mock_scan_iter(match):
                yield b"run:state:run-1"
                raise Exception("Scan error")
            
            mock_redis_client.scan_iter = mock_scan_iter
            
            # Должен вернуть количество восстановленных до ошибки
            count = await recover_all_crashed_runs()
            assert count == 0

