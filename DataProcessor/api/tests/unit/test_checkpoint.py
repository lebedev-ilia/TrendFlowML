"""
Unit тесты для Checkpoint Service
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from api.services.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    determine_last_processor,
    get_checkpoint_info,
    delete_checkpoint,
    PROCESSOR_ORDER
)


class TestSaveCheckpoint:
    """Тесты для функции save_checkpoint."""
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_success(self, mock_storage, mock_key_layout):
        """Успешное сохранение checkpoint'а."""
        with patch("api.services.checkpoint.retry_storage_operation", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = None
            
            result = await save_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id",
                last_processor="segmenter",
                status="running"
            )
            
            assert result is True
            mock_retry.assert_called_once()
            # Проверить что данные корректны
            call_args = mock_retry.call_args
            checkpoint_bytes = call_args[0][1]
            checkpoint_data = json.loads(checkpoint_bytes.decode("utf-8"))
            assert checkpoint_data["run_id"] == "test-run-id"
            assert checkpoint_data["platform_id"] == "youtube"
            assert checkpoint_data["video_id"] == "test_video"
            assert checkpoint_data["status"] == "running"
            assert checkpoint_data["last_processor"] == "segmenter"
            assert checkpoint_data["processor_order"] == PROCESSOR_ORDER
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_no_last_processor(self, mock_storage, mock_key_layout):
        """Сохранение checkpoint'а без последнего процессора."""
        with patch("api.services.checkpoint.retry_storage_operation", new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = None
            
            result = await save_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id",
                last_processor=None,
                status="running"
            )
            
            assert result is True
            call_args = mock_retry.call_args
            checkpoint_bytes = call_args[0][1]
            checkpoint_data = json.loads(checkpoint_bytes.decode("utf-8"))
            assert checkpoint_data["last_processor"] is None
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_error(self, mock_storage, mock_key_layout):
        """Ошибка при сохранении checkpoint'а."""
        with patch("api.services.checkpoint.retry_storage_operation", new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = Exception("Storage error")
            
            result = await save_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is False


class TestLoadCheckpoint:
    """Тесты для функции load_checkpoint."""
    
    def test_load_checkpoint_success(self, mock_storage, mock_key_layout):
        """Успешная загрузка checkpoint'а."""
        checkpoint_data = {
            "run_id": "test-run-id",
            "platform_id": "youtube",
            "video_id": "test_video",
            "status": "running",
            "last_processor": "segmenter",
            "processor_order": PROCESSOR_ORDER
        }
        
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            # Первый вызов - exists, второй - read_bytes
            mock_retry.side_effect = [
                True,  # exists
                json.dumps(checkpoint_data).encode("utf-8")  # read_bytes
            ]
            
            result = load_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is not None
            assert result["run_id"] == "test-run-id"
            assert result["last_processor"] == "segmenter"
    
    def test_load_checkpoint_not_found(self, mock_storage, mock_key_layout):
        """Checkpoint не найден."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.return_value = False  # exists возвращает False
            
            result = load_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is None
    
    def test_load_checkpoint_invalid_json(self, mock_storage, mock_key_layout):
        """Невалидный JSON в checkpoint'е."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.side_effect = [
                True,  # exists
                b"Invalid JSON {"  # read_bytes
            ]
            
            result = load_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is None
    
    def test_load_checkpoint_error(self, mock_storage, mock_key_layout):
        """Ошибка при загрузке checkpoint'а."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.side_effect = Exception("Storage error")
            
            result = load_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is None


class TestDetermineLastProcessor:
    """Тесты для функции determine_last_processor."""
    
    def test_determine_last_processor_segmenter_success(self):
        """Определение последнего процессора - segmenter успешно выполнен."""
        processors = {
            "segmenter": {
                "processor": {"status": "success"},
                "status": "success"
            }
        }
        
        last_processor = determine_last_processor(processors)
        assert last_processor == "segmenter"
    
    def test_determine_last_processor_visual_running(self):
        """Определение последнего процессора - visual выполняется."""
        processors = {
            "segmenter": {"processor": {"status": "success"}},
            "audio": {"processor": {"status": "success"}},
            "text": {"processor": {"status": "success"}},
            "visual": {"processor": {"status": "running"}}
        }
        
        last_processor = determine_last_processor(processors)
        assert last_processor == "visual"
    
    def test_determine_last_processor_order(self):
        """Определение последнего процессора в правильном порядке."""
        processors = {
            "visual": {"processor": {"status": "success"}},
            "segmenter": {"processor": {"status": "success"}},
            "audio": {"processor": {"status": "success"}}
        }
        
        # Должен вернуть visual, так как он последний в порядке выполнения
        last_processor = determine_last_processor(processors)
        assert last_processor == "visual"
    
    def test_determine_last_processor_error(self):
        """Определение последнего процессора - процессор завершился с ошибкой."""
        processors = {
            "segmenter": {"processor": {"status": "success"}},
            "audio": {"processor": {"status": "error"}}
        }
        
        last_processor = determine_last_processor(processors)
        assert last_processor == "audio"
    
    def test_determine_last_processor_none(self):
        """Нет выполненного процессора."""
        processors = {
            "segmenter": {"processor": {"status": "waiting"}},
            "audio": {"processor": {"status": "pending"}}
        }
        
        last_processor = determine_last_processor(processors)
        assert last_processor is None
    
    def test_determine_last_processor_empty(self):
        """Пустой словарь процессоров."""
        processors = {}
        
        last_processor = determine_last_processor(processors)
        assert last_processor is None
    
    def test_determine_last_processor_status_in_state(self):
        """Статус процессора в state, а не в processor."""
        processors = {
            "segmenter": {"status": "success"}
        }
        
        last_processor = determine_last_processor(processors)
        assert last_processor == "segmenter"


class TestGetCheckpointInfo:
    """Тесты для функции get_checkpoint_info."""
    
    def test_get_checkpoint_info_success(self, mock_storage, mock_key_layout):
        """Успешное получение информации о checkpoint'е."""
        checkpoint_data = {
            "run_id": "test-run-id",
            "status": "running",
            "last_processor": "segmenter"
        }
        
        processor_state = {
            "processor": {"status": "success"},
            "status": "success"
        }
        
        with patch("api.services.checkpoint.load_checkpoint", return_value=checkpoint_data):
            with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
                # Мокаем проверку существования и чтение состояний процессоров
                mock_retry.side_effect = [
                    True,  # segmenter exists
                    json.dumps(processor_state).encode("utf-8"),  # segmenter read
                    False,  # audio exists
                    False,  # text exists
                    False   # visual exists
                ]
                
                result = get_checkpoint_info(
                    storage=mock_storage,
                    key_layout=mock_key_layout,
                    platform_id="youtube",
                    video_id="test_video",
                    run_id="test-run-id"
                )
                
                assert result is not None
                assert result["checkpoint"] == checkpoint_data
                assert result["last_processor"] == "segmenter"
                assert result["can_resume"] is True
    
    def test_get_checkpoint_info_not_found(self, mock_storage, mock_key_layout):
        """Checkpoint не найден."""
        with patch("api.services.checkpoint.load_checkpoint", return_value=None):
            result = get_checkpoint_info(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is None
    
    def test_get_checkpoint_info_cannot_resume(self, mock_storage, mock_key_layout):
        """Checkpoint найден, но нельзя возобновить (статус не running)."""
        checkpoint_data = {
            "run_id": "test-run-id",
            "status": "success",  # Завершенный run
            "last_processor": "visual"
        }
        
        with patch("api.services.checkpoint.load_checkpoint", return_value=checkpoint_data):
            with patch("api.services.checkpoint.retry_storage_operation", return_value=False):
                result = get_checkpoint_info(
                    storage=mock_storage,
                    key_layout=mock_key_layout,
                    platform_id="youtube",
                    video_id="test_video",
                    run_id="test-run-id"
                )
                
                assert result is not None
                assert result["can_resume"] is False
    
    def test_get_checkpoint_info_no_last_processor(self, mock_storage, mock_key_layout):
        """Checkpoint найден, но нет последнего процессора."""
        checkpoint_data = {
            "run_id": "test-run-id",
            "status": "running",
            "last_processor": None
        }
        
        with patch("api.services.checkpoint.load_checkpoint", return_value=checkpoint_data):
            with patch("api.services.checkpoint.retry_storage_operation", return_value=False):
                result = get_checkpoint_info(
                    storage=mock_storage,
                    key_layout=mock_key_layout,
                    platform_id="youtube",
                    video_id="test_video",
                    run_id="test-run-id"
                )
                
                assert result is not None
                assert result["can_resume"] is False


class TestDeleteCheckpoint:
    """Тесты для функции delete_checkpoint."""
    
    def test_delete_checkpoint_success(self, mock_storage, mock_key_layout):
        """Успешное удаление checkpoint'а."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.return_value = True  # exists возвращает True
            
            result = delete_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is True
    
    def test_delete_checkpoint_not_found(self, mock_storage, mock_key_layout):
        """Checkpoint не найден для удаления."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.return_value = False  # exists возвращает False
            
            result = delete_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is True  # Все равно возвращает True
    
    def test_delete_checkpoint_error(self, mock_storage, mock_key_layout):
        """Ошибка при удалении checkpoint'а."""
        with patch("api.services.checkpoint.retry_storage_operation") as mock_retry:
            mock_retry.side_effect = Exception("Storage error")
            
            result = delete_checkpoint(
                storage=mock_storage,
                key_layout=mock_key_layout,
                platform_id="youtube",
                video_id="test_video",
                run_id="test-run-id"
            )
            
            assert result is False

