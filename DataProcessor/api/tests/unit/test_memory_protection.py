"""
Unit тесты для Memory Protection

Тесты для функций защиты от превышения памяти:
- subprocess memory monitoring
- kill при превышении лимита
- обновление метрик

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2378-2416)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any

from api.services.processor import ProcessorService
from api.schemas.requests import ProcessRequest
from api.config import config


@pytest.fixture
def mock_process():
    """Mock subprocess процесс для тестов."""
    process = AsyncMock()
    process.pid = 12345
    process.returncode = None
    process.kill = Mock()
    process.communicate = AsyncMock(return_value=(b"stdout", b"stderr"))
    return process


@pytest.fixture
def processor_service():
    """Создать ProcessorService для тестов."""
    return ProcessorService()


@pytest.fixture
def sample_request():
    """Создать sample ProcessRequest для тестов."""
    return ProcessRequest(
        run_id="test-run-123",
        video_id="video-123",
        platform_id="youtube",
        video_path="/path/to/video.mp4",
        profile_config={}
    )


class TestMemoryMonitoring:
    """Тесты для memory monitoring."""
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_no_psutil(self, processor_service, mock_process):
        """Тест: если psutil не доступен, мониторинг не запускается."""
        with patch("api.services.processor.psutil", side_effect=ImportError("No module named 'psutil'")):
            # Должен вернуться без ошибок
            await processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            # Процесс не должен быть убит
            mock_process.kill.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_no_limit(self, processor_service, mock_process):
        """Тест: если лимит не установлен, мониторинг не запускается."""
        with patch("api.services.processor.config") as mock_config:
            mock_config.subprocess_memory_limit_mb = None
            
            await processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            mock_process.kill.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_normal_usage(self, processor_service, mock_process):
        """Тест: нормальное использование памяти не убивает процесс."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            mock_config.subprocess_memory_limit_mb = 8000  # 8GB
            
            # Mock psutil процесс
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=4 * 1024 * 1024 * 1024)  # 4GB
            mock_psutil.Process.return_value = mock_proc
            
            # Mock процесс завершится после первой проверки
            mock_process.returncode = 0
            
            # Запустить мониторинг
            task = asyncio.create_task(
                processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            )
            
            # Подождать немного
            await asyncio.sleep(0.1)
            
            # Отменить задачу
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Процесс не должен быть убит
            mock_process.kill.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_exceeds_limit(self, processor_service, mock_process):
        """Тест: превышение лимита убивает процесс."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil, \
             patch("api.services.processor.logger") as mock_logger, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            mock_config.subprocess_memory_limit_mb = 8000  # 8GB
            
            # Mock psutil процесс с превышением лимита
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=9 * 1024 * 1024 * 1024)  # 9GB > 8GB
            mock_psutil.Process.return_value = mock_proc
            
            # Запустить мониторинг
            task = asyncio.create_task(
                processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            )
            
            # Подождать немного для выполнения проверки
            await asyncio.sleep(0.1)
            
            # Отменить задачу
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Процесс должен быть убит
            mock_process.kill.assert_called_once()
            mock_logger.warning.assert_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_process_finished(self, processor_service, mock_process):
        """Тест: если процесс завершился, мониторинг останавливается."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil:
            
            mock_config.subprocess_memory_limit_mb = 8000
            
            # Процесс уже завершился
            mock_process.returncode = 0
            
            await processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            
            # Процесс не должен быть убит (уже завершился)
            mock_process.kill.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_no_such_process(self, processor_service, mock_process):
        """Тест: если процесс не найден, мониторинг останавливается."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil:
            
            mock_config.subprocess_memory_limit_mb = 8000
            
            # psutil.NoSuchProcess исключение
            from psutil import NoSuchProcess
            mock_psutil.Process.side_effect = NoSuchProcess(12345)
            
            await processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            
            # Процесс не должен быть убит (уже не существует)
            mock_process.kill.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_metrics_update(self, processor_service, mock_process):
        """Тест: метрики обновляются при мониторинге."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil, \
             patch("api.services.processor.metrics") as mock_metrics:
            
            mock_config.subprocess_memory_limit_mb = 8000
            
            # Mock psutil процесс
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=4 * 1024 * 1024 * 1024)  # 4GB
            mock_psutil.Process.return_value = mock_proc
            
            # Mock метрики
            mock_memory_usage = MagicMock()
            mock_memory_usage.labels.return_value = mock_memory_usage
            mock_memory_usage.set = Mock()
            mock_metrics.memory_usage = mock_memory_usage
            
            # Процесс завершится после первой проверки
            mock_process.returncode = 0
            
            task = asyncio.create_task(
                processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            )
            
            await asyncio.sleep(0.1)
            
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Метрика должна быть обновлена
            mock_memory_usage.labels.assert_called_with(run_id="test-run-123")
            mock_memory_usage.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_check_interval(self, processor_service, mock_process):
        """Тест: проверка выполняется каждые 10 секунд."""
        with patch("api.services.processor.config") as mock_config, \
             patch("api.services.processor.psutil") as mock_psutil, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            
            mock_config.subprocess_memory_limit_mb = 8000
            
            # Mock psutil процесс
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=4 * 1024 * 1024 * 1024)  # 4GB
            mock_psutil.Process.return_value = mock_proc
            
            # Процесс завершится после нескольких проверок
            call_count = 0
            def returncode_side_effect():
                nonlocal call_count
                call_count += 1
                return None if call_count < 3 else 0
            
            mock_process.returncode = None
            mock_process.__getattribute__ = lambda self, name: returncode_side_effect() if name == "returncode" else object.__getattribute__(self, name)
            
            task = asyncio.create_task(
                processor_service._monitor_subprocess_memory(mock_process, "test-run-123")
            )
            
            await asyncio.sleep(0.2)
            
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # asyncio.sleep должен быть вызван с интервалом 10 секунд
            # (хотя бы один раз для первой проверки)
            assert mock_sleep.called


class TestMemoryProtectionIntegration:
    """Интеграционные тесты для memory protection."""
    
    @pytest.mark.asyncio
    async def test_memory_protection_in_processor(self, processor_service, sample_request):
        """Тест: memory protection интегрирован в ProcessorService."""
        with patch("api.services.processor.asyncio.create_subprocess_exec") as mock_create, \
             patch("api.services.processor.config") as mock_config:
            
            mock_config.subprocess_memory_limit_mb = 8000
            
            # Mock subprocess
            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"stdout", b"stderr"))
            mock_create.return_value = mock_process
            
            # Mock мониторинг памяти
            with patch.object(processor_service, "_monitor_subprocess_memory", new_callable=AsyncMock) as mock_monitor:
                result = await processor_service._run_main_py_async(sample_request)
                
                # Мониторинг должен быть запущен
                assert mock_monitor.called
                assert result["success"] is True

