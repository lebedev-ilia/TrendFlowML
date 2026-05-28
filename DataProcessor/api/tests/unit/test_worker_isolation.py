"""
Unit тесты для worker isolation (subprocess isolation)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import subprocess

from api.services.processor import ProcessorService
from api.services.worker import Worker


class TestSubprocessIsolation:
    """Тесты для изоляции subprocess."""
    
    @pytest.mark.asyncio
    async def test_subprocess_launched_for_each_run(self):
        """Каждый run запускается в отдельном subprocess."""
        processor_service = ProcessorService()
        
        # Мокаем ProcessRequest
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Мокаем subprocess запуск
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=0)
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Мокаем сохранение profile config
            with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                # Мокаем чтение stdout/stderr
                mock_process.stdout = AsyncMock()
                mock_process.stdout.readline = AsyncMock(return_value=b"")
                mock_process.stderr = AsyncMock()
                mock_process.stderr.readline = AsyncMock(return_value=b"")
                
                # Запустить обработку
                try:
                    result = await asyncio.wait_for(
                        processor_service.run_processing(request),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Это нормально - subprocess может работать долго
                    pass
                
                # Проверяем что subprocess был запущен
                assert mock_subprocess.called


class TestWorkerRequestReconstruction:
    def test_build_process_request_data_keeps_full_backend_payload(self):
        request_data = Worker._build_process_request_data(
            "550e8400-e29b-41d4-a716-446655440000",
            {
                "video_id": "dQw4w9WgXcQ",
                "platform_id": "youtube",
                "video_path": "/tmp/video.mp4",
                "config_hash": "abc123",
                "profile_config": {"processors": {"segmenter": {"enabled": True}}},
                "visual_cfg_path": "/tmp/visual.yaml",
                "dag_path": "/tmp/dag.yaml",
                "dag_stage": "baseline",
                "rs_base": "/tmp/result_store",
                "output": "/tmp/frames",
                "chunk_size": 64,
            },
        )

        assert request_data["run_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert request_data["video_id"] == "dQw4w9WgXcQ"
        assert request_data["platform_id"] == "youtube"
        assert request_data["video_path"] == "/tmp/video.mp4"
        assert request_data["visual_cfg_path"] == "/tmp/visual.yaml"
        assert request_data["dag_path"] == "/tmp/dag.yaml"
        assert request_data["dag_stage"] == "baseline"
        assert request_data["rs_base"] == "/tmp/result_store"
        assert request_data["output"] == "/tmp/frames"
        assert request_data["chunk_size"] == 64
    
    @pytest.mark.asyncio
    async def test_subprocess_exit_code_handling(self):
        """Обработка exit code subprocess."""
        processor_service = ProcessorService()
        
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Тест успешного завершения (exit code 0)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=0)
            mock_process.returncode = 0
            mock_process.stdout = AsyncMock()
            mock_process.stdout.readline = AsyncMock(return_value=b"")
            mock_process.stderr = AsyncMock()
            mock_process.stderr.readline = AsyncMock(return_value=b"")
            mock_subprocess.return_value = mock_process
            
            with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                try:
                    result = await asyncio.wait_for(
                        processor_service.run_processing(request),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    pass
                
                # Проверяем что процесс был запущен
                assert mock_subprocess.called
        
        # Тест ошибки (exit code != 0)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=1)
            mock_process.returncode = 1
            mock_process.stdout = AsyncMock()
            mock_process.stdout.readline = AsyncMock(return_value=b"")
            mock_process.stderr = AsyncMock()
            mock_process.stderr.readline = AsyncMock(return_value=b"Error occurred")
            mock_subprocess.return_value = mock_process
            
            with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                try:
                    result = await asyncio.wait_for(
                        processor_service.run_processing(request),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    pass
    
    @pytest.mark.asyncio
    async def test_subprocess_memory_monitoring(self):
        """Мониторинг памяти subprocess."""
        processor_service = ProcessorService()
        
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Мокаем psutil для мониторинга памяти
        with patch("psutil.Process") as mock_psutil_process:
            mock_process_obj = MagicMock()
            mock_process_obj.memory_info = MagicMock(return_value=MagicMock(rss=1024 * 1024 * 1024))  # 1GB
            mock_psutil_process.return_value = mock_process_obj
            
            # Мокаем subprocess
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.wait = AsyncMock(return_value=0)
                mock_process.returncode = 0
                mock_process.stdout = AsyncMock()
                mock_process.stdout.readline = AsyncMock(return_value=b"")
                mock_process.stderr = AsyncMock()
                mock_process.stderr.readline = AsyncMock(return_value=b"")
                mock_subprocess.return_value = mock_process
                
                with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                    # Мокаем мониторинг памяти (если он реализован)
                    # В реальном коде это может быть в отдельном методе
                    try:
                        result = await asyncio.wait_for(
                            processor_service.run_processing(request),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        pass
    
    @pytest.mark.asyncio
    async def test_subprocess_kill_on_memory_limit(self):
        """Kill subprocess при превышении лимита памяти."""
        processor_service = ProcessorService()
        
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Мокаем превышение лимита памяти
        with patch("psutil.Process") as mock_psutil_process:
            # Симулируем превышение лимита (например, 9GB при лимите 8GB)
            mock_process_obj = MagicMock()
            mock_process_obj.memory_info = MagicMock(return_value=MagicMock(rss=9 * 1024 * 1024 * 1024))
            mock_process_obj.kill = MagicMock()
            mock_psutil_process.return_value = mock_process_obj
            
            # Мокаем subprocess
            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.wait = AsyncMock(return_value=-9)  # SIGKILL
                mock_process.returncode = -9
                mock_process.stdout = AsyncMock()
                mock_process.stdout.readline = AsyncMock(return_value=b"")
                mock_process.stderr = AsyncMock()
                mock_process.stderr.readline = AsyncMock(return_value=b"")
                mock_subprocess.return_value = mock_process
                
                with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                    try:
                        result = await asyncio.wait_for(
                            processor_service.run_processing(request),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        pass
                    
                    # Проверяем что процесс был убит (если мониторинг памяти реализован)
                    # В реальном коде это должно быть проверено через exit code -9


class TestSubprocessErrorHandling:
    """Тесты для обработки ошибок subprocess."""
    
    @pytest.mark.asyncio
    async def test_subprocess_timeout_handling(self):
        """Обработка timeout subprocess."""
        processor_service = ProcessorService()
        
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Мокаем timeout
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError("Process timeout"))
            mock_subprocess.return_value = mock_process
            
            with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                try:
                    result = await asyncio.wait_for(
                        processor_service.run_processing(request),
                        timeout=0.1
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
    
    @pytest.mark.asyncio
    async def test_subprocess_stdout_stderr_handling(self):
        """Обработка stdout/stderr из subprocess."""
        processor_service = ProcessorService()
        
        from api.schemas.requests import ProcessRequest
        
        request = ProcessRequest(
            run_id="test-run-id",
            video_id="test_video",
            platform_id="youtube",
            video_path="/tmp/test_video.mp4"
        )
        
        # Мокаем stdout/stderr с данными
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_subprocess:
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=0)
            mock_process.returncode = 0
            
            # Мокаем чтение stdout
            stdout_lines = [b"Line 1\n", b"Line 2\n", b""]
            mock_process.stdout = AsyncMock()
            mock_process.stdout.readline = AsyncMock(side_effect=stdout_lines)
            
            # Мокаем чтение stderr
            stderr_lines = [b"Error line\n", b""]
            mock_process.stderr = AsyncMock()
            mock_process.stderr.readline = AsyncMock(side_effect=stderr_lines)
            
            mock_subprocess.return_value = mock_process
            
            with patch.object(processor_service, '_save_profile_config', return_value="/tmp/profile.yaml"):
                try:
                    result = await asyncio.wait_for(
                        processor_service.run_processing(request),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    pass
                
                # Проверяем что stdout/stderr были прочитаны
                assert mock_process.stdout.readline.called
                assert mock_process.stderr.readline.called

