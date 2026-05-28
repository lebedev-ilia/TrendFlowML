"""
Unit тесты для TaskManager
"""

import pytest
from datetime import datetime
from api.services.task_manager import TaskManager
from api.schemas.state import RunStatus


class TestTaskManager:
    """Тесты для TaskManager."""
    
    @pytest.fixture
    def task_manager(self):
        """Создать TaskManager для тестов."""
        return TaskManager(max_concurrent_runs=4)
    
    def test_register_run(self, task_manager):
        """Регистрация нового run'а."""
        run_id = "test-run-1"
        metadata = {"video_id": "test_video", "platform_id": "youtube"}
        
        task_manager.register_run(run_id, metadata)
        
        assert task_manager.is_run_active(run_id) is True
        run_info = task_manager.get_run(run_id)
        assert run_info is not None
        assert run_info["status"] == RunStatus.QUEUED.value
    
    def test_register_run_with_custom_status(self, task_manager):
        """Регистрация run'а с кастомным статусом."""
        run_id = "test-run-2"
        metadata = {"video_id": "test_video"}
        
        task_manager.register_run(run_id, metadata, initial_status=RunStatus.PENDING)
        
        run_info = task_manager.get_run(run_id)
        assert run_info["status"] == RunStatus.PENDING.value
    
    def test_update_run_status(self, task_manager):
        """Обновление статуса run'а."""
        run_id = "test-run-3"
        metadata = {"video_id": "test_video"}
        
        task_manager.register_run(run_id, metadata)
        task_manager.update_run_status(run_id, RunStatus.RUNNING, started_at=1234567890.0)
        
        run_info = task_manager.get_run(run_id)
        assert run_info["status"] == RunStatus.RUNNING.value
        assert run_info.get("started_at") == 1234567890.0
    
    def test_update_run_status_invalid_transition(self, task_manager):
        """Обновление статуса с невалидным переходом."""
        run_id = "test-run-4"
        metadata = {"video_id": "test_video"}
        
        task_manager.register_run(run_id, metadata)
        task_manager.update_run_status(run_id, RunStatus.SUCCESS)
        task_manager.update_run_status(run_id, RunStatus.SUCCESS)  # Повторное обновление допустимо
        
        # Попытка перейти из SUCCESS в RUNNING должна вызвать ValueError
        with pytest.raises(ValueError, match="Invalid status transition"):
            task_manager.update_run_status(run_id, RunStatus.RUNNING)
    
    def test_get_run_not_found(self, task_manager):
        """Получение несуществующего run'а."""
        run_info = task_manager.get_run("non-existent-run")
        assert run_info is None
    
    def test_is_run_active(self, task_manager):
        """Проверка активности run'а."""
        run_id = "test-run-5"
        metadata = {"video_id": "test_video"}
        
        assert task_manager.is_run_active(run_id) is False
        
        task_manager.register_run(run_id, metadata)
        assert task_manager.is_run_active(run_id) is True
        
        task_manager.update_run_status(run_id, RunStatus.SUCCESS)
        # SUCCESS - финальное состояние, но run всё ещё считается активным в TaskManager
        # (это может быть изменено в будущем)
        assert task_manager.is_run_active(run_id) is True
    
    def test_can_accept_new_run(self, task_manager):
        """Проверка возможности принять новый run."""
        # TaskManager с max_concurrent_runs=4
        assert task_manager.can_accept_new_run() is True
        
        # Зарегистрировать 4 run'а
        for i in range(4):
            run_id = f"test-run-{i}"
            task_manager.register_run(run_id, {"video_id": f"video_{i}"})
            task_manager.update_run_status(run_id, RunStatus.RUNNING)
        
        # Теперь не должно быть возможности принять новый run
        assert task_manager.can_accept_new_run() is False
    
    def test_get_all_runs(self, task_manager):
        """Получение всех run'ов."""
        # Зарегистрировать несколько run'ов
        for i in range(3):
            run_id = f"test-run-{i}"
            task_manager.register_run(run_id, {"video_id": f"video_{i}"})
        
        all_runs = task_manager.get_all_runs()
        assert len(all_runs) == 3
    
    def test_concurrent_runs_limit(self, task_manager):
        """Проверка лимита параллельных run'ов."""
        # TaskManager с max_concurrent_runs=4
        # Зарегистрировать run'ы в разных статусах
        for i in range(2):
            run_id = f"running-run-{i}"
            task_manager.register_run(run_id, {"video_id": f"video_{i}"})
            task_manager.update_run_status(run_id, RunStatus.RUNNING)
        
        for i in range(2):
            run_id = f"queued-run-{i}"
            task_manager.register_run(run_id, {"video_id": f"video_{i+2}"})
            # Оставить в QUEUED
        
        # Должно быть 2 активных (RUNNING) и 2 в очереди (QUEUED)
        # can_accept_new_run должен учитывать только RUNNING
        assert task_manager.can_accept_new_run() is True  # 2 < 4
        
        # Добавить ещё 2 RUNNING
        for i in range(2):
            run_id = f"running-run-{i+2}"
            task_manager.register_run(run_id, {"video_id": f"video_{i+4}"})
            task_manager.update_run_status(run_id, RunStatus.RUNNING)
        
        # Теперь 4 RUNNING, лимит достигнут
        assert task_manager.can_accept_new_run() is False

