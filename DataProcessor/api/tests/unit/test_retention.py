"""
Unit тесты для Retention Policy Service

Тесты для функций очистки старых данных:
- cleanup_redis_state
- cleanup_storage
- run_retention_cleanup

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2349-2376)
"""

import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from api.services.retention import (
    cleanup_redis_state,
    cleanup_storage,
    run_retention_cleanup,
    delete_run_storage,
    REDIS_STATE_RETENTION,
    STORAGE_RETENTION
)
from storage.base import Storage
from storage.paths import KeyLayout


@pytest.fixture
def mock_redis_client():
    """Mock Redis клиент для тестов."""
    redis_client = AsyncMock()
    redis_client.scan_iter = AsyncMock()
    redis_client.ttl = AsyncMock(return_value=-1)
    redis_client.get = AsyncMock()
    redis_client.delete = AsyncMock()
    return redis_client


@pytest.fixture
def mock_storage():
    """Mock Storage для тестов."""
    storage = MagicMock(spec=Storage)
    storage.list = MagicMock(return_value=[])
    storage.exists = MagicMock(return_value=True)
    storage.read_bytes = MagicMock()
    storage.__class__.__name__ = "FileSystemStorage"
    storage._abs = MagicMock(return_value="/data/result_store/youtube/video123/run-uuid")
    return storage


@pytest.fixture
def mock_key_layout():
    """Mock KeyLayout для тестов."""
    key_layout = MagicMock(spec=KeyLayout)
    key_layout.result_store_prefix = MagicMock(return_value="result_store")
    return key_layout


class TestCleanupRedisState:
    """Тесты для cleanup_redis_state."""
    
    @pytest.mark.asyncio
    async def test_cleanup_redis_state_no_redis(self):
        """Тест: если Redis недоступен, возвращает пустые результаты."""
        with patch("api.services.retention.get_redis_client", return_value=None):
            results = await cleanup_redis_state()
            
            assert results == {"checked": 0, "deleted": 0, "errors": 0}
    
    @pytest.mark.asyncio
    async def test_cleanup_redis_state_no_keys(self, mock_redis_client):
        """Тест: если нет ключей для проверки, возвращает пустые результаты."""
        mock_redis_client.scan_iter.return_value = []
        
        with patch("api.services.retention.get_redis_client", return_value=mock_redis_client):
            results = await cleanup_redis_state()
            
            assert results == {"checked": 0, "deleted": 0, "errors": 0}
    
    @pytest.mark.asyncio
    async def test_cleanup_redis_state_with_ttl(self, mock_redis_client):
        """Тест: ключи с TTL > 0 не удаляются (будут удалены автоматически)."""
        # Создать mock ключи
        keys = [b"run:state:run-1", b"run:state:run-2"]
        mock_redis_client.scan_iter.return_value = iter(keys)
        mock_redis_client.ttl.return_value = 3600  # TTL > 0
        
        with patch("api.services.retention.get_redis_client", return_value=mock_redis_client):
            results = await cleanup_redis_state()
            
            assert results["checked"] == 2
            assert results["deleted"] == 0
            assert results["errors"] == 0
            # Проверить что delete не вызывался
            mock_redis_client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cleanup_redis_state_old_key(self, mock_redis_client):
        """Тест: старые ключи без TTL удаляются."""
        cutoff = time.time() - REDIS_STATE_RETENTION - 100  # Старый ключ
        old_state = {
            "status": "success",
            "updated_at": cutoff - 1000  # Очень старый
        }
        
        keys = [b"run:state:run-1"]
        mock_redis_client.scan_iter.return_value = iter(keys)
        mock_redis_client.ttl.return_value = -1  # Нет TTL
        mock_redis_client.get.return_value = json.dumps(old_state).encode("utf-8")
        
        with patch("api.services.retention.get_redis_client", return_value=mock_redis_client):
            results = await cleanup_redis_state(cutoff_timestamp=cutoff)
            
            assert results["checked"] == 1
            assert results["deleted"] == 1
            assert results["errors"] == 0
            mock_redis_client.delete.assert_called_once_with("run:state:run-1")
    
    @pytest.mark.asyncio
    async def test_cleanup_redis_state_recent_key(self, mock_redis_client):
        """Тест: недавние ключи не удаляются."""
        cutoff = time.time() - REDIS_STATE_RETENTION - 100
        recent_state = {
            "status": "success",
            "updated_at": time.time()  # Недавний
        }
        
        keys = [b"run:state:run-1"]
        mock_redis_client.scan_iter.return_value = iter(keys)
        mock_redis_client.ttl.return_value = -1
        mock_redis_client.get.return_value = json.dumps(recent_state).encode("utf-8")
        
        with patch("api.services.retention.get_redis_client", return_value=mock_redis_client):
            results = await cleanup_redis_state(cutoff_timestamp=cutoff)
            
            assert results["checked"] == 1
            assert results["deleted"] == 0  # Не удалён
            assert results["errors"] == 0
            mock_redis_client.delete.assert_not_called()


class TestCleanupStorage:
    """Тесты для cleanup_storage."""
    
    @pytest.mark.asyncio
    async def test_cleanup_storage_no_runs(self, mock_storage, mock_key_layout):
        """Тест: если нет run'ов, возвращает пустые результаты."""
        mock_storage.list.return_value = []
        
        results = await cleanup_storage(mock_storage, mock_key_layout)
        
        assert results == {"checked": 0, "deleted": 0, "errors": 0}
    
    @pytest.mark.asyncio
    async def test_cleanup_storage_old_run(self, mock_storage, mock_key_layout):
        """Тест: старые run'ы удаляются."""
        cutoff = time.time() - STORAGE_RETENTION - 100
        
        # Mock структура storage
        from storage.base import ObjectInfo
        
        # Platform: youtube
        platform_obj = ObjectInfo(key="result_store/youtube", size_bytes=0)
        # Video: video123
        video_obj = ObjectInfo(key="result_store/youtube/video123", size_bytes=0)
        # Run: run-uuid
        run_obj = ObjectInfo(key="result_store/youtube/video123/run-uuid", size_bytes=0)
        # Manifest
        manifest_obj = ObjectInfo(key="result_store/youtube/video123/run-uuid/manifest.json", size_bytes=100)
        
        old_manifest = {
            "run_id": "run-uuid",
            "finished_at": cutoff - 1000  # Очень старый
        }
        
        # Настроить mock для list
        def list_side_effect(prefix):
            if prefix == "result_store":
                return [platform_obj]
            elif prefix == "result_store/youtube":
                return [video_obj]
            elif prefix == "result_store/youtube/video123":
                return [run_obj]
            elif prefix == "result_store/youtube/video123/run-uuid":
                return [manifest_obj]
            return []
        
        mock_storage.list.side_effect = list_side_effect
        mock_storage.exists.return_value = True
        mock_storage.read_bytes.return_value = json.dumps(old_manifest).encode("utf-8")
        
        # Mock для delete_run_storage
        with patch("api.services.retention.delete_run_storage", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            
            results = await cleanup_storage(mock_storage, mock_key_layout, cutoff_timestamp=cutoff)
            
            assert results["checked"] == 1
            assert results["deleted"] == 1
            assert results["errors"] == 0
            mock_delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_storage_recent_run(self, mock_storage, mock_key_layout):
        """Тест: недавние run'ы не удаляются."""
        cutoff = time.time() - STORAGE_RETENTION - 100
        
        from storage.base import ObjectInfo
        
        platform_obj = ObjectInfo(key="result_store/youtube", size_bytes=0)
        video_obj = ObjectInfo(key="result_store/youtube/video123", size_bytes=0)
        run_obj = ObjectInfo(key="result_store/youtube/video123/run-uuid", size_bytes=0)
        manifest_obj = ObjectInfo(key="result_store/youtube/video123/run-uuid/manifest.json", size_bytes=100)
        
        recent_manifest = {
            "run_id": "run-uuid",
            "finished_at": time.time()  # Недавний
        }
        
        def list_side_effect(prefix):
            if prefix == "result_store":
                return [platform_obj]
            elif prefix == "result_store/youtube":
                return [video_obj]
            elif prefix == "result_store/youtube/video123":
                return [run_obj]
            elif prefix == "result_store/youtube/video123/run-uuid":
                return [manifest_obj]
            return []
        
        mock_storage.list.side_effect = list_side_effect
        mock_storage.exists.return_value = True
        mock_storage.read_bytes.return_value = json.dumps(recent_manifest).encode("utf-8")
        
        with patch("api.services.retention.delete_run_storage", new_callable=AsyncMock) as mock_delete:
            results = await cleanup_storage(mock_storage, mock_key_layout, cutoff_timestamp=cutoff)
            
            assert results["checked"] == 1
            assert results["deleted"] == 0  # Не удалён
            assert results["errors"] == 0
            mock_delete.assert_not_called()


class TestDeleteRunStorage:
    """Тесты для delete_run_storage."""
    
    @pytest.mark.asyncio
    async def test_delete_run_storage_filesystem(self):
        """Тест: удаление из FileSystemStorage."""
        storage = MagicMock()
        storage.__class__.__name__ = "FileSystemStorage"
        storage._abs = MagicMock(return_value="/data/result_store/youtube/video123/run-uuid")
        
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isdir", return_value=True), \
             patch("shutil.rmtree") as mock_rmtree:
            
            result = await delete_run_storage(storage, "result_store/youtube/video123/run-uuid")
            
            assert result is True
            mock_rmtree.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_run_storage_s3(self):
        """Тест: удаление из S3Storage."""
        storage = MagicMock()
        storage.__class__.__name__ = "S3Storage"
        storage._client = MagicMock()
        storage._k = MagicMock(side_effect=lambda x: x)
        storage.bucket = "test-bucket"
        
        from storage.base import ObjectInfo
        
        obj1 = ObjectInfo(key="result_store/youtube/video123/run-uuid/manifest.json", size_bytes=100)
        obj2 = ObjectInfo(key="result_store/youtube/video123/run-uuid/component1.npz", size_bytes=200)
        
        storage.list = MagicMock(return_value=[obj1, obj2])
        storage._client.delete_objects = MagicMock(return_value={})
        
        result = await delete_run_storage(storage, "result_store/youtube/video123/run-uuid")
        
        assert result is True
        storage._client.delete_objects.assert_called_once()
        # Проверить что delete_objects был вызван с правильными параметрами
        call_args = storage._client.delete_objects.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert len(call_args[1]["Delete"]["Objects"]) == 2


class TestRunRetentionCleanup:
    """Тесты для run_retention_cleanup."""
    
    @pytest.mark.asyncio
    async def test_run_retention_cleanup_success(self, mock_storage, mock_key_layout):
        """Тест: успешный запуск полной очистки."""
        redis_results = {"checked": 10, "deleted": 2, "errors": 0}
        storage_results = {"checked": 5, "deleted": 1, "errors": 0}
        
        with patch("api.services.retention.cleanup_redis_state", new_callable=AsyncMock) as mock_redis, \
             patch("api.services.retention.cleanup_storage", new_callable=AsyncMock) as mock_storage_cleanup:
            
            mock_redis.return_value = redis_results
            mock_storage_cleanup.return_value = storage_results
            
            results = await run_retention_cleanup(mock_storage, mock_key_layout)
            
            assert results["redis"] == redis_results
            assert results["storage"] == storage_results
            assert "timestamp" in results
            assert "elapsed_seconds" in results
            assert results["elapsed_seconds"] >= 0
    
    @pytest.mark.asyncio
    async def test_run_retention_cleanup_with_errors(self, mock_storage, mock_key_layout):
        """Тест: очистка с ошибками."""
        redis_results = {"checked": 10, "deleted": 2, "errors": 1}
        storage_results = {"checked": 5, "deleted": 1, "errors": 1}
        
        with patch("api.services.retention.cleanup_redis_state", new_callable=AsyncMock) as mock_redis, \
             patch("api.services.retention.cleanup_storage", new_callable=AsyncMock) as mock_storage_cleanup:
            
            mock_redis.return_value = redis_results
            mock_storage_cleanup.return_value = storage_results
            
            results = await run_retention_cleanup(mock_storage, mock_key_layout)
            
            assert results["redis"]["errors"] == 1
            assert results["storage"]["errors"] == 1

