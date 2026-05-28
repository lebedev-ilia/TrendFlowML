"""
Pytest конфигурация и фикстуры для тестов API
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.config import APIConfig
from api.services.task_manager import TaskManager
from api.services.processor import ProcessorService
from storage.fs import FileSystemStorage
from state.managers import KeyLayout


@pytest.fixture(scope="session")
def event_loop():
    """Создать event loop для тестов."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def api_config() -> APIConfig:
    """Фикстура для конфигурации API."""
    return APIConfig(
        max_concurrent_runs=4,
        storage_type="fs",
        storage_root="/tmp/test_storage",
        redis_url=None,  # Отключить Redis для unit тестов
        api_key="test_api_key",
        auth_type="api_key"
    )


@pytest.fixture
def mock_storage() -> Mock:
    """Фикстура для мок Storage."""
    storage = Mock(spec=FileSystemStorage)
    storage.exists = AsyncMock(return_value=True)
    storage.read_bytes = AsyncMock(return_value=b'{"test": "data"}')
    storage.write_bytes = AsyncMock(return_value=None)
    storage.atomic_write_bytes = AsyncMock(return_value=None)
    storage.list = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_key_layout() -> Mock:
    """Фикстура для мок KeyLayout."""
    key_layout = Mock(spec=KeyLayout)
    key_layout.result_store_run_prefix = Mock(return_value="test/prefix")
    return key_layout


@pytest.fixture
def task_manager() -> TaskManager:
    """Фикстура для TaskManager."""
    return TaskManager(max_concurrent_runs=4)


@pytest.fixture
def processor_service() -> ProcessorService:
    """Фикстура для ProcessorService."""
    return ProcessorService()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Фикстура для FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_redis_client():
    """Фикстура для мок Redis клиента."""
    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.setex = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=1)
    redis_client.exists = AsyncMock(return_value=False)
    redis_client.hgetall = AsyncMock(return_value={})
    redis_client.hset = AsyncMock(return_value=1)
    redis_client.xadd = AsyncMock(return_value=b"0-0")
    redis_client.xread = AsyncMock(return_value=[])
    redis_client.xreadgroup = AsyncMock(return_value=[])
    redis_client.xack = AsyncMock(return_value=1)
    redis_client.xpending_range = AsyncMock(return_value=[])
    redis_client.xlen = AsyncMock(return_value=0)
    redis_client.lpush = AsyncMock(return_value=1)
    redis_client.ltrim = AsyncMock(return_value=True)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.lrange = AsyncMock(return_value=[])
    redis_client.ping = AsyncMock(return_value=True)
    return redis_client


@pytest.fixture
def sample_process_request():
    """Фикстура для примера ProcessRequest."""
    return {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "video_id": "test_video",
        "platform_id": "youtube",
        "video_path": "/tmp/test_video.mp4",
        "config_hash": "test_hash",
        "profile_config": {
            "processors": {
                "segmenter": {"enabled": True, "required": True},
                "visual": {"enabled": True, "required": True}
            }
        }
    }

