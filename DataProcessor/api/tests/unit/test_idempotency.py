"""
Unit тесты для идемпотентности processors.

Тесты проверяют:
- Проверку существующих результатов
- Использование кэша при повторном запуске
- Обработку ошибок при проверке кэша
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from api.services.idempotency import check_existing_result, check_component_result
from storage.base import Storage
from storage.paths import KeyLayout


@pytest.fixture
def mock_storage():
    """Фикстура для мока Storage."""
    storage = Mock(spec=Storage)
    storage.exists = AsyncMock()
    storage.read_bytes = AsyncMock()
    return storage


@pytest.fixture
def mock_key_layout():
    """Фикстура для мока KeyLayout."""
    key_layout = Mock(spec=KeyLayout)
    key_layout.result_store_run_prefix = Mock(return_value="result_store/youtube/video123/run456")
    return key_layout


@pytest.mark.asyncio
async def test_check_existing_result_success(mock_storage, mock_key_layout):
    """Тест проверки существующего результата - успешный кэш."""
    # Настроить моки
    mock_storage.exists.return_value = True
    manifest_data = {
        "schema_version": "manifest_v1",
        "run": {
            "run_id": "run456",
            "video_id": "video123",
            "platform_id": "youtube",
            "status": "success"
        },
        "components": [
            {
                "name": "core_clip",
                "status": "success",
                "artifacts": [
                    {
                        "path": "core_clip/features.npz",
                        "size_bytes": 1024
                    }
                ]
            }
        ]
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    # Проверить существование артефакта
    mock_storage.exists.side_effect = [
        True,  # manifest.json существует
        True   # artifact существует
    ]
    
    result = await check_existing_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456"
    )
    
    assert result is not None
    assert result["success"] is True
    assert result["run_id"] == "run456"
    assert result["from_cache"] is True
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_check_existing_result_no_manifest(mock_storage, mock_key_layout):
    """Тест проверки существующего результата - manifest не найден."""
    mock_storage.exists.return_value = False
    
    result = await check_existing_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_check_existing_result_incomplete(mock_storage, mock_key_layout):
    """Тест проверки существующего результата - run не завершен."""
    mock_storage.exists.return_value = True
    manifest_data = {
        "run": {
            "run_id": "run456",
            "status": "running"  # Не завершен
        },
        "components": []
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    result = await check_existing_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_check_existing_result_missing_artifact(mock_storage, mock_key_layout):
    """Тест проверки существующего результата - артефакт отсутствует."""
    mock_storage.exists.side_effect = [
        True,  # manifest.json существует
        False  # artifact не существует
    ]
    manifest_data = {
        "run": {
            "run_id": "run456",
            "status": "success"
        },
        "components": [
            {
                "name": "core_clip",
                "status": "success",
                "artifacts": [
                    {
                        "path": "core_clip/features.npz"
                    }
                ]
            }
        ]
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    result = await check_existing_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_check_existing_result_error(mock_storage, mock_key_layout):
    """Тест проверки существующего результата - ошибка при проверке."""
    mock_storage.exists.side_effect = Exception("Storage error")
    
    result = await check_existing_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_check_component_result_success(mock_storage, mock_key_layout):
    """Тест проверки результата компонента - успешный кэш."""
    mock_storage.exists.side_effect = [
        True,  # manifest.json существует
        True   # artifact существует
    ]
    manifest_data = {
        "components": [
            {
                "name": "core_clip",
                "status": "success",
                "artifacts": [
                    {
                        "path": "core_clip/features.npz"
                    }
                ]
            }
        ]
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    result = await check_component_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456",
        "core_clip"
    )
    
    assert result is not None
    assert result["component"] == "core_clip"
    assert result["status"] == "success"
    assert result["from_cache"] is True


@pytest.mark.asyncio
async def test_check_component_result_not_found(mock_storage, mock_key_layout):
    """Тест проверки результата компонента - компонент не найден."""
    mock_storage.exists.return_value = True
    manifest_data = {
        "components": []
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    result = await check_component_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456",
        "core_clip"
    )
    
    assert result is None


@pytest.mark.asyncio
async def test_check_component_result_incomplete(mock_storage, mock_key_layout):
    """Тест проверки результата компонента - компонент не завершен."""
    mock_storage.exists.return_value = True
    manifest_data = {
        "components": [
            {
                "name": "core_clip",
                "status": "running",  # Не завершен
                "artifacts": []
            }
        ]
    }
    mock_storage.read_bytes.return_value = json.dumps(manifest_data).encode("utf-8")
    
    result = await check_component_result(
        mock_storage,
        mock_key_layout,
        "youtube",
        "video123",
        "run456",
        "core_clip"
    )
    
    assert result is None

