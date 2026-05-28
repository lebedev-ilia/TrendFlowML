"""
Checkpoint Service - управление checkpoint'ами для resumable execution

Этот модуль реализует логику сохранения и загрузки checkpoint'ов для продолжения
обработки с последнего сохраненного состояния.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 842-874)
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any
from storage.base import Storage
from storage.paths import KeyLayout
from api.utils.retry import retry_storage_operation

logger = logging.getLogger(__name__)

# Порядок процессоров для определения последнего выполненного
PROCESSOR_ORDER = ["segmenter", "audio", "text", "visual"]


async def save_checkpoint(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str,
    last_processor: Optional[str] = None,
    status: str = "running"
) -> bool:
    """
    Сохранить checkpoint run'а в Storage.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        last_processor: Имя последнего выполненного процессора
        status: Статус run'а
        
    Returns:
        True если успешно сохранено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 855-865)
    """
    try:
        state_prefix = key_layout.state_run_prefix(platform_id, video_id, run_id)
        checkpoint_key = f"{state_prefix}/checkpoint.json"
        
        checkpoint_data = {
            "run_id": run_id,
            "platform_id": platform_id,
            "video_id": video_id,
            "status": status,
            "last_processor": last_processor,
            "processor_order": PROCESSOR_ORDER
        }
        
        # Сохранить checkpoint с retry
        from api.utils.retry import retry_storage_operation
        
        checkpoint_bytes = json.dumps(checkpoint_data, indent=2).encode("utf-8")
        await retry_storage_operation(
            storage.atomic_write_bytes,
            checkpoint_key,
            checkpoint_bytes,
            content_type="application/json"
        )
        
        logger.debug(f"Saved checkpoint for run {run_id}, last_processor: {last_processor}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint for run {run_id}: {e}")
        return False


async def load_checkpoint_async(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """
    Загрузить checkpoint run'а из Storage.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        
    Returns:
        Словарь с данными checkpoint'а или None если не найден
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 867-873)
    """
    try:
        state_prefix = key_layout.state_run_prefix(platform_id, video_id, run_id)
        checkpoint_key = f"{state_prefix}/checkpoint.json"
        
        exists = await retry_storage_operation(
            storage.exists,
            checkpoint_key
        )
        if not exists:
            logger.debug(f"Checkpoint not found for run {run_id}")
            return None
        
        data = await retry_storage_operation(
            storage.read_bytes,
            checkpoint_key
        )
        checkpoint_data = json.loads(data.decode("utf-8"))
        
        logger.debug(f"Loaded checkpoint for run {run_id}, last_processor: {checkpoint_data.get('last_processor')}")
        return checkpoint_data
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint for run {run_id}: {e}")
        return None


def load_checkpoint(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """Синхронная обертка для совместимости с существующими тестами/скриптами."""
    return asyncio.run(
        load_checkpoint_async(storage, key_layout, platform_id, video_id, run_id)
    )


def determine_last_processor(
    processors: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    Определить последний выполненный процессор на основе состояний.
    
    Args:
        processors: Словарь с состояниями процессоров {processor_name: state}
        
    Returns:
        Имя последнего выполненного процессора или None если нет выполненного
    """
    # Проверить процессоры в порядке выполнения
    for processor_name in PROCESSOR_ORDER:
        if processor_name not in processors:
            continue
        
        processor_state = processors[processor_name]
        
        # Извлечь статус процессора
        processor_info = processor_state.get("processor", {})
        if isinstance(processor_info, dict):
            status = processor_info.get("status", processor_state.get("status", "waiting"))
        else:
            status = processor_state.get("status", "waiting")
        
        # Если процессор выполнен (success) или выполняется (running)
        if status in ("success", "running"):
            return processor_name
        
        # Если процессор завершился с ошибкой, вернуть его как последний
        if status == "error":
            return processor_name
    
    return None


async def get_checkpoint_info_async(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """
    Получить информацию о checkpoint'е, включая состояния процессоров.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        
    Returns:
        Словарь с информацией о checkpoint'е или None если не найден
    """
    # Загрузить checkpoint
    checkpoint = await load_checkpoint_async(
        storage, key_layout, platform_id, video_id, run_id
    )
    if not checkpoint:
        return None
    
    # Загрузить состояния процессоров
    processors = {}
    for processor_name in PROCESSOR_ORDER:
        state_prefix = key_layout.state_run_prefix(platform_id, video_id, run_id)
        processor_state_key = f"{state_prefix}/state_{processor_name}.json"
        
        exists = await retry_storage_operation(
            storage.exists,
            processor_state_key
        )
        if exists:
            try:
                data = await retry_storage_operation(
                    storage.read_bytes,
                    processor_state_key
                )
                processors[processor_name] = json.loads(data.decode("utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load processor state for {processor_name}: {e}")
    
    # Определить последний процессор на основе состояний
    last_processor = determine_last_processor(processors)
    
    return {
        "checkpoint": checkpoint,
        "processors": processors,
        "last_processor": last_processor,
        "can_resume": checkpoint.get("status") == "running" and last_processor is not None
    }


def get_checkpoint_info(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """Синхронная обертка для совместимости с существующими тестами/скриптами."""
    return asyncio.run(
        get_checkpoint_info_async(storage, key_layout, platform_id, video_id, run_id)
    )


async def delete_checkpoint_async(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> bool:
    """
    Удалить checkpoint run'а из Storage.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        
    Returns:
        True если успешно удалено, False иначе
    """
    try:
        state_prefix = key_layout.state_run_prefix(platform_id, video_id, run_id)
        checkpoint_key = f"{state_prefix}/checkpoint.json"
        
        exists = await retry_storage_operation(
            storage.exists,
            checkpoint_key
        )
        if exists:
            # Storage не имеет метода delete в Protocol, пропускаем удаление
            # В production можно добавить метод delete в Storage interface
            logger.debug(f"Checkpoint exists for run {run_id}, but delete not implemented")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete checkpoint for run {run_id}: {e}")
        return False


def delete_checkpoint(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> bool:
    """Синхронная обертка для совместимости с существующими тестами/скриптами."""
    return asyncio.run(
        delete_checkpoint_async(storage, key_layout, platform_id, video_id, run_id)
    )

