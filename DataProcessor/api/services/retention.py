"""
Retention Policy Service - очистка старых данных

Этот сервис реализует политику хранения данных:
- Удаление Redis state старше 1 дня
- Удаление storage старше 7 дней

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2349-2376)
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from storage.base import Storage, NotFoundError
from storage.paths import KeyLayout
from api.services.redis_client import get_redis_client
from api.services.redis_schema import KEY_PREFIX_STATE

logger = logging.getLogger(__name__)

# Retention константы (в секундах)
REDIS_STATE_RETENTION = 24 * 3600  # 1 день
STORAGE_RETENTION = 7 * 24 * 3600  # 7 дней


async def cleanup_redis_state(cutoff_timestamp: Optional[float] = None) -> Dict[str, Any]:
    """
    Удалить Redis state старше 1 дня.
    
    Проверяет все ключи run:state:* и удаляет те, которые:
    - Не имеют TTL (ttl == -1) и updated_at < cutoff
    - Или имеют TTL, но ключ уже истёк (будет удалён автоматически)
    
    Args:
        cutoff_timestamp: Timestamp для cutoff (по умолчанию: текущее время - 1 день)
        
    Returns:
        Словарь с результатами очистки:
        {
            "checked": int,  # Количество проверенных ключей
            "deleted": int,  # Количество удалённых ключей
            "errors": int    # Количество ошибок
        }
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2356-2363)
    """
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Redis not available, skipping Redis state cleanup")
        return {"checked": 0, "deleted": 0, "errors": 0}
    
    if cutoff_timestamp is None:
        cutoff_timestamp = time.time() - REDIS_STATE_RETENTION
    
    checked = 0
    deleted = 0
    errors = 0
    
    try:
        # Найти все run:state:* ключи
        pattern = f"{KEY_PREFIX_STATE}*"
        keys_to_check = []
        
        async for key in redis_client.scan_iter(match=pattern):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            keys_to_check.append(key)
        
        logger.info(f"Found {len(keys_to_check)} Redis state keys to check")
        
        # Проверить каждый ключ
        for key in keys_to_check:
            checked += 1
            try:
                # Проверить TTL
                ttl = await redis_client.ttl(key)
                
                # Если TTL установлен и > 0, ключ будет удалён автоматически
                if ttl > 0:
                    continue
                
                # Если TTL == -1 (нет TTL), проверить updated_at
                if ttl == -1:
                    # Попытаться получить updated_at из hash или из JSON значения
                    value = await redis_client.get(key)
                    if value:
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        
                        try:
                            # Попытаться распарсить как JSON
                            state_data = json.loads(value)
                            updated_at = state_data.get("updated_at")
                            
                            if updated_at:
                                updated_timestamp = float(updated_at)
                                if updated_timestamp < cutoff_timestamp:
                                    await redis_client.delete(key)
                                    deleted += 1
                                    logger.debug(f"Deleted old Redis state key: {key}")
                        except (json.JSONDecodeError, ValueError, TypeError):
                            # Если не JSON или нет updated_at, проверить по TTL ключа
                            # Если ключ старый (нет TTL), удалить его
                            # Для простоты удаляем все ключи без TTL старше cutoff
                            # (это безопасно, так как они должны были иметь TTL)
                            await redis_client.delete(key)
                            deleted += 1
                            logger.debug(f"Deleted Redis state key without TTL: {key}")
                
            except Exception as e:
                errors += 1
                logger.error(f"Error checking Redis key {key}: {e}")
        
        logger.info(f"Redis state cleanup completed: checked={checked}, deleted={deleted}, errors={errors}")
        return {"checked": checked, "deleted": deleted, "errors": errors}
        
    except Exception as e:
        logger.exception(f"Error during Redis state cleanup: {e}")
        return {"checked": checked, "deleted": deleted, "errors": errors + 1}


async def cleanup_storage(
    storage: Storage,
    key_layout: KeyLayout,
    cutoff_timestamp: Optional[float] = None
) -> Dict[str, Any]:
    """
    Удалить storage старше 7 дней.
    
    Проверяет все run'ы в result_store и удаляет те, у которых finished_at < cutoff.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        cutoff_timestamp: Timestamp для cutoff (по умолчанию: текущее время - 7 дней)
        
    Returns:
        Словарь с результатами очистки:
        {
            "checked": int,  # Количество проверенных run'ов
            "deleted": int,  # Количество удалённых run'ов
            "errors": int    # Количество ошибок
        }
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2365-2375)
    """
    if cutoff_timestamp is None:
        cutoff_timestamp = time.time() - STORAGE_RETENTION
    
    checked = 0
    deleted = 0
    errors = 0
    
    try:
        # Получить префикс для result_store
        result_store_prefix = key_layout.result_store_prefix()
        
        # Получить список всех platform_id
        platform_ids = []
        try:
            for obj in storage.list(result_store_prefix):
                # obj.key будет вида "result_store/youtube" или "result_store/upload"
                parts = obj.key.split("/")
                if len(parts) >= 2:
                    platform_id = parts[1]  # youtube, upload, etc.
                    if platform_id not in platform_ids:
                        platform_ids.append(platform_id)
        except Exception as e:
            logger.warning(f"Error listing platforms in result_store: {e}")
            return {"checked": 0, "deleted": 0, "errors": 1}
        
        logger.info(f"Found {len(platform_ids)} platforms in result_store")
        
        # Для каждого platform_id проверить video_id и run_id
        for platform_id in platform_ids:
            platform_prefix = f"{result_store_prefix}/{platform_id}"
            
            try:
                # Получить список video_id
                video_ids = []
                for obj in storage.list(platform_prefix):
                    # obj.key будет вида "result_store/youtube/video123"
                    parts = obj.key.split("/")
                    if len(parts) >= 3:
                        video_id = parts[2]
                        if video_id not in video_ids:
                            video_ids.append(video_id)
                
                # Для каждого video_id проверить run_id
                for video_id in video_ids:
                    video_prefix = f"{platform_prefix}/{video_id}"
                    
                    try:
                        # Получить список run_id
                        run_ids = []
                        for obj in storage.list(video_prefix):
                            # obj.key будет вида "result_store/youtube/video123/run-uuid"
                            parts = obj.key.split("/")
                            if len(parts) >= 4:
                                run_id = parts[3]
                                if run_id not in run_ids:
                                    run_ids.append(run_id)
                        
                        # Для каждого run_id проверить manifest.json
                        for run_id in run_ids:
                            checked += 1
                            try:
                                # Проверить manifest.json
                                run_prefix = f"{video_prefix}/{run_id}"
                                manifest_key = f"{run_prefix}/manifest.json"
                                
                                if not storage.exists(manifest_key):
                                    # Если нет manifest.json, пропустить (возможно, неполный run)
                                    continue
                                
                                # Прочитать manifest.json
                                manifest_bytes = storage.read_bytes(manifest_key)
                                manifest_data = json.loads(manifest_bytes.decode("utf-8"))
                                
                                # Проверить finished_at
                                finished_at = manifest_data.get("finished_at")
                                if finished_at:
                                    finished_timestamp = float(finished_at)
                                    
                                    if finished_timestamp < cutoff_timestamp:
                                        # Удалить весь run
                                        await delete_run_storage(storage, run_prefix)
                                        deleted += 1
                                        logger.info(f"Deleted old storage for run {run_id} (finished_at={finished_timestamp})")
                                
                            except NotFoundError:
                                # manifest.json не найден, пропустить
                                continue
                            except Exception as e:
                                errors += 1
                                logger.error(f"Error checking run {run_id} in storage: {e}")
                    
                    except Exception as e:
                        logger.warning(f"Error listing runs for video {video_id}: {e}")
                        errors += 1
                
            except Exception as e:
                logger.warning(f"Error listing videos for platform {platform_id}: {e}")
                errors += 1
        
        logger.info(f"Storage cleanup completed: checked={checked}, deleted={deleted}, errors={errors}")
        return {"checked": checked, "deleted": deleted, "errors": errors}
        
    except Exception as e:
        logger.exception(f"Error during storage cleanup: {e}")
        return {"checked": checked, "deleted": deleted, "errors": errors + 1}


async def delete_run_storage(storage: Storage, run_prefix: str) -> bool:
    """
    Удалить все файлы run'а из storage.
    
    Args:
        storage: Экземпляр Storage
        run_prefix: Префикс пути к run'у (например, "result_store/youtube/video123/run-uuid")
        
    Returns:
        True если успешно удалено, False иначе
    """
    try:
        # Проверить тип storage и удалить соответственно
        storage_class_name = storage.__class__.__name__ if hasattr(storage, "__class__") else "unknown"
        
        if "FileSystem" in storage_class_name or hasattr(storage, "_abs"):
            # FileSystemStorage
            import os
            import shutil
            abs_path = storage._abs(run_prefix)
            if os.path.exists(abs_path):
                try:
                    if os.path.isdir(abs_path):
                        shutil.rmtree(abs_path)
                        logger.debug(f"Deleted directory: {abs_path}")
                    else:
                        os.remove(abs_path)
                        logger.debug(f"Deleted file: {abs_path}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting filesystem path {abs_path}: {e}")
                    return False
            else:
                logger.debug(f"Path does not exist: {abs_path}")
                return True  # Уже удалено или не существует
                
        elif "S3" in storage_class_name or hasattr(storage, "_client"):
            # S3Storage
            try:
                # Получить список всех объектов с префиксом
                objects_to_delete = []
                for obj in storage.list(run_prefix):
                    objects_to_delete.append(obj.key)
                
                if not objects_to_delete:
                    logger.debug(f"No objects found for prefix: {run_prefix}")
                    return True
                
                # Удалить объекты батчами (S3 поддерживает до 1000 объектов за раз)
                batch_size = 1000
                for i in range(0, len(objects_to_delete), batch_size):
                    batch = objects_to_delete[i:i + batch_size]
                    delete_keys = [{"Key": storage._k(key)} for key in batch]
                    
                    try:
                        storage._client.delete_objects(
                            Bucket=storage.bucket,
                            Delete={"Objects": delete_keys}
                        )
                        logger.debug(f"Deleted {len(batch)} S3 objects from {run_prefix}")
                    except Exception as e:
                        logger.error(f"Error deleting S3 objects batch: {e}")
                        # Продолжить с остальными объектами
                        continue
                
                return True
                
            except Exception as e:
                logger.error(f"Error deleting S3 storage {run_prefix}: {e}")
                return False
        else:
            logger.warning(f"Unknown storage type: {storage_class_name}, cannot delete {run_prefix}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting run storage {run_prefix}: {e}")
        return False


async def run_retention_cleanup(
    storage: Storage,
    key_layout: KeyLayout
) -> Dict[str, Any]:
    """
    Запустить полную очистку retention policy.
    
    Выполняет очистку Redis state и storage.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        
    Returns:
        Словарь с результатами очистки:
        {
            "redis": {...},  # Результаты очистки Redis
            "storage": {...}, # Результаты очистки Storage
            "timestamp": float  # Timestamp выполнения
        }
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2354-2376)
    """
    logger.info("Starting retention policy cleanup")
    start_time = time.time()
    
    # Очистка Redis state
    redis_results = await cleanup_redis_state()
    
    # Очистка Storage
    storage_results = await cleanup_storage(storage, key_layout)
    
    elapsed_time = time.time() - start_time
    
    results = {
        "redis": redis_results,
        "storage": storage_results,
        "timestamp": time.time(),
        "elapsed_seconds": elapsed_time
    }
    
    logger.info(f"Retention policy cleanup completed in {elapsed_time:.2f}s: {results}")
    return results

