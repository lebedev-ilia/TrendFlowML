"""
FastAPI Dependencies для DataProcessor API

Этот модуль содержит зависимости FastAPI, которые используются в endpoints:
- Storage dependency
- StateReader dependency
- TaskManager dependency
- И другие общие зависимости

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1364)
"""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from api.config import config
from storage.base import Storage
from storage.fs import FileSystemStorage
from storage.s3 import S3Storage
from storage.paths import KeyLayout
from storage.settings import load_storage_settings


@lru_cache()
def get_storage_settings():
    """
    Получить настройки Storage из переменных окружения.
    """
    return load_storage_settings()


@lru_cache()
def get_storage() -> Storage:
    """
    Получить экземпляр Storage.
    
    В зависимости от TREND_STORAGE_BACKEND возвращает FileSystemStorage или S3Storage.
    """
    settings = get_storage_settings()
    
    if settings.backend == "s3":
        # Инициализация S3Storage
        from storage.s3 import S3Storage
        return S3Storage(
            endpoint_url=settings.s3_endpoint,
            bucket=settings.s3_bucket,
            region=settings.aws_region
        )
    else:
        # FileSystemStorage по умолчанию
        return FileSystemStorage(root_dir=settings.fs_root)


@lru_cache()
def get_key_layout() -> KeyLayout:
    """
    Получить экземпляр KeyLayout для работы с путями в Storage.
    """
    settings = get_storage_settings()
    prefix = settings.s3_prefix if settings.backend == "s3" else ""
    return KeyLayout(prefix=prefix)


# Типы для использования в endpoints
StorageDep = Annotated[Storage, Depends(get_storage)]
KeyLayoutDep = Annotated[KeyLayout, Depends(get_key_layout)]


def get_state_reader(storage: StorageDep, key_layout: KeyLayoutDep):
    """
    Получить экземпляр StateReader.
    
    StateReader автоматически получает Redis клиент для кэширования.
    """
    from api.services.state_reader import StateReader
    from api.services.task_manager import TaskManager
    from api.services.redis_client import get_redis_client
    # Получить TaskManager для доступа к метаданным активных run'ов
    task_manager = get_task_manager()
    # Получить Redis клиент для кэширования
    redis_client = get_redis_client()
    return StateReader(
        storage=storage,
        key_layout=key_layout,
        task_manager=task_manager,
        redis_client=redis_client
    )


@lru_cache()
def get_task_manager():
    """
    Получить singleton TaskManager.

    Важно: один экземпляр TaskManager на процесс, чтобы registry run'ов
    был общим для всех обработчиков внутри этого процесса.
    """
    from api.services.task_manager import TaskManager
    return TaskManager()


@lru_cache()
def get_processor_service():
    """
    Получить экземпляр ProcessorService (singleton через lru_cache).
    
    Использует lru_cache для создания единственного экземпляра ProcessorService
    на всё время работы приложения.
    
    Returns:
        Экземпляр ProcessorService
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1381)
    """
    from api.services.processor import ProcessorService
    return ProcessorService()


StateReaderDep = Annotated[object, Depends(get_state_reader)]
TaskManagerDep = Annotated[object, Depends(get_task_manager)]
ProcessorServiceDep = Annotated[object, Depends(get_processor_service)]

