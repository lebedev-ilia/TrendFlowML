"""
Валидаторы для DataProcessor API

Дополнительная валидация payload помимо Pydantic моделей.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1388, 1663-1692, 2768-2774)
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from api.utils.errors import InvalidPayloadError
from api.config import config


def validate_video_path(video_path: str) -> None:
    """
    Валидация пути к видео файлу.
    
    Проверяет:
    - Файл существует
    - Это файл (не директория)
    - Путь находится в разрешённых директориях
    - Размер файла не превышает лимит (опционально)
    
    Args:
        video_path: Путь к видео файлу
        
    Raises:
        InvalidPayloadError: Если валидация не прошла
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2768-2774)
    """
    path = Path(video_path)
    
    if not path.exists():
        raise InvalidPayloadError(
            f"Video file not found: {video_path}",
            details={"field": "video_path", "value": video_path}
        )
    
    if not path.is_file():
        raise InvalidPayloadError(
            f"Path is not a file: {video_path}",
            details={"field": "video_path", "value": video_path}
        )
    
    # Проверка, что путь находится в разрешённых директориях
    allowed_paths = _get_allowed_video_paths()
    if allowed_paths:
        path_resolved = path.resolve()
        is_allowed = False
        
        for allowed_path_str in allowed_paths:
            allowed_path = Path(allowed_path_str).resolve()
            try:
                # Проверка, что путь является подпутем разрешённой директории
                if path_resolved.is_relative_to(allowed_path):
                    is_allowed = True
                    break
            except (ValueError, AttributeError):
                # Python < 3.9 не поддерживает is_relative_to, используем альтернативный метод
                try:
                    path_resolved.relative_to(allowed_path)
                    is_allowed = True
                    break
                except ValueError:
                    continue
        
        if not is_allowed:
            raise InvalidPayloadError(
                f"Video path outside allowed directories: {video_path}",
                details={
                    "field": "video_path",
                    "value": video_path,
                    "allowed_paths": allowed_paths
                }
            )
    
    # Проверка размера файла (опционально, можно добавить лимит)
    from api.config import config
    file_size = path.stat().st_size
    max_size = config.max_video_size_bytes
    
    if file_size > max_size:
        raise InvalidPayloadError(
            f"Video file too large: {file_size} bytes (max: {max_size})",
            details={
                "field": "video_path",
                "value": video_path,
                "file_size": file_size,
                "max_size": max_size
            }
        )


def _get_allowed_video_paths() -> List[str]:
    """
    Получить список разрешённых директорий для video_path.
    
    Returns:
        Список разрешённых путей (может быть пустым, если не настроено)
    """
    # Использовать свойство allowed_video_paths_list из config
    paths = config.allowed_video_paths_list
    
    # Проверить, что пути существуют
    valid_paths = []
    for path_str in paths:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            valid_paths.append(str(path.resolve()))
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Allowed video path does not exist or is not a directory: {path_str}")
    
    return valid_paths


def validate_profile_config(profile_config: Dict[str, Any]) -> None:
    """
    Валидация конфигурации профиля.
    
    Проверяет:
    - profile_config является словарём
    - Содержит ключ 'processors'
    - Структура processors валидна
    
    Args:
        profile_config: Конфигурация профиля
        
    Raises:
        InvalidPayloadError: Если валидация не прошла
    """
    if not isinstance(profile_config, dict):
        raise InvalidPayloadError(
            "profile_config must be a dictionary",
            details={"field": "profile_config", "value": type(profile_config).__name__}
        )
    
    if "processors" not in profile_config:
        raise InvalidPayloadError(
            "profile_config must contain 'processors'",
            details={"field": "profile_config", "missing_key": "processors"}
        )
    
    processors = profile_config.get("processors", {})
    if not isinstance(processors, dict):
        raise InvalidPayloadError(
            "profile_config.processors must be a dictionary",
            details={"field": "profile_config.processors", "value": type(processors).__name__}
        )
    
    # Валидация структуры каждого процессора
    valid_processors = ["segmenter", "audio", "text", "visual"]
    for proc_name, proc_config in processors.items():
        if proc_name not in valid_processors:
            raise InvalidPayloadError(
                f"Unknown processor: {proc_name}",
                details={
                    "field": f"profile_config.processors.{proc_name}",
                    "value": proc_name,
                    "valid_processors": valid_processors
                }
            )
        
        if not isinstance(proc_config, dict):
            raise InvalidPayloadError(
                f"Processor config for {proc_name} must be a dictionary",
                details={
                    "field": f"profile_config.processors.{proc_name}",
                    "value": type(proc_config).__name__
                }
            )


def validate_run_id(run_id: str) -> None:
    """
    Валидация UUID run_id.
    
    Проверяет формат UUID (36 символов с дефисами).
    
    Args:
        run_id: UUID run'а
        
    Raises:
        InvalidPayloadError: Если валидация не прошла
    """
    import re
    
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    
    if not re.match(uuid_pattern, run_id.lower()):
        raise InvalidPayloadError(
            f"Invalid run_id format: {run_id}",
            details={
                "field": "run_id",
                "value": run_id,
                "expected_format": "UUID (36 characters with dashes)"
            }
        )


def validate_platform_id(platform_id: str) -> None:
    """
    Валидация platform_id.
    
    Проверяет что platform_id один из допустимых значений.
    
    Args:
        platform_id: ID платформы
        
    Raises:
        InvalidPayloadError: Если валидация не прошла
    """
    valid_platforms = ["youtube", "upload"]
    
    if platform_id not in valid_platforms:
        raise InvalidPayloadError(
            f"Invalid platform_id: {platform_id}",
            details={
                "field": "platform_id",
                "value": platform_id,
                "valid_platforms": valid_platforms
            }
        )

