"""
Контекст для изоляции данных каждого видео при batch processing.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import os
import json


@dataclass
class VideoContext:
    """
    Контекст для изоляции данных одного видео при batch processing.
    
    Обеспечивает изоляцию:
    - Кадров (frames_dir)
    - Артефактов (rs_path)
    - Метаданных (metadata.json)
    - Результатов (result_store paths)
    """
    # Идентификатор видео (уникальный для каждого видео)
    video_id: str
    
    # Путь к директории с кадрами (frames_dir)
    frames_dir: str
    
    # Путь к result_store для этого видео
    rs_path: str
    
    # Путь к metadata.json для этого видео
    metadata_path: Optional[str] = None
    
    # Загруженные метаданные (если уже загружены)
    metadata: Optional[Dict[str, Any]] = None
    
    # Дополнительные метаданные (platform_id, run_id, config_hash, etc.)
    platform_id: Optional[str] = None
    run_id: Optional[str] = None
    config_hash: Optional[str] = None
    sampling_policy_version: Optional[str] = None
    dataprocessor_version: Optional[str] = None
    
    def __post_init__(self):
        """Создание директорий при инициализации."""
        # Создаем директории, если они не существуют
        Path(self.rs_path).mkdir(parents=True, exist_ok=True)
        
        # Автоматически определяем metadata_path если не задан
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.frames_dir, "metadata.json")
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Загружает метаданные из metadata.json.
        
        Returns:
            Словарь с метаданными
            
        Raises:
            FileNotFoundError: Если metadata.json не найден
            ValueError: Если метаданные некорректны
        """
        if self.metadata is not None:
            return self.metadata
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"VideoContext | metadata.json не найден: {self.metadata_path}"
            )
        
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            # Извлекаем дополнительные метаданные если они есть
            if self.platform_id is None:
                self.platform_id = self.metadata.get("platform_id")
            if self.run_id is None:
                self.run_id = self.metadata.get("run_id")
            if self.config_hash is None:
                self.config_hash = self.metadata.get("config_hash")
            if self.sampling_policy_version is None:
                self.sampling_policy_version = self.metadata.get("sampling_policy_version")
            if self.dataprocessor_version is None:
                self.dataprocessor_version = self.metadata.get("dataprocessor_version")
            
            return self.metadata
        except json.JSONDecodeError as e:
            raise ValueError(
                f"VideoContext | Ошибка парсинга metadata.json: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"VideoContext | Ошибка загрузки metadata.json: {e}"
            ) from e
    
    def get_component_rs_path(self, component_name: str) -> str:
        """
        Возвращает путь к директории результатов для конкретного компонента.
        
        Args:
            component_name: Имя компонента (например, "core_clip", "shot_quality")
            
        Returns:
            Путь к директории результатов компонента
        """
        component_path = os.path.join(self.rs_path, component_name)
        Path(component_path).mkdir(parents=True, exist_ok=True)
        return component_path
    
    def __repr__(self) -> str:
        return (
            f"VideoContext(video_id={self.video_id}, "
            f"frames_dir={self.frames_dir}, rs_path={self.rs_path})"
        )

