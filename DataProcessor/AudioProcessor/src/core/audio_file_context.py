"""
Контекст для изоляции данных каждого аудио файла при batch processing.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class AudioFileContext:
    """
    Контекст для изоляции данных одного аудио файла при batch processing.
    
    Обеспечивает изоляцию:
    - Временных файлов (tmp_path)
    - Артефактов (artifacts_dir)
    - Сегментов (segments.json)
    - Результатов (result_store paths)
    """
    # Идентификатор файла (уникальный для каждого видео)
    file_id: str
    
    # Путь к входному аудио файлу (audio/audio.wav из Segmenter output)
    input_uri: str
    
    # Путь к временной директории для этого файла
    tmp_path: str
    
    # Путь к директории артефактов для этого файла
    artifacts_dir: str
    
    # Путь к segments.json для этого файла
    segments_json_path: Optional[str] = None
    
    # Загруженные сегменты (если уже загружены)
    segments: Optional[List[Dict[str, Any]]] = None
    
    # Families из segments.json (если уже загружены)
    families: Optional[Dict[str, Any]] = None
    
    # Дополнительные метаданные
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Создание директорий при инициализации."""
        # Создаем директории, если они не существуют
        Path(self.tmp_path).mkdir(parents=True, exist_ok=True)
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
    
    def get_segment_family(self, family_name: str) -> List[Dict[str, Any]]:
        """
        Получить сегменты для указанного family из segments.json.
        
        Args:
            family_name: Имя family (например, 'primary', 'clap', 'tempo', 'asr', 'diarization', 'emotion')
        
        Returns:
            Список сегментов для указанного family
        """
        if not self.families:
            return []
        
        return self.families.get(family_name, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать контекст в словарь для передачи в extractors."""
        return {
            "file_id": self.file_id,
            "input_uri": self.input_uri,
            "tmp_path": self.tmp_path,
            "artifacts_dir": self.artifacts_dir,
            "segments_json_path": self.segments_json_path,
            "segments": self.segments,
            "families": self.families,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioFileContext":
        """Создать контекст из словаря."""
        return cls(
            file_id=data["file_id"],
            input_uri=data["input_uri"],
            tmp_path=data["tmp_path"],
            artifacts_dir=data["artifacts_dir"],
            segments_json_path=data.get("segments_json_path"),
            segments=data.get("segments"),
            families=data.get("families"),
            metadata=data.get("metadata"),
        )

