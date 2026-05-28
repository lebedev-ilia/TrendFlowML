"""
Request Models для DataProcessor API

Pydantic модели для входящих запросов к API.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1375, 1669-1692)
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, Any, Optional, Annotated
from pathlib import Path


class ProcessRequest(BaseModel):
    """
    Запрос на запуск обработки видео.

    Поддерживаются два способа указания видео (Phase 3: Backend ↔ Fetcher):
    - video_path — локальный путь (должен быть в allowed_video_paths);
    - video_url — URL для скачивания (DataProcessor скачивает в кэш и использует как video_path).
    Передавать ровно один из них. См. docs/PHASE3_ARTIFACTS_CONTRACT.md.

    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1055-1086, 1669-1692)
    """
    run_id: Annotated[
        str,
        Field(
            ...,
            description="UUID run'а (формат: 550e8400-e29b-41d4-a716-446655440000)",
            pattern=r"^[0-9a-f-]{36}$",
        ),
    ]
    video_id: str = Field(..., min_length=1, description="ID видео")
    platform_id: Annotated[
        str,
        Field(
            ...,
            pattern=r"^(youtube|upload)$",
            description="Платформа: 'youtube' или 'upload'",
        ),
    ]
    video_path: Optional[str] = Field(None, description="Путь к видео файлу (обязателен, если не передан video_url)")
    video_url: Optional[str] = Field(None, description="URL для скачивания видео (Phase 3: Fetcher signed URL)")
    config_hash: str = Field(..., description="Хеш конфигурации профиля")
    profile_config: Dict[str, Any] = Field(..., description="Конфигурация профиля обработки")
    
    # Поля версионирования профилей
    profile_version: Optional[str] = Field(None, description="Версия профиля (например, 'v1', 'v2')")
    feature_schema_version: Optional[str] = Field(None, description="Версия схемы фич (например, 'v1', 'v2')")
    pipeline_version: Optional[str] = Field(None, description="Версия pipeline (например, 'dev', 'prod')")
    
    # Опциональные поля для расширенной конфигурации
    sampling_policy_version: Optional[str] = Field(None, description="Версия sampling policy")
    dataprocessor_version: Optional[str] = Field(None, description="Версия DataProcessor")
    analysis_fps: Optional[float] = Field(None, description="FPS для анализа")
    analysis_width: Optional[int] = Field(None, description="Ширина для анализа")
    analysis_height: Optional[int] = Field(None, description="Высота для анализа")
    chunk_size: Optional[int] = Field(None, description="Размер чанка")
    visual_cfg_path: Optional[str] = Field(None, description="Путь к конфигурации visual процессора")
    dag_path: Optional[str] = Field(None, description="Путь к DAG файлу")
    dag_stage: Optional[str] = Field(None, description="Стадия DAG")
    rs_base: Optional[str] = Field(None, description="Базовый путь result_store")
    output: Optional[str] = Field(None, description="Путь к выходной директории")
    run_audio: Optional[bool] = Field(None, description="Запускать ли audio процессор")
    run_text: Optional[bool] = Field(None, description="Запускать ли text процессор")
    global_config_path: Optional[str] = Field(
        None,
        description="Путь к unified global_config YAML (--global-config в main.py): все процессоры и экстракторы",
    )
    
    @root_validator(skip_on_failure=True)
    def require_video_path_or_url(cls, values):
        """Требуется хотя бы один из video_path или video_url (Phase 3). Приоритет у video_url."""
        path = values.get("video_path")
        url = values.get("video_url")
        if url:
            return values
        if path and str(path).strip():
            return values
        raise ValueError("Either video_path or video_url must be provided")

    @validator("video_path")
    def validate_video_path(cls, v):
        """Валидация пути к видео файлу (пропуск, если передан video_url — путь заполнится после скачивания)."""
        if v is None or not str(v).strip():
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Video file not found: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return str(path.absolute())

    @validator("profile_config")
    def validate_profile_config(cls, v):
        """Валидация структуры профиля."""
        if not isinstance(v, dict):
            raise ValueError("profile_config must be a dictionary")
        if "processors" not in v:
            raise ValueError("profile_config must contain 'processors'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "video_id": "dQw4w9WgXcQ",
                "platform_id": "youtube",
                "video_path": "/data/videos/dQw4w9WgXcQ.mp4",
                "config_hash": "abc123def456",
                "profile_config": {
                    "processors": {
                        "segmenter": {"enabled": True, "required": True},
                        "audio": {"enabled": True, "required": False},
                        "text": {"enabled": False, "required": False},
                        "visual": {"enabled": True, "required": True}
                    }
                },
                "profile_version": "v1",
                "feature_schema_version": "v1",
                "pipeline_version": "prod",
                "sampling_policy_version": "v1",
                "dataprocessor_version": "dev"
            }
        }

