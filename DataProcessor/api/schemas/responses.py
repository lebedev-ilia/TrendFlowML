"""
Response Models для DataProcessor API

Pydantic модели для ответов API.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1376, 1089-1350)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class ProcessResponse(BaseModel):
    """
    Ответ на запрос запуска обработки.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1089-1098)
    """
    run_id: str = Field(..., description="UUID run'а")
    status: str = Field(..., description="Статус: 'queued', 'running', 'success', 'error'")
    message: str = Field(..., description="Сообщение о статусе")
    status_url: str = Field(..., description="URL для получения статуса")
    estimated_duration_seconds: Optional[int] = Field(None, description="Оценка длительности в секундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Processing started",
                "status_url": "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status",
                "estimated_duration_seconds": 300
            }
        }


class RunMetadataResponse(BaseModel):
    """
    Метаданные run'а.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1109-1122)
    """
    run_id: str = Field(..., description="UUID run'а")
    video_id: str = Field(..., description="ID видео")
    platform_id: str = Field(..., description="Платформа")
    config_hash: str = Field(..., description="Хеш конфигурации")
    status: str = Field(..., description="Статус run'а")
    created_at: Optional[datetime] = Field(None, description="Время создания")
    started_at: Optional[datetime] = Field(None, description="Время начала обработки")
    updated_at: Optional[datetime] = Field(None, description="Время последнего обновления")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")


class ComponentProgress(BaseModel):
    """Прогресс компонента."""
    status: str = Field(..., description="Статус: 'waiting', 'running', 'success', 'error', 'skipped'")
    progress: float = Field(..., ge=0.0, le=1.0, description="Прогресс от 0.0 до 1.0")
    started_at: Optional[datetime] = Field(None, description="Время начала")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")
    duration_ms: Optional[int] = Field(None, description="Длительность в миллисекундах")
    error: Optional[str] = Field(None, description="Сообщение об ошибке процессора (если status=error)")
    error_code: Optional[str] = Field(None, description="Код ошибки процессора")
    current_component: Optional[str] = Field(None, description="Текущий подкомпонент")
    components: Optional[Dict[str, Any]] = Field(None, description="Детали подкомпонентов")
    done: Optional[int] = Field(None, description="Количество обработанных элементов")
    total: Optional[int] = Field(None, description="Общее количество элементов")


class ProgressInfo(BaseModel):
    """Информация о прогрессе обработки."""
    overall: float = Field(..., ge=0.0, le=1.0, description="Общий прогресс")
    current_processor: Optional[str] = Field(None, description="Текущий процессор")
    current_component: Optional[str] = Field(None, description="Текущий компонент")
    components: Dict[str, ComponentProgress] = Field(default_factory=dict, description="Прогресс компонентов")


class RunStatusResponse(BaseModel):
    """
    Детальный статус обработки.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1131-1184)
    """
    run_id: str = Field(..., description="UUID run'а")
    video_id: str = Field(..., description="ID видео")
    platform_id: str = Field(..., description="Платформа")
    status: str = Field(..., description="Статус: 'pending', 'queued', 'running', 'success', 'error', 'cancelled'")
    stage: Optional[str] = Field(None, description="Текущая стадия: 'segmenter', 'audio', 'text', 'visual'")
    progress: ProgressInfo = Field(..., description="Информация о прогрессе")
    started_at: Optional[datetime] = Field(None, description="Время начала обработки")
    updated_at: Optional[datetime] = Field(..., description="Время последнего обновления")
    estimated_finish: Optional[datetime] = Field(None, description="Оценка времени завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    error_code: Optional[str] = Field(None, description="Код ошибки")


class HealthResponse(BaseModel):
    """
    Ответ health check endpoint.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1292-1313)
    """
    status: str = Field(..., description="Общий статус: 'healthy', 'degraded', 'unhealthy'")
    api: str = Field(..., description="Статус API: 'healthy', 'unhealthy'")
    storage: str = Field(..., description="Статус Storage: 'healthy', 'unhealthy', 'unknown'")
    version: str = Field(..., description="Версия API")
    uptime_seconds: float = Field(..., description="Время работы сервиса в секундах")
    dependencies: Dict[str, Any] = Field(default_factory=dict, description="Статус зависимостей")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Метрики сервиса")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Время проверки")


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке."""
    error: str = Field(..., description="Описание ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")
    run_id: Optional[str] = Field(None, description="UUID run'а (если применимо)")


class ManifestArtifact(BaseModel):
    """Артефакт компонента в manifest."""
    path: str = Field(..., description="Путь к артефакту")
    size_bytes: Optional[int] = Field(None, description="Размер артефакта в байтах")
    schema_version: Optional[str] = Field(None, description="Версия схемы артефакта")


class ManifestComponent(BaseModel):
    """Компонент в manifest."""
    name: str = Field(..., description="Имя компонента")
    kind: Optional[str] = Field(None, description="Тип компонента: 'core', 'module', 'other'")
    status: str = Field(..., description="Статус: 'ok', 'empty', 'error'")
    empty_reason: Optional[str] = Field(None, description="Причина пустого результата")
    started_at: Optional[str] = Field(None, description="Время начала обработки (ISO 8601)")
    finished_at: Optional[str] = Field(None, description="Время завершения обработки (ISO 8601)")
    duration_ms: Optional[int] = Field(None, description="Длительность обработки в миллисекундах")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Список артефактов")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    error_code: Optional[str] = Field(None, description="Код ошибки")
    notes: Optional[str] = Field(None, description="Дополнительные заметки")
    warnings: Optional[List[str]] = Field(None, description="Предупреждения")
    producer_version: Optional[str] = Field(None, description="Версия производителя")
    schema_version: Optional[str] = Field(None, description="Версия схемы")
    device_used: Optional[str] = Field(None, description="Использованное устройство")


class ManifestResponse(BaseModel):
    """
    Ответ с manifest.json run'а.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1226-1253)
    """
    schema_version: Optional[str] = Field(None, description="Версия схемы manifest")
    run_id: str = Field(..., description="UUID run'а")
    video_id: str = Field(..., description="ID видео")
    platform_id: str = Field(..., description="Платформа")
    config_hash: Optional[str] = Field(None, description="Хеш конфигурации")
    sampling_policy_version: Optional[str] = Field(None, description="Версия sampling policy")
    dataprocessor_version: Optional[str] = Field(None, description="Версия DataProcessor")
    created_at: Optional[str] = Field(None, description="Время создания (ISO 8601)")
    finished_at: Optional[str] = Field(None, description="Время завершения (ISO 8601)")
    updated_at: Optional[str] = Field(None, description="Время последнего обновления (ISO 8601)")
    components: Dict[str, ManifestComponent] = Field(default_factory=dict, description="Компоненты обработки")


class CancelResponse(BaseModel):
    """Ответ на запрос отмены run'а."""
    run_id: str = Field(..., description="UUID run'а")
    status: str = Field(..., description="Новый статус run'а")
    message: str = Field(..., description="Сообщение о результате отмены")

