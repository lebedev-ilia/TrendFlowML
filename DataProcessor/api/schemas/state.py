"""
State Models для DataProcessor API

Pydantic модели для работы с состоянием обработки.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1377)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class RunStatus(str, Enum):
    """
    Статусы выполнения run'а.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 912-919)
    """
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    RECOVERING = "recovering"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class ComponentStatus(str, Enum):
    """
    Статусы компонента (из state/enums.py).
    """
    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"
    SKIPPED = "skipped"


class ProcessorState(BaseModel):
    """Состояние процессора."""
    name: str = Field(..., description="Имя процессора")
    status: ComponentStatus = Field(..., description="Статус процессора")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Прогресс от 0.0 до 1.0")
    started_at: Optional[datetime] = Field(None, description="Время начала")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    components: Dict[str, Any] = Field(default_factory=dict, description="Состояния подкомпонентов")


class RunState(BaseModel):
    """Полное состояние run'а."""
    run_id: str = Field(..., description="UUID run'а")
    video_id: str = Field(..., description="ID видео")
    platform_id: str = Field(..., description="Платформа")
    status: RunStatus = Field(..., description="Статус run'а")
    stage: Optional[str] = Field(None, description="Текущая стадия")
    processors: Dict[str, ProcessorState] = Field(default_factory=dict, description="Состояния процессоров")
    created_at: Optional[datetime] = Field(None, description="Время создания")
    started_at: Optional[datetime] = Field(None, description="Время начала")
    updated_at: Optional[datetime] = Field(None, description="Время последнего обновления")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    error_code: Optional[str] = Field(None, description="Код ошибки")

