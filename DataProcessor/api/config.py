"""
Настройки API сервера DataProcessor

Этот модуль содержит конфигурацию для API сервера, включая:
- Настройки подключения к Storage
- Настройки Redis (для будущего использования)
- Лимиты параллелизма
- Пути и директории

Все параметры читаются из переменных окружения через pydantic-settings.
Значения по умолчанию указаны в полях класса.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1363)
"""

from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """Конфигурация API сервера DataProcessor."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Игнорировать дополнительные переменные окружения
    )
    
    # Основные настройки
    api_host: str = Field(default="0.0.0.0", description="Хост API сервера")
    api_port: int = Field(default=8000, ge=1, le=65535, description="Порт API сервера")
    api_workers: int = Field(default=1, ge=1, description="Количество worker процессов")
    api_version: str = Field(default="0.1.0", description="Версия API")
    debug: bool = Field(default=False, description="Режим отладки")
    
    # Лимиты параллелизма (для MVP)
    max_concurrent_runs: int = Field(
        default=4,
        ge=1,
        description="Максимальное количество одновременных run'ов"
    )
    
    # Лимит длины очереди (backpressure)
    # Если очередь превышает этот лимит, API возвращает 503 Service Unavailable
    max_queue_length: int = Field(
        default=100,
        ge=1,
        description="Максимальная длина очереди для backpressure"
    )
    
    # Memory limits для subprocess (в MB, опционально)
    # Если установлен, subprocess будет убит при превышении лимита
    subprocess_memory_limit_mb: Optional[int] = Field(
        default=None,
        ge=1,
        description="Лимит памяти для subprocess в MB"
    )
    
    # SSE (Server-Sent Events) настройки
    # Максимальное количество одновременных SSE соединений на run_id
    max_sse_connections_per_run: int = Field(
        default=10,
        ge=1,
        description="Максимальное количество SSE соединений на run_id"
    )
    # Таймаут для чтения новых событий из Redis Streams (мс)
    sse_stream_read_timeout: int = Field(
        default=5000,
        ge=1000,
        description="Таймаут чтения событий из Redis Streams в миллисекундах"
    )
    
    # Storage настройки
    storage_type: str = Field(
        default="fs",
        description="Тип storage: 'fs' или 's3'"
    )
    storage_root: Optional[str] = Field(
        default=None,
        description="Корневая директория для файловой системы storage"
    )
    
    # Redis настройки (для Этапа 2)
    redis_host: Optional[str] = Field(default=None, description="Хост Redis")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Порт Redis")
    redis_db: int = Field(default=0, ge=0, description="Номер базы данных Redis")
    redis_password: Optional[str] = Field(default=None, description="Пароль Redis")
    redis_url: Optional[str] = Field(
        default=None,
        description="URL подключения к Redis (приоритет над отдельными параметрами)"
    )
    
    # Логирование
    log_level: str = Field(
        default="INFO",
        description="Уровень логирования: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_format: str = Field(
        default="json",
        description="Формат логов: 'json' или 'text'"
    )
    
    # CORS настройки
    cors_origins: str = Field(
        default="*",
        description="Разрешённые origins для CORS (разделённые запятой)"
    )
    
    # Triton настройки (опционально)
    triton_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint для Triton Inference Server"
    )
    
    # Аутентификация
    api_key: Optional[str] = Field(
        default=None,
        description="API ключ для аутентификации"
    )
    auth_type: str = Field(
        default="api_key",
        description="Тип аутентификации: 'api_key' или 'mtls'"
    )
    
    # Security настройки
    # Разрешённые директории для video_path (разделённые запятой)
    allowed_video_paths: str = Field(
        default="/data/videos,/data/uploads",
        description="Разрешённые директории для video_path (разделённые запятой)"
    )
    # Включить audit log
    audit_log_enabled: bool = Field(
        default=True,
        description="Включить audit log"
    )
    # TTL для audit log записей в Redis (в секундах, по умолчанию 30 дней)
    audit_log_ttl: int = Field(
        default=30 * 24 * 3600,
        ge=0,
        description="TTL для audit log записей в Redis в секундах"
    )
    
    # OpenTelemetry tracing настройки (опционально)
    enable_tracing: bool = Field(
        default=False,
        description="Включить distributed tracing"
    )
    tracing_exporter: str = Field(
        default="jaeger",
        description="Экспортер трейсов: 'jaeger' или 'otlp'"
    )
    jaeger_agent_host: str = Field(
        default="localhost",
        description="Хост Jaeger agent"
    )
    jaeger_agent_port: int = Field(
        default=6831,
        ge=1,
        le=65535,
        description="Порт Jaeger agent"
    )
    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP endpoint для экспорта трейсов"
    )
    service_name: str = Field(
        default="dataprocessor-api",
        description="Имя сервиса для трейсинга"
    )
    service_version: str = Field(
        default="0.1.0",
        description="Версия сервиса для трейсинга"
    )
    
    # Дополнительные параметры
    worker_id: Optional[str] = Field(
        default=None,
        description="Уникальный ID worker'а (если не указан, генерируется автоматически)"
    )
    max_video_size_bytes: int = Field(
        default=10737418240,  # 10GB
        ge=0,
        description="Максимальный размер видео файла в байтах"
    )

    # Кэш для видео, скачанных по URL (Phase 3: Backend ↔ Fetcher)
    # См. docs/PHASE3_ARTIFACTS_CONTRACT.md
    video_url_cache_dir: Optional[str] = Field(
        default=None,
        description="Директория для кэша видео по video_url. Если None — используется {первая из allowed_video_paths}/_url_cache",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Валидация уровня логирования."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Валидация формата логов."""
        if v.lower() not in ["json", "text"]:
            raise ValueError("log_format must be 'json' or 'text'")
        return v.lower()
    
    @field_validator("storage_type")
    @classmethod
    def validate_storage_type(cls, v: str) -> str:
        """Валидация типа storage."""
        if v.lower() not in ["fs", "s3"]:
            raise ValueError("storage_type must be 'fs' or 's3'")
        return v.lower()
    
    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, v: str) -> str:
        """Валидация типа аутентификации."""
        if v.lower() not in ["api_key", "mtls"]:
            raise ValueError("auth_type must be 'api_key' or 'mtls'")
        return v.lower()
    
    @field_validator("tracing_exporter")
    @classmethod
    def validate_tracing_exporter(cls, v: str) -> str:
        """Валидация экспортера трейсов."""
        if v.lower() not in ["jaeger", "otlp"]:
            raise ValueError("tracing_exporter must be 'jaeger' or 'otlp'")
        return v.lower()
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Получить список разрешённых origins для CORS."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def allowed_video_paths_list(self) -> List[str]:
        """Получить список разрешённых директорий для video_path."""
        return [path.strip() for path in self.allowed_video_paths.split(",")]

    def get_video_url_cache_dir(self) -> str:
        """Директория кэша для видео по URL (Phase 3). По умолчанию — поддиректория первой из allowed_video_paths."""
        if self.video_url_cache_dir:
            return self.video_url_cache_dir
        allowed = self.allowed_video_paths_list
        if allowed:
            return str(Path(allowed[0]).resolve() / "_url_cache")
        return "/tmp/dataprocessor_url_cache"


# Глобальный экземпляр конфигурации
config = APIConfig()

