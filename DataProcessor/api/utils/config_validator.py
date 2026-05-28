"""
Валидация конфигурации API при старте.

Проверяет значения переменных окружения / настроек `APIConfig`
и останавливает приложение при критичных ошибках конфигурации.

Ссылка: DataProcessor/docs/API_DEVELOPMENT_CHECKLIST.md (раздел 7.3)
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

from api.config import APIConfig, config

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Критичная ошибка конфигурации API."""


def _validate_basic(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Проверка базовых настроек API."""
    errors: List[str] = []
    warnings: List[str] = []

    # API_HOST
    if not cfg.api_host:
        errors.append("API_HOST must not be empty")

    # API_PORT
    if not (1 <= cfg.api_port <= 65535):
        errors.append(f"API_PORT must be between 1 and 65535 (got {cfg.api_port})")

    # MAX_CONCURRENT_RUNS
    if cfg.max_concurrent_runs <= 0:
        errors.append("MAX_CONCURRENT_RUNS must be > 0")
    elif cfg.max_concurrent_runs > 1000:
        warnings.append(
            f"MAX_CONCURRENT_RUNS is very high ({cfg.max_concurrent_runs}), "
            "this may overload the system"
        )

    # MAX_QUEUE_LENGTH
    if cfg.max_queue_length <= 0:
        errors.append("MAX_QUEUE_LENGTH must be > 0")

    # LOG_LEVEL
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if cfg.log_level.upper() not in valid_log_levels:
        errors.append(
            f"LOG_LEVEL must be one of {sorted(valid_log_levels)} "
            f"(got {cfg.log_level})"
        )

    # LOG_FORMAT
    if cfg.log_format not in {"json", "text"}:
        errors.append(f"LOG_FORMAT must be 'json' or 'text' (got {cfg.log_format})")

    return errors, warnings


def _validate_storage(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Проверка настроек Storage."""
    errors: List[str] = []
    warnings: List[str] = []

    if cfg.storage_type not in {"fs", "s3"}:
        errors.append(f"STORAGE_TYPE must be 'fs' or 's3' (got {cfg.storage_type})")
        return errors, warnings

    if cfg.storage_type == "fs":
        if not cfg.storage_root:
            errors.append("STORAGE_ROOT must be set when STORAGE_TYPE='fs'")
        else:
            root_path = Path(cfg.storage_root)
            if not root_path.exists():
                errors.append(
                    f"STORAGE_ROOT directory does not exist: {cfg.storage_root}"
                )
            elif not root_path.is_dir():
                errors.append(
                    f"STORAGE_ROOT must be a directory (got file): {cfg.storage_root}"
                )
    else:
        # STORAGE_TYPE == "s3"
        # Здесь можно добавить более строгую проверку настроек S3
        # Пока ограничимся предупреждением, если отсутствуют базовые переменные.
        s3_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        missing = [name for name in s3_env_vars if not os.getenv(name)]
        if missing:
            warnings.append(
                "S3 storage selected but some AWS_* environment variables are missing: "
                + ", ".join(missing)
            )

    return errors, warnings


def _validate_redis(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Проверка настроек Redis."""
    errors: List[str] = []
    warnings: List[str] = []

    if cfg.redis_url:
        # Если задан redis_url, отдельные host/port игнорируются.
        return errors, warnings

    # Если redis_url не задан, но указан host, проверим порт.
    if cfg.redis_host:
        if not (1 <= cfg.redis_port <= 65535):
            errors.append(f"REDIS_PORT must be between 1 and 65535 (got {cfg.redis_port})")
    else:
        # Redis может быть опционален — лишь предупреждение.
        warnings.append(
            "Redis is not configured (REDIS_URL/REDIS_HOST not set). "
            "Some features (queue, state cache) may be disabled."
        )

    return errors, warnings


def _validate_security(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Проверка security-настроек."""
    errors: List[str] = []
    warnings: List[str] = []

    # API key: предупреждение в production, если не установлен.
    is_production = not cfg.debug
    if is_production and not cfg.api_key:
        warnings.append(
            "DATAPROCESSOR_API_KEY is not set while DEBUG=false. "
            "This is insecure for production."
        )

    # ALLOWED_VIDEO_PATHS
    allowed_paths_str = cfg.allowed_video_paths or ""
    if allowed_paths_str:
        paths = [p.strip() for p in allowed_paths_str.split(",") if p.strip()]
        valid_paths = []
        for p in paths:
            path_obj = Path(p)
            if path_obj.exists() and path_obj.is_dir():
                valid_paths.append(str(path_obj.resolve()))
            else:
                warnings.append(
                    f"ALLOWED_VIDEO_PATHS contains non-existent or non-directory path: {p}"
                )
        if is_production and not valid_paths:
            errors.append(
                "ALLOWED_VIDEO_PATHS is set but no valid directories found. "
                "Video path validation will always fail."
            )
    else:
        warnings.append(
            "ALLOWED_VIDEO_PATHS is empty. Video path validation will only check file existence."
        )

    return errors, warnings


def _validate_tracing(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Проверка настроек OpenTelemetry."""
    errors: List[str] = []
    warnings: List[str] = []

    if not cfg.enable_tracing:
        return errors, warnings

    if cfg.tracing_exporter not in {"jaeger", "otlp"}:
        errors.append(
            f"TRACING_EXPORTER must be 'jaeger' or 'otlp' (got {cfg.tracing_exporter})"
        )
        return errors, warnings

    if cfg.tracing_exporter == "jaeger":
        if not cfg.jaeger_agent_host:
            errors.append("JAEGER_AGENT_HOST must be set when TRACING_EXPORTER='jaeger'")
        if not (1 <= cfg.jaeger_agent_port <= 65535):
            errors.append(
                f"JAEGER_AGENT_PORT must be between 1 and 65535 (got {cfg.jaeger_agent_port})"
            )
    elif cfg.tracing_exporter == "otlp":
        if not cfg.otlp_endpoint:
            errors.append("OTLP_ENDPOINT must be set when TRACING_EXPORTER='otlp'")

    return errors, warnings


def _aggregate_validations(cfg: APIConfig) -> Tuple[List[str], List[str]]:
    """Запустить все проверки и объединить ошибки/предупреждения."""
    errors: List[str] = []
    warnings: List[str] = []

    for validator in (
        _validate_basic,
        _validate_storage,
        _validate_redis,
        _validate_security,
        _validate_tracing,
    ):
        v_errors, v_warnings = validator(cfg)
        errors.extend(v_errors)
        warnings.extend(v_warnings)

    return errors, warnings


def validate_config(cfg: Optional[APIConfig] = None) -> None:
    """
    Валидировать конфигурацию API.

    Логирует предупреждения и выбрасывает ConfigValidationError при наличии ошибок.
    """
    cfg = cfg or config

    errors, warnings = _aggregate_validations(cfg)

    for w in warnings:
        logger.warning("Config warning: %s", w)

    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        raise ConfigValidationError(
            "Configuration validation failed with errors: " + "; ".join(errors)
        )

    logger.info("Configuration validation completed successfully")


__all__ = ["ConfigValidationError", "validate_config"]


