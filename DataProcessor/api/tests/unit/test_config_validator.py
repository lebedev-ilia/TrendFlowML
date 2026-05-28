"""
Unit тесты для Config Validator.
"""

import pytest
from unittest.mock import patch

from api.config import APIConfig
from api.utils.config_validator import (
    validate_config,
    ConfigValidationError,
)


class TestConfigValidatorBasic:
    """Тесты базовой валидации конфигурации."""

    def test_validate_config_success(self):
        """Валидная конфигурация проходит без ошибок."""
        cfg = APIConfig(
            api_host="0.0.0.0",
            api_port=8000,
            max_concurrent_runs=4,
            max_queue_length=100,
            log_level="INFO",
            log_format="json",
            storage_type="fs",
            storage_root="/tmp",
            redis_host=None,
            redis_url=None,
            api_key="test_key",
            allowed_video_paths="/tmp",
        )

        # Не должно выбрасывать исключение
        validate_config(cfg)

    def test_invalid_port_raises_error(self):
        """Невалидный порт вызывает ошибку конфигурации."""
        cfg = APIConfig(
            api_host="0.0.0.0",
            api_port=70000,  # вне допустимого диапазона
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "API_PORT must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_log_level_raises_error(self):
        """Невалидный уровень логирования вызывает ошибку."""
        cfg = APIConfig(
            log_level="INVALID",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "LOG_LEVEL must be one of" in str(exc_info.value)

    def test_invalid_storage_type_raises_error(self):
        """Невалидный STORAGE_TYPE вызывает ошибку."""
        cfg = APIConfig(
            storage_type="invalid",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "STORAGE_TYPE must be 'fs' or 's3'" in str(exc_info.value)

    def test_fs_storage_without_root_raises_error(self):
        """STORAGE_TYPE=fs без STORAGE_ROOT вызывает ошибку."""
        cfg = APIConfig(
            storage_type="fs",
            storage_root=None,
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "STORAGE_ROOT must be set when STORAGE_TYPE='fs'" in str(exc_info.value)


class TestConfigValidatorSecurity:
    """Тесты валидации security-настроек."""

    def test_missing_api_key_in_production_warns_only(self, caplog):
        """
        Отсутствие API key в production должно давать предупреждение,
        но не критичную ошибку.
        """
        cfg = APIConfig(
            debug=False,
            api_key=None,
        )

        with caplog.at_level("WARNING"):
            # Не должно выбрасывать ConfigValidationError
            validate_config(cfg)

        assert any("DATAPROCESSOR_API_KEY is not set" in m for m in caplog.messages)

    def test_allowed_video_paths_invalid_in_production_errors(self, tmp_path):
        """ALLOWED_VIDEO_PATHS без валидных директорий в production вызывает ошибку."""
        cfg = APIConfig(
            debug=False,
            allowed_video_paths="/nonexistent/path1,/nonexistent/path2",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "ALLOWED_VIDEO_PATHS is set but no valid directories found" in str(
            exc_info.value
        )


class TestConfigValidatorTracing:
    """Тесты валидации настроек OpenTelemetry."""

    def test_invalid_tracing_exporter_raises_error(self):
        """Невалидный TRACING_EXPORTER вызывает ошибку."""
        cfg = APIConfig(
            enable_tracing=True,
            tracing_exporter="invalid",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "TRACING_EXPORTER must be 'jaeger' or 'otlp'" in str(exc_info.value)

    def test_jaeger_without_host_raises_error(self):
        """TRACING_EXPORTER=jaeger без host вызывает ошибку."""
        cfg = APIConfig(
            enable_tracing=True,
            tracing_exporter="jaeger",
            jaeger_agent_host="",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "JAEGER_AGENT_HOST must be set" in str(exc_info.value)

    def test_otlp_without_endpoint_raises_error(self):
        """TRACING_EXPORTER=otlp без endpoint вызывает ошибку."""
        cfg = APIConfig(
            enable_tracing=True,
            tracing_exporter="otlp",
            otlp_endpoint="",
        )

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(cfg)

        assert "OTLP_ENDPOINT must be set" in str(exc_info.value)


