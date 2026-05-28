"""
Контрактные тесты Backend ↔ DataProcessor.

Проверяют, что payload, который формирует backend для POST /api/v1/process,
содержит все обязательные поля и типы, ожидаемые DataProcessor API.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pytest

from app.services.dataprocessor import (
    resolve_run_paths,
)


# Минимальный контракт (без валидации video_path.exists и profile_config.processors),
# чтобы не тянуть зависимость на DataProcessor в backend.
# Полная валидация — в DataProcessor в test_backend_request_contract.
pytestmark = pytest.mark.contract

REQUIRED_PROCESS_FIELDS = {
    "run_id": (str, re.compile(r"^[0-9a-f-]{36}$")),
    "video_id": (str, lambda s: len(s) >= 1),
    "platform_id": (str, re.compile(r"^(youtube|upload)$")),
    "video_path": (str, None),
    "config_hash": (str, None),
    "profile_config": (dict, None),
}

OPTIONAL_PROCESS_FIELDS = [
    "rs_base", "output", "visual_cfg_path", "dag_path", "dag_stage",
    "sampling_policy_version", "dataprocessor_version", "chunk_size",
    "profile_version", "feature_schema_version", "pipeline_version",
]


class TestBackendProcessPayloadContract:
    """Payload от backend для /api/v1/process соответствует контракту."""

    def test_run_dataprocessor_async_payload_shape(self, sample_process_payload):
        """Структура payload содержит обязательные поля и типы."""
        for field, (expected_type, validator) in REQUIRED_PROCESS_FIELDS.items():
            assert field in sample_process_payload, f"Missing required field: {field}"
            value = sample_process_payload[field]
            assert isinstance(
                value, expected_type
            ), f"{field} should be {expected_type.__name__}, got {type(value).__name__}"
            if validator is not None and callable(validator):
                if hasattr(validator, "match"):
                    assert validator.match(str(value)), f"{field} format invalid: {value}"
                else:
                    assert validator(value), f"{field} validation failed: {value}"

    def test_profile_config_has_processors(self, sample_process_payload):
        """DataProcessor требует profile_config.processors."""
        pc = sample_process_payload.get("profile_config") or {}
        assert "processors" in pc, "profile_config must contain 'processors'"

    def test_backend_sends_optional_paths(self, sample_process_payload):
        """Backend передаёт rs_base, output, dag_path для DataProcessor."""
        for key in ["rs_base", "output", "dag_path", "dag_stage"]:
            assert key in sample_process_payload
            assert isinstance(sample_process_payload[key], str)


class TestRunPathsContract:
    """Пути результатов backend совпадают с соглашением DataProcessor."""

    def test_manifest_and_state_events_paths(self, tmp_path):
        """run_rs_path/platform_id/video_id/run_id и state/.../state_events.jsonl."""
        paths = resolve_run_paths(
            platform_id="upload",
            video_id="local-video-1",
            run_id="550e8400-e29b-41d4-a716-446655440000",
            result_store_base=tmp_path / "result_store",
        )
        assert paths.manifest_path == paths.run_rs_path / "manifest.json"
        assert paths.state_events_path.name == "state_events.jsonl"
        assert "upload" in str(paths.run_rs_path)
        assert "local-video-1" in str(paths.run_rs_path)
        assert "550e8400" in str(paths.run_rs_path)
