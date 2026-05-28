"""
Pytest конфигурация и фикстуры для тестов Backend ↔ DataProcessor интеграции.

Запуск из корня репозитория:
  cd backend && pytest tests/ -v

Или из backend:
  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Добавить backend в PYTHONPATH для импорта app
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


@pytest.fixture
def mock_settings(monkeypatch):
    """Подменить настройки backend для тестов DataProcessor API."""
    monkeypatch.setenv("TF_BACKEND_DATAPROCESSOR_API_URL", "http://dataprocessor-test:8001")
    monkeypatch.setenv("TF_BACKEND_DATAPROCESSOR_API_KEY", "test-api-key")
    monkeypatch.setenv("TF_BACKEND_DATAPROCESSOR_POLL_INTERVAL", "1")
    monkeypatch.setenv("TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS", "60")


@pytest.fixture
def sample_profile_config():
    """Профиль обработки в формате, который передаёт backend в DataProcessor."""
    return {
        "config_hash": "abc123def456",
        "visual": {"cfg_path": "/opt/visual/config.yaml"},
        "processors": {
            "segmenter": {"enabled": True, "required": True},
            "audio": {"enabled": False, "required": False},
            "text": {"enabled": False, "required": False},
            "visual": {"enabled": True, "required": True},
        },
    }


@pytest.fixture
def sample_run_id():
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def sample_process_payload(sample_run_id, sample_profile_config, tmp_path):
    """Payload как формирует backend для POST /api/v1/process."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return {
        "run_id": sample_run_id,
        "video_id": "dQw4w9WgXcQ",
        "platform_id": "youtube",
        "video_path": str(video_file.absolute()),
        "config_hash": sample_profile_config["config_hash"],
        "profile_config": sample_profile_config,
        "rs_base": str(tmp_path / "result_store"),
        "output": str(tmp_path / "frames"),
        "visual_cfg_path": str(tmp_path / "visual.yaml"),
        "dag_path": str(tmp_path / "dag.yaml"),
        "dag_stage": "baseline",
        "sampling_policy_version": "v1",
        "dataprocessor_version": "dev",
        "chunk_size": 64,
    }
