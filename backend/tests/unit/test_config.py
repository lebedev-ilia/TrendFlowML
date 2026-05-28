"""
Unit-тесты конфигурации Backend (Settings, resolve_paths).

Проверяют загрузку настроек из env (TF_BACKEND_*), значения по умолчанию
и формирование путей хранилища и DataProcessor.

См. backend/docs/CONFIGURATION.md, TESTING_PLAN.md § 3.5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, ResolvedPaths


pytestmark = pytest.mark.unit


class TestSettings:
    """Загрузка настроек приложения."""

    def test_default_app_name(self):
        """По умолчанию app_name = TrendFlow Backend."""
        s = Settings()
        assert s.app_name == "TrendFlow Backend"

    def test_env_prefix(self):
        """Используется префикс TF_BACKEND_ для переменных окружения."""
        assert Settings.model_config.get("env_prefix") == "TF_BACKEND_"

    def test_dataprocessor_api_url_default(self):
        """По умолчанию dataprocessor_api_url = http://localhost:8001."""
        s = Settings()
        assert s.dataprocessor_api_url == "http://localhost:8001"

    def test_dataprocessor_api_key_optional(self):
        """dataprocessor_api_key опционален (None)."""
        s = Settings()
        assert s.dataprocessor_api_key is None

    def test_dataprocessor_poll_and_timeout_defaults(self):
        """Дефолтные poll_interval и timeout_seconds."""
        s = Settings()
        assert s.dataprocessor_poll_interval == 5
        assert s.dataprocessor_timeout_seconds == 3600

    def test_cors_origins_default_star(self):
        s = Settings()
        assert s.cors_allow_origins() == ["*"]

    def test_cors_origins_comma_list(self, monkeypatch):
        monkeypatch.setenv(
            "TF_BACKEND_CORS_ORIGINS",
            "http://localhost:3000, https://app.example.com ",
        )
        s = Settings()
        assert s.cors_allow_origins() == [
            "http://localhost:3000",
            "https://app.example.com",
        ]


class TestResolvePaths:
    """Формирование путей хранилища и DataProcessor."""

    def test_resolve_paths_returns_resolved_paths(self):
        """resolve_paths() возвращает объект ResolvedPaths."""
        s = Settings()
        paths = s.resolve_paths()
        assert isinstance(paths, ResolvedPaths)

    def test_resolve_paths_all_attributes_set(self):
        """Без явного storage_root все пути заданы относительно repo."""
        s = Settings()
        paths = s.resolve_paths()
        assert paths.storage_root is not None
        assert paths.result_store_base is not None
        assert paths.frames_dir_base is not None
        assert paths.raw_uploads_dir is not None
        assert paths.dataproc_root is not None
        assert paths.visual_cfg_default is not None

    def test_resolve_paths_explicit_storage_root(self, monkeypatch, tmp_path):
        """При заданном storage_root остальные пути строятся от него."""
        storage = tmp_path / "my_storage"
        storage.mkdir()
        monkeypatch.setenv("TF_BACKEND_STORAGE_ROOT", str(storage))
        s = Settings()
        paths = s.resolve_paths()
        assert paths.storage_root == storage
        assert paths.result_store_base == storage / "result_store"
        assert paths.frames_dir_base == storage / "frames_dir"
        assert paths.raw_uploads_dir == storage / "raw"

    def test_resolve_paths_dataproc_root(self):
        """dataproc_root: при наличии каталога DataProcessor у корня репо — он, иначе сам repo_root (автономный репо)."""
        s = Settings()
        paths = s.resolve_paths()
        repo_root = paths.repo_root
        assert paths.dataproc_root.is_absolute()
        if (repo_root / "DataProcessor").is_dir():
            assert paths.dataproc_root == repo_root / "DataProcessor"
        else:
            assert paths.dataproc_root == repo_root

    def test_resolve_paths_visual_cfg_default(self):
        """visual_cfg_default указывает на yaml в dataproc или переопределён."""
        s = Settings()
        paths = s.resolve_paths()
        assert paths.visual_cfg_default is not None
        assert paths.visual_cfg_default.suffix in (".yaml", ".yml") or "visual" in str(paths.visual_cfg_default)
