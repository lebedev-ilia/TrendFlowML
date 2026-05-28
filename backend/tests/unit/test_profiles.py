"""
Unit-тесты, связанные с профилями анализа: запись profile YAML, дефолтная структура,
compute_config_hash, seed_public_profiles из YAML (§ 3.7.3).

Нормализация профиля (visual.cfg_path, processors по умолчанию) и config_hash
покрыты в tests/test_dataprocessor_adapter.py через _resolve_processing_config
и prepare_dataprocessor_payload.

См. backend/docs/TESTING_PLAN.md § 3.7, PROFILES.md.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.dataprocessor import build_profile_yaml
from app.services.profiles import compute_config_hash, seed_public_profiles


pytestmark = pytest.mark.unit


class TestComputeConfigHash:
    """Детерминированный config_hash от JSON с сортировкой ключей."""

    def test_deterministic(self):
        """Одинаковый конфиг даёт один и тот же hash."""
        config = {"a": 1, "b": 2, "processors": {"visual": {"enabled": True}}}
        assert compute_config_hash(config) == compute_config_hash(config)

    def test_order_independent(self):
        """Порядок ключей не влияет (сортировка внутри)."""
        c1 = {"b": 2, "a": 1}
        c2 = {"a": 1, "b": 2}
        assert compute_config_hash(c1) == compute_config_hash(c2)

    def test_different_config_different_hash(self):
        """Разный конфиг — разный hash."""
        a = compute_config_hash({"x": 1})
        b = compute_config_hash({"x": 2})
        assert a != b
        assert len(a) == 64
        assert all(c in "0123456789abcdef" for c in a)


class TestSeedPublicProfiles:
    """Загрузка публичных профилей из YAML (мок БД)."""

    def test_creates_profiles_from_yaml_files(self, tmp_path: Path):
        """Из директории с *.yaml создаются записи AnalysisProfile (is_public=True)."""
        (tmp_path / "default.yaml").write_text(
            "config_hash: h1\nprocessors:\n  visual:\n    enabled: true\n",
            encoding="utf-8",
        )
        (tmp_path / "minimal.yaml").write_text(
            "processors: {}\n",
            encoding="utf-8",
        )
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        n = seed_public_profiles(db, tmp_path)
        assert n == 2
        assert db.add.call_count == 2
        calls = [c[0][0] for c in db.add.call_args_list]
        names = {p.name for p in calls}
        assert names == {"default", "minimal"}
        for p in calls:
            assert p.is_public is True
            assert p.user_id is None
            assert p.config_hash

    def test_skips_existing_public_profile_by_name(self, tmp_path: Path):
        """Если публичный профиль с таким именем уже есть — не создаём дубликат."""
        (tmp_path / "default.yaml").write_text(
            "processors: {}\n",
            encoding="utf-8",
        )
        db = MagicMock()
        existing = MagicMock()
        # Первый вызов first() — для "default" возвращаем существующий профиль
        db.query.return_value.filter.return_value.first.side_effect = [existing]
        n = seed_public_profiles(db, tmp_path)
        assert n == 0
        db.add.assert_not_called()

    def test_nonexistent_dir_returns_zero(self):
        """Если директории нет — возвращается 0."""
        db = MagicMock()
        n = seed_public_profiles(db, Path("/nonexistent/profiles"))
        assert n == 0
        db.add.assert_not_called()


class TestBuildProfileYaml:
    """build_profile_yaml: запись JSON-конфига в YAML для DataProcessor."""

    def test_writes_yaml_file(self, tmp_path: Path):
        """Записывает YAML по пути, родительская директория создаётся."""
        out_path = tmp_path / "sub" / "profile.yaml"
        config = {
            "config_hash": "abc",
            "processors": {
                "segmenter": {"enabled": True},
                "visual": {"enabled": True},
            },
        }
        build_profile_yaml(config, out_path)
        assert out_path.exists()
        assert out_path.parent.exists()

    def test_yaml_roundtrip(self, tmp_path: Path):
        """Записанный YAML можно прочитать и получить эквивалентную структуру."""
        config = {
            "config_hash": "h1",
            "visual": {"cfg_path": "/opt/visual.yaml"},
            "processors": {
                "audio": {"enabled": False},
                "text": {"enabled": False},
            },
        }
        out_path = tmp_path / "profile.yaml"
        build_profile_yaml(config, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded["config_hash"] == config["config_hash"]
        assert loaded["visual"]["cfg_path"] == config["visual"]["cfg_path"]
        assert "processors" in loaded
        assert loaded["processors"]["audio"]["enabled"] is False
