"""
Интеграционные тесты: парсинг manifest.json и регистрация артефактов.

Проверяют _sync_from_manifest_v2 (чтение manifest, создание Prediction)
и _scan_and_register_artifacts / _register_artifact с mock DB.
См. backend/docs/TESTING_PLAN.md § 3.8 (storage/artifacts).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.tasks import (
    _register_artifact,
    _scan_and_register_artifacts,
    _sync_from_manifest_v2,
)
from app.dbv2 import models as v2_models


pytestmark = pytest.mark.integration


def _mock_analysis_job():
    job = MagicMock(spec=v2_models.AnalysisJob)
    job.id = uuid4()
    job.model_version_id = None
    return job


def _mock_payload():
    payload = MagicMock()
    payload.video_id = "vid-1"
    payload.platform_id = "youtube"
    return payload


class TestManifestParsing:
    """Контракт manifest.json: ожидаемые ключи и структура."""

    def test_manifest_structure_run_components_predictions(self, tmp_path):
        """manifest содержит run, components, predictions (для _sync_from_manifest_v2)."""
        manifest = {
            "run": {"run_id": "r1", "video_id": "v1"},
            "components": [{"name": "face"}, {"name": "scene"}],
            "predictions": [
                {"horizon_days": 7, "predicted_views": 1000.0, "predicted_likes": 50.0},
            ],
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "run" in data
        assert "components" in data
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["horizon_days"] == 7


class TestSyncFromManifestV2:
    """Синхронизация данных из manifest в AnalysisJob и Prediction."""

    def test_missing_manifest_returns_empty(self):
        """Если manifest не существует, возвращается {}."""
        db = MagicMock()
        job = _mock_analysis_job()
        payload = _mock_payload()
        result = _sync_from_manifest_v2(
            db, job, Path("/nonexistent/manifest.json"), payload
        )
        assert result == {}

    def test_manifest_with_predictions_adds_to_db(self, tmp_path):
        """manifest с predictions создаёт записи Prediction через db.add."""
        manifest = {
            "run": {},
            "components": [],
            "predictions": [
                {"horizon_days": 7, "predicted_views": 2000.0, "predicted_likes": 100.0},
                {"horizon_days": 30, "predicted_views": 5000.0},
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        db = MagicMock()
        job = _mock_analysis_job()
        payload = _mock_payload()
        # Чтобы не создавать дубликаты, first() возвращает None (нет существующей prediction)
        db.query.return_value.filter.return_value.first.return_value = None

        result = _sync_from_manifest_v2(db, job, tmp_path / "manifest.json", payload)

        assert "predictions" in result
        assert len(result["predictions"]) == 2
        assert db.add.called
        # Должны быть добавлены 2 Prediction
        add_calls = [c for c in db.add.call_args_list if c[0][0].__class__.__name__ == "Prediction"]
        assert len(add_calls) == 2


class TestRegisterArtifact:
    """Регистрация одного артефакта в таблице artifacts."""

    def test_register_artifact_adds_when_not_exists(self, tmp_path):
        """_register_artifact добавляет Artifact, если такой object_key ещё нет."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        path = tmp_path / "comp" / "out.html"
        path.parent.mkdir(parents=True)
        path.write_text("<html/>")
        _register_artifact(db, "run-123", "comp", path)
        assert db.add.called
        (arg,) = db.add.call_args[0]
        assert arg.run_id == "run-123"
        assert arg.component_name == "comp"
        assert arg.kind == "html"
        assert path.as_posix() in arg.object_key
        assert arg.size_bytes == len("<html/>")

    def test_register_artifact_skips_duplicate(self, tmp_path):
        """При существующей записи с тем же run_id и object_key add не вызывается."""
        db = MagicMock()
        existing = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = existing
        path = tmp_path / "file.json"
        path.write_text("{}")
        _register_artifact(db, "run-1", "c", path)
        db.add.assert_not_called()


class TestScanAndRegisterArtifacts:
    """Сканирование run_rs_path и регистрация артефактов (.npz, .json, .html)."""

    def test_scan_skips_manifest_json(self, tmp_path):
        """manifest.json внутри компонента не регистрируется как артефакт."""
        (tmp_path / "comp").mkdir()
        (tmp_path / "comp" / "manifest.json").write_text("{}")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        _scan_and_register_artifacts(db, "run-1", tmp_path)
        # Только manifest.json — регистраций быть не должно (он пропускается)
        db.add.assert_not_called()

    def test_scan_registers_npz_and_html(self, tmp_path):
        """Регистрируются файлы с расширениями .npz, .json, .html."""
        (tmp_path / "face").mkdir()
        (tmp_path / "face" / "data.npz").write_bytes(b"x")
        (tmp_path / "face" / "report_quality.html").write_text("<html/>")
        (tmp_path / "scene").mkdir()
        (tmp_path / "scene" / "meta.json").write_text("{}")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        _scan_and_register_artifacts(db, "run-1", tmp_path)
        assert db.add.called
        keys = [c[0][0].object_key for c in db.add.call_args_list]
        assert any("data.npz" in k for k in keys)
        assert any("report_quality.html" in k for k in keys)
        assert any("meta.json" in k for k in keys)

    def test_scan_ignores_nonexistent_path(self):
        """Если run_rs_path не существует, функция ничего не делает."""
        db = MagicMock()
        _scan_and_register_artifacts(db, "run-1", Path("/nonexistent/run/path"))
        db.add.assert_not_called()
