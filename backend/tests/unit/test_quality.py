"""
Unit-тесты модуля quality: discover_quality_scripts, find_component_npz,
build_quality_command, run_quality_reports (с моком subprocess).

См. backend/docs/TESTING_PLAN.md § 3.8 (storage/artifacts, quality reports).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.quality import (
    QualityScript,
    build_quality_command,
    discover_quality_scripts,
    find_component_npz,
    run_quality_reports,
)


pytestmark = pytest.mark.unit


class TestDiscoverQualityScripts:
    """Поиск скриптов quality_report/demo_*_quality.py."""

    def test_empty_root_returns_empty(self, tmp_path):
        """Пустой dataproc_root → пустой словарь."""
        assert discover_quality_scripts(tmp_path) == {}

    def test_discovers_script_under_modules(self, tmp_path):
        """Скрипт в modules/component_name/quality_report/demo_*_quality.py обнаруживается."""
        script_dir = tmp_path / "modules" / "my_component" / "quality_report"
        script_dir.mkdir(parents=True)
        script_file = script_dir / "demo_my_component_quality.py"
        script_file.write_text("# --out-html --npz-path\n")
        scripts = discover_quality_scripts(tmp_path)
        assert "my_component" in scripts
        assert scripts["my_component"].script_path == script_file
        assert scripts["my_component"].flags == {"out-html", "npz-path"}

    def test_ignores_non_matching_glob(self, tmp_path):
        """Файлы не demo_*_quality.py не учитываются."""
        (tmp_path / "modules" / "x" / "quality_report").mkdir(parents=True)
        (tmp_path / "modules" / "x" / "quality_report" / "other.py").write_text("--out-html")
        assert discover_quality_scripts(tmp_path) == {}


class TestFindComponentNpz:
    """Поиск .npz файла в директории компонента."""

    def test_no_dir_returns_none(self, tmp_path):
        """Нет директории компонента → None."""
        assert find_component_npz(tmp_path, "missing") is None

    def test_no_npz_returns_none(self, tmp_path):
        """Директория есть, но без .npz → None."""
        (tmp_path / "comp").mkdir()
        (tmp_path / "comp" / "data.json").write_text("{}")
        assert find_component_npz(tmp_path, "comp") is None

    def test_returns_npz_prefer_component_name(self, tmp_path):
        """Возвращается .npz; предпочитается файл с именем компонента."""
        (tmp_path / "face").mkdir()
        other = tmp_path / "face" / "other.npz"
        other.write_bytes(b"")
        face_npz = tmp_path / "face" / "face_output.npz"
        face_npz.write_bytes(b"")
        result = find_component_npz(tmp_path, "face")
        assert result is not None
        assert "face" in result.name
        assert result.suffix == ".npz"

    def test_returns_first_npz_if_no_name_match(self, tmp_path):
        """Если нет .npz с именем компонента — возвращается первый по сортировке."""
        (tmp_path / "comp").mkdir()
        first = tmp_path / "comp" / "a.npz"
        first.write_bytes(b"")
        (tmp_path / "comp" / "b.npz").write_bytes(b"")
        assert find_component_npz(tmp_path, "comp") == first


class TestBuildQualityCommand:
    """Сборка командной строки для quality-скрипта."""

    def test_basic_command_has_python_and_script(self, tmp_path):
        """Команда содержит python и путь к скрипту."""
        script_path = tmp_path / "demo_quality.py"
        script_path.write_bytes(b"")
        script = QualityScript(component_name="test", script_path=script_path, flags=set())
        cmd = build_quality_command(
            script,
            run_rs_path=tmp_path,
            frames_dir=None,
            video_path=None,
            out_dir=tmp_path / "out",
        )
        assert cmd is not None
        assert "python" in cmd[0].lower() or "python3" in cmd[0]
        assert str(script_path) in cmd

    def test_out_html_adds_flag_and_creates_out_dir(self, tmp_path):
        """Флаг out-html добавляет --out-html и путь к html."""
        script_path = tmp_path / "demo_quality.py"
        script_path.write_bytes(b"")
        out_dir = tmp_path / "comp" / "quality_report"
        script = QualityScript(
            component_name="comp",
            script_path=script_path,
            flags={"out-html"},
        )
        cmd = build_quality_command(
            script,
            run_rs_path=tmp_path,
            frames_dir=None,
            video_path=None,
            out_dir=out_dir,
        )
        assert cmd is not None
        assert "--out-html" in cmd
        assert "comp_quality.html" in " ".join(cmd)
        assert out_dir.exists()

    def test_npz_path_required_returns_none_without_npz(self, tmp_path):
        """При флаге npz-path и отсутствии .npz команда не строится (None)."""
        script_path = tmp_path / "demo_quality.py"
        script_path.write_bytes(b"")
        script = QualityScript(
            component_name="no_npz",
            script_path=script_path,
            flags={"out-html", "npz-path"},
        )
        out_dir = tmp_path / "no_npz" / "quality_report"
        out_dir.mkdir(parents=True)
        cmd = build_quality_command(
            script,
            run_rs_path=tmp_path,
            frames_dir=None,
            video_path=None,
            out_dir=out_dir,
        )
        assert cmd is None

    def test_rs_path_and_frames_dir_in_command(self, tmp_path):
        """Флаги rs-path и frames-dir попадают в команду при заданных путях."""
        script_path = tmp_path / "demo_quality.py"
        script_path.write_bytes(b"")
        frames = tmp_path / "frames"
        frames.mkdir()
        script = QualityScript(
            component_name="c",
            script_path=script_path,
            flags={"rs-path", "frames-dir"},
        )
        cmd = build_quality_command(
            script,
            run_rs_path=tmp_path,
            frames_dir=frames,
            video_path=None,
            out_dir=tmp_path / "out",
        )
        assert cmd is not None
        assert "--rs-path" in cmd
        assert "--frames-dir" in cmd
        assert str(tmp_path) in cmd
        assert str(frames) in cmd


class TestRunQualityReports:
    """Запуск quality reports с моком subprocess."""

    def test_run_quality_reports_mocked_subprocess(self, tmp_path):
        """run_quality_reports вызывает скрипт и возвращает сгенерированные html."""
        comp = "face"
        (tmp_path / comp / "quality_report").mkdir(parents=True)
        html_path = tmp_path / comp / "quality_report" / "face_quality.html"
        html_path.write_text("<html/>")
        script_path = tmp_path / "demo_face_quality.py"
        script_path.write_text("# --out-html")
        scripts = {
            comp: QualityScript(component_name=comp, script_path=script_path, flags={"out-html"}),
        }
        with patch("app.services.quality.subprocess.run") as run_mock:
            run_mock.return_value = type("R", (), {"returncode": 0})()
            generated = run_quality_reports(
                scripts,
                run_rs_path=tmp_path,
                frames_dir=None,
                video_path=None,
                components=[comp],
            )
        assert run_mock.called
        assert len(generated) == 1
        assert generated[0][0] == comp
        assert generated[0][1] == html_path

    def test_run_quality_skips_missing_component(self, tmp_path):
        """Компонент без скрипта пропускается."""
        scripts = {}
        generated = run_quality_reports(
            scripts,
            run_rs_path=tmp_path,
            frames_dir=None,
            video_path=None,
            components=["nonexistent"],
        )
        assert generated == []
