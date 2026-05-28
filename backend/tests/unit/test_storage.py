"""
Unit-тесты модуля storage: пути, ensure_dirs, move_upload_to_storage, sha256_file.

Проверяют создание директорий хранилища, перенос загрузок и хеширование файлов.
См. backend/docs/TESTING_PLAN.md § 3.8 (storage/artifacts).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import ResolvedPaths
from app.services.storage import ensure_dirs, move_upload_to_storage, sha256_file


pytestmark = pytest.mark.unit


def _make_paths(root: Path) -> ResolvedPaths:
    return ResolvedPaths(
        repo_root=root / "repo",
        storage_root=root / "storage",
        result_store_base=root / "storage" / "result_store",
        frames_dir_base=root / "storage" / "frames_dir",
        raw_uploads_dir=root / "storage" / "raw",
        example_videos_dir=root / "repo" / "example" / "example_videos",
        dataproc_root=root / "DataProcessor",
        visual_cfg_default=root / "DataProcessor" / "configs" / "visual.yaml",
    )


class TestEnsureDirs:
    """Создание директорий хранилища."""

    def test_ensure_dirs_creates_all_paths(self, tmp_path):
        """ensure_dirs создаёт storage_root, raw, frames_dir, result_store, example_videos."""
        paths = _make_paths(tmp_path)
        with patch("app.services.storage.Settings") as Settings:
            Settings.return_value.resolve_paths.return_value = paths
            ensure_dirs()
        assert paths.storage_root.exists()
        assert paths.raw_uploads_dir.exists()
        assert paths.frames_dir_base.exists()
        assert paths.result_store_base.exists()
        assert paths.example_videos_dir.exists()


class TestMoveUploadToStorage:
    """Перенос загруженного файла в хранилище."""

    def test_move_upload_creates_video_dir_and_moves_file(self, tmp_path):
        """Файл перемещается в raw/{video_id}/video{ext}, копия в example_videos."""
        upload_file = tmp_path / "upload.mp4"
        upload_file.write_bytes(b"fake video content")
        paths = _make_paths(tmp_path)
        paths.storage_root.mkdir(parents=True)
        paths.raw_uploads_dir.mkdir(parents=True)
        paths.example_videos_dir.mkdir(parents=True)

        with patch("app.services.storage.Settings") as Settings:
            Settings.return_value.resolve_paths.return_value = paths
            with patch("app.services.storage.ensure_dirs"):
                out_path, example_path = move_upload_to_storage(
                    str(upload_file), "vid-123", filename="video.mp4"
                )
        assert not upload_file.exists()
        assert Path(out_path).exists()
        assert Path(out_path).read_bytes() == b"fake video content"
        assert Path(out_path).parent.name == "vid-123"
        assert Path(out_path).name == "video.mp4"
        assert Path(example_path).exists()
        assert Path(example_path).name == "vid-123.mp4"

    def test_move_upload_default_extension_mp4(self, tmp_path):
        """Без filename используется расширение из пути или .mp4."""
        upload_file = tmp_path / "upload.mov"
        upload_file.write_bytes(b"x")
        paths = _make_paths(tmp_path)
        paths.storage_root.mkdir(parents=True)
        paths.raw_uploads_dir.mkdir(parents=True)
        paths.example_videos_dir.mkdir(parents=True)

        with patch("app.services.storage.Settings") as Settings:
            Settings.return_value.resolve_paths.return_value = paths
            with patch("app.services.storage.ensure_dirs"):
                out_path, _ = move_upload_to_storage(str(upload_file), "vid-456", filename=None)
        assert Path(out_path).suffix == ".mov"


class TestSha256File:
    """Хеширование файла SHA-256."""

    def test_sha256_file_known_content(self, tmp_path):
        """sha256_file возвращает hex-дайджест для содержимого файла."""
        f = tmp_path / "f.bin"
        f.write_bytes(b"hello\n")
        digest = sha256_file(str(f))
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
        expected = hashlib.sha256(b"hello\n").hexdigest()
        assert digest == expected
