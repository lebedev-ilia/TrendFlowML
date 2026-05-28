"""Валидатор manifest.json для Fetcher.

Проверяет корректность manifest перед сохранением в storage.
Соответствует Quality Assurance Checklist (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Optional

from schemas.manifest import FetcherManifest
from .storage import storage_client

logger = logging.getLogger(__name__)


def validate_manifest(manifest: FetcherManifest) -> tuple[bool, Optional[str]]:
    """Валидировать manifest перед сохранением.

    Проверяет:
    - обязательные поля присутствуют
    - artifact paths не пустые
    - checksums имеют правильный формат (sha256:...)
    - size_bytes > 0 для всех артефактов

    Args:
        manifest: FetcherManifest для валидации

    Returns:
        Tuple (is_valid, error_message)
        is_valid=True если manifest корректен, False иначе
        error_message содержит описание проблемы, если is_valid=False
    """
    # 1) Проверка обязательных полей (Pydantic уже проверит, но добавим явные проверки)
    if not manifest.video_id:
        return False, "video_id is required"
    if not manifest.platform:
        return False, "platform is required"
    if manifest.duration_seconds <= 0:
        return False, f"duration_seconds must be > 0, got {manifest.duration_seconds}"

    # 2) Проверка artifacts
    if not manifest.artifacts.video_file:
        return False, "artifacts.video_file is required"
    if not manifest.artifacts.meta_file:
        return False, "artifacts.meta_file is required"

    # 3) Проверка video_file
    video_file = manifest.artifacts.video_file
    if not video_file.path:
        return False, "artifacts.video_file.path is required"
    if video_file.size_bytes is not None and video_file.size_bytes <= 0:
        return False, f"artifacts.video_file.size_bytes must be > 0, got {video_file.size_bytes}"
    if video_file.checksum and not video_file.checksum.startswith("sha256:"):
        return False, f"artifacts.video_file.checksum must start with 'sha256:', got {video_file.checksum}"

    # 4) Проверка meta_file
    meta_file = manifest.artifacts.meta_file
    if not meta_file.path:
        return False, "artifacts.meta_file.path is required"
    if meta_file.size_bytes is not None and meta_file.size_bytes <= 0:
        return False, f"artifacts.meta_file.size_bytes must be > 0, got {meta_file.size_bytes}"
    if meta_file.checksum and not meta_file.checksum.startswith("sha256:"):
        return False, f"artifacts.meta_file.checksum must start with 'sha256:', got {meta_file.checksum}"

    # 5) Проверка comments_file (опциональный)
    if manifest.artifacts.comments_file:
        comments_file = manifest.artifacts.comments_file
        if not comments_file.path:
            return False, "artifacts.comments_file.path is required if comments_file is present"
        if comments_file.size_bytes is not None and comments_file.size_bytes <= 0:
            return False, f"artifacts.comments_file.size_bytes must be > 0, got {comments_file.size_bytes}"
        if comments_file.checksum and not comments_file.checksum.startswith("sha256:"):
            return False, f"artifacts.comments_file.checksum must start with 'sha256:', got {comments_file.checksum}"

    # 6) Проверка manifest_version
    if not manifest.manifest_version:
        return False, "manifest_version is required"

    return True, None


def validate_manifest_artifacts_exist(
    manifest: FetcherManifest, bucket: str = "video-analytics-raw"
) -> tuple[bool, Optional[str]]:
    """Проверить, что все артефакты, указанные в manifest, существуют в storage.

    Args:
        manifest: FetcherManifest для проверки
        bucket: Bucket для проверки (по умолчанию video-analytics-raw)

    Returns:
        Tuple (all_exist, error_message)
        all_exist=True если все артефакты существуют, False иначе
        error_message содержит путь к отсутствующему артефакту, если all_exist=False
    """
    # Проверяем video_file
    if not storage_client.object_exists(bucket, manifest.artifacts.video_file.path):
        return False, f"Video file not found in storage: {manifest.artifacts.video_file.path}"

    # Проверяем meta_file
    if not storage_client.object_exists(bucket, manifest.artifacts.meta_file.path):
        return False, f"Meta file not found in storage: {manifest.artifacts.meta_file.path}"

    # Проверяем comments_file (если есть)
    if manifest.artifacts.comments_file:
        if not storage_client.object_exists(bucket, manifest.artifacts.comments_file.path):
            return False, f"Comments file not found in storage: {manifest.artifacts.comments_file.path}"

    return True, None


__all__ = ["validate_manifest", "validate_manifest_artifacts_exist"]

