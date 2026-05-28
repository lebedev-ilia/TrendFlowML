"""Проверка идемпотентности для Fetcher.

Обеспечивает, что повторные вызовы pipeline stages не создают дубликаты
и корректно используют существующие артефакты.
Соответствует Quality Assurance Checklist (Idempotency of pipeline stages).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .db import session_scope
from .models import Artifact, Video
from .storage import storage_client

logger = logging.getLogger(__name__)


def check_video_exists(platform: str, platform_video_id: str) -> Optional[str]:
    """Проверить, существует ли видео в кеше.

    Args:
        platform: Платформа (например, "youtube")
        platform_video_id: ID видео на платформе

    Returns:
        UUID видео (str) если существует, None иначе. Возвращает примитив, чтобы не передавать detached ORM-объекты.
    """
    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == platform,
                Video.platform_video_id == platform_video_id,
            )
            .first()
        )
        return str(video.id) if video else None


def check_artifact_exists(
    video_id: str, artifact_type: str, status: str = "COMPLETED"
) -> Optional[Tuple[str, Optional[str]]]:
    """Проверить, существует ли артефакт для видео.

    Args:
        video_id: UUID видео
        artifact_type: Тип артефакта (video_file, metadata_file, comments_file)
        status: Статус артефакта (по умолчанию COMPLETED)

    Returns:
        (storage_path, checksum) если артефакт существует, None иначе. Примитивы, чтобы не передавать detached ORM.
    """
    from uuid import UUID

    video_uuid = UUID(video_id) if isinstance(video_id, str) else video_id

    with session_scope() as db:
        artifact: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video_uuid,
                Artifact.artifact_type == artifact_type,
                Artifact.status == status,
            )
            .first()
        )
        if artifact is None:
            return None
        return (artifact.storage_path or "", artifact.checksum)


def check_artifact_in_storage(storage_path: str, bucket: str) -> bool:
    """Проверить, существует ли артефакт в storage.

    Args:
        storage_path: Путь к артефакту в storage
        bucket: Bucket для проверки

    Returns:
        True если артефакт существует в storage, False иначе
    """
    if not storage_path:
        return False

    try:
        return storage_client.object_exists(bucket, storage_path)
    except Exception as e:
        logger.warning(f"Failed to check artifact existence in storage: {e}")
        return False


def validate_artifact_checksum(
    storage_path: str, checksum: Optional[str], bucket: str
) -> tuple[bool, Optional[str]]:
    """Проверить checksum артефакта в storage.

    Скачивает артефакт, вычисляет checksum и сравнивает с сохранённым.

    Args:
        storage_path: Путь к артефакту в storage
        checksum: Ожидаемый checksum (например, sha256:...)
        bucket: Bucket для проверки

    Returns:
        Tuple (is_valid, error_message)
        is_valid=True если checksum совпадает, False иначе
        error_message содержит описание проблемы, если is_valid=False
    """
    if not storage_path:
        return False, "Artifact has no storage_path"

    if not checksum:
        # Если checksum не сохранён, считаем валидным (для обратной совместимости)
        return True, None

    try:
        import tempfile
        from pathlib import Path

        from .checksums import compute_sha256

        # Скачиваем артефакт во временный файл
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            storage_client.download_file(bucket, storage_path, tmp_path)

            # Вычисляем checksum
            computed_checksum_hex = compute_sha256(tmp_path)
            computed_checksum = f"sha256:{computed_checksum_hex}"

            # Сравниваем с сохранённым
            if computed_checksum != checksum:
                return (
                    False,
                    f"Checksum mismatch: expected {checksum}, got {computed_checksum}",
                )

            return True, None
        finally:
            # Удаляем временный файл
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    except Exception as e:
        return False, f"Failed to validate checksum: {str(e)}"


def is_stage_idempotent(
    platform: str, platform_video_id: str, stage: str, validate_checksum: bool = False
) -> tuple[bool, Optional[str]]:
    """Проверить, можно ли пропустить stage (идемпотентность).

    Args:
        platform: Платформа
        platform_video_id: ID видео на платформе (может быть URL, будет нормализован)
        stage: Название stage (metadata, download, comments)
        validate_checksum: Проверять ли checksum артефакта (по умолчанию False)

    Returns:
        Tuple (can_skip, reason)
        can_skip=True если stage можно пропустить (уже выполнен), False иначе
        reason содержит причину, почему можно/нельзя пропустить
    """
    # Если platform_video_id это URL, пытаемся извлечь video_id
    # Для простоты проверяем по platform_video_id как есть (YouTubeAdapter сам нормализует)
    video_id = check_video_exists(platform, platform_video_id)
    if not video_id:
        return False, "Video not found in cache"

    # Маппинг stage -> artifact_type
    stage_to_artifact = {
        "metadata": "metadata_file",
        "download": "video_file",
        "comments": "comments_file",
    }

    artifact_type = stage_to_artifact.get(stage)
    if not artifact_type:
        return False, f"Unknown stage: {stage}"

    artifact_info = check_artifact_exists(video_id, artifact_type)
    if not artifact_info:
        return False, f"Artifact {artifact_type} not found"

    storage_path, checksum = artifact_info

    # Проверяем, что артефакт существует в storage
    from .config import settings

    if not check_artifact_in_storage(storage_path, settings.bucket_raw):
        return False, f"Artifact {artifact_type} not found in storage"

    # Опциональная проверка checksum
    if validate_checksum and checksum:
        is_valid, error_msg = validate_artifact_checksum(
            storage_path, checksum, settings.bucket_raw
        )
        if not is_valid:
            return False, f"Artifact {artifact_type} checksum validation failed: {error_msg}"

    return True, f"Stage {stage} already completed, artifact exists"


__all__ = [
    "check_video_exists",
    "check_artifact_exists",
    "check_artifact_in_storage",
    "validate_artifact_checksum",
    "is_stage_idempotent",
]

