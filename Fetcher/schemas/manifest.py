from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArtifactInfo(BaseModel):
    """Информация об одном артефакте, записанном Fetcher в object storage.

    Соответствует разделу `artifacts.*` в BACKEND_CONTRACTS.md.
    """

    path: str = Field(
        ...,
        description=(
            "Относительный путь внутри bucket'а Fetcher "
            "(например, raw/youtube/2026/03/05/VIDEO_ID/video.mp4)."
        ),
    )
    checksum: Optional[str] = Field(
        None,
        description="Контрольная сумма файла (обычно sha256:...), опционально для MVP.",
    )
    size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Размер файла в байтах, если известен.",
    )
    comment_count: Optional[int] = Field(
        None,
        ge=0,
        description="Количество комментариев в comments.json (актуально только для comments_file).",
    )


class ManifestArtifacts(BaseModel):
    """Группа артефактов, необходимых DataProcessor.

    Обязательные ключи:
    - video_file
    - meta_file
    - comments_file (может быть None, если комментарии отключены флагом)
    """

    video_file: ArtifactInfo
    meta_file: ArtifactInfo
    comments_file: Optional[ArtifactInfo] = None


class FetcherManifest(BaseModel):
    """Контракт manifest.json между Fetcher и DataProcessor.

    См. раздел "3. Контракт manifest.json и версионирование" в BACKEND_CONTRACTS.md.
    """

    manifest_version: str = Field(
        "1.0",
        description="Версия контракта manifest.json (мажорная часть ломает совместимость).",
    )
    run_id: UUID = Field(
        ...,
        description="run_id, созданный Backend и используемый в Fetcher/DataProcessor.",
    )
    video_id: str = Field(
        ...,
        description=(
            "Нормализованный идентификатор видео (обычно совпадает с platform_video_id)."
        ),
    )
    platform: str = Field(
        ...,
        description='Платформа видео (например, "youtube").',
    )
    duration_seconds: float = Field(
        ...,
        gt=0,
        description="Длительность видео в секундах.",
    )
    storage_layout_version: str = Field(
        "1.0",
        description=(
            "Версия схемы путей в object storage (layout). "
            "Позволяет эволюционировать layout без смены логического контракта."
        ),
    )
    artifacts: ManifestArtifacts = Field(
        ...,
        description="Описание артефактов, необходимых для запуска DataProcessor.",
    )


__all__ = [
    "ArtifactInfo",
    "ManifestArtifacts",
    "FetcherManifest",
]


