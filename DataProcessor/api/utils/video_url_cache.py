"""
Кэш для видео, скачанных по video_url (Phase 3: Backend ↔ Fetcher).

При запросе с video_url DataProcessor скачивает файл в директорию кэша
(внутри allowed_video_paths) и использует полученный путь как video_path.
См. docs/PHASE3_ARTIFACTS_CONTRACT.md.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

from api.config import config

logger = logging.getLogger(__name__)


def get_video_cache_path(run_id: str, extension: str = "mp4") -> Path:
    """Путь к файлу в кэше по run_id (уникальное имя для избежания коллизий)."""
    cache_dir = Path(config.get_video_url_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{run_id}.{extension.lstrip('.')}"


async def download_video_url_to_cache(
    video_url: str,
    run_id: str,
    *,
    timeout_seconds: float = 300.0,
) -> Path:
    """
    Скачивает видео по URL в кэш и возвращает путь к файлу.

    Имя файла: {run_id}.mp4 (расширение из URL не используется для простоты).

    Args:
        video_url: URL для скачивания (например signed URL от Fetcher).
        run_id: UUID run'а (используется как имя файла).
        timeout_seconds: Таймаут HTTP-запроса.

    Returns:
        Path к скачанному файлу в кэше.

    Raises:
        httpx.HTTPStatusError: При 4xx/5xx от сервера.
        httpx.RequestError: При сетевой ошибке или таймауте.
    """
    out_path = get_video_cache_path(run_id, "mp4")
    cache_dir = out_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading video from URL to cache",
        extra={"run_id": run_id, "cache_path": str(out_path)},
    )
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        out_path.write_bytes(response.content)

    logger.info(
        "Video downloaded to cache",
        extra={"run_id": run_id, "path": str(out_path), "size_bytes": out_path.stat().st_size},
    )
    return out_path
