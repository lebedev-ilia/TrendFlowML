"""
Адаптер для преобразования V2 моделей (AnalysisJob) в формат, понятный DataProcessor.

DataProcessor ожидает legacy формат (run_id, video_id, platform_id, config_hash),
но мы хотим использовать новую доменную модель (workspaces, channels, analysis_jobs).

Phase 3: для ingestion-run'ов из Fetcher — build_ingestion_payload_from_fetcher формирует
payload с video_url (signed URL); Backend передаёт его в DataProcessor без скачивания.
См. docs/PHASE3_ARTIFACTS_CONTRACT.md, backend/docs/FETCHER_INTEGRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ..config import ResolvedPaths, Settings
from ..dbv2 import models as v2_models
from ..models import AnalysisProfile
from .fetcher_client import get_run_artifacts, get_run_manifest


def resolve_dataprocessor_global_config_path(
    settings: Settings, paths: ResolvedPaths
) -> Optional[Path]:
    """
    Путь к YAML для --global-config в DataProcessor main.py.

    Приоритет:
    1) ``<storage_root>/e2e_full_max/active_global_config`` (первая строка — абсолютный путь к YAML).
    2) Settings.dataprocessor_global_config_path (TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG).
    """
    marker = paths.storage_root / "e2e_full_max" / "active_global_config"
    if marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
        if raw:
            line = raw.splitlines()[0].strip()
            if line and not line.startswith("#"):
                p = Path(line).expanduser()
                if p.is_file():
                    return p.resolve()
    cfg = settings.dataprocessor_global_config_path
    if cfg:
        p = Path(str(cfg)).expanduser()
        if p.is_file():
            return p.resolve()
    return None


@dataclass
class DataProcessorPayload:
    """Payload для запуска DataProcessor в legacy формате."""

    run_id: str  # UUID строка (для обратной совместимости используем analysis_job.id)
    video_id: str  # external_video_id или id из Video
    platform_id: str  # из Channel.platform
    config_hash: str  # из ProcessingConfig или AnalysisProfile
    video_path: Path  # путь к видеофайлу
    profile_config: Dict[str, Any]  # полный JSON профиля


def prepare_dataprocessor_payload(
    db: Session,
    analysis_job: v2_models.AnalysisJob,
) -> DataProcessorPayload:
    """
    Преобразует AnalysisJob (v2) в формат, понятный DataProcessor (legacy).

    Args:
        db: SQLAlchemy session
        analysis_job: AnalysisJob из core.analysis_jobs

    Returns:
        DataProcessorPayload с параметрами для DataProcessor

    Raises:
        ValueError: если не удалось найти необходимые данные
    """
    # Загружаем связанные сущности
    video = db.query(v2_models.Video).filter(v2_models.Video.id == analysis_job.video_id).first()
    if not video:
        raise ValueError(f"Video not found: {analysis_job.video_id}")

    channel = db.query(v2_models.Channel).filter(v2_models.Channel.id == video.channel_id).first()
    if not channel:
        raise ValueError(f"Channel not found: {video.channel_id}")

    # Определяем platform_id
    platform_id = channel.platform

    # Определяем video_id (external_video_id или id)
    video_id = video.external_video_id or str(video.id)

    # Определяем video_path
    # TODO: В v2 нужно хранить путь к видеофайлу в Video.storage_path
    # Пока используем legacy VideoFile для обратной совместимости
    video_path = _resolve_video_path(db, video)

    # Определяем config_hash и profile_config
    # Временное решение: используем legacy AnalysisProfile по processing_config_id
    # TODO: Создать core.processing_configs и мигрировать на него
    config_hash, profile_config = _resolve_processing_config(db, analysis_job.processing_config_id)

    return DataProcessorPayload(
        run_id=str(analysis_job.id),  # Используем analysis_job.id как run_id для DataProcessor
        video_id=video_id,
        platform_id=platform_id,
        config_hash=config_hash,
        video_path=video_path,
        profile_config=profile_config,
    )


def _resolve_video_path(db: Session, video: v2_models.Video) -> Path:
    """
    Определяет путь к видеофайлу.

    Приоритет:
    1. Video.storage_path (если заполнено)
    2. Legacy VideoFile.object_key (для обратной совместимости)

    Args:
        db: SQLAlchemy session
        video: Video из core.videos

    Returns:
        Path к видеофайлу

    Raises:
        ValueError: если путь не найден
    """
    if video.storage_path:
        path = Path(video.storage_path)
        if path.exists():
            return path

    # Fallback: ищем в legacy VideoFile по checksum или external_video_id
    # TODO: Это временное решение, нужно мигрировать на Video.storage_path
    from ..models import VideoFile

    if video.checksum:
        file_row = db.query(VideoFile).filter(VideoFile.sha256_hex == video.checksum).first()
        if file_row and Path(file_row.object_key).exists():
            return Path(file_row.object_key)

    raise ValueError(f"Video file not found for video {video.id}")


def _resolve_processing_config(
    db: Session,
    processing_config_id: UUID,
) -> tuple[str, Dict[str, Any]]:
    """
    Определяет config_hash и profile_config по processing_config_id.

    Временное решение: используем legacy AnalysisProfile.
    TODO: Создать core.processing_configs и мигрировать на него.

    Args:
        db: SQLAlchemy session
        processing_config_id: UUID processing config (пока маппится на AnalysisProfile.id)

    Returns:
        Tuple (config_hash, profile_config)

    Raises:
        ValueError: если профиль не найден
    """
    # Временное решение: processing_config_id маппится на AnalysisProfile.id
    profile = db.query(AnalysisProfile).filter(AnalysisProfile.id == str(processing_config_id)).first()
    if not profile:
        raise ValueError(f"Processing config not found: {processing_config_id}")

    config_hash = profile.config_hash
    profile_config = dict(profile.config_json) if profile.config_json else {}

    # Нормализуем профиль (как в legacy коде)
    settings = Settings()
    paths = settings.resolve_paths()
    if "visual" not in profile_config:
        profile_config["visual"] = {"cfg_path": str(paths.visual_cfg_default)}
    if isinstance(profile_config.get("visual"), dict) and not profile_config["visual"].get("cfg_path"):
        profile_config["visual"]["cfg_path"] = str(paths.visual_cfg_default)
    if "processors" not in profile_config:
        profile_config["processors"] = {
            "audio": {"enabled": False, "required": False},
            "text": {"enabled": False, "required": False},
        }

    return config_hash, profile_config


def resolve_run_paths_v2(
    *,
    platform_id: str,
    video_id: str,
    analysis_job_id: UUID,
    result_store_base: Path,
) -> Dict[str, Path]:
    """
    Определяет пути для результатов анализа (v2).

    Использует тот же формат путей, что и legacy, для совместимости с DataProcessor.

    Args:
        platform_id: Platform ID (из Channel.platform)
        video_id: Video ID (external_video_id или id)
        analysis_job_id: AnalysisJob ID (используется как run_id)
        result_store_base: Base path для result store

    Returns:
        Dict с путями:
        - run_rs_path: Path к директории результатов
        - manifest_path: Path к manifest.json
        - state_events_path: Path к state_events.jsonl
    """
    run_id = str(analysis_job_id)  # Используем analysis_job_id как run_id
    run_rs_path = result_store_base / platform_id / video_id / run_id
    manifest_path = run_rs_path / "manifest.json"
    runs_root = result_store_base.parent
    state_events_path = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"

    return {
        "run_rs_path": run_rs_path,
        "manifest_path": manifest_path,
        "state_events_path": state_events_path,
    }


# -----------------------------------------------------------------------------
# Phase 3: Ingestion run из Fetcher → payload для DataProcessor
# -----------------------------------------------------------------------------


@dataclass
class IngestionPayloadFromFetcher:
    """
    Payload для запуска DataProcessor по run'у ингестии (Fetcher).

    Содержит данные из manifest и signed URL видео из GET .../artifacts.
    Передаётся в run_dataprocessor_async: либо video_url (вариант B — DataProcessor
    скачивает сам), либо video_path после скачивания в Backend (вариант A, fallback).
    """

    run_id: str
    platform_id: str
    video_id: str
    profile_config: Dict[str, Any]
    video_url: Optional[str] = None  # signed URL для video_file
    video_path: Optional[Path] = None  # локальный путь (если Backend уже скачал)


def build_ingestion_payload_from_fetcher(
    run_id: str,
    *,
    settings: Optional[Settings] = None,
) -> IngestionPayloadFromFetcher:
    """
    Формирует payload для DataProcessor по run_id ингестии из Fetcher.

    Запрашивает у Fetcher manifest и artifacts, извлекает signed URL для video_file.
    Не скачивает видео — вызывающая сторона может передать video_url в DataProcessor
    (вариант B) или скачать во временный файл и передать video_path (вариант A).

    Args:
        run_id: UUID run'а (IngestionRun.run_id).
        settings: Настройки Backend; если None — создаётся Settings().

    Returns:
        IngestionPayloadFromFetcher с platform_id, video_id, profile_config, video_url.

    Raises:
        ValueError: если run_id не UUID или нет video_file в artifacts.
        httpx.HTTPStatusError: при ошибке запроса к Fetcher.
    """
    s = settings or Settings()
    run_uuid = UUID(run_id)
    manifest = get_run_manifest(run_uuid, settings=s)
    platform_id = manifest.get("platform", "youtube")
    video_id = manifest.get("video_id", run_id)

    artifacts_resp = get_run_artifacts(run_uuid, settings=s)
    artifacts_list = artifacts_resp.get("artifacts") or []
    video_url: Optional[str] = None
    for item in artifacts_list:
        if item.get("artifact_type") == "video_file" and item.get("download_url"):
            video_url = item["download_url"]
            break
    if not video_url:
        raise ValueError(f"No video_file download_url in artifacts for run_id={run_id}")

    paths = s.resolve_paths()
    if resolve_dataprocessor_global_config_path(s, paths):
        profile_config = _ingestion_profile_with_global_config_yaml()
    else:
        profile_config = _default_ingestion_profile_config(s)
    return IngestionPayloadFromFetcher(
        run_id=run_id,
        platform_id=platform_id,
        video_id=video_id,
        profile_config=profile_config,
        video_url=video_url,
        video_path=None,
    )


def _default_ingestion_profile_config(settings: Settings) -> Dict[str, Any]:
    """Лёгкий profile_config для ingestion run'ов локального E2E."""
    paths = settings.resolve_paths()
    return {
        # Для E2E держим профиль намеренно лёгким: segmenter обязан пройти
        # end-to-end, а тяжёлые audio/text/visual пайплайны не блокируют smoke run.
        "config_hash": "ingestion-e2e-segmenter-only",
        "visual": {"cfg_path": str(paths.visual_cfg_default)},
        "processors": {
            "segmenter": {"enabled": True, "required": True},
            "audio": {"enabled": False, "required": False},
            "text": {"enabled": False, "required": False},
            "visual": {"enabled": False, "required": False},
        },
    }


def _ingestion_profile_with_global_config_yaml() -> Dict[str, Any]:
    """Параллельный profile к unified global_config: флаги процессоров (детали — в YAML)."""
    return {
        "config_hash": "ingestion-e2e-full-max-global-yaml",
        "visual": {},
        "processors": {
            "segmenter": {"enabled": True, "required": True},
            "audio": {"enabled": True, "required": False},
            "text": {"enabled": True, "required": False},
            "visual": {"enabled": True, "required": False},
        },
    }

