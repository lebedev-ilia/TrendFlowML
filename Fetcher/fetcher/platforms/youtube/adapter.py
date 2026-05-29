from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yt_dlp
from yt_dlp.utils import DownloadError

from fetcher.checksums import compute_sha256
from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.cookies import apply_cookiefile
from fetcher.db import session_scope
from fetcher.metrics import fetcher_youtube_403_total, fetcher_youtube_429_total
from fetcher.models import Artifact, ChannelMetadata, Comment, Video, VideoMetadata
from fetcher.pii import mask_pii
from fetcher.rate_limiter import acquire_token, acquire_video_lock, release_video_lock
from fetcher.proxies import get_next_proxy, record_proxy_result
from fetcher.snapshots import create_initial_snapshot_from_info
from fetcher.storage import storage_client
from fetcher.platforms.base import PlatformAdapter
from fetcher.services.youtube_data_client import (
    CommentDto,
    VideoMetadataDto,
    YouTubeDataClient,
)
from sqlalchemy.exc import IntegrityError


def _ffprobe_duration_seconds(path: Path) -> Optional[int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            return None
        return int(float((r.stdout or "").strip()))
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None


def _resolve_mock_sample_video_path(platform_video_id: str) -> Path:
    """Файл для FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD: {id}.mp4, затем sample_N.mp4, иначе единственный .mp4 в каталоге."""
    sample_dir = Path(
        settings.youtube_mock_sample_video_dir or "/tmp/fetcher_sample_videos"
    )
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Mock sample video directory not found: {sample_dir}")

    named = sample_dir / f"{platform_video_id}.mp4"
    if named.is_file():
        return named

    index = int(
        hashlib.sha256(platform_video_id.encode("utf-8")).hexdigest(),
        16,
    ) % max(settings.youtube_mock_sample_video_count, 1)
    legacy = sample_dir / f"sample_{index}.mp4"
    if legacy.is_file():
        return legacy

    mp4s = sorted(sample_dir.glob("*.mp4"))
    if len(mp4s) == 1:
        return mp4s[0]

    raise FileNotFoundError(
        f"Mock sample video not found for id={platform_video_id!r} under {sample_dir} "
        f"(expected {named.name} or {legacy.name}, or exactly one *.mp4)"
    )


def _synthetic_youtube_info_dict(
    platform_video_id: str, duration_sec: Optional[int]
) -> dict[str, Any]:
    dur = int(duration_sec) if duration_sec is not None else 120
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return {
        "id": platform_video_id,
        "title": f"mock:{platform_video_id}",
        "description": "Offline E2E fixture (FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD)",
        "duration": dur,
        "channel_id": "mock_channel",
        "channel": "mock_channel_title",
        "upload_date": today,
        "language": "en",
    }


class YouTubeAdapter(PlatformAdapter):
    """Скелет YouTubeAdapter.

    В текущем виде реализует только структуру методов и точки интеграции с
    БД и storage. Конкретные вызовы yt-dlp / YouTube API будут добавлены позже.
    """

    def __init__(self) -> None:
        # В будущем сюда можно передать:
        # - proxy‑pool
        # - rate‑лимитер
        # - logger
        self._data_client: YouTubeDataClient | None = None

    @property
    def data_client(self) -> YouTubeDataClient:
        if self._data_client is None:
            self._data_client = YouTubeDataClient()
        return self._data_client

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        """Загрузить метаданные видео и канала и сохранить их в БД/хранилище.

        - вызывает yt-dlp для получения `info_dict`;
        - создаёт/обновляет записи в таблицах `videos`, `video_metadata`, `channel_metadata`;
        - сохраняет `meta.json` в `video-analytics-raw` и регистрирует артефакт.
        """

        # 1) Circuit breaker: проверяем, не заблокирована ли операция
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")

        dto: VideoMetadataDto | None = None
        info: dict[str, Any]
        duration: int | None
        # Полностью офлайн: мок видео + синтетические meta (без yt-dlp / YouTube API).
        if settings.youtube_mock_video_download:
            platform_video_id = source
            sample_path = _resolve_mock_sample_video_path(platform_video_id)
            duration = _ffprobe_duration_seconds(sample_path)
            info = _synthetic_youtube_info_dict(platform_video_id, duration)
            if duration is None:
                duration = int(info["duration"])
        # Если включен режим YouTube Data API — используем его как источник правды.
        elif settings.youtube_data_enabled:
            dto = self.data_client.get_video_metadata(source)
            platform_video_id = dto.video_id
            info = dto.raw_json
            duration = dto.duration_seconds
        else:
            # 2) Rate limiting: metadata window per IP/proxy (MVP — один глобальный ключ)
            # Для MVP не учитываем IP/proxy_id, используем один ключ.
            if not acquire_token(
                key="rate:youtube:metadata:default",
                limit=settings.youtube_metadata_limit_per_window,
                window_sec=settings.youtube_metadata_window_sec,
            ):
                breaker.record_failure("rate_limit_exceeded")
                raise RuntimeError("YouTube metadata rate limit exceeded")

            # 3) Получаем info_dict через yt-dlp
            # get_next_proxy сам учитывает настройки enable_proxies / список proxies
            proxy = get_next_proxy()
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "proxy": proxy,
            }
            apply_cookiefile(ydl_opts)
            start_time = time.time()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                latency_ms = int((time.time() - start_time) * 1000)
                if proxy:
                    record_proxy_result(
                        proxy, success=True, operation="metadata", latency_ms=latency_ms
                    )
            except DownloadError as e:
                latency_ms = int((time.time() - start_time) * 1000)
                msg = str(e)
                error_reason = "unknown"
                if "HTTP Error 429" in msg:
                    fetcher_youtube_429_total.labels(operation="metadata").inc()
                    error_reason = "429"
                if "HTTP Error 403" in msg:
                    fetcher_youtube_403_total.labels(
                        operation="metadata", error_code="HTTP_403"
                    ).inc()
                    error_reason = "403"
                if proxy:
                    record_proxy_result(
                        proxy,
                        False,
                        operation="metadata",
                        latency_ms=latency_ms,
                        error_message=msg[:500],  # Ограничиваем длину сообщения
                    )
                breaker.record_failure(error_reason)
                raise

            platform_video_id = info.get("id")
            if not platform_video_id:
                breaker.record_failure("no_video_id")
                raise ValueError("yt-dlp did not return video id")
            duration = info.get("duration")

        # Записываем успех в circuit breaker
        breaker.record_success()

        # 3) Создаём/обновляем записи в БД
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .first()
            )
            if video is None:
                video = Video(platform="youtube", platform_video_id=platform_video_id)
                db.add(video)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    video = (
                        db.query(Video)
                        .filter(
                            Video.platform == "youtube",
                            Video.platform_video_id == platform_video_id,
                        )
                        .one()
                    )

            video.channel_id = info.get("channel_id") or video.channel_id
            if isinstance(duration, int):
                video.duration_seconds = duration

            # video_metadata
            vm: Optional[VideoMetadata] = (
                db.query(VideoMetadata).filter(VideoMetadata.video_id == video.id).first()
            )
            if vm is None:
                vm = VideoMetadata(video_id=video.id)
                db.add(vm)

            if dto is not None:
                vm.title = dto.title
                vm.description = dto.description
                vm.language = None
                vm.duration_seconds = dto.duration_seconds
                vm.published_at = dto.published_at
                vm.raw_json = dto.raw_json if settings.retain_raw_meta else None
            else:
                vm.title = info.get("title")
                vm.description = info.get("description")
                vm.language = info.get("language")
                if isinstance(duration, int):
                    vm.duration_seconds = duration

                upload_date = info.get("upload_date")  # формат YYYYMMDD
                if isinstance(upload_date, str) and len(upload_date) == 8:
                    try:
                        vm.published_at = datetime.strptime(
                            upload_date, "%Y%m%d"
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
                # Если retain_raw_meta=False, не сохраняем raw_json
                vm.raw_json = info if settings.retain_raw_meta else None

            # channel_metadata
            cm: Optional[ChannelMetadata] = (
                db.query(ChannelMetadata)
                .filter(ChannelMetadata.video_id == video.id)
                .first()
            )
            if cm is None:
                cm = ChannelMetadata(video_id=video.id)
                db.add(cm)

            if dto is not None:
                cm.channel_id = dto.channel_id
                cm.channel_title = dto.channel_title
                # Статистика по каналу YouTube Data API в этом клиенте пока не тянется.
            else:
                cm.channel_id = info.get("channel_id")
                cm.channel_title = info.get("channel")
                cm.subscriber_count = info.get("channel_follower_count")
                cm.video_count = info.get("channel_video_count")
                cm.view_count = info.get("channel_view_count")
                cm.raw_json = info.get("channel") or cm.raw_json

            db.flush()

        # 4) Создаём начальный snapshot (snapshot_index=0) на основе info_dict, если включено
        if settings.enable_snapshots:
            create_initial_snapshot_from_info("youtube", platform_video_id, info)

        # 5) Сохраняем meta.json в object storage и регистрируем артефакт
        # layout: raw/youtube/YYYY/MM/DD/VIDEO_ID/meta.json
        # Если retain_raw_meta=False, сохраняем только базовые поля (без description, tags и т.д.)
        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/youtube/{date_prefix}/{platform_video_id}/meta.json"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_meta"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}_meta.json"

        # Если retain_raw_meta=False, удаляем чувствительные поля из info / dto
        if settings.youtube_data_enabled:
            src = dto.raw_json
            meta_to_save = src.copy() if settings.retain_raw_meta else {
                "id": src.get("id"),
                "title": src.get("snippet", {}).get("title"),
                "duration": src.get("contentDetails", {}).get("duration"),
                "upload_date": src.get("snippet", {}).get("publishedAt"),
                "channel": src.get("snippet", {}).get("channelTitle"),
                "channel_id": src.get("snippet", {}).get("channelId"),
                "view_count": src.get("statistics", {}).get("viewCount"),
                "like_count": src.get("statistics", {}).get("likeCount"),
                "comment_count": src.get("statistics", {}).get("commentCount"),
            }
        else:
            meta_to_save = info.copy() if settings.retain_raw_meta else {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "upload_date": info.get("upload_date"),
                "channel": info.get("channel"),
                "channel_id": info.get("channel_id"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "comment_count": info.get("comment_count"),
                "channel_follower_count": info.get("channel_follower_count"),
                "channel_video_count": info.get("channel_video_count"),
                "channel_view_count": info.get("channel_view_count"),
            }

        tmp_path.write_text(json.dumps(meta_to_save, ensure_ascii=False), encoding="utf-8")
        # Размер и checksum meta.json до upload
        size_bytes = tmp_path.stat().st_size
        checksum_hex = compute_sha256(tmp_path)
        checksum = f"sha256:{checksum_hex}"
        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == "metadata_file",
                )
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="metadata_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def download_video(self, source: str, *, run_id: str) -> None:
        """Скачать видео и загрузить его в object storage.

        MVP‑реализация:
        - использует yt-dlp для скачивания лучшего потока ≤720p в /tmp;
        - (опционально) может вызывать ffmpeg для нормализации контейнера;
        - загружает файл в bucket `video-analytics-raw` по layout из STORAGE_LAYOUT.md;
        - регистрирует/обновляет артефакт `video_file` в таблице `artifacts`.
        """

        # Если включён мок‑режим — используем sample‑видео вместо реального скачивания.
        if settings.youtube_mock_video_download:
            self._download_video_mock(source)
            return

        # 1) Circuit breaker: проверяем, не заблокирована ли операция
        breaker = get_circuit_breaker("download")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for download operations")

        # 2) Rate limiting: download window per IP/proxy (MVP — один глобальный ключ)
        if not acquire_token(
            key="rate:youtube:download:default",
            limit=settings.youtube_download_limit_per_window,
            window_sec=settings.youtube_download_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("YouTube download rate limit exceeded")

        # 3) Получаем info_dict, чтобы узнать id и duration
        proxy = get_next_proxy()
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "proxy": proxy,
        }
        apply_cookiefile(ydl_opts)
        start_time = time.time()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
            latency_ms = int((time.time() - start_time) * 1000)
            if proxy:
                record_proxy_result(proxy, success=True, operation="download", latency_ms=latency_ms)
        except DownloadError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            msg = str(e)
            error_reason = "unknown"
            if "HTTP Error 429" in msg:
                fetcher_youtube_429_total.labels(operation="download").inc()
                error_reason = "429"
            if "HTTP Error 403" in msg:
                fetcher_youtube_403_total.labels(operation="download", error_code="HTTP_403").inc()
                error_reason = "403"
            if proxy:
                record_proxy_result(
                    proxy,
                    success=False,
                    operation="download",
                    latency_ms=latency_ms,
                    error_message=msg[:500],
                )
            breaker.record_failure(error_reason)
            raise

        platform_video_id = info.get("id")
        if not platform_video_id:
            breaker.record_failure("no_video_id")
            raise ValueError("yt-dlp did not return video id")

        # 3) Проверяем, не существует ли уже COMPLETED video_file
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .first()
            )
            if video is None:
                video = Video(platform="youtube", platform_video_id=platform_video_id)
                db.add(video)
                db.flush()

            existing_artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == "video_file",
                    Artifact.status == "COMPLETED",
                )
                .first()
            )
            if existing_artifact is not None:
                # Видео уже скачано и загружено — выходим.
                return

        # 4) Distributed lock перед скачиванием, чтобы избежать дубликатов
        # Для MVP используем один глобальный lock на видео.
        if not acquire_video_lock("youtube", platform_video_id):
            # Кто-то уже скачивает это видео — просто выходим, артефакт появится позже.
            return

        # 5) Скачиваем видео во временный файл через yt-dlp
        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_template = str(tmp_dir / f"{platform_video_id}.%(ext)s")

        proxy = get_next_proxy() if settings.enable_proxies else None
        ydl_download_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best",
            "outtmpl": tmp_template,
            "quiet": True,
            "merge_output_format": "mp4",
            "proxy": proxy,
        }
        apply_cookiefile(ydl_download_opts)
        download_start_time = time.time()
        try:
            with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
                info = ydl.extract_info(source, download=True)
            download_latency_ms = int((time.time() - download_start_time) * 1000)
            if proxy:
                record_proxy_result(proxy, success=True, operation="download", latency_ms=download_latency_ms)
        except DownloadError as e:
            download_latency_ms = int((time.time() - download_start_time) * 1000)
            msg = str(e)
            error_reason = "unknown"
            if "HTTP Error 429" in msg:
                fetcher_youtube_429_total.labels(operation="download").inc()
                error_reason = "429"
            if "HTTP Error 403" in msg:
                fetcher_youtube_403_total.labels(operation="download", error_code="HTTP_403").inc()
                error_reason = "403"
            if proxy:
                record_proxy_result(
                    proxy,
                    success=False,
                    operation="download",
                    latency_ms=download_latency_ms,
                    error_message=msg[:500],
                )
            breaker.record_failure(error_reason)
            # Снимаем lock при ошибке скачивания
            release_video_lock("youtube", platform_video_id)
            raise

        # yt-dlp вернёт фактический путь
        downloaded_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

        # 6) (Опционально) можно вызвать ffmpeg для дополнительной нормализации
        # Для MVP считаем, что yt-dlp с merge_output_format=mp4 достаточно.

        # 7) Загружаем файл в object storage
        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/youtube/{date_prefix}/{platform_video_id}/video.mp4"

        # Размер и checksum видео до upload
        try:
            size_bytes = downloaded_path.stat().st_size
            checksum_hex = compute_sha256(downloaded_path)
            checksum = f"sha256:{checksum_hex}"
        except FileNotFoundError:
            # В юнит‑тестах или при нестандартных сценариях файл может быть замокан или уже удалён.
            # Для таких случаев пропускаем вычисление размера и checksum.
            size_bytes = None
            checksum = None

        try:
            storage_client.upload_file(downloaded_path, bucket=settings.bucket_raw, key=storage_key)
            # Записываем успех в circuit breaker после успешного upload
            breaker.record_success()
        finally:
            try:
                downloaded_path.unlink()
            except FileNotFoundError:
                pass
            # Снимаем lock независимо от успеха upload (TTL защитит от зависаний).
            release_video_lock("youtube", platform_video_id)

        # 8) Обновляем/создаём артефакт video_file
        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == "video_file",
                )
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="video_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        """Загрузить комментарии к видео и сохранить их в БД/JSON (MVP).

        Ограничения:
        - опирается на то, что yt-dlp вернёт поле `comments` в info_dict (зависит от версии/опций);
        - для production потребуется более надёжная интеграция с YouTube API или расширенными опциями yt-dlp.
        """

        # 1) Circuit breaker: проверяем, не заблокирована ли операция
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")

        if settings.youtube_mock_video_download:
            platform_video_id = source
            comments_iter = []
        elif settings.youtube_data_enabled:
            platform_video_id = source
            comments_iter = self.data_client.iter_comments(
                video_id=platform_video_id,
                max_count=min(limit, settings.youtube_data_max_comments),
            )
        else:
            # 2) Получаем info_dict с комментариями (если поддерживается)
            proxy = get_next_proxy() if settings.enable_proxies else None
            ydl_opts = {
                "skip_download": True,
                "quiet": True,
                "proxy": proxy,
            }
            apply_cookiefile(ydl_opts)
            comments_start_time = time.time()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                comments_latency_ms = int((time.time() - comments_start_time) * 1000)
                if proxy:
                    record_proxy_result(
                        proxy,
                        success=True,
                        operation="comments",
                        latency_ms=comments_latency_ms,
                    )
            except DownloadError as e:
                comments_latency_ms = int((time.time() - comments_start_time) * 1000)
                msg = str(e)
                error_reason = "unknown"
                if "HTTP Error 429" in msg:
                    fetcher_youtube_429_total.labels(operation="comments").inc()
                    error_reason = "429"
                if "HTTP Error 403" in msg:
                    fetcher_youtube_403_total.labels(
                        operation="comments", error_code="HTTP_403"
                    ).inc()
                    error_reason = "403"
                if proxy:
                    record_proxy_result(
                        proxy,
                        success=False,
                        operation="comments",
                        latency_ms=comments_latency_ms,
                        error_message=msg[:500],
                    )
                breaker.record_failure(error_reason)
                raise

            platform_video_id = info.get("id")
            if not platform_video_id:
                breaker.record_failure("no_video_id")
                raise ValueError("yt-dlp did not return video id")

            comments = info.get("comments") or []
            if not comments:
                return
            comments = comments[:limit]
            comments_iter = comments

        # Записываем успех в circuit breaker
        breaker.record_success()

        # 3) Сохраняем в БД (таблица comments)
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one_or_none()
            )
            if video is None:
                # Возможна гонка с другим воркером, который уже создал Video.
                from sqlalchemy.exc import IntegrityError

                video = Video(platform="youtube", platform_video_id=platform_video_id)
                db.add(video)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    video = (
                        db.query(Video)
                        .filter(
                            Video.platform == "youtube",
                            Video.platform_video_id == platform_video_id,
                        )
                        .one()
                    )

            persisted_comments: list[dict] = []

            for item in comments_iter:
                if settings.youtube_data_enabled:
                    c: CommentDto = item
                    text = c.text_original or ""
                    filtered_text = mask_pii(text) if settings.enable_pii_filtering else text
                    comment_text = filtered_text if settings.retain_raw_comments else ""
                    raw_json = c.raw_json if settings.retain_raw_comments else None
                    like_count = c.like_count
                    reply_count = None
                    published_at = c.published_at
                else:
                    c = item
                    author = c.get("author")
                    text = c.get("text") or ""
                    filtered_text = mask_pii(text) if settings.enable_pii_filtering else text
                    like_count = c.get("like_count")
                    reply_count = c.get("reply_count")
                    published_at = None
                    ts = c.get("timestamp")
                    if isinstance(ts, int):
                        published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
                    comment_text = filtered_text if settings.retain_raw_comments else ""
                    raw_json = c if settings.retain_raw_comments else None

                db_comment = Comment(
                    video_id=video.id,
                    author=(c.author_display_name if settings.youtube_data_enabled else c.get("author")),
                    text=comment_text,
                    like_count=like_count,
                    reply_count=reply_count,
                    published_at=published_at,
                    raw_json=raw_json,
                )
                db.add(db_comment)

                # Для артефакта comments.json накапливаем компактное представление
                persisted_comments.append(
                    {
                        "author": (
                            c.author_display_name
                            if settings.youtube_data_enabled
                            else c.get("author")
                        ),
                        "text": text if settings.retain_raw_comments else "",
                        "like_count": like_count,
                        "reply_count": reply_count,
                        "published_at": (
                            published_at.isoformat() if published_at is not None else None
                        ),
                    }
                )

            db.flush()

        # 4) Сохраняем comments.json в storage и регистрируем артефакт comments_file
        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/youtube/{date_prefix}/{platform_video_id}/comments.json"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_comments"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}_comments.json"

        tmp_path.write_text(json.dumps(persisted_comments, ensure_ascii=False), encoding="utf-8")
        size_bytes = tmp_path.stat().st_size
        checksum_hex = compute_sha256(tmp_path)
        checksum = f"sha256:{checksum_hex}"

        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == "comments_file",
                )
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="comments_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def _download_video_mock(self, source: str) -> None:
        """Моковое скачивание YouTube‑видео для e2e/локальной разработки.

        Берёт локальный mp4 (см. ``_resolve_mock_sample_video_path``), копирует во временный файл
        и загружает в storage как обычный video_file артефакт.
        """
        platform_video_id = source
        sample_path = _resolve_mock_sample_video_path(platform_video_id)

        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/youtube/{date_prefix}/{platform_video_id}/video.mp4"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video_mock"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}.mp4"
        tmp_path.write_bytes(sample_path.read_bytes())

        try:
            size_bytes = tmp_path.stat().st_size
            checksum_hex = compute_sha256(tmp_path)
            checksum = f"sha256:{checksum_hex}"
        except FileNotFoundError:
            size_bytes = None
            checksum = None

        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one_or_none()
            )
            if video is None:
                video = Video(platform="youtube", platform_video_id=platform_video_id)
                db.add(video)
                db.flush()

            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == "video_file",
                )
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="video_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    # Примеры вспомогательных методов, которые будут реализованы позже:

    def _get_or_create_video(self, platform_video_id: str) -> Video:
        """Найти или создать запись Video по (platform, platform_video_id)."""
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(
                    Video.platform == "youtube",
                    Video.platform_video_id == platform_video_id,
                )
                .one_or_none()
            )
            if video is None:
                video = Video(platform="youtube", platform_video_id=platform_video_id)
                db.add(video)
                db.flush()
            return video

    def _ensure_artifact(
        self, video: Video, artifact_type: str, storage_path: str
    ) -> Artifact:
        """Создать или вернуть существующий артефакт."""
        with session_scope() as db:
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(
                    Artifact.video_id == video.id,
                    Artifact.artifact_type == artifact_type,
                )
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type=artifact_type,
                    storage_path=storage_path,
                )
                db.add(artifact)
                db.flush()
            return artifact


