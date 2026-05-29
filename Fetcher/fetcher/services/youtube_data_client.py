from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import httpx
from pydantic import BaseModel, ValidationError

from fetcher.config import settings
from fetcher.metrics import fetcher_youtube_403_total, fetcher_youtube_429_total


BASE_URL = "https://youtube.googleapis.com/youtube/v3"


class VideoNotFoundError(Exception):
    """Видео не найдено в YouTube Data API."""


class QuotaExceededError(Exception):
    """Превышен суточный лимит квоты YouTube Data API."""


class YouTubeAPIError(Exception):
    """Общий класс ошибок YouTube Data API."""


class VideoMetadataDto(BaseModel):
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    duration_seconds: int
    view_count: int
    like_count: int
    comment_count: int
    published_at: datetime
    raw_json: dict


class CommentDto(BaseModel):
    comment_id: str
    author_display_name: str
    text_original: str
    like_count: int
    replies_count: int = 0
    published_at: datetime
    updated_at: Optional[datetime] = None
    raw_json: dict


class YouTubeSearchItemDto(BaseModel):
    video_id: str
    title: str = ""
    channel_id: str = ""
    channel_title: str = ""
    published_at: Optional[datetime] = None
    raw_json: dict


class YouTubeSearchResult(BaseModel):
    items: List[YouTubeSearchItemDto]
    next_page_token: Optional[str] = None
    raw_json: dict


class ChannelMetadataDto(BaseModel):
    channel_id: str
    title: str = ""
    subscriber_count: Optional[int] = None
    video_count: Optional[int] = None
    view_count: Optional[int] = None
    raw_json: dict


@dataclass
class QuotaTracker:
    """Простейший трекер квоты в памяти для одного процесса.

    На уровне одного контейнера этого достаточно, для горизонтального
    масштабирования можно будет вынести в Redis/БД.
    """

    daily_limit: int
    used_units: int = 0
    reset_at: Optional[datetime] = None

    def _ensure_window(self) -> None:
        now = datetime.now(timezone.utc)
        if self.reset_at is None or now >= self.reset_at:
            # Окно начинается в полночь UTC и длится до следующей полуночи.
            self.reset_at = datetime(
                year=now.year, month=now.month, day=now.day, tzinfo=timezone.utc
            ).replace(hour=23, minute=59, second=59)
            self.used_units = 0

    def consume(self, units: int) -> None:
        self._ensure_window()
        if self.used_units + units > self.daily_limit:
            raise QuotaExceededError(
                f"YouTube Data API daily quota exceeded: "
                f"used={self.used_units}, limit={self.daily_limit}, requested={units}"
            )
        self.used_units += units


class YouTubeDataClient:
    """Клиент для YouTube Data API v3.

    Оборачивает httpx‑клиент, реализует базовый retry с backoff и трекинг квоты.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        rate_limit_rps: Optional[int] = None,
        daily_quota_limit: Optional[int] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.youtube_data_api_key
        if not self.api_key:
            raise ValueError("YouTube Data API key is not configured")

        self.timeout = timeout
        client_kwargs: dict[str, object] = {"timeout": timeout}
        if proxy:
            client_kwargs["proxy"] = proxy
        self.client = httpx.Client(**client_kwargs)
        self.rate_limit_rps = rate_limit_rps or settings.youtube_rate_limit_rps
        self._min_interval = 1.0 / float(self.rate_limit_rps) if self.rate_limit_rps > 0 else 0.0
        self._last_request_ts: float = 0.0

        self.quota_tracker = QuotaTracker(
            daily_limit=daily_quota_limit or settings.youtube_daily_quota_limit
        )
        # Простой in‑memory кэш метадаты в пределах одного процесса.
        self._metadata_cache: Dict[str, Tuple[VideoMetadataDto, datetime]] = {}
        cache_ttl_seconds = getattr(settings, "youtube_metadata_cache_ttl_seconds", 24 * 60 * 60)
        if not isinstance(cache_ttl_seconds, (int, float)):
            cache_ttl_seconds = 24 * 60 * 60
        self._metadata_cache_ttl = timedelta(seconds=cache_ttl_seconds)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def get_video_metadata(self, video_id: str) -> VideoMetadataDto:
        """Получить метаданные видео через videos.list."""
        cached = self._metadata_cache.get(video_id)
        if cached is not None:
            dto, fetched_at = cached
            if datetime.now(timezone.utc) - fetched_at < self._metadata_cache_ttl:
                return dto

        params = {
            "part": "snippet,contentDetails,statistics,status,recordingDetails",
            "id": video_id,
            "key": self.api_key,
        }

        data = self._request_json("videos", params=params, quota_units=1, operation="metadata")
        items = data.get("items") or []
        if not items:
            raise VideoNotFoundError(video_id)

        item = items[0]
        snippet = item.get("snippet") or {}
        content_details = item.get("contentDetails") or {}
        statistics = item.get("statistics") or {}

        duration_iso = content_details.get("duration") or "PT0S"
        duration_seconds = self._parse_iso8601_duration(duration_iso)

        published_at_str = snippet.get("publishedAt")
        published_at = (
            datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
            if isinstance(published_at_str, str)
            else datetime.now(timezone.utc)
        )

        dto = VideoMetadataDto(
            video_id=item.get("id") or video_id,
            title=snippet.get("title") or "",
            description=snippet.get("description") or "",
            channel_id=snippet.get("channelId") or "",
            channel_title=snippet.get("channelTitle") or "",
            duration_seconds=duration_seconds,
            view_count=int(statistics.get("viewCount") or 0),
            like_count=int(statistics.get("likeCount") or 0),
            comment_count=int(statistics.get("commentCount") or 0),
            published_at=published_at,
            raw_json=item,
        )
        # Обновляем кэш
        self._metadata_cache[video_id] = (dto, datetime.now(timezone.utc))
        return dto

    def get_videos_metadata_batch(self, video_ids: Iterable[str]) -> List[VideoMetadataDto]:
        """Получить метаданные видео батчами YouTube videos.list до 50 ID."""
        ids = [video_id for video_id in dict.fromkeys(video_ids) if video_id]
        result: List[VideoMetadataDto] = []
        for start in range(0, len(ids), 50):
            chunk = ids[start : start + 50]
            if not chunk:
                continue
            data = self._request_json(
                "videos",
                params={
                    "part": "snippet,contentDetails,statistics,status,recordingDetails",
                    "id": ",".join(chunk),
                    "key": self.api_key,
                },
                quota_units=1,
                operation="metadata",
            )
            for item in data.get("items") or []:
                dto = self._parse_video_metadata_item(item)
                self._metadata_cache[dto.video_id] = (dto, datetime.now(timezone.utc))
                result.append(dto)
        return result

    def search_videos(
        self,
        query: str,
        *,
        page_token: Optional[str] = None,
        max_results: int = 50,
        order: str = "relevance",
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        relevance_language: Optional[str] = None,
        region_code: Optional[str] = None,
        safe_search: str = "none",
    ) -> YouTubeSearchResult:
        """Search YouTube videos with search.list. Costs 100 quota units."""
        params: dict[str, object] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(max_results, 50)),
            "order": order,
            "safeSearch": safe_search,
            "key": self.api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        if published_after:
            params["publishedAfter"] = published_after.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if published_before:
            params["publishedBefore"] = published_before.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if relevance_language:
            params["relevanceLanguage"] = relevance_language
        if region_code:
            params["regionCode"] = region_code

        data = self._request_json("search", params=params, quota_units=100, operation="search")
        items = []
        for item in data.get("items") or []:
            video_id = ((item.get("id") or {}).get("videoId")) or ""
            if not video_id:
                continue
            snippet = item.get("snippet") or {}
            published_at = snippet.get("publishedAt")
            items.append(
                YouTubeSearchItemDto(
                    video_id=video_id,
                    title=snippet.get("title") or "",
                    channel_id=snippet.get("channelId") or "",
                    channel_title=snippet.get("channelTitle") or "",
                    published_at=(
                        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                        if isinstance(published_at, str)
                        else None
                    ),
                    raw_json=item,
                )
            )
        return YouTubeSearchResult(
            items=items,
            next_page_token=data.get("nextPageToken"),
            raw_json=data,
        )

    def get_channels_metadata_batch(self, channel_ids: Iterable[str]) -> List[ChannelMetadataDto]:
        """Получить статистику каналов батчами channels.list до 50 ID."""
        ids = [channel_id for channel_id in dict.fromkeys(channel_ids) if channel_id]
        result: List[ChannelMetadataDto] = []
        for start in range(0, len(ids), 50):
            chunk = ids[start : start + 50]
            if not chunk:
                continue
            data = self._request_json(
                "channels",
                params={
                    "part": "snippet,statistics",
                    "id": ",".join(chunk),
                    "key": self.api_key,
                },
                quota_units=1,
                operation="channels",
            )
            for item in data.get("items") or []:
                statistics = item.get("statistics") or {}
                snippet = item.get("snippet") or {}
                hidden_subs = bool(statistics.get("hiddenSubscriberCount"))
                result.append(
                    ChannelMetadataDto(
                        channel_id=item.get("id") or "",
                        title=snippet.get("title") or "",
                        subscriber_count=(
                            None if hidden_subs else int(statistics.get("subscriberCount") or 0)
                        ),
                        video_count=int(statistics.get("videoCount") or 0),
                        view_count=int(statistics.get("viewCount") or 0),
                        raw_json=item,
                    )
                )
        return result

    def iter_comments(self, video_id: str, *, max_count: int) -> Iterator[CommentDto]:
        """Итерировать комментарии к видео через commentThreads.list.

        Если у видео отключены комментарии, YouTube API возвращает 403 с сообщением
        "videoId parameter has disabled comments" — в этом случае считаем, что
        комментариев нет, и завершаем без ошибки.
        """
        consumed = 0
        page_token: Optional[str] = None

        while True:
            if consumed >= max_count:
                return

            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "key": self.api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                data = self._request_json(
                    "commentThreads",
                    params=params,
                    quota_units=1,
                    operation="comments",
                )
            except YouTubeAPIError as exc:
                msg = str(exc)
                if "has disabled comments" in msg:
                    # Комментарии отключены — возвращаем пустой результат без ошибки.
                    return
                raise
            items = data.get("items") or []
            if not items:
                return

            for item in items:
                try:
                    dto = self._parse_comment_thread(item)
                except ValidationError:
                    # Пропускаем некорректные записи, не заваливая весь pipeline
                    continue
                yield dto
                consumed += 1
                if consumed >= max_count:
                    return

            page_token = data.get("nextPageToken")
            if not page_token:
                return

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _request_json(
        self,
        path: str,
        *,
        params: dict[str, object],
        quota_units: int,
        operation: str,
    ) -> dict:
        """Сделать HTTP‑запрос к YouTube Data API с retry и backoff."""
        url = f"{BASE_URL}/{path}"
        self.quota_tracker.consume(quota_units)

        # Простое client-side rate limiting
        if self._min_interval > 0:
            now = time.time()
            elapsed = now - self._last_request_ts
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

        backoff_seconds = [1, 2, 4]
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate([0, *backoff_seconds]):
            if delay > 0:
                time.sleep(delay)

            try:
                self._last_request_ts = time.time()
                resp = self.client.get(url, params=params)
            except httpx.RequestError as exc:
                last_exc = exc
                # retriable сетевые ошибки — пробуем ещё раз
                continue

            # Обработка rate limit / quota
            if resp.status_code == 429:
                fetcher_youtube_429_total.labels(operation=operation).inc()
                last_exc = YouTubeAPIError(f"429 Too Many Requests: {resp.text[:500]}")
                continue

            if resp.status_code == 403:
                fetcher_youtube_403_total.labels(operation=operation, error_code="HTTP_403").inc()
                last_exc = YouTubeAPIError(f"403 Forbidden: {resp.text[:500]}")
                # Часть 403 может быть неретраибельной (quotaExceeded), но для простоты
                # ограничимся несколькими попытками.
                continue

            if 500 <= resp.status_code < 600:
                last_exc = YouTubeAPIError(
                    f"{resp.status_code} Server error from YouTube: {resp.text[:500]}"
                )
                continue

            if resp.status_code != 200:
                raise YouTubeAPIError(
                    f"Unexpected status {resp.status_code} from YouTube Data API: {resp.text[:500]}"
                )

            try:
                data = resp.json()
            except ValueError as exc:
                raise YouTubeAPIError("Failed to decode JSON from YouTube Data API") from exc

            # В случае, если сама API вернула ошибку в теле
            error = data.get("error")
            if error:
                message = error.get("message") or "Unknown error"
                raise YouTubeAPIError(f"YouTube Data API error: {message}")

            return data

        # Если дошли сюда — все попытки исчерпаны
        if isinstance(last_exc, QuotaExceededError):
            raise last_exc
        raise YouTubeAPIError(f"Failed to call YouTube Data API after retries: {last_exc}")

    @staticmethod
    def _parse_iso8601_duration(value: str) -> int:
        """Очень упрощённый парсер ISO8601‑длительности вида PT#H#M#S."""
        if not isinstance(value, str) or not value.startswith("PT"):
            return 0

        # Пример: PT1H2M3S, PT15M33S, PT2H, PT45S
        seconds = 0
        num = ""
        for ch in value[2:]:
            if ch.isdigit():
                num += ch
                continue
            if not num:
                continue
            if ch == "H":
                seconds += int(num) * 3600
            elif ch == "M":
                seconds += int(num) * 60
            elif ch == "S":
                seconds += int(num)
            num = ""
        return seconds

    @classmethod
    def _parse_video_metadata_item(cls, item: dict) -> VideoMetadataDto:
        snippet = item.get("snippet") or {}
        content_details = item.get("contentDetails") or {}
        statistics = item.get("statistics") or {}
        duration_iso = content_details.get("duration") or "PT0S"
        published_at_str = snippet.get("publishedAt")
        published_at = (
            datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
            if isinstance(published_at_str, str)
            else datetime.now(timezone.utc)
        )
        return VideoMetadataDto(
            video_id=item.get("id") or "",
            title=snippet.get("title") or "",
            description=snippet.get("description") or "",
            channel_id=snippet.get("channelId") or "",
            channel_title=snippet.get("channelTitle") or "",
            duration_seconds=cls._parse_iso8601_duration(duration_iso),
            view_count=int(statistics.get("viewCount") or 0),
            like_count=int(statistics.get("likeCount") or 0),
            comment_count=int(statistics.get("commentCount") or 0),
            published_at=published_at,
            raw_json=item,
        )

    @staticmethod
    def _parse_comment_thread(item: dict) -> CommentDto:
        """Распарсить одну commentThread запись в CommentDto."""
        snippet = (item.get("snippet") or {}).get("topLevelComment") or {}
        top_snippet = snippet.get("snippet") or {}
        thread_snippet = item.get("snippet") or {}

        published_at = top_snippet.get("publishedAt")
        updated_at = top_snippet.get("updatedAt")

        published_dt = (
            datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            if isinstance(published_at, str)
            else datetime.now(timezone.utc)
        )
        updated_dt = (
            datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if isinstance(updated_at, str)
            else None
        )

        return CommentDto(
            comment_id=(snippet.get("id") or item.get("id") or ""),
            author_display_name=top_snippet.get("authorDisplayName") or "",
            text_original=top_snippet.get("textOriginal") or "",
            like_count=int(top_snippet.get("likeCount") or 0),
            replies_count=int(thread_snippet.get("totalReplyCount") or 0),
            published_at=published_dt,
            updated_at=updated_dt,
            raw_json=item,
        )


__all__ = [
    "YouTubeDataClient",
    "VideoMetadataDto",
    "CommentDto",
    "YouTubeSearchItemDto",
    "YouTubeSearchResult",
    "ChannelMetadataDto",
    "VideoNotFoundError",
    "QuotaExceededError",
    "YouTubeAPIError",
]

