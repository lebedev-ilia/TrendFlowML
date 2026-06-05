from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


SourceProvider = Literal["api", "sdk", "merged"]


def _parse_upload_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


class PlatformVideoDto(BaseModel):
    """Каноническая модель метаданных видео для всех платформ."""

    video_id: str
    title: str = ""
    description: str = ""
    duration_seconds: Optional[int] = None
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    published_at: Optional[datetime] = None
    channel_id: str = ""
    channel_title: str = ""
    media_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    webpage_url: Optional[str] = None
    language: Optional[str] = None
    raw_json: dict = Field(default_factory=dict)
    source_provider: SourceProvider = "api"

    def merge_from(self, other: "PlatformVideoDto") -> "PlatformVideoDto":
        """Дополнить пустые поля значениями из другого источника (parallel mode)."""
        data = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        other_data = other.model_dump() if hasattr(other, "model_dump") else other.dict()
        for field, value in other_data.items():
            if field in {"raw_json", "source_provider"}:
                continue
            current = data.get(field)
            if current in (None, "", 0) and value not in (None, "", 0):
                data[field] = value
        merged_raw = dict(self.raw_json)
        merged_raw.update(other.raw_json)
        data["raw_json"] = merged_raw
        data["source_provider"] = "merged"
        if hasattr(PlatformVideoDto, "model_validate"):
            return PlatformVideoDto.model_validate(data)
        return PlatformVideoDto.parse_obj(data)

    def to_info_dict(self) -> dict[str, Any]:
        """Совместимость с legacy info_dict / snapshots."""
        upload_date = None
        if self.published_at:
            upload_date = self.published_at.strftime("%Y%m%d")
        return {
            "id": self.video_id,
            "title": self.title,
            "description": self.description,
            "duration": self.duration_seconds,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "upload_date": upload_date,
            "channel_id": self.channel_id,
            "channel": self.channel_title,
            "uploader": self.channel_title,
            "uploader_id": self.channel_id,
            "thumbnail": self.thumbnail_url,
            "url": self.media_url or self.webpage_url,
            "webpage_url": self.webpage_url,
            "language": self.language,
            **self.raw_json,
        }


class PlatformCommentDto(BaseModel):
    comment_id: str
    author_display_name: str = ""
    text_original: str = ""
    like_count: int = 0
    replies_count: int = 0
    published_at: Optional[datetime] = None
    raw_json: dict = Field(default_factory=dict)


def from_ytdlp(info: dict[str, Any], *, source_provider: SourceProvider = "sdk") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=str(info.get("id") or ""),
        title=str(info.get("title") or ""),
        description=str(info.get("description") or ""),
        duration_seconds=info.get("duration") if isinstance(info.get("duration"), int) else None,
        view_count=_safe_int(info.get("view_count")),
        like_count=_safe_int(info.get("like_count")),
        comment_count=_safe_int(info.get("comment_count")),
        published_at=_parse_upload_date(info.get("upload_date") or info.get("timestamp")),
        channel_id=str(info.get("channel_id") or info.get("uploader_id") or ""),
        channel_title=str(info.get("channel") or info.get("uploader") or ""),
        media_url=info.get("url"),
        thumbnail_url=info.get("thumbnail"),
        webpage_url=info.get("webpage_url") or info.get("original_url"),
        language=info.get("language"),
        raw_json=info,
        source_provider=source_provider,
    )


def from_tiktok_api(item: dict[str, Any], *, source_provider: SourceProvider = "api") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=str(item.get("id") or ""),
        title=str(item.get("title") or item.get("video_description") or ""),
        description=str(item.get("video_description") or item.get("title") or ""),
        duration_seconds=_safe_int(item.get("duration"), default=0) or None,
        view_count=_safe_int(item.get("view_count")),
        like_count=_safe_int(item.get("like_count")),
        comment_count=_safe_int(item.get("comment_count")),
        published_at=_parse_upload_date(item.get("create_time")),
        channel_id=str(item.get("creator_id") or item.get("open_id") or ""),
        channel_title=str(item.get("creator_nickname") or ""),
        media_url=item.get("share_url"),
        thumbnail_url=item.get("cover_image_url"),
        webpage_url=item.get("share_url"),
        raw_json=item,
        source_provider=source_provider,
    )


def from_tiktok_sdk(video: Any, *, source_provider: SourceProvider = "sdk") -> PlatformVideoDto:
    data = getattr(video, "as_dict", None) or {}
    if callable(data):
        data = data()
    if not isinstance(data, dict):
        data = {}
    stats = data.get("stats") or {}
    author = data.get("author") or {}
    return PlatformVideoDto(
        video_id=str(getattr(video, "id", None) or data.get("id") or ""),
        title=str(data.get("desc") or ""),
        description=str(data.get("desc") or ""),
        duration_seconds=_safe_int(data.get("duration"), default=0) or None,
        view_count=_safe_int(stats.get("playCount")),
        like_count=_safe_int(stats.get("diggCount")),
        comment_count=_safe_int(stats.get("commentCount")),
        published_at=_parse_upload_date(data.get("createTime")),
        channel_id=str(author.get("id") or ""),
        channel_title=str(author.get("nickname") or author.get("uniqueId") or ""),
        webpage_url=data.get("share_url"),
        raw_json=data,
        source_provider=source_provider,
    )


def from_instagram_graph(item: dict[str, Any], *, source_provider: SourceProvider = "api") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=str(item.get("id") or ""),
        title=str(item.get("caption") or "")[:200],
        description=str(item.get("caption") or ""),
        view_count=_safe_int(item.get("view_count") or item.get("play_count")),
        like_count=_safe_int(item.get("like_count")),
        comment_count=_safe_int(item.get("comments_count")),
        published_at=_parse_upload_date(item.get("timestamp")),
        media_url=item.get("media_url"),
        thumbnail_url=item.get("thumbnail_url"),
        webpage_url=item.get("permalink"),
        raw_json=item,
        source_provider=source_provider,
    )


def from_instaloader_post(post: Any, *, source_provider: SourceProvider = "sdk") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=str(getattr(post, "mediaid", None) or getattr(post, "shortcode", "") or ""),
        title=str(getattr(post, "title", None) or (getattr(post, "caption", None) or ""))[:200],
        description=str(getattr(post, "caption", None) or ""),
        view_count=_safe_int(getattr(post, "video_view_count", 0)),
        like_count=_safe_int(getattr(post, "likes", 0)),
        comment_count=_safe_int(getattr(post, "comments", 0)),
        published_at=getattr(post, "date_utc", None),
        channel_id=str(getattr(post, "owner_id", None) or ""),
        channel_title=str(getattr(post, "owner_username", None) or ""),
        media_url=getattr(post, "video_url", None) if getattr(post, "is_video", False) else None,
        thumbnail_url=getattr(post, "url", None),
        webpage_url=f"https://www.instagram.com/p/{getattr(post, 'shortcode', '')}/",
        raw_json={"shortcode": getattr(post, "shortcode", None)},
        source_provider=source_provider,
    )


def from_twitch_helix(item: dict[str, Any], *, source_provider: SourceProvider = "api") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=str(item.get("id") or ""),
        title=str(item.get("title") or ""),
        description=str(item.get("description") or ""),
        view_count=_safe_int(item.get("view_count")),
        published_at=_parse_upload_date(item.get("published_at") or item.get("created_at")),
        channel_id=str(item.get("user_id") or ""),
        channel_title=str(item.get("user_name") or item.get("user_login") or ""),
        thumbnail_url=item.get("thumbnail_url"),
        webpage_url=item.get("url"),
        raw_json=item,
        source_provider=source_provider,
    )


def from_youtube_api(dto: Any, *, source_provider: SourceProvider = "api") -> PlatformVideoDto:
    return PlatformVideoDto(
        video_id=dto.video_id,
        title=dto.title,
        description=dto.description,
        duration_seconds=dto.duration_seconds,
        view_count=dto.view_count,
        like_count=dto.like_count,
        comment_count=dto.comment_count,
        published_at=dto.published_at,
        channel_id=dto.channel_id,
        channel_title=dto.channel_title,
        raw_json=dto.raw_json,
        source_provider=source_provider,
    )


__all__ = [
    "PlatformCommentDto",
    "PlatformVideoDto",
    "SourceProvider",
    "from_instagram_graph",
    "from_instaloader_post",
    "from_tiktok_api",
    "from_tiktok_sdk",
    "from_twitch_helix",
    "from_youtube_api",
    "from_ytdlp",
]
