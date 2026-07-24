from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

# YouTube quota resets at midnight Pacific Time (not UTC midnight).
# PDT (summer, Mar–Nov): UTC-7 → 07:00 UTC; PST (winter): UTC-8 → 08:00 UTC.
_PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.proxy import ProxyRotator
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow
from fetcher.services.youtube_data_client import (
    ChannelMetadataDto,
    QuotaExceededError,
    YouTubeAPIError,
    YouTubeDataClient,
    is_comments_disabled_error,
)


@dataclass
class YouTubeKeyState:
    api_key: str
    used_units: int = 0
    disabled_until: Optional[str] = None
    last_error: Optional[str] = None
    quota_date: Optional[str] = None


class YouTubeKeyPool:
    def __init__(
        self,
        api_keys: Iterable[str],
        *,
        state_path: Path | None = None,
        daily_quota_limit: int = 10_000,
        proxy_rotator: ProxyRotator | None = None,
    ) -> None:
        self.state_path = state_path
        self.daily_quota_limit = daily_quota_limit
        self.proxy_rotator = proxy_rotator
        self.states: Dict[str, YouTubeKeyState] = {
            key: YouTubeKeyState(api_key=key) for key in api_keys if key
        }
        if not self.states:
            raise ValueError("YouTubeKeyPool requires at least one API key")
        self._load()

    @staticmethod
    def _today_pacific() -> str:
        """Return today's date in Pacific Time (America/Los_Angeles).
        YouTube quota resets at midnight PT, so this is the correct boundary
        for rolling over used_units — NOT UTC date (which is 7-8h off)."""
        return datetime.now(_PACIFIC_TZ).date().isoformat()

    def _reset_if_new_day(self, state: YouTubeKeyState) -> bool:
        """Google resets each key's daily quota at midnight Pacific Time.
        Our persisted used_units must roll over at the same boundary, otherwise
        a key that ever crossed daily_quota_limit stays excluded from the pool
        forever (see bug report: all 49 keys permanently exhausted after ~2 days).
        Uses Pacific date (not UTC) so reset fires at 07:00/08:00 UTC, matching
        Google's actual reset time."""
        today = self._today_pacific()
        if state.quota_date != today:
            state.used_units = 0
            state.disabled_until = None
            state.last_error = None
            state.quota_date = today
            return True
        return False

    def _reset_all_if_new_day(self) -> None:
        changed = False
        for state in self.states.values():
            if self._reset_if_new_day(state):
                changed = True
        if changed:
            self._save()

    def get_client(self) -> YouTubeDataClient:
        state = self._select_key()
        remaining = max(self.daily_quota_limit - state.used_units, 1)
        rotator = self.proxy_rotator
        return YouTubeDataClient(
            api_key=state.api_key,
            daily_quota_limit=remaining,
            proxy=rotator.next() if rotator else None,
            on_proxy_success=rotator.record_success if rotator else None,
            on_proxy_failure=rotator.record_failure if rotator else None,
        )

    def record_success(self, api_key: str, units: int | None = None) -> None:
        state = self.states[api_key]
        self._reset_if_new_day(state)
        state.used_units += units or 0
        state.last_error = None
        self._save()

    def record_failure(self, api_key: str, error: Exception) -> None:
        if is_comments_disabled_error(error):
            return
        state = self.states[api_key]
        self._reset_if_new_day(state)
        state.last_error = str(error)[:500]
        err_str = str(error)
        # Баг найден 2026-07-24 (владелец: "проверь ещё раз fetcher main" — 27 попыток подряд,
        # 0 успехов за 10+ часов): "Consumer '...' has been suspended" — это ПОСТОЯННЫЙ бан ключа
        # Google (не дневная квота, не временный rate-limit), но раньше попадал в тот же 15-минутный
        # бэкофф, что 429/обычный 403. used_units у забаненного ключа почти всегда ~0 (он падает
        # раньше, чем успевает потратить квоту) -> _select_key() (сортировка по used_units) выбирает
        # ИМЕННО его первым, как только 15 минут проходят — а внешний retry snapshot-poll (90с/30мин)
        # почти всегда ждёт ДОЛЬШЕ 15 минут, то есть ключ гарантированно успевает разблокироваться и
        # выбраться СНОВА. Живое наблюдение: 20+ из 27 попыток подряд били в один и тот же
        # заблокированный ключ, здоровые ~47 из 49 ни разу не пробовались. Теперь suspended — тот же
        # длинный бэкофф, что и реальная квота (до полуночи Pacific) — гораздо безопаснее, чем 15 мин.
        is_suspended = "suspended" in err_str.lower()
        if isinstance(error, QuotaExceededError) or "quota" in err_str.lower() or is_suspended:
            # Disable until next Pacific midnight (= actual Google quota reset time).
            # Old code used 23:59:59 UTC which is wrong: it's 7-8h after Google resets,
            # so the key stays locked for up to 31h instead of being freed at reset time.
            pacific_now = datetime.now(_PACIFIC_TZ)
            next_pacific_midnight = (pacific_now + timedelta(days=1)).replace(
                hour=0, minute=0, second=5, microsecond=0
            )
            state.disabled_until = next_pacific_midnight.astimezone(timezone.utc).isoformat()
        elif "429" in err_str or "403" in err_str:
            state.disabled_until = (utcnow() + timedelta(minutes=15)).isoformat()
        else:
            # Generic error (network issue, invalid key, etc.) — short backoff to
            # prevent tight loops. If persistent, all keys eventually get disabled
            # and _select_key raises QuotaExceededError, exiting the discover loop.
            state.disabled_until = (utcnow() + timedelta(minutes=2)).isoformat()
        self._save()

    def _select_key(self) -> YouTubeKeyState:
        self._reset_all_if_new_day()
        now = utcnow()
        candidates = []
        for state in self.states.values():
            if state.disabled_until:
                disabled_until = datetime.fromisoformat(state.disabled_until)
                if disabled_until > now:
                    continue
            if state.used_units < self.daily_quota_limit:
                candidates.append(state)
        if not candidates:
            raise QuotaExceededError("All YouTube API keys are exhausted or disabled")
        return sorted(candidates, key=lambda item: item.used_units)[0]

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        text = self.state_path.read_text(encoding="utf-8").strip()
        if not text:
            # Пустой файл (например, write был прерван) — начинаем с чистого состояния.
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Повреждённый файл — игнорируем, начинаем заново без сохранённого состояния.
            return
        for key, raw in data.get("keys", {}).items():
            if key in self.states:
                self.states[key] = YouTubeKeyState(api_key=key, **{k: v for k, v in raw.items() if k != "api_key"})

    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"keys": {key: state.__dict__ for key, state in self.states.items()}}
        # Атомарная запись через temp-файл + replace — предотвращает появление 0-байтных файлов
        # при аварийном завершении процесса в момент write_text.
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def quota_stats(self) -> dict[str, int]:
        self._reset_all_if_new_day()
        now = utcnow()
        available = 0
        total_used = 0
        for state in self.states.values():
            total_used += int(state.used_units or 0)
            if state.disabled_until:
                disabled_until = datetime.fromisoformat(state.disabled_until)
                if disabled_until > now:
                    continue
            if int(state.used_units or 0) < self.daily_quota_limit:
                available += 1
        return {
            "keys_available": available,
            "keys_total": len(self.states),
            "quota_used_total": total_used,
        }


class YouTubeDiscoveryAdapter:
    platform = "youtube"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=True,
        downloads=True,
    )

    def __init__(self, key_pool: YouTubeKeyPool) -> None:
        self.key_pool = key_pool

    def discover(
        self,
        *,
        category: str,
        query: str,
        limit: int,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        time_interval: Optional[str] = None,
        relevance_language: Optional[str] = None,
        region_code: Optional[str] = None,
    ) -> Iterable[CollectedVideo]:
        collected = 0
        page_token: Optional[str] = None
        while collected < limit:
            channels: dict[str, ChannelMetadataDto] = {}
            client = self.key_pool.get_client()
            api_key = client.api_key
            try:
                search = client.search_videos(
                    query,
                    page_token=page_token,
                    max_results=min(50, limit - collected),
                    published_after=published_after,
                    published_before=published_before,
                    relevance_language=relevance_language,
                    region_code=region_code,
                    order="date" if published_after or published_before else "relevance",
                )
                video_ids = [item.video_id for item in search.items]
                metadata = client.get_videos_metadata_batch(video_ids)
                channel_ids = [item.channel_id for item in metadata if item.channel_id]
                if channel_ids:
                    channels = {
                        item.channel_id: item
                        for item in client.get_channels_metadata_batch(channel_ids)
                    }
                self.key_pool.record_success(api_key, client.quota_tracker.used_units)
            except (QuotaExceededError, YouTubeAPIError) as exc:
                self.key_pool.record_failure(api_key, exc)
                continue

            for item in metadata:
                channel = channels.get(item.channel_id)
                snapshot = self._snapshot_from_metadata(item, channel, snapshot_index=0)
                yield CollectedVideo(
                    platform=self.platform,
                    video_id=item.video_id,
                    url=f"https://www.youtube.com/watch?v={item.video_id}",
                    category=category,
                    query=query,
                    metadata=self._metadata_dict(item, channel),
                    snapshot_0=snapshot,
                    time_interval=time_interval,
                    discovered_at=utcnow(),
                    platform_capabilities=self.capabilities.__dict__,
                )
                collected += 1
                if collected >= limit:
                    break
            page_token = search.next_page_token
            if not page_token:
                break

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        client = self.key_pool.get_client()
        api_key = client.api_key
        try:
            metadata = client.get_video_metadata(video_id)
            channel = None
            if metadata.channel_id:
                channels = client.get_channels_metadata_batch([metadata.channel_id])
                if channels:
                    channel = channels[0]
            comments = [
                {
                    "commentId": comment.comment_id,
                    "text": comment.text_original,
                    "likeCount": comment.like_count,
                    "publishedAt": comment.published_at.isoformat(),
                    "authorName": comment.author_display_name,
                }
                for comment in client.iter_comments(video_id, max_count=comments_limit)
            ]
            self.key_pool.record_success(api_key, client.quota_tracker.used_units)
        except (QuotaExceededError, YouTubeAPIError) as exc:
            self.key_pool.record_failure(api_key, exc)
            raise
        snapshot = self._snapshot_from_metadata(metadata, channel, snapshot_index=snapshot_index)
        snapshot.comments = comments
        if snapshot_index > 0:
            from fetcher.dataset_collector.schemas import compact_follow_up_snapshot

            return compact_follow_up_snapshot(snapshot)
        return snapshot

    def collect_comments(self, video_id: str, *, comments_limit: int, attempts: int = 5) -> list[dict]:
        last_error: Exception | None = None
        for _ in range(max(attempts, 1)):
            client = self.key_pool.get_client()
            api_key = client.api_key
            try:
                comments = [
                    {
                        "text": comment.text_original,
                        "likeCount": comment.like_count,
                        "repliesCount": comment.replies_count,
                        "publishedAt": comment.published_at.isoformat().replace("+00:00", "Z"),
                        "authorName": comment.author_display_name,
                    }
                    for comment in client.iter_comments(video_id, max_count=comments_limit)
                ]
                self.key_pool.record_success(api_key, client.quota_tracker.used_units)
                return comments
            except (QuotaExceededError, YouTubeAPIError) as exc:
                if is_comments_disabled_error(exc):
                    self.key_pool.record_success(api_key, client.quota_tracker.used_units)
                    return []
                last_error = exc
                self.key_pool.record_failure(api_key, exc)
                continue
        if last_error is not None:
            raise last_error
        return []

    @staticmethod
    def _metadata_dict(item, channel: ChannelMetadataDto | None) -> dict:
        data = {
            "title": item.title,
            "description": item.description,
            "duration_seconds": item.duration_seconds,
            "publishedAt": item.published_at.isoformat(),
            "channelTitle": item.channel_title,
            "channel_id": item.channel_id,
            "raw": item.raw_json,
        }
        if channel:
            data["channel"] = channel.raw_json
        return data

    @staticmethod
    def _snapshot_from_metadata(item, channel: ChannelMetadataDto | None, *, snapshot_index: int) -> Snapshot:
        now = utcnow()
        return Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str(item.view_count),
            likeCount=str(item.like_count),
            commentCount=str(item.comment_count),
            subscriberCount=channel.subscriber_count if channel else None,
            videoCount=channel.video_count if channel else None,
            viewCount_channel=channel.view_count if channel else None,
            raw=(
                {"video": item.raw_json, "channel": channel.raw_json if channel else None}
                if snapshot_index == 0
                else {}
            ),
        )
