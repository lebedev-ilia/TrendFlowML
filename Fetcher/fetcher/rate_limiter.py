from __future__ import annotations

"""Redis-based rate limiting and distributed locks for Fetcher.

Реализация следует дизайну из `Fetcher/docs/RATE_LIMITING_AND_LOCKS.md`:
- простейший счетчик + TTL для rate limiting;
- SET NX EX для distributed locks.
"""

from typing import Optional

import redis

from .config import settings


_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Лениво инициализировать Redis‑клиент на основе settings.redis_url.

    Поддерживает TLS через rediss:// URL или настройки redis_ssl.
    """
    global _redis_client
    if _redis_client is None:
        # Определяем SSL из URL или настроек
        use_ssl = settings.redis_ssl
        if settings.redis_url.startswith("rediss://"):
            use_ssl = True
        elif settings.redis_url.startswith("redis://"):
            use_ssl = False

        # Парсим SSL cert_reqs
        ssl_cert_reqs_map = {
            "none": None,
            "optional": "optional",
            "required": "required",
        }
        ssl_cert_reqs = ssl_cert_reqs_map.get(settings.redis_ssl_cert_reqs.lower(), "required")

        if use_ssl:
            _redis_client = redis.from_url(
                settings.redis_url,
                ssl=use_ssl,
                ssl_cert_reqs=ssl_cert_reqs,
            )
        else:
            _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def acquire_token(key: str, limit: int, window_sec: int) -> bool:
    """Простейший rate‑лимитер (fixed window).

    Инкрементирует счетчик в Redis и устанавливает TTL для первого инкремента.
    Возвращает True, если текущий счётчик не превышает лимит.
    При любых ошибках Redis — ведём себя консервативно и возвращаем True,
    чтобы не ломать ingestion, но логика ошибок может быть ужесточена позже.
    """
    client = get_redis_client()
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, window_sec)
        return count <= limit
    except Exception:
        # TODO: добавить логирование и отдельные метрики ошибок Redis
        return True


def acquire_video_lock(platform: str, video_id: str, ttl_sec: int = 1800) -> bool:
    """Получить distributed lock для скачивания видео.

    Ключ: lock:video:{platform}:{platform_video_id}
    Используется перед download_video, чтобы избежать параллельных скачиваний.
    """
    client = get_redis_client()
    key = f"lock:video:{platform}:{video_id}"
    try:
        # SET NX EX — атомарная установка lock с TTL
        return bool(client.set(key, "1", nx=True, ex=ttl_sec))
    except Exception:
        # При проблемах с Redis не блокируем скачивание, но это можно усилить позже.
        return True


def release_video_lock(platform: str, video_id: str) -> None:
    """Снять lock для видео (опционально, можно полагаться на TTL)."""
    client = get_redis_client()
    key = f"lock:video:{platform}:{video_id}"
    try:
        client.delete(key)
    except Exception:
        # Ошибки при снятии lock не критичны.
        return


def acquire_artifact_lock(video_id: str, artifact_type: str, ttl_sec: int = 600) -> bool:
    """Получить distributed lock для upload'а артефакта.

    Ключ: lock:artifact:{video_id}:{artifact_type}
    Предотвращает двойной upload одного и того же артефакта.
    """
    client = get_redis_client()
    key = f"lock:artifact:{video_id}:{artifact_type}"
    try:
        return bool(client.set(key, "1", nx=True, ex=ttl_sec))
    except Exception:
        return True


def release_artifact_lock(video_id: str, artifact_type: str) -> None:
    """Снять lock для артефакта."""
    client = get_redis_client()
    key = f"lock:artifact:{video_id}:{artifact_type}"
    try:
        client.delete(key)
    except Exception:
        return


__all__ = [
    "get_redis_client",
    "acquire_token",
    "acquire_video_lock",
    "release_video_lock",
    "acquire_artifact_lock",
    "release_artifact_lock",
]


