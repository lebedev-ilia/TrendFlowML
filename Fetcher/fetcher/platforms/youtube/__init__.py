"""
Скелет YouTubeAdapter для Fetcher.

Полная реализация будет использовать:
- yt-dlp для получения метаданных и скачивания видео;
- proxy‑pool и rate limiting (см. RATE_LIMITING_AND_LOCKS.md);
- модели БД и storage‑клиент Fetcher.
"""

from .adapter import YouTubeAdapter

__all__ = ["YouTubeAdapter"]


