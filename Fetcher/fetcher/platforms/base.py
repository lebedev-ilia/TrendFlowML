from __future__ import annotations

from abc import ABC, abstractmethod


class PlatformAdapter(ABC):
    """Базовый интерфейс адаптера платформы для Fetcher.

    Должен реализовывать операции:
    - fetch_metadata
    - download_video
    - fetch_comments

    Контракт описан в `Fetcher/docs/PLATFORM_ADAPTERS.md`.
    """

    @abstractmethod
    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        """Загрузить и сохранить метаданные видео и канала."""

    @abstractmethod
    def download_video(self, source: str, *, run_id: str) -> None:
        """Скачать и загрузить видео в object storage."""

    @abstractmethod
    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        """Загрузить комментарии и сохранить их в БД/JSON."""


__all__ = ["PlatformAdapter"]


