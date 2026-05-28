"""
Fetcher service package.

Здесь живут:
- конфигурация Fetcher;
- модели БД и data-access слой;
- storage‑клиент для S3/MinIO;
- платформенные адаптеры (YouTube, TikTok, ...);
- orchestration/state machine и воркеры ingestion‑pipeline’а.

Код должен соответствовать контрактам из `Fetcher/docs/*.md`.
"""

__all__ = []


