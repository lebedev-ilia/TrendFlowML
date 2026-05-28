"""Webhooks для Fetcher.

Отправка уведомлений о завершении runs с HMAC-SHA256 подписью.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import httpx

from .config import settings

logger = logging.getLogger(__name__)


def compute_webhook_signature(payload: str, secret: str) -> str:
    """Вычислить HMAC-SHA256 подпись для webhook payload.

    Args:
        payload: JSON payload в виде строки
        secret: Секретный ключ для подписи

    Returns:
        Hex-строка подписи (sha256=...)
    """
    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


async def send_webhook(
    webhook_url: str,
    run_id: UUID | str,
    status: str,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    error: Optional[str] = None,
    max_retries: int = 3,
    retry_delays: list[int] = None,
) -> bool:
    """Отправить webhook уведомление о завершении run'а.

    Args:
        webhook_url: URL для отправки webhook
        run_id: UUID run'а
        status: Статус run'а (COMPLETED, FAILED, etc.)
        platform: Платформа видео (опционально)
        platform_video_id: ID видео на платформе (опционально)
        error: Сообщение об ошибке (если есть)
        max_retries: Максимальное количество попыток
        retry_delays: Задержки между попытками в секундах (exponential backoff)

    Returns:
        True если webhook успешно отправлен, False иначе
    """
    if retry_delays is None:
        retry_delays = [1, 5, 30]  # Exponential backoff: 1s, 5s, 30s

    # Формируем payload
    payload_data: dict[str, Any] = {
        "run_id": str(run_id),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if platform:
        payload_data["platform"] = platform
    if platform_video_id:
        payload_data["platform_video_id"] = platform_video_id
    if error:
        payload_data["error"] = error

    payload = json.dumps(payload_data, sort_keys=True)

    # Получаем webhook secret из настроек
    webhook_secret = getattr(settings, "webhook_secret", None)
    if not webhook_secret:
        logger.warning("webhook_secret not configured, webhook will be sent without signature")
        webhook_secret = ""

    # Вычисляем подпись
    signature = compute_webhook_signature(payload, webhook_secret) if webhook_secret else None

    # Формируем заголовки
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "Fetcher/1.0",
    }
    if signature:
        headers["X-Fetcher-Signature"] = signature
    headers["X-Fetcher-Event"] = f"run.{status.lower()}"
    headers["X-Fetcher-Timestamp"] = str(int(time.time()))

    # Отправляем webhook с retry logic
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook_url,
                    content=payload,
                    headers=headers,
                )
                response.raise_for_status()

                logger.info(
                    f"Webhook sent successfully for run {run_id} to {webhook_url} (attempt {attempt + 1})"
                )
                return True

        except httpx.HTTPStatusError as e:
            # 4xx ошибки не ретраим (кроме 429)
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                logger.error(
                    f"Webhook failed for run {run_id} with status {e.response.status_code}: {e.response.text}"
                )
                return False

            # Для 429 и 5xx делаем retry
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    f"Webhook failed for run {run_id} with status {e.response.status_code}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Webhook failed for run {run_id} after {max_retries} attempts: {e.response.text}"
                )
                return False

        except Exception as e:
            # Для других ошибок (network, timeout) делаем retry
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.warning(
                    f"Webhook error for run {run_id}: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Webhook failed for run {run_id} after {max_retries} attempts: {e}")
                return False

    return False


__all__ = ["send_webhook", "compute_webhook_signature"]

