"""API Authentication и Rate Limiting для Fetcher API.

Поддерживает:
- API Key authentication (X-API-Key header или api_key query parameter)
- IP-based и API key-based rate limiting
- Rate limit headers (X-RateLimit-*)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import settings
from .rate_limiter import acquire_token, get_redis_client

logger = logging.getLogger(__name__)


# Список путей, которые не требуют аутентификации
PUBLIC_PATHS = {
    "/",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
}


def get_api_key_from_request(request: Request) -> Optional[str]:
    """Извлечь API key из запроса.

    Проверяет:
    1. X-API-Key header
    2. api_key query parameter

    Args:
        request: FastAPI Request объект

    Returns:
        API key или None если не найден
    """
    # Проверяем header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    # Проверяем query parameter
    api_key = request.query_params.get("api_key")
    if api_key:
        return api_key

    return None


def verify_api_key(api_key: str) -> bool:
    """Проверить валидность API key.

    Args:
        api_key: API key для проверки

    Returns:
        True если API key валиден, False иначе
    """
    # Получаем список валидных API keys из настроек
    valid_keys = getattr(settings, "api_keys", None)
    if not valid_keys:
        # Если API keys не настроены, аутентификация отключена
        return True

    # Поддерживаем как список, так и строку (comma-separated)
    if isinstance(valid_keys, str):
        valid_keys = [k.strip() for k in valid_keys.split(",")]

    return api_key in valid_keys


class APIAuthMiddleware(BaseHTTPMiddleware):
    """Middleware для API authentication и rate limiting."""

    def __init__(self, app, require_auth: bool = True):
        """Инициализировать middleware.

        Args:
            app: ASGI приложение
            require_auth: Требовать ли аутентификацию (если False, проверка пропускается если API keys не настроены)
        """
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next):
        """Обработать запрос: проверка аутентификации и rate limiting."""
        # Пропускаем публичные пути
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Проверка аутентификации
        api_key = get_api_key_from_request(request)
        api_keys_configured = bool(getattr(settings, "api_keys", None))

        if self.require_auth and api_keys_configured:
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Provide X-API-Key header or api_key query parameter.",
                )

            if not verify_api_key(api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )

        # Rate limiting
        rate_limit_key = api_key or request.client.host if request.client else "unknown"
        rate_limit_key = f"api:rate_limit:{rate_limit_key}"

        # Получаем лимиты из настроек
        api_rate_limit = getattr(settings, "api_rate_limit_per_minute", 60)
        api_rate_limit_window = 60  # 1 минута

        # Проверяем rate limit
        allowed = acquire_token(rate_limit_key, api_rate_limit, api_rate_limit_window)

        if not allowed:
            # Вычисляем время до сброса (TTL ключа)
            redis_client = get_redis_client()
            try:
                ttl = redis_client.ttl(rate_limit_key)
                reset_time = int(time.time()) + (ttl if ttl > 0 else api_rate_limit_window)
            except Exception:
                reset_time = int(time.time()) + api_rate_limit_window

            response = Response(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content='{"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Rate limit exceeded"}}',
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(api_rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )
            return response

        # Выполняем запрос
        response = await call_next(request)

        # Добавляем rate limit headers
        redis_client = get_redis_client()
        try:
            current_count = redis_client.get(rate_limit_key)
            current_count = int(current_count) if current_count else 0
            remaining = max(0, api_rate_limit - current_count)

            ttl = redis_client.ttl(rate_limit_key)
            reset_time = int(time.time()) + (ttl if ttl > 0 else api_rate_limit_window)
        except Exception:
            remaining = api_rate_limit - 1
            reset_time = int(time.time()) + api_rate_limit_window

        response.headers["X-RateLimit-Limit"] = str(api_rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        return response


__all__ = ["APIAuthMiddleware", "get_api_key_from_request", "verify_api_key"]

