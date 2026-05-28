"""
HTTP-клиент Backend → Fetcher API (Phase 0 интеграции).

Предоставляет синхронный и асинхронный вызовы к Fetcher:
- POST /api/v1/runs — создание run (передача run_id и source_url)
- GET /api/v1/runs/{run_id} — получение статуса run
- GET /api/v1/runs/{run_id}/manifest — получение manifest (для последующей передачи в DataProcessor)
- GET /api/v1/runs/{run_id}/artifacts — получение артефактов с signed URLs

Контракт API: Fetcher/docs/BACKEND_CONTRACTS.md, Fetcher/fetcher/schemas/api.py.
Документация интеграции: backend/docs/FETCHER_INTEGRATION.md, docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)

# Retry при создании run: только при временных ошибках (timeout, 503, connection)
_CREATE_RUN_RETRY_MAX_ATTEMPTS = 3
_CREATE_RUN_RETRY_BACKOFF_SECONDS = 1.0
_CREATE_RUN_RETRYABLE_STATUS_CODES = {503, 502, 504}
_CREATE_RUN_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.ReadError,
)
_GET_RUN_RETRY_MAX_ATTEMPTS = 4
_GET_RUN_RETRY_BACKOFF_SECONDS = 1.0
_GET_RUN_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_GET_RUN_RETRYABLE_EXCEPTIONS = _CREATE_RUN_RETRYABLE_EXCEPTIONS

# -----------------------------------------------------------------------------
# Хелперы: общая подготовка запросов (убираем дублирование sync/async)
# -----------------------------------------------------------------------------


def _default_headers(api_key: Optional[str], idempotency_key: Optional[str] = None) -> Dict[str, str]:
    """Заголовки для запросов к Fetcher: X-API-Key, опционально Idempotency-Key."""
    headers: Dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _get_client_kwargs(settings: Settings) -> Dict[str, Any]:
    """Базовые параметры для httpx-клиента (timeout)."""
    return {"timeout": settings.fetcher_timeout_seconds}


def _create_run_payload(
    run_id: UUID,
    source_url: str,
    *,
    platform: Optional[str] = None,
    priority: str = "normal",
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Тело запроса POST /api/v1/runs (общее для sync/async)."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "source_url": source_url,
        "priority": priority,
    }
    if platform is not None:
        payload["platform"] = platform
    if webhook_url is not None:
        payload["webhook_url"] = webhook_url
    return payload


def _build_create_run_request(
    run_id: UUID,
    source_url: str,
    *,
    platform: Optional[str] = None,
    priority: str = "normal",
    webhook_url: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    settings: Settings,
) -> tuple[str, Dict[str, Any], Dict[str, str]]:
    """Возвращает (url, payload, headers) для POST /api/v1/runs."""
    base_url = settings.fetcher_api_url.rstrip("/")
    url = f"{base_url}/api/v1/runs"
    payload = _create_run_payload(
        run_id, source_url, platform=platform, priority=priority, webhook_url=webhook_url
    )
    headers = _default_headers(settings.fetcher_api_key, idempotency_key=idempotency_key)
    return url, payload, headers


def _run_url(settings: Settings, path_suffix: str, run_id: UUID) -> str:
    """Базовый URL для GET-запросов по run_id (status, manifest, artifacts)."""
    base = settings.fetcher_api_url.rstrip("/")
    return f"{base}/api/v1/runs/{run_id}{path_suffix}"


def _is_retryable_for_create_run(exc: BaseException) -> bool:
    """Решать, стоит ли повторять create_run при данной ошибке."""
    if isinstance(exc, httpx.HTTPStatusError):
        return getattr(exc.response, "status_code", None) in _CREATE_RUN_RETRYABLE_STATUS_CODES
    return isinstance(exc, _CREATE_RUN_RETRYABLE_EXCEPTIONS)


def _retry_delay_seconds(response: Optional[httpx.Response], attempt: int, base_delay: float) -> float:
    """Вычислить задержку retry с учетом Retry-After от Fetcher."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), base_delay)
            except ValueError:
                pass
    return base_delay * attempt


def _is_retryable_for_get_run(exc: BaseException) -> bool:
    """Решать, стоит ли повторять GET /runs* запросы к Fetcher."""
    if isinstance(exc, httpx.HTTPStatusError):
        return getattr(exc.response, "status_code", None) in _GET_RUN_RETRYABLE_STATUS_CODES
    return isinstance(exc, _GET_RUN_RETRYABLE_EXCEPTIONS)


def _get_json_with_retry(
    url: str,
    headers: Dict[str, str],
    *,
    settings: Settings,
    operation: str,
) -> Dict[str, Any]:
    """Выполнить GET к Fetcher с retry на 429/5xx и transient network errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, _GET_RUN_RETRY_MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(**_get_client_kwargs(settings)) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            last_exc = e
            if attempt < _GET_RUN_RETRY_MAX_ATTEMPTS and _is_retryable_for_get_run(e):
                response = e.response if isinstance(e, httpx.HTTPStatusError) else None
                delay = _retry_delay_seconds(response, attempt, _GET_RUN_RETRY_BACKOFF_SECONDS)
                logger.warning(
                    "Fetcher %s retry attempt=%s url=%s error=%s delay=%.1fs",
                    operation,
                    attempt,
                    url,
                    e,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation}: unexpected state")


async def _get_json_with_retry_async(
    url: str,
    headers: Dict[str, str],
    *,
    settings: Settings,
    operation: str,
) -> Dict[str, Any]:
    """Async GET к Fetcher с retry на 429/5xx и transient network errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, _GET_RUN_RETRY_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(**_get_client_kwargs(settings)) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            last_exc = e
            if attempt < _GET_RUN_RETRY_MAX_ATTEMPTS and _is_retryable_for_get_run(e):
                response = e.response if isinstance(e, httpx.HTTPStatusError) else None
                delay = _retry_delay_seconds(response, attempt, _GET_RUN_RETRY_BACKOFF_SECONDS)
                logger.warning(
                    "Fetcher %s async retry attempt=%s url=%s error=%s delay=%.1fs",
                    operation,
                    attempt,
                    url,
                    e,
                    delay,
                )
                await _async_sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation}_async: unexpected state")


# -----------------------------------------------------------------------------
# Синхронный API
# -----------------------------------------------------------------------------


def create_run(
    run_id: UUID,
    source_url: str,
    *,
    platform: Optional[str] = None,
    priority: str = "normal",
    webhook_url: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Создать run в Fetcher (POST /api/v1/runs).

    Fetcher запускает ingestion (metadata, video, comments) асинхронно;
    ответ возвращается сразу после постановки задачи.
    При временной недоступности (timeout, 503, 502, 504) выполняются до 3 попыток с backoff.

    Args:
        run_id: UUID run'а (генерируется в Backend, source of truth).
        source_url: URL видео (например YouTube).
        platform: Платформа (youtube, tiktok, …) или None (Fetcher определит по URL).
        priority: low | normal | high.
        webhook_url: URL для webhook при завершении run'а (опционально).
        idempotency_key: Ключ идемпотентности (заголовок Idempotency-Key).
        settings: Настройки Backend; если None — создаётся Settings().

    Returns:
        Ответ Fetcher (CreateRunResponse): run_id, status, source_url, platform,
        created_at, message; при идемпотентности может быть existing_run_id.

    Raises:
        httpx.HTTPStatusError: При 4xx/5xx от Fetcher (после исчерпания retry).
        httpx.RequestError: При сетевой ошибке или таймауте (после retry).
    """
    s = settings or Settings()
    url, payload, headers = _build_create_run_request(
        run_id, source_url,
        platform=platform, priority=priority, webhook_url=webhook_url,
        idempotency_key=idempotency_key, settings=s,
    )
    logger.info(
        "Fetcher create_run request",
        extra={"run_id": str(run_id), "source_url": source_url, "url": url},
    )
    last_exc: Optional[Exception] = None
    for attempt in range(1, _CREATE_RUN_RETRY_MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(**_get_client_kwargs(s)) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            logger.info(
                "Fetcher create_run response",
                extra={"run_id": str(run_id), "status": data.get("status"), "message": data.get("message")},
            )
            return data
        except Exception as e:
            last_exc = e
            if attempt < _CREATE_RUN_RETRY_MAX_ATTEMPTS and _is_retryable_for_create_run(e):
                delay = _CREATE_RUN_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Fetcher create_run retry attempt=%s run_id=%s error=%s delay=%.1fs",
                    attempt, str(run_id), e, delay,
                )
                time.sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("create_run: unexpected state")


def get_run(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Получить статус run из Fetcher (GET /api/v1/runs/{run_id}).

    Args:
        run_id: UUID run'а.
        settings: Настройки Backend; если None — создаётся Settings().

    Returns:
        Ответ Fetcher (RunResponse): run_id, status, source_url, platform,
        platform_video_id, created_at, started_at, finished_at, error, error_code,
        artifacts, progress и т.д.

    Raises:
        httpx.HTTPStatusError: При 4xx/5xx (в т.ч. 404 если run не найден).
        httpx.RequestError: При сетевой ошибке или таймауте.
    """
    s = settings or Settings()
    url = _run_url(s, "", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return _get_json_with_retry(url, headers, settings=s, operation="get_run")


def get_run_manifest(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Получить manifest run'а из Fetcher (GET /api/v1/runs/{run_id}/manifest).

    Manifest — контракт между Fetcher и DataProcessor (пути к video_file,
    meta_file, comments_file в object storage). Доступен после успешного
    завершения ingestion (status=COMPLETED).

    Args:
        run_id: UUID run'а.
        settings: Настройки Backend; если None — создаётся Settings().

    Returns:
        Ответ Fetcher (manifest): manifest_version, run_id, video_id, platform,
        duration_seconds, storage_layout_version, artifacts (paths, checksums, sizes).

    Raises:
        httpx.HTTPStatusError: При 4xx/5xx (404 если run не готов или не найден).
        httpx.RequestError: При сетевой ошибке или таймауте.
    """
    s = settings or Settings()
    url = _run_url(s, "/manifest", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return _get_json_with_retry(url, headers, settings=s, operation="get_run_manifest")


def get_run_artifacts(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Получить артефакты run'а с signed URLs (GET /api/v1/runs/{run_id}/artifacts).

    Args:
        run_id: UUID run'а.
        settings: Настройки Backend; если None — создаётся Settings().

    Returns:
        Ответ Fetcher: список артефактов с download_url (signed), expires_at, size_bytes, checksum.

    Raises:
        httpx.HTTPStatusError: При 4xx/5xx.
        httpx.RequestError: При сетевой ошибке или таймауте.
    """
    s = settings or Settings()
    url = _run_url(s, "/artifacts", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return _get_json_with_retry(url, headers, settings=s, operation="get_run_artifacts")


# -----------------------------------------------------------------------------
# Асинхронный API
# -----------------------------------------------------------------------------


async def create_run_async(
    run_id: UUID,
    source_url: str,
    *,
    platform: Optional[str] = None,
    priority: str = "normal",
    webhook_url: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Асинхронно создать run в Fetcher (POST /api/v1/runs).

    Параметры и возвращаемое значение совпадают с create_run().
    При временной недоступности (timeout, 503, 502, 504) — до 3 попыток с backoff.
    """
    s = settings or Settings()
    url, payload, headers = _build_create_run_request(
        run_id, source_url,
        platform=platform, priority=priority, webhook_url=webhook_url,
        idempotency_key=idempotency_key, settings=s,
    )
    logger.info(
        "Fetcher create_run_async request",
        extra={"run_id": str(run_id), "source_url": source_url, "url": url},
    )
    last_exc: Optional[Exception] = None
    for attempt in range(1, _CREATE_RUN_RETRY_MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(**_get_client_kwargs(s)) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            logger.info(
                "Fetcher create_run_async response",
                extra={"run_id": str(run_id), "status": data.get("status"), "message": data.get("message")},
            )
            return data
        except Exception as e:
            last_exc = e
            if attempt < _CREATE_RUN_RETRY_MAX_ATTEMPTS and _is_retryable_for_create_run(e):
                delay = _CREATE_RUN_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Fetcher create_run_async retry attempt=%s run_id=%s error=%s delay=%.1fs",
                    attempt, str(run_id), e, delay,
                )
                await _async_sleep(delay)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("create_run_async: unexpected state")


async def _async_sleep(seconds: float) -> None:
    """Небольшая задержка в async контексте без блокировки event loop."""
    import asyncio
    await asyncio.sleep(seconds)


async def get_run_async(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Асинхронно получить статус run из Fetcher (GET /api/v1/runs/{run_id})."""
    s = settings or Settings()
    url = _run_url(s, "", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return await _get_json_with_retry_async(url, headers, settings=s, operation="get_run")


async def get_run_manifest_async(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Асинхронно получить manifest run'а (GET /api/v1/runs/{run_id}/manifest)."""
    s = settings or Settings()
    url = _run_url(s, "/manifest", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return await _get_json_with_retry_async(
        url, headers, settings=s, operation="get_run_manifest"
    )


async def get_run_artifacts_async(
    run_id: UUID,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Асинхронно получить артефакты run'а (GET /api/v1/runs/{run_id}/artifacts)."""
    s = settings or Settings()
    url = _run_url(s, "/artifacts", run_id)
    headers = _default_headers(s.fetcher_api_key)
    return await _get_json_with_retry_async(
        url, headers, settings=s, operation="get_run_artifacts"
    )
