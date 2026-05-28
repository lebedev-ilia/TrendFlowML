"""
HTTP API для Fetcher.

Предоставляет REST API для управления runs, получения статусов и manifest.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import session_scope
from .health import (
    check_database_health,
    check_redis_health,
    check_storage_health,
    get_uptime_seconds,
    set_startup_time,
)
from .kafka_producer import init_producer
from .lifecycle import run_lifecycle_cleanup
from .logging import setup_logging
from .metrics import get_metrics, get_metrics_content_type
from .models import Artifact, FetchJob, FetchLog, Run, Video, VideoSource
from .orchestrator import normalize_source
from .schemas import HealthResponse
from .schemas.api import (
    ArtifactsResponse,
    ArtifactItem,
    BulkCreateRunsRequest,
    BulkCreateRunsResponse,
    CreateRunRequest,
    CreateRunResponse,
    ErrorResponse,
    LimitsResponse,
    LogsUrlResponse,
    ManifestResponse,
    QueueResponse,
    RetryRunResponse,
    RunArtifactsInfo,
    RunEventItem,
    RunEventsResponse,
    RunListItem,
    RunListResponse,
    RunProgress,
    RunResponse,
    VideoCacheResponse,
    StatsResponse,
    UpdateRunRequest,
    UpdateRunResponse,
)
from .state_machine import (
    RUN_STATUS_PENDING,
    RUN_STATUS_NORMALIZING_SOURCE,
    RUN_STATUS_CHECKING_CACHE,
    RUN_STATUS_FETCHING_METADATA,
    RUN_STATUS_FETCHING_CHANNEL,
    RUN_STATUS_FETCHING_COMMENTS,
    RUN_STATUS_DOWNLOADING_VIDEO,
    RUN_STATUS_UPLOADING_ARTIFACTS,
    RUN_STATUS_FINALIZING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
)
from .storage import storage_client
from .tasks import fetch_metadata_task
from .validation import (
    validate_circuit_breaker_cooldown,
    validate_proxy_rotation,
    validate_rate_limiter_enforcement,
)
from .api_auth import APIAuthMiddleware

logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Fetcher Service API",
    description="""
    Production-grade ingestion platform for YouTube and other video platforms.
    
    ## Features
    
    * **Event-driven ingestion**: API publishes events to queues (Celery/Kafka), doesn't run ingestion synchronously
    * **Cursor-based pagination**: Scalable pagination for large datasets
    * **Signed URLs**: Secure access to artifacts via presigned URLs
    * **Webhooks**: Asynchronous notifications with HMAC-SHA256 signatures
    * **Idempotency**: Support for Idempotency-Key header
    * **Bulk ingestion**: Efficient processing of multiple runs
    
    ## Authentication
    
    API Key authentication is supported via `X-API-Key` header or `api_key` query parameter.
    
    ## Rate Limiting
    
    Rate limiting is applied per IP and API key. Check `X-RateLimit-*` headers for current limits.
    
    ## Error Responses
    
    All errors follow a standardized format:
    ```json
    {
      "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable error message",
        "details": {}
      }
    }
    ```
    """,
    version="1.0.0",
    contact={
        "name": "Fetcher Service",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
        {
            "url": "https://api.example.com",
            "description": "Production server",
        },
    ],
    openapi_tags=[
        {
            "name": "api",
            "description": "Main API endpoints for runs management",
        },
        {
            "name": "runs",
            "description": "Run management endpoints",
        },
        {
            "name": "artifacts",
            "description": "Artifact access endpoints",
        },
        {
            "name": "logs",
            "description": "Logs access endpoints",
        },
        {
            "name": "events",
            "description": "Run events history endpoints",
        },
        {
            "name": "monitoring",
            "description": "Monitoring and statistics endpoints",
        },
        {
            "name": "health",
            "description": "Health check endpoints",
        },
        {
            "name": "metrics",
            "description": "Prometheus metrics endpoint",
        },
        {
            "name": "admin",
            "description": "Administrative endpoints",
        },
    ],
)

# CORS middleware (для development, в production настроить правильно)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production указать конкретные origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Authentication и Rate Limiting middleware
app.add_middleware(
    APIAuthMiddleware,
    require_auth=settings.api_require_auth,
)

# Exception handlers для стандартизированных error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTPException для стандартизированных error responses."""
    error_code = "HTTP_ERROR"
    if exc.status_code == 404:
        error_code = "NOT_FOUND"
    elif exc.status_code == 400:
        error_code = "BAD_REQUEST"
    elif exc.status_code == 409:
        error_code = "CONFLICT"
    elif exc.status_code == 429:
        error_code = "RATE_LIMIT_EXCEEDED"
    elif exc.status_code == 500:
        error_code = "INTERNAL_SERVER_ERROR"
    elif exc.status_code == 503:
        error_code = "SERVICE_UNAVAILABLE"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": error_code,
                "message": exc.detail or "An error occurred",
                "details": {
                    "path": request.url.path,
                    "method": request.method,
                },
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Обработчик RequestValidationError для стандартизированных error responses."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {
                    "errors": exc.errors(),
                    "path": request.url.path,
                    "method": request.method,
                },
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик общих исключений для стандартизированных error responses."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "details": {
                    "path": request.url.path,
                    "method": request.method,
                },
            }
        },
    )


# Настройка логирования при старте
setup_logging()

# Установить время запуска при импорте модуля
set_startup_time()

# Инициализация Kafka producer (если включено)
if settings.kafka_enabled and settings.kafka_bootstrap_servers:
    try:
        bootstrap_servers = settings.kafka_bootstrap_servers
        if isinstance(bootstrap_servers, str):
            bootstrap_servers = [s.strip() for s in bootstrap_servers.split(",")]
        init_producer(
            bootstrap_servers=bootstrap_servers,
            topic_prefix=settings.kafka_topic_prefix,
        )
        logger.info(f"Kafka producer initialized: bootstrap_servers={bootstrap_servers}")
    except Exception as e:
        logger.warning(f"Failed to initialize Kafka producer: {e}")


@app.get("/metrics", tags=["metrics"])
async def prometheus_metrics() -> Response:
    """Prometheus metrics endpoint.

    Возвращает метрики в формате Prometheus text format.

    Returns:
        Response с метриками в формате text/plain; version=0.0.4; charset=utf-8

    Пример ответа:
        # HELP fetcher_videos_downloaded_total Total number of successfully downloaded videos.
        # TYPE fetcher_videos_downloaded_total counter
        fetcher_videos_downloaded_total{platform="youtube"} 42.0
    """
    try:
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type=get_metrics_content_type(),
        )
    except Exception as e:
        logger.exception(f"Error generating metrics: {e}")
        # Возвращаем пустые метрики при ошибке
        return Response(
            content=b"# Error generating metrics\n",
            media_type=get_metrics_content_type(),
        )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse | JSONResponse:
    """Health check endpoint.

    Проверяет состояние Fetcher и его зависимостей:
    - API сервер
    - PostgreSQL база данных
    - Redis (для rate limiting/locks)
    - S3/MinIO storage

    Returns:
        HealthResponse с информацией о здоровье сервиса.
        Или JSONResponse с 503 статусом если unhealthy.

    Статусы:
    - healthy: Все критичные сервисы работают
    - degraded: API работает, но некритичные сервисы недоступны
    - unhealthy: Критичные сервисы недоступны (возвращается 503)
    """
    try:
        # Базовая проверка API
        api_status = "healthy"

        # Проверка зависимостей
        db_health = check_database_health()
        redis_health = check_redis_health()
        storage_health = check_storage_health()

        dependencies = {
            "database": db_health,
            "redis": redis_health,
            "storage": storage_health,
        }

        # Определение общего статуса
        # Критичные: database, storage
        # Некритичные: redis (может быть not_configured)
        critical_healthy = (
            db_health.get("status") == "healthy"
            and storage_health.get("status") == "healthy"
        )

        if not critical_healthy:
            overall_status = "unhealthy"
        elif redis_health.get("status") == "unhealthy":
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Метрики (пока пустые, можно добавить активные runs и т.д.)
        metrics: dict[str, Any] = {}

        response = HealthResponse(
            status=overall_status,
            api=api_status,
            version="0.1.0",
            uptime_seconds=get_uptime_seconds(),
            dependencies=dependencies,
            metrics=metrics,
        )

        # Возвращаем 503 если unhealthy
        if overall_status == "unhealthy":
            return JSONResponse(
                status_code=503,
                content=response.dict(),
            )

        return response

    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        # Возвращаем unhealthy при любой неожиданной ошибке
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "api": "unhealthy",
                "version": "0.1.0",
                "uptime_seconds": get_uptime_seconds(),
                "dependencies": {},
                "metrics": {},
                "error": str(e),
            },
        )


@app.get("/admin/validation", tags=["admin", "validation"])
async def validation_check() -> dict[str, Any]:
    """Проверить корректность работы proxy rotation, rate limiter и circuit breaker.

    Выполняет валидацию:
    - Proxy rotation correctness (равномерное распределение)
    - Rate limiter enforcement (не превышаются лимиты)
    - Circuit breaker cooldown (корректное снятие после cooldown)

    Returns:
        Результаты валидации для каждого компонента
    """
    results: dict[str, Any] = {}

    # 1. Proxy rotation validation
    try:
        is_valid, distribution = validate_proxy_rotation(num_requests=100)
        results["proxy_rotation"] = {
            "valid": is_valid,
            "distribution": distribution,
        }
    except Exception as e:
        results["proxy_rotation"] = {
            "valid": False,
            "error": str(e),
        }

    # 2. Rate limiter validation
    try:
        is_valid, stats = validate_rate_limiter_enforcement(
            key="rate:youtube:metadata:validation",
            limit=10,
            window_sec=60,
            num_requests=20,
        )
        results["rate_limiter"] = {
            "valid": is_valid,
            "stats": stats,
        }
    except Exception as e:
        results["rate_limiter"] = {
            "valid": False,
            "error": str(e),
        }

    # 3. Circuit breaker cooldown validation
    try:
        is_valid, timings = validate_circuit_breaker_cooldown(
            operation="metadata",
            cooldown_seconds=300,
        )
        results["circuit_breaker"] = {
            "valid": is_valid,
            "timings": timings,
        }
    except Exception as e:
        results["circuit_breaker"] = {
            "valid": False,
            "error": str(e),
        }

    return results


@app.post("/admin/lifecycle/cleanup", tags=["admin", "lifecycle"])
async def lifecycle_cleanup() -> dict[str, Any]:
    """Запустить lifecycle cleanup вручную.

    Запускает очистку старых артефактов согласно retention policies:
    - Raw видео старше N дней (по умолчанию 30)
    - Temp файлы старше N дней (по умолчанию 7)
    - Failed runs старше N дней (по умолчанию 7)

    **Примечание**: В production рекомендуется добавить аутентификацию для этого endpoint.

    Returns:
        Dict с результатами очистки по каждому типу:
        {
            "raw_videos": {"checked": int, "deleted": int, "errors": int},
            "temp_files": {"checked": int, "deleted": int, "errors": int},
            "failed_runs": {"checked": int, "deleted": int, "errors": int},
            "timestamp": float,
            "elapsed_seconds": float
        }

    Raises:
        HTTPException: При ошибке выполнения очистки
    """
    try:
        start_time = time.time()

        results = run_lifecycle_cleanup(
            raw_video_retention_days=settings.raw_video_retention_days,
            raw_comments_retention_days=settings.raw_comments_retention_days,
            raw_comments_hard_cap_days=settings.raw_comments_hard_cap_days,
            temp_files_retention_days=settings.temp_files_retention_days,
            failed_runs_retention_days=settings.failed_runs_retention_days,
        )

        elapsed_seconds = time.time() - start_time

        return {
            **results,
            "timestamp": start_time,
            "elapsed_seconds": elapsed_seconds,
        }
    except Exception as e:
        logger.exception(f"Lifecycle cleanup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lifecycle cleanup failed: {str(e)}",
        )


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Информация о сервисе Fetcher.
    """
    return {
        "service": "Fetcher",
        "version": "0.1.0",
        "description": "Production-grade ingestion platform for YouTube and other platforms",
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "validation": "/admin/validation",
            "lifecycle_cleanup": "/admin/lifecycle/cleanup",
            "api": {
                "v1": {
                    "runs": "/api/v1/runs",
                    "run_detail": "/api/v1/runs/{run_id}",
                    "manifest": "/api/v1/runs/{run_id}/manifest",
                }
            },
        },
    }


# ============================================================================
# API v1 Endpoints
# ============================================================================


@app.post(
    "/api/v1/runs",
    response_model=CreateRunResponse,
    status_code=201,
    tags=["api", "runs"],
)
async def create_run(
    request: CreateRunRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> CreateRunResponse:
    """Создать новый run и запустить ingestion (event-driven).

    Args:
        request: Данные для создания run'а
        idempotency_key: Опциональный ключ для идемпотентности

    Returns:
        CreateRunResponse с информацией о созданном run'е

    Raises:
        HTTPException: При ошибках валидации или создания run'а
    """
    run_uuid = request.run_id

    try:
        with session_scope() as db:
            # Проверка Idempotency-Key (если передан)
            if idempotency_key:
                existing_run = db.query(Run).filter(Run.id == run_uuid).first()
                if existing_run:
                    logger.info(
                        f"Idempotency key {idempotency_key} matched existing run {run_uuid}"
                    )
                    return CreateRunResponse(
                        run_id=existing_run.id,
                        status=existing_run.status,
                        source_url=existing_run.source_url,
                        platform=None,  # Будет заполнено из video_sources
                        created_at=existing_run.created_at,
                        message="Run already exists (idempotency key matched)",
                    )

            # Проверка существования run'а
            existing_run = db.query(Run).filter(Run.id == run_uuid).first()
            if existing_run:
                raise HTTPException(
                    status_code=409,
                    detail=f"Run with id {run_uuid} already exists",
                )

            # Нормализацию и проверку кеша выполняет orchestrator.fetch_video.
            # Здесь сохраняем только исходный URL и platform (если указана явно).
            final_platform = request.platform or "youtube"

            # TODO: дублирование по canonical video_id можно вернуть позже,
            # когда normalize_source будет вызываться до create_run или результат
            # нормализации будет кэшироваться отдельно.
            existing_run_id: Optional[UUID] = None

            # Создание записи в таблице runs (PENDING: полный пайплайн запускается через fetch_video)
            run = Run(
                id=run_uuid,
                source_type="video",
                source_url=str(request.source_url),
                status=RUN_STATUS_PENDING,
            )
            db.add(run)

            # Создание записи в таблице video_sources
            video_source = VideoSource(
                run_id=run_uuid,
                platform=final_platform,
                url=str(request.source_url),
                normalized_video_id=None,
            )
            db.add(video_source)

            db.flush()  # Чтобы получить created_at

            # Commit до Celery: иначе worker может стартовать раньше конца session_scope
            # и не увидеть run (ValueError: Run ... not found).
            db.commit()

            # Публикация события в очередь (event-driven): запускаем orchestrator.fetch_video
            priority = request.priority or "normal"
            queue_name = f"fetcher.{priority}"  # fetcher.high, fetcher.normal, fetcher.low

            from .tasks import fetch_video_task

            fetch_video_task.apply_async(
                args=[str(run_uuid)],
                queue=queue_name,
            )

            logger.info(
                f"Created run {run_uuid} for {final_platform}, queued to Celery"
            )

            return CreateRunResponse(
                run_id=run_uuid,
                status=run.status,
                source_url=run.source_url,
                platform=final_platform,
                created_at=run.created_at,
                message="Run created and ingestion queued",
                existing_run_id=existing_run_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create run {run_uuid}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create run: {str(e)}",
        )


@app.get(
    "/api/v1/runs/{run_id}",
    response_model=RunResponse,
    tags=["api", "runs"],
)
async def get_run(run_id: str) -> RunResponse:
    """Получить информацию о run'е.

    Args:
        run_id: UUID run'а

    Returns:
        RunResponse с полной информацией о run'е

    Raises:
        HTTPException: Если run не найден
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Получаем video_source для platform и platform_video_id
            video_source = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )

            platform = video_source.platform if video_source else None
            platform_video_id = (
                video_source.normalized_video_id if video_source else None
            )

            # Получаем video_id если есть
            video_id: Optional[UUID] = None
            if platform and platform_video_id:
                video = (
                    db.query(Video)
                    .filter(
                        Video.platform == platform,
                        Video.platform_video_id == platform_video_id,
                    )
                    .first()
                )
                if video:
                    video_id = video.id

            # Получаем информацию об артефактах
            artifacts_info: Optional[RunArtifactsInfo] = None
            if video_id:
                artifacts = (
                    db.query(Artifact)
                    .filter(Artifact.video_id == video_id)
                    .all()
                )
                if artifacts:
                    artifacts_dict: dict[str, Optional[str]] = {
                        "video_file": None,
                        "meta_file": None,
                        "comments_file": None,
                        "manifest_file": None,
                    }
                    for art in artifacts:
                        if art.artifact_type == "video_file":
                            artifacts_dict["video_file"] = art.storage_path
                        elif art.artifact_type == "metadata_file":
                            artifacts_dict["meta_file"] = art.storage_path
                        elif art.artifact_type == "comments_file":
                            artifacts_dict["comments_file"] = art.storage_path
                        elif art.artifact_type == "manifest_file":
                            artifacts_dict["manifest_file"] = art.storage_path

                    artifacts_info = RunArtifactsInfo(**artifacts_dict)

            # Определяем прогресс (упрощённая версия)
            # run.status в БД хранится в верхнем регистре (константы state_machine)
            progress: Optional[RunProgress] = None
            completed_stages: list[str] = []
            current_stage: Optional[str] = None
            status_upper = (run.status or "").upper()

            if status_upper == RUN_STATUS_PENDING:
                current_stage = "pending"
            elif status_upper == RUN_STATUS_NORMALIZING_SOURCE:
                current_stage = "normalize_source"
            elif status_upper == RUN_STATUS_CHECKING_CACHE:
                completed_stages = ["normalize_source"]
                current_stage = "check_cache"
            elif status_upper == RUN_STATUS_FETCHING_METADATA:
                completed_stages = ["normalize_source", "check_cache"]
                current_stage = "fetch_metadata"
            elif status_upper == RUN_STATUS_DOWNLOADING_VIDEO:
                completed_stages = ["normalize_source", "check_cache", "fetch_metadata"]
                current_stage = "download_video"
            elif status_upper in (
                RUN_STATUS_FETCHING_CHANNEL,
                RUN_STATUS_FETCHING_COMMENTS,
            ):
                completed_stages = [
                    "normalize_source",
                    "check_cache",
                    "fetch_metadata",
                    "download_video",
                ]
                current_stage = "fetch_comments"
            elif status_upper == RUN_STATUS_UPLOADING_ARTIFACTS:
                completed_stages = [
                    "normalize_source",
                    "check_cache",
                    "fetch_metadata",
                    "download_video",
                    "fetch_comments",
                ]
                current_stage = "finalize"
            elif status_upper == RUN_STATUS_FINALIZING:
                completed_stages = [
                    "normalize_source",
                    "check_cache",
                    "fetch_metadata",
                    "download_video",
                    "fetch_comments",
                ]
                current_stage = "finalize"
            elif status_upper in (RUN_STATUS_COMPLETED, RUN_STATUS_FAILED):
                completed_stages = [
                    "normalize_source",
                    "check_cache",
                    "fetch_metadata",
                    "download_video",
                    "fetch_comments",
                    "finalize",
                ]
                current_stage = None

            if current_stage or completed_stages:
                progress = RunProgress(
                    stage=current_stage,
                    completed_stages=completed_stages,
                    total_stages=7,
                )

            return RunResponse(
                run_id=run.id,
                status=status_upper or run.status,
                source_url=run.source_url,
                platform=platform,
                platform_video_id=platform_video_id,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                error=run.error,
                error_code=getattr(run, "error_code", None),
                cancel_requested=getattr(run, "cancel_requested", False),
                video_id=video_id,
                artifacts=artifacts_info,
                progress=progress,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get run: {str(e)}",
        )


@app.get(
    "/api/v1/runs/{run_id}/manifest",
    response_model=ManifestResponse,
    tags=["api", "runs", "manifest"],
)
async def get_run_manifest(run_id: str) -> ManifestResponse:
    """Получить manifest.json для run'а.

    Args:
        run_id: UUID run'а

    Returns:
        ManifestResponse с manifest данными

    Raises:
        HTTPException: Если run не найден или manifest не готов
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Проверяем статус run'а (используем state machine статусы в верхнем регистре)
            status_upper = (run.status or "").upper()
            if status_upper not in (RUN_STATUS_COMPLETED, RUN_STATUS_FINALIZING):
                raise HTTPException(
                    status_code=503,
                    detail=f"Manifest not ready yet. Run status: {status_upper}",
                )

            # Получаем video_source для определения platform и video_id
            video_source = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )

            if not video_source:
                raise HTTPException(
                    status_code=404,
                    detail=f"Video source not found for run {run_id}",
                )

            platform = video_source.platform
            video_id = video_source.normalized_video_id

            # Формируем путь к manifest в storage
            # Manifest сохраняется по пути: raw/{platform}/{YYYY/MM/DD}/{video_id}/manifest.json
            # Пробуем найти manifest по дате создания run'а
            run_date = run.created_at.strftime("%Y/%m/%d")
            storage_key = f"raw/{platform}/{run_date}/{video_id}/manifest.json"

            # Проверяем существование manifest в storage
            if not storage_client.object_exists(
                bucket=settings.bucket_raw, key=storage_key
            ):
                # Пробуем найти manifest по другим возможным датам (в пределах последних 7 дней)
                found = False
                for days_ago in range(7):
                    check_date = (run.created_at - timedelta(days=days_ago)).strftime(
                        "%Y/%m/%d"
                    )
                    check_key = f"raw/{platform}/{check_date}/{video_id}/manifest.json"
                    if storage_client.object_exists(
                        bucket=settings.bucket_raw, key=check_key
                    ):
                        storage_key = check_key
                        found = True
                        break

                if not found:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Manifest not found in storage for run {run_id}",
                    )

            # Скачиваем manifest из storage
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
                tmp_path = Path(tmp.name)
                try:
                    storage_client.download_file(
                        bucket=settings.bucket_raw,
                        key=storage_key,
                        local_path=tmp_path,
                    )

                    # Читаем и парсим manifest
                    manifest_data = json.loads(tmp_path.read_text(encoding="utf-8"))

                    # Преобразуем в ManifestResponse
                    return ManifestResponse(
                        manifest_version=manifest_data.get("manifest_version", "1.0"),
                        run_id=UUID(manifest_data["run_id"]),
                        video_id=manifest_data["video_id"],
                        platform=manifest_data["platform"],
                        duration_seconds=manifest_data["duration_seconds"],
                        storage_layout_version=manifest_data.get(
                            "storage_layout_version", "1.0"
                        ),
                        artifacts=manifest_data["artifacts"],
                        created_at=run.finished_at or run.created_at,
                    )
                finally:
                    # Удаляем временный файл
                    if tmp_path.exists():
                        tmp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get manifest for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get manifest: {str(e)}",
        )


@app.get(
    "/api/v1/runs",
    response_model=RunListResponse,
    tags=["api", "runs"],
)
async def list_runs(
    status: Optional[str] = None,
    platform: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
) -> RunListResponse:
    """Получить список runs с фильтрацией и cursor-based пагинацией.

    Args:
        status: Фильтр по статусу (PENDING, FETCHING_METADATA, COMPLETED, FAILED, etc.)
        platform: Фильтр по платформе (youtube, tiktok, etc.)
        created_after: Фильтр по дате создания (ISO 8601)
        created_before: Фильтр по дате создания (ISO 8601)
        limit: Количество результатов (default: 50, max: 100)
        cursor: Cursor для пагинации (base64 encoded JSON)

    Returns:
        RunListResponse со списком runs и информацией о пагинации

    Raises:
        HTTPException: При ошибках валидации или запроса
    """
    # Ограничиваем limit
    limit = min(max(limit, 1), 100)

    try:
        with session_scope() as db:
            # Начинаем запрос
            query = db.query(Run)

            # Применяем фильтры
            if status:
                query = query.filter(Run.status == status)
            if created_after:
                query = query.filter(Run.created_at >= created_after)
            if created_before:
                query = query.filter(Run.created_at <= created_before)

            # Фильтр по платформе через join с video_sources
            if platform:
                query = query.join(VideoSource).filter(VideoSource.platform == platform)

            # Cursor-based pagination
            if cursor:
                try:
                    cursor_data = json.loads(base64.b64decode(cursor).decode("utf-8"))
                    cursor_created_at = datetime.fromisoformat(
                        cursor_data["created_at"].replace("Z", "+00:00")
                    )
                    cursor_run_id = UUID(cursor_data["run_id"])

                    # Фильтруем: created_at < cursor_created_at ИЛИ
                    # (created_at == cursor_created_at AND run_id < cursor_run_id)
                    query = query.filter(
                        (Run.created_at < cursor_created_at)
                        | (
                            (Run.created_at == cursor_created_at)
                            & (Run.id < cursor_run_id)
                        )
                    )
                except Exception as e:
                    logger.warning(f"Invalid cursor format: {e}")
                    raise HTTPException(
                        status_code=400, detail=f"Invalid cursor format: {str(e)}"
                    )

            # Сортировка: по created_at DESC, затем по run_id DESC
            query = query.order_by(Run.created_at.desc(), Run.id.desc())

            # Получаем limit + 1 для проверки has_more
            runs = query.limit(limit + 1).all()

            has_more = len(runs) > limit
            if has_more:
                runs = runs[:limit]

            # Формируем список runs
            run_items: list[RunListItem] = []
            for run in runs:
                # Получаем platform из video_source
                video_source = (
                    db.query(VideoSource)
                    .filter(VideoSource.run_id == run.id)
                    .order_by(VideoSource.created_at)
                    .first()
                )
                platform_value = video_source.platform if video_source else None

                run_items.append(
                    RunListItem(
                        run_id=run.id,
                        status=run.status,
                        platform=platform_value,
                        created_at=run.created_at,
                        finished_at=run.finished_at,
                    )
                )

            # Формируем next_cursor если есть ещё результаты
            next_cursor: Optional[str] = None
            if has_more and runs:
                last_run = runs[-1]
                cursor_data = {
                    "created_at": last_run.created_at.isoformat(),
                    "run_id": str(last_run.id),
                }
                next_cursor = base64.b64encode(
                    json.dumps(cursor_data).encode("utf-8")
                ).decode("utf-8")

            return RunListResponse(
                runs=run_items,
                pagination={
                    "limit": limit,
                    "has_more": has_more,
                    "next_cursor": next_cursor,
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to list runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list runs: {str(e)}",
        )


@app.get(
    "/api/v1/runs/{run_id}/artifacts",
    response_model=ArtifactsResponse,
    tags=["api", "runs", "artifacts"],
)
async def get_run_artifacts(
    run_id: str, expires_in: int = 3600
) -> ArtifactsResponse:
    """Получить список артефактов для run'а с signed URLs.

    Args:
        run_id: UUID run'а
        expires_in: Время жизни signed URL в секундах (default: 3600, max: 86400)

    Returns:
        ArtifactsResponse со списком артефактов и signed URLs

    Raises:
        HTTPException: Если run не найден
    """
    # Ограничиваем expires_in
    expires_in = min(max(expires_in, 60), 86400)  # От 1 минуты до 24 часов

    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Получаем video_source для определения video_id
            video_source = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )

            if not video_source:
                # Если нет video_source, возвращаем пустой список
                return ArtifactsResponse(run_id=run_uuid, artifacts=[])

            # Получаем video
            video = (
                db.query(Video)
                .filter(
                    Video.platform == video_source.platform,
                    Video.platform_video_id == video_source.normalized_video_id,
                )
                .first()
            )

            if not video:
                return ArtifactsResponse(run_id=run_uuid, artifacts=[])

            # Получаем артефакты
            artifacts = (
                db.query(Artifact).filter(Artifact.video_id == video.id).all()
            )

            artifact_items: list[ArtifactItem] = []
            for art in artifacts:
                # Определяем artifact_status на основе status
                if art.status == "COMPLETED":
                    artifact_status = "READY"
                elif art.status == "FAILED":
                    artifact_status = "FAILED"
                else:
                    artifact_status = "PENDING"

                # Генерируем signed URL только если artifact_status == READY
                download_url: Optional[str] = None
                download_url_expires_at: Optional[datetime] = None

                if artifact_status == "READY":
                    try:
                        # Извлекаем bucket и key из storage_path
                        # storage_path может быть в формате "raw/platform/date/video_id/file.ext"
                        # или полный путь "s3://bucket/key"
                        storage_path = art.storage_path
                        if storage_path.startswith("s3://"):
                            # Парсим s3://bucket/key
                            parts = storage_path[5:].split("/", 1)
                            bucket = parts[0]
                            key = parts[1] if len(parts) > 1 else ""
                        else:
                            # Используем bucket из settings и storage_path как key
                            bucket = settings.bucket_raw
                            key = storage_path

                        download_url = storage_client.generate_presigned_url(
                            bucket=bucket,
                            key=key,
                            expires_in=expires_in,
                        )
                        download_url_expires_at = datetime.now(timezone.utc) + timedelta(
                            seconds=expires_in
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate presigned URL for artifact {art.id}: {e}"
                        )
                        # Продолжаем без signed URL

                artifact_items.append(
                    ArtifactItem(
                        artifact_type=art.artifact_type,
                        download_url=download_url,
                        download_url_expires_at=download_url_expires_at,
                        size_bytes=art.size_bytes,
                        checksum=art.checksum,
                        artifact_status=artifact_status,
                        status=art.status,
                        created_at=art.created_at,
                    )
                )

            return ArtifactsResponse(run_id=run_uuid, artifacts=artifact_items)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get artifacts for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get artifacts: {str(e)}",
        )


@app.get(
    "/api/v1/videos/{platform}/{video_id}",
    response_model=VideoCacheResponse,
    tags=["api", "videos", "cache"],
)
async def get_video_from_cache(platform: str, video_id: str) -> VideoCacheResponse:
    """Получить информацию о видео из кеша Fetcher.

    Возвращает данные по (platform, platform_video_id): наличие артефактов,
    количество снэпшотов и комментариев. 404 если видео не найдено в кеше.

    Args:
        platform: Платформа (youtube, tiktok, ...)
        video_id: ID видео на платформе (например YouTube video_id)

    Returns:
        VideoCacheResponse с полями video_id, artifacts_available, snapshots_count, comments_count

    Raises:
        HTTPException: 404 если видео не в кеше
    """
    try:
        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(
                    Video.platform == platform.lower(),
                    Video.platform_video_id == video_id,
                )
                .first()
            )
            if not video:
                raise HTTPException(
                    status_code=404,
                    detail=f"Video not found in cache: {platform}/{video_id}",
                )

            required = {"video_file", "metadata_file", "comments_file"}
            completed = {
                a.artifact_type
                for a in video.artifacts
                if a.status == "COMPLETED"
            }
            artifacts_available = required <= completed

            snapshots_count = len(video.snapshots) if video.snapshots else 0
            comments_count = len(video.comments) if video.comments else 0

            return VideoCacheResponse(
                video_id=video.id,
                platform=video.platform,
                platform_video_id=video.platform_video_id,
                artifacts_available=artifacts_available,
                snapshots_count=snapshots_count,
                comments_count=comments_count,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get video from cache {platform}/{video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video: {str(e)}",
        )


@app.get(
    "/api/v1/runs/{run_id}/logs_url",
    response_model=LogsUrlResponse,
    tags=["api", "runs", "logs"],
)
async def get_run_logs_url(run_id: str) -> LogsUrlResponse:
    """Получить URL для доступа к логам run'а.

    Логи хранятся в централизованном хранилище (Loki, Elasticsearch, CloudWatch),
    не в БД. API возвращает URL для доступа к логам через Grafana или другой интерфейс.

    Args:
        run_id: UUID run'а

    Returns:
        LogsUrlResponse с URL для доступа к логам

    Raises:
        HTTPException: Если run не найден или логирование не настроено
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Проверяем, настроено ли централизованное логирование
            if not settings.logging_backend:
                return LogsUrlResponse(
                    run_id=run_uuid,
                    logs_url=None,
                    logs_backend=None,
                    message="Centralized logging is not configured. Logs are only available in database.",
                )

            logs_backend = settings.logging_backend
            logs_url: Optional[str] = None

            # Формируем URL в зависимости от backend'а
            if logs_backend == "loki" and settings.logging_loki_url:
                # Формируем Grafana Explore URL для Loki
                query = f'{{run_id="{run_id}"}}'
                grafana_url = settings.logging_loki_url.replace("/loki/api/v1", "")
                # Упрощённая версия - в production нужно использовать реальный Grafana URL
                logs_url = f"{grafana_url}/explore?query={urllib.parse.quote(query)}"

            elif logs_backend == "elasticsearch" and settings.logging_elasticsearch_url:
                # Формируем Kibana URL
                es_url = settings.logging_elasticsearch_url
                index = settings.logging_elasticsearch_index
                # Упрощённая версия - в production нужно использовать реальный Kibana URL
                logs_url = f"{es_url}/_search?q=run_id:{run_id}&index={index}"

            elif logs_backend == "cloudwatch":
                # Для CloudWatch возвращаем информацию о log group
                log_group = settings.logging_cloudwatch_log_group
                region = settings.logging_cloudwatch_region or "us-east-1"
                logs_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{urllib.parse.quote(log_group)}"

            if not logs_url:
                return LogsUrlResponse(
                    run_id=run_uuid,
                    logs_url=None,
                    logs_backend=logs_backend,
                    message=f"Logs backend {logs_backend} is configured but URL generation is not implemented.",
                )

            return LogsUrlResponse(
                run_id=run_uuid,
                logs_url=logs_url,
                logs_backend=logs_backend,
                message="Logs are available in centralized logging system",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get logs URL for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get logs URL: {str(e)}",
        )


@app.post(
    "/api/v1/runs/{run_id}/retry",
    response_model=RetryRunResponse,
    tags=["api", "runs"],
)
async def retry_run(run_id: str) -> RetryRunResponse:
    """Перезапустить ingestion для существующего run'а (event-driven).

    Args:
        run_id: UUID run'а

    Returns:
        RetryRunResponse с информацией о перезапуске

    Raises:
        HTTPException: Если run не найден или нельзя перезапустить
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Проверяем, можно ли перезапустить run
            if run.status == "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot retry completed run {run_id}",
                )

            # Сбрасываем статус на PENDING
            from .state_machine import RUN_STATUS_PENDING, validate_transition

            try:
                validate_transition(run.status, RUN_STATUS_PENDING, run_id=run_id)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot retry run {run_id}: {str(e)}",
                )

            run.status = RUN_STATUS_PENDING
            run.started_at = None
            run.finished_at = None
            run.error = None
            db.flush()
            db.commit()

            # Публикация события в очередь для перезапуска (event-driven)
            # Используем дефолтную очередь (fetcher.normal) для retry
            fetch_metadata_task.apply_async(
                args=[str(run_uuid)],
                queue="fetcher.normal",
            )

            logger.info(f"Retrying run {run_id}, queued to Celery")

            return RetryRunResponse(
                run_id=run_uuid,
                status=run.status,
                message="Ingestion queued for retry",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to retry run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry run: {str(e)}",
        )


@app.patch(
    "/api/v1/runs/{run_id}",
    response_model=UpdateRunResponse,
    tags=["api", "runs"],
)
async def update_run(
    run_id: str, request: UpdateRunRequest
) -> UpdateRunResponse:
    """Обновить run (например, запросить отмену).

    Args:
        run_id: UUID run'а
        request: Данные для обновления

    Returns:
        UpdateRunResponse с информацией об обновлении

    Raises:
        HTTPException: Если run не найден или нельзя обновить
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Проверяем, можно ли отменить run
            if run.status in ("completed", "cancelled", "failed"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot update run {run_id} in status {run.status}",
                )

            cancel_requested_value: Optional[bool] = None
            message = "Run updated"

            # Обрабатываем cancel_requested
            if request.cancel_requested is not None:
                if request.cancel_requested:
                    run.cancel_requested = True
                    cancel_requested_value = True
                    message = "Cancellation requested. Run will be cancelled at next checkpoint."
                else:
                    run.cancel_requested = False
                    cancel_requested_value = False
                    message = "Cancellation request removed"

            db.flush()

            return UpdateRunResponse(
                run_id=run_uuid,
                status=run.status,
                cancel_requested=cancel_requested_value,
                message=message,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update run: {str(e)}",
        )


@app.post(
    "/api/v1/runs/bulk",
    response_model=BulkCreateRunsResponse,
    tags=["api", "runs"],
)
async def bulk_create_runs(
    request: BulkCreateRunsRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> BulkCreateRunsResponse:
    """Создать несколько runs в одном запросе (bulk ingestion).

    Args:
        request: Список runs для создания
        idempotency_key: Опциональный ключ идемпотентности

    Returns:
        BulkCreateRunsResponse с результатами создания

    Raises:
        HTTPException: При ошибках валидации или создания
    """
    created: list[CreateRunResponse] = []
    duplicates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        with session_scope() as db:
            for run_item in request.runs:
                try:
                    # Нормализуем source URL
                    normalized = normalize_source(str(run_item.source_url))
                    platform = run_item.platform or normalized["platform"]
                    canonical_video_id = normalized.get("canonical_video_id")

                    # Проверка на дубликаты по canonical video ID
                    existing_run_id: Optional[UUID] = None
                    if canonical_video_id:
                        existing_video_source = (
                            db.query(VideoSource)
                            .filter(
                                VideoSource.platform == platform,
                                VideoSource.platform_video_id_canonical == canonical_video_id,
                            )
                            .first()
                        )
                        if existing_video_source:
                            existing_run_id = existing_video_source.run_id

                    if existing_run_id:
                        duplicates.append(
                            {
                                "requested_run_id": str(run_item.run_id),
                                "existing_run_id": str(existing_run_id),
                            }
                        )
                        continue

                    # Создаём run
                    run = Run(
                        id=run_item.run_id,
                        source_type="url",
                        source_url=str(run_item.source_url),
                        status=RUN_STATUS_PENDING,
                    )
                    db.add(run)

                    # Создаём video_source
                    video_source = VideoSource(
                        run_id=run_item.run_id,
                        platform=platform,
                        source_url=str(run_item.source_url),
                        normalized_video_id=normalized["video_id"],
                        platform_video_id_canonical=canonical_video_id,
                    )
                    db.add(video_source)
                    db.flush()
                    db.commit()

                    # Публикация события в очередь
                    # Определяем очередь на основе priority
                    priority = run_item.priority or "normal"
                    queue_name = f"fetcher.{priority}"  # fetcher.high, fetcher.normal, fetcher.low

                    fetch_metadata_task.apply_async(
                        args=[str(run_item.run_id)],
                        queue=queue_name,
                    )

                    created.append(
                        CreateRunResponse(
                            run_id=run_item.run_id,
                            status=RUN_STATUS_PENDING,
                            source_url=str(run_item.source_url),
                            platform=platform,
                            created_at=run.created_at,
                            message="Run created and queued for ingestion",
                            existing_run_id=None,
                        )
                    )

                except Exception as e:
                    logger.exception(f"Failed to create run {run_item.run_id}: {e}")
                    errors.append(
                        {
                            "run_id": str(run_item.run_id),
                            "error": str(e),
                        }
                    )

            return BulkCreateRunsResponse(
                created=created,
                duplicates=duplicates,
                errors=errors,
                total_requested=len(request.runs),
                total_created=len(created),
            )

    except Exception as e:
        logger.exception(f"Failed to bulk create runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to bulk create runs: {str(e)}",
        )


@app.get(
    "/api/v1/runs/{run_id}/events",
    response_model=RunEventsResponse,
    tags=["api", "runs", "events"],
)
async def get_run_events(run_id: str, limit: int = 100) -> RunEventsResponse:
    """Получить историю событий для run'а.

    Args:
        run_id: UUID run'а
        limit: Количество событий (default: 100, max: 500)

    Returns:
        RunEventsResponse с историей событий

    Raises:
        HTTPException: Если run не найден
    """
    limit = min(max(limit, 1), 500)

    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")

    try:
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run with id {run_id} not found",
                )

            # Получаем логи из FetchLog
            logs = (
                db.query(FetchLog)
                .filter(FetchLog.run_id == run_uuid)
                .order_by(FetchLog.created_at.desc())
                .limit(limit)
                .all()
            )

            # Получаем jobs для истории статусов
            jobs = (
                db.query(FetchJob)
                .filter(FetchJob.run_id == run_uuid)
                .order_by(FetchJob.created_at.desc())
                .all()
            )

            events: list[RunEventItem] = []

            # Добавляем события из логов
            for log in reversed(logs):  # В хронологическом порядке
                events.append(
                    RunEventItem(
                        event_type="log.line",
                        timestamp=log.created_at,
                        stage=log.stage,
                        message=log.message,
                        level=log.level,
                    )
                )

            # Добавляем события из jobs (status changes)
            for job in reversed(jobs):
                if job.started_at:
                    events.append(
                        RunEventItem(
                            event_type="job.started",
                            timestamp=job.started_at,
                            stage=job.job_type,
                            message=f"Job {job.job_type} started",
                        )
                    )
                if job.finished_at:
                    events.append(
                        RunEventItem(
                            event_type="job.finished" if job.status == "completed" else "job.failed",
                            timestamp=job.finished_at,
                            stage=job.job_type,
                            status=job.status,
                            message=f"Job {job.job_type} {job.status}",
                        )
                    )

            # Сортируем по времени
            events.sort(key=lambda e: e.timestamp)

            return RunEventsResponse(
                run_id=run_uuid,
                events=events,
                total=len(events),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get events for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get events: {str(e)}",
        )


@app.get(
    "/api/v1/queue",
    response_model=QueueResponse,
    tags=["api", "monitoring"],
)
async def get_queue_info() -> QueueResponse:
    """Получить информацию о глубине очередей (читает из Redis и обновляет метрики).

    Returns:
        QueueResponse с информацией о каждой очереди (pending из Redis LLEN)
    """
    try:
        from .celery_queues import (
            CELERY_QUEUE_NAMES,
            get_celery_queue_lengths,
            update_celery_queue_metrics,
        )

        lengths = get_celery_queue_lengths()
        update_celery_queue_metrics()

        queues = [
            QueueInfo(
                queue_name=name,
                pending=lengths.get(name, 0),
                running=0,
                retry=0,
            )
            for name in CELERY_QUEUE_NAMES
        ]

        kafka_lag: Optional[int] = None
        # При включённом Kafka lag можно получать из Kafka Admin API / consumer group metrics

        return QueueResponse(
            queues=queues,
            total_pending=sum(q.pending for q in queues),
            total_running=sum(q.running for q in queues),
            total_retry=sum(q.retry for q in queues),
            kafka_consumer_lag=kafka_lag,
        )

    except Exception as e:
        logger.exception(f"Failed to get queue info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get queue info: {str(e)}",
        )


@app.get(
    "/api/v1/limits",
    response_model=LimitsResponse,
    tags=["api", "monitoring"],
)
async def get_limits() -> LimitsResponse:
    """Получить информацию о системных лимитах.

    Returns:
        LimitsResponse с информацией о лимитах
    """
    try:
        # Получаем текущее использование из БД
        with session_scope() as db:
            now = datetime.now(timezone.utc)
            one_hour_ago = now - timedelta(hours=1)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            runs_this_hour = (
                db.query(Run)
                .filter(Run.created_at >= one_hour_ago)
                .count()
            )
            runs_today = (
                db.query(Run)
                .filter(Run.created_at >= today_start)
                .count()
            )

        return LimitsResponse(
            rate_limits={
                "runs_per_minute": settings.runs_per_minute,
                "runs_per_hour": settings.runs_per_hour,
                "runs_per_day": settings.runs_per_day,
            },
            resource_limits={
                "max_video_size_mb": settings.max_video_size_mb,
                "max_video_duration_seconds": settings.max_video_duration_seconds,
                "max_comments_per_video": settings.max_comments_per_video,
            },
            platform_limits={
                "youtube": {
                    "max_requests_per_minute": settings.youtube_metadata_limit_per_window // settings.youtube_metadata_window_sec * 60,
                },
            },
            current_usage={
                "runs_today": runs_today,
                "runs_this_hour": runs_this_hour,
            },
        )

    except Exception as e:
        logger.exception(f"Failed to get limits: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get limits: {str(e)}",
        )


@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["api", "monitoring"],
)
async def get_stats(period: str = "24h") -> StatsResponse:
    """Получить статистику по ingestion (читает из prepared cache).

    Args:
        period: Период статистики (1h, 24h, 7d, 30d)

    Returns:
        StatsResponse со статистикой

    Raises:
        HTTPException: При ошибках чтения статистики
    """
    try:
        from .rate_limiter import get_redis_client

        redis_client = get_redis_client()
        cache_key = f"fetcher:stats:{period}"

        # Пытаемся получить из cache
        cached_stats = redis_client.get(cache_key)
        if cached_stats:
            import json
            return StatsResponse(**json.loads(cached_stats))

        # Fallback: вычисляем из БД используя stats_aggregator
        from .stats_aggregator import compute_stats_for_period

        stats = compute_stats_for_period(period)

        # Сохраняем в cache на 5 минут
        if redis_client:
            import json
            redis_client.setex(
                cache_key,
                300,  # 5 минут
                json.dumps(stats.dict()),
            )

        return stats

    except Exception as e:
        logger.exception(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}",
        )


__all__ = ["app"]

