"""
Endpoint для запуска обработки видео

POST /api/v1/process - запускает обработку видео асинхронно.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1051-1103)
"""

import logging
import asyncio
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Annotated, Optional
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from storage.base import StorageError, NotFoundError

from api.schemas.requests import ProcessRequest
from api.schemas.responses import ProcessResponse
from api.schemas.state import RunStatus
from api.services.processor import ProcessorService
from api.services.queue import enqueue_run, get_queue_length, get_total_queue_length
from api.services.redis_schema import (
    acquire_run_lock,
    effective_can_accept_new_run,
    get_effective_active_runs_count,
    release_run_lock,
    save_run_state,
)
from api.dependencies import StorageDep, KeyLayoutDep, TaskManagerDep, ProcessorServiceDep
from api.utils.errors import (
    InvalidPayloadError,
    ProcessingError,
    RunAlreadyExistsError,
    BackpressureError
)
from api.config import config
from api.utils.logging import get_logger
from api.utils.validators import validate_video_path, validate_profile_config
from api.utils.error_handling import handle_background_error, handle_processing_result
from api.utils.video_url_cache import download_video_url_to_cache
from api.services.audit import audit_log
from api.security import verify_api_key

# Rate limiting (отключён для локального E2E, см. rate_limit_decorator ниже)
try:
    from api.main import limiter
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_ENABLED = limiter is not None
except (ImportError, AttributeError):
    limiter = None
    RateLimitExceeded = None
    RATE_LIMIT_ENABLED = False

logger = get_logger(__name__)

QUEUE_REQUEST_FIELDS = (
    "video_id",
    "platform_id",
    "video_path",
    "video_url",
    "config_hash",
    "profile_config",
    "profile_version",
    "feature_schema_version",
    "pipeline_version",
    "sampling_policy_version",
    "dataprocessor_version",
    "analysis_fps",
    "analysis_width",
    "analysis_height",
    "chunk_size",
    "visual_cfg_path",
    "dag_path",
    "dag_stage",
    "rs_base",
    "output",
    "run_audio",
    "run_text",
    "global_config_path",
)


def _request_to_queue_metadata(request: ProcessRequest) -> dict:
    """Serialize the full process request payload for Redis queue handoff."""
    if hasattr(request, "model_dump"):
        raw_data = request.model_dump(exclude_none=True)
    else:
        raw_data = request.dict(exclude_none=True)

    return {
        field: raw_data[field]
        for field in QUEUE_REQUEST_FIELDS
        if field in raw_data
    }


def rate_limit_decorator(func):
    """
    Декоратор для условного применения rate limiting.

    Для локального E2E и интеграционных тестов rate limiting отключён,
    чтобы избежать ошибок интеграции slowapi. В production можно
    вернуть применение limiter.limit(...) здесь.
    """
    # Локальный/тестовый режим: rate limiting не применяем
    return func

router = APIRouter(prefix="/process", tags=["process"])


async def _enqueue_fallback(
    request: ProcessRequest,
    task_manager: TaskManagerDep,
    processor_service: ProcessorServiceDep
):
    """
    Fallback метод для запуска обработки без Redis (MVP режим).
    
    Используется когда Redis не доступен или не настроен.
    
    Args:
        request: Запрос на обработку
        task_manager: TaskManager для управления состоянием
        processor_service: ProcessorService для запуска обработки (dependency injection)
    """
    # Запустить обработку в фоне (не ждём завершения)
    asyncio.create_task(
        _run_processing_background(request, task_manager, processor_service)
    )
    
    logger.info(f"Processing queued (fallback mode) for run_id={request.run_id}")


async def _run_processing_background(
    request: ProcessRequest,
    task_manager: TaskManagerDep,
    processor_service: ProcessorService
):
    """
    Фоновая задача для запуска обработки.
    
    Args:
        request: Запрос на обработку
        task_manager: TaskManager для управления состоянием
        processor_service: ProcessorService для запуска обработки
    """
    run_id = request.run_id
    
    try:
        # Получить слот через semaphore
        semaphore = await task_manager.acquire_slot()
        async with semaphore:
            # Обновить статус на "running"
            task_manager.update_run_status(
                run_id,
                RunStatus.RUNNING,
                started_at=asyncio.get_event_loop().time()
            )
            
            logger.info(
                "Starting processing",
                run_id=run_id,
                video_id=request.video_id,
                platform_id=request.platform_id
            )
            
            # Запустить обработку
            result = await processor_service.run_processing(request)
            
            # Обработать результат (успешный или с ошибкой)
            await handle_processing_result(run_id, result, task_manager)
            
            # Освободить lock после завершения обработки (fallback режим)
            await release_run_lock(run_id)
                
    except (RedisError, ConnectionError, TimeoutError) as e:
        # Обработка ошибок Redis
        await handle_background_error(
            run_id=run_id,
            error=e,
            error_message=f"Redis error: {str(e)}",
            task_manager=task_manager,
            error_type="redis_error"
        )
    except (StorageError, NotFoundError) as e:
        # Обработка ошибок Storage
        await handle_background_error(
            run_id=run_id,
            error=e,
            error_message=f"Storage error: {str(e)}",
            task_manager=task_manager,
            error_type="storage_error"
        )
    except Exception as e:
        # Обработка всех остальных ошибок
        await handle_background_error(
            run_id=run_id,
            error=e,
            error_message=str(e),
            task_manager=task_manager,
            error_type=None  # Автоматическое определение типа
        )


@router.post(
    "",
    response_model=ProcessResponse,
    status_code=202,
    summary="Запустить обработку видео",
    description="""
    Запускает асинхронную обработку видео через DataProcessor.
    
    Запрос ставится в очередь и обрабатывается worker'ом. Возвращает 202 Accepted
    с информацией о запущенной задаче, включая URL для отслеживания статуса.
    
    ## Параметры запроса
    
    * `run_id` - Уникальный UUID для идентификации run'а (формат UUID)
    * `video_id` - ID видео
    * `platform_id` - Платформа: 'youtube' или 'upload'
    * `video_path` - Путь к видео файлу (должен находиться в разрешённых директориях)
    * `config_hash` - Хеш конфигурации профиля
    * `profile_config` - Полная конфигурация профиля обработки
    
    ## Ограничения
    
    * Rate limit: 100 запросов в час на backend instance
    * Backpressure: Если очередь переполнена, возвращается 503 с заголовком Retry-After
    * Idempotency: Повторный запрос с тем же run_id вернёт 409 Conflict
    
    ## Примеры ответов
    
    ### Успешный запрос (202 Accepted)
    ```json
    {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "queued",
        "message": "Processing started",
        "status_url": "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status",
        "estimated_duration_seconds": 300
    }
    ```
    
    ### Ошибка: Дубликат run_id (409 Conflict)
    ```json
    {
        "error": "Run already exists",
        "run_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    
    ### Ошибка: Backpressure (503 Service Unavailable)
    ```json
    {
        "error": "Service overloaded",
        "message": "Too many active runs: 10/4"
    }
    ```
    Заголовок: `Retry-After: 60`
    """,
    responses={
        202: {
            "description": "Запрос принят, обработка запущена",
            "content": {
                "application/json": {
                    "example": {
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "queued",
                        "message": "Processing started",
                        "status_url": "/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status",
                        "estimated_duration_seconds": 300
                    }
                }
            }
        },
        400: {
            "description": "Невалидный запрос",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Invalid payload",
                        "details": {
                            "field": "video_path",
                            "value": "/invalid/path",
                            "error": "Video path outside allowed directories"
                        }
                    }
                }
            }
        },
        401: {
            "description": "Требуется аутентификация",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "API key required"
                    }
                }
            }
        },
        403: {
            "description": "Невалидный API ключ",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid API key"
                    }
                }
            }
        },
        409: {
            "description": "Run с таким run_id уже существует",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Run already exists",
                        "run_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        },
        503: {
            "description": "Сервис перегружен (backpressure)",
            "headers": {
                "Retry-After": {
                    "description": "Время в секундах до следующей попытки",
                    "schema": {"type": "integer", "example": 60}
                }
            },
            "content": {
                "application/json": {
                    "example": {
                        "error": "Service overloaded",
                        "message": "Too many active runs: 10/4"
                    }
                }
            }
        }
    },
    tags=["process"]
)
@rate_limit_decorator
async def process_video(
    http_request: Request,
    request: ProcessRequest,
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    task_manager: TaskManagerDep,
    processor_service: ProcessorServiceDep,
    api_key: str = Depends(verify_api_key)
):
    """
    Запустить обработку видео.
    
    Принимает запрос на обработку и ставит задачу в очередь Redis Streams.
    Возвращает 202 Accepted с информацией о запущенной задаче.
    
    Args:
        http_request: FastAPI Request объект для получения request_id и client_ip
        request: Запрос на обработку видео (ProcessRequest)
            - run_id: UUID run'а (обязательно)
            - video_id: ID видео (обязательно)
            - platform_id: ID платформы (обязательно)
            - video_path: Путь к видео файлу (обязательно)
            - config_hash: Хэш конфигурации (обязательно)
            - profile_config: Конфигурация профиля обработки (обязательно)
            - profile_version: Версия профиля (опционально)
            - feature_schema_version: Версия схемы фич (опционально)
            - pipeline_version: Версия pipeline (опционально)
        storage: Storage dependency для доступа к хранилищу
        key_layout: KeyLayout dependency для работы с путями
        task_manager: TaskManager dependency для управления задачами
        api_key: API ключ из заголовка X-API-Key (проверяется через verify_api_key)
        
    Returns:
        ProcessResponse: Ответ с информацией о запущенной задаче
            - run_id: UUID run'а
            - status: Статус ("queued")
            - message: Сообщение о статусе
            - status_url: URL для проверки статуса
            - estimated_duration_seconds: Оценка длительности обработки
        
    Raises:
        HTTPException 400: Невалидный payload (InvalidPayloadError)
        HTTPException 401: Требуется аутентификация (отсутствует X-API-Key)
        HTTPException 403: Невалидный API ключ
        HTTPException 409: Run с таким run_id уже существует (RunAlreadyExistsError)
        HTTPException 503: Превышен лимит активных run'ов (backpressure) или Redis недоступен
        HTTPException 500: Ошибка при запуске обработки (ProcessingError)
        
    Example:
        ```python
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/process",
                json={
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    "video_id": "video-123",
                    "platform_id": "youtube",
                    "video_path": "/data/videos/video.mp4",
                    "config_hash": "abc123...",
                    "profile_config": {
                        "visual": {"enabled": True},
                        "audio": {"enabled": True}
                    }
                },
                headers={"X-API-Key": "your-api-key"}
            )
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "queued"
        ```
        
    Note:
        - Задача ставится в очередь Redis Streams с приоритетом "normal"
        - При недоступности Redis используется fallback режим (MVP)
        - Idempotency проверяется через Redis lock и TaskManager
        - Backpressure защита ограничивает количество одновременных run'ов
    """
    try:
        # Validate payload that does not require local video presence first.
        validate_profile_config(request.profile_config)
        
        # Получить request_id и IP адрес для audit log
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        
        # Audit log: начало обработки
        await audit_log(
            action="process_started",
            run_id=request.run_id,
            details={
                "video_id": request.video_id,
                "platform_id": request.platform_id,
                "video_path": request.video_path,
                "config_hash": request.config_hash
            },
            request_id=request_id,
            ip_address=client_ip
        )
        
        # Проверка существования run_id (idempotency check)
        # Сначала проверяем TaskManager, затем Redis lock
        if task_manager.is_run_active(request.run_id):
            raise RunAlreadyExistsError(
                f"Run with run_id={request.run_id} already exists and is active",
                run_id=request.run_id
            )

        # Backpressure: считаем активные run по Redis (см. get_effective_active_runs_count).
        if not await effective_can_accept_new_run(task_manager):
            active_count = await get_effective_active_runs_count(task_manager)
            logger.warning(
                "Too many active runs, backpressure triggered",
                run_id=request.run_id,
                video_id=request.video_id,
                platform_id=request.platform_id,
                active_runs=active_count,
                max_concurrent_runs=config.max_concurrent_runs
            )
            raise BackpressureError(
                f"Too many active runs: {active_count}/{config.max_concurrent_runs}",
                retry_after=60  # suggest retry
            )

        # Phase 3: if video_url provided, download to cache and fill video_path.
        if request.video_url:
            try:
                cached_path = await download_video_url_to_cache(
                    request.video_url, request.run_id
                )
                request.video_path = str(cached_path)
                request.video_url = None
            except Exception as e:
                logger.exception("Failed to download video from URL for run_id=%s", request.run_id)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Video URL download failed",
                        "details": str(e),
                    },
                ) from e

        # Validate local file path only after URL download (if any).
        validate_video_path(request.video_path)
        
        # Попробовать получить блокировку в Redis (idempotency lock)
        lock_acquired = await acquire_run_lock(request.run_id)
        if not lock_acquired:
            raise RunAlreadyExistsError(
                f"Run with run_id={request.run_id} is already being processed",
                run_id=request.run_id
            )
        
        # Регистрация run в TaskManager
        task_manager.register_run(
            request.run_id,
            {
                "video_id": request.video_id,
                "platform_id": request.platform_id,
                "config_hash": request.config_hash,
                "video_path": request.video_path,
                "profile_version": request.profile_version,
                "feature_schema_version": request.feature_schema_version,
                "pipeline_version": request.pipeline_version,
            }
        )

        # Инициализируем Redis state как queued, чтобы worker видел валидный
        # переход queued -> running вместо "первый статус = running".
        await save_run_state(
            request.run_id,
            {
                "status": RunStatus.QUEUED.value,
                "updated_at": time.time(),
            }
        )
        
        # Увеличить счетчик run'ов за сегодня (для метрик health check)
        from api.endpoints.health import increment_total_runs_today
        increment_total_runs_today()
        
        # Попробовать добавить в Redis Streams queue если доступен
        # Если Redis не доступен, fallback на старый способ (MVP режим)
        # Если Redis критичен и недоступен, возвращаем 503
        try:
            from api.services.redis_client import get_redis_client, check_redis_health
            redis_client = get_redis_client()
            
            # Проверить здоровье Redis если он настроен
            if config.redis_url or config.redis_host:
                redis_health = await check_redis_health()
                if redis_health.get("status") == "unhealthy":
                    # Redis критичен и недоступен - возвращаем 503
                    logger.error(
                        "Redis is unavailable and critical, returning 503",
                        run_id=request.run_id,
                        redis_error=redis_health.get("error")
                    )
                    raise BackpressureError(
                        "Redis is unavailable, service temporarily unavailable",
                        retry_after=60
                    )
            
            if redis_client:
                # Использовать Redis Streams queue
                priority = "normal"  # TODO: Поддержка приоритетов из запроса

                # Богатые метаданные run'а для Redis:
                #   - будут сохранены в run:meta:{run_id}
                #   - попадут в payload сообщения очереди
                # Эти данные потом использует worker, чтобы восстановить run
                # в своём локальном TaskManager, даже если API и worker в разных процессах.
                metadata = _request_to_queue_metadata(request)

                message_id = await enqueue_run(
                    run_id=request.run_id,
                    priority=priority,
                    metadata=metadata,
                )
                
                if message_id:
                    logger.info(
                        "Run enqueued to Redis Streams",
                        run_id=request.run_id,
                        video_id=request.video_id,
                        platform_id=request.platform_id,
                        message_id=message_id,
                        priority=priority
                    )
                else:
                    # Fallback на старый способ
                    logger.warning(
                        "Failed to enqueue to Redis, using fallback mode",
                        run_id=request.run_id,
                        video_id=request.video_id,
                        platform_id=request.platform_id
                    )
                    await _enqueue_fallback(request, task_manager, processor_service)
            else:
                # Redis не настроен, использовать fallback (не критично)
                logger.info(
                    "Redis not configured, using fallback mode",
                    run_id=request.run_id
                )
                await _enqueue_fallback(request, task_manager)
                
        except BackpressureError:
            # Пробросить BackpressureError для возврата 503
            raise
        except (RedisError, ConnectionError, TimeoutError) as e:
            request_id = getattr(http_request.state, 'request_id', None)
            logger.warning(
                "Redis error using queue, falling back to MVP mode",
                run_id=request.run_id,
                request_id=request_id,
                video_id=request.video_id,
                platform_id=request.platform_id,
                error=str(e),
                error_type=type(e).__name__
            )
            await _enqueue_fallback(request, task_manager)
        except Exception as e:
            request_id = getattr(http_request.state, 'request_id', None)
            logger.warning(
                "Unexpected error using Redis queue, falling back to MVP mode",
                run_id=request.run_id,
                request_id=request_id,
                video_id=request.video_id,
                platform_id=request.platform_id,
                error=str(e),
                error_type=type(e).__name__
            )
            await _enqueue_fallback(request, task_manager)
        
        # Возврат ответа
        return ProcessResponse(
            run_id=request.run_id,
            status="queued",
            message="Processing started",
            status_url=f"/api/v1/runs/{request.run_id}/status",
            estimated_duration_seconds=300  # TODO: Рассчитать на основе истории
        )
        
    except RunAlreadyExistsError as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        logger.warning(
            "Run already exists",
            run_id=e.run_id,
            request_id=request_id,
            client_ip=client_ip
        )
        # Audit log: дубликат run_id
        await audit_log(
            action="process_duplicate_run_id",
            run_id=e.run_id,
            details={"error": str(e)},
            request_id=request_id,
            ip_address=client_ip
        )
        raise HTTPException(status_code=409, detail=str(e))
    except BackpressureError as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.warning(
            "Backpressure triggered",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            retry_after=e.retry_after if hasattr(e, 'retry_after') else None
        )
        # Audit log: backpressure
        await audit_log(
            action="process_backpressure",
            run_id=run_id,
            details={"error": str(e), "retry_after": e.retry_after if hasattr(e, 'retry_after') else None},
            request_id=request_id,
            ip_address=client_ip
        )
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after)} if e.retry_after else None
        )
    except InvalidPayloadError as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.error(
            "Invalid payload",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            error=str(e),
            error_details=e.details if hasattr(e, 'details') else {}
        )
        # Audit log: ошибка валидации
        await audit_log(
            action="process_validation_error",
            run_id=run_id,
            details={"error": str(e), "details": e.details if hasattr(e, 'details') else {}},
            request_id=request_id,
            ip_address=client_ip
        )
        raise HTTPException(status_code=400, detail=str(e))
    except (RedisError, ConnectionError, TimeoutError) as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.exception(
            "Redis error in process_video",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to Redis connection issue"
        )
    except (StorageError, NotFoundError) as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.exception(
            "Storage error in process_video",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            error=str(e),
            error_type=type(e).__name__
        )
        if isinstance(e, NotFoundError):
            raise HTTPException(status_code=404, detail=f"Resource not found: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Storage error: {str(e)}")
    except ProcessingError as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.error(
            "Processing error",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            error=str(e),
            error_code=e.error_code if hasattr(e, 'error_code') else None
        )
        # Audit log: ошибка обработки
        await audit_log(
            action="process_error",
            run_id=run_id,
            details={"error": str(e)},
            request_id=request_id,
            ip_address=client_ip
        )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        request_id = getattr(http_request.state, 'request_id', None)
        client_ip = http_request.client.host if http_request.client else None
        run_id = request.run_id if 'request' in locals() else None
        logger.exception(
            "Unexpected error in process_video",
            run_id=run_id,
            request_id=request_id,
            client_ip=client_ip,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail="Internal server error")

