"""
Health Check Endpoint

GET /api/v1/health - проверка здоровья API сервера

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1287-1328)
"""

import logging
import time
import uuid
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Annotated, Dict, Any, Optional
import httpx

from api.schemas.responses import HealthResponse
from api.dependencies import StorageDep, TaskManagerDep
from api.main import get_uptime_seconds
from api.config import config
from api.services.redis_client import check_redis_health, get_redis_client
from api.services.redis_schema import get_effective_active_runs_count
from api.services.queue import get_total_queue_length
from api.utils.retry import retry_storage_operation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

# Глобальная переменная для отслеживания общего количества run'ов за сегодня
# В production это должно быть в Redis или БД
_total_runs_today: Dict[str, int] = {}  # date -> count


async def check_triton_health() -> Dict[str, Any]:
    """
    Проверить здоровье Triton Inference Server.
    
    Выполняет HTTP запрос к Triton health endpoint с retry логикой.
    Timeout 30 сек, retry 3 раза с exponential backoff.
    
    Returns:
        Словарь с информацией о статусе Triton
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2241)
    """
    from api.utils.retry import retry_triton_operation, is_triton_timeout_error
    
    triton_endpoint = getattr(config, 'triton_endpoint', None)
    
    if not triton_endpoint:
        return {
            "status": "not_configured",
            "message": "Triton endpoint not configured"
        }
    
    async def _check_triton():
        """Внутренняя функция для проверки Triton с retry."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{triton_endpoint}/v2/health/ready")
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "endpoint": triton_endpoint
                }
            else:
                # Не transient ошибка - не повторяем
                return {
                    "status": "unhealthy",
                    "endpoint": triton_endpoint,
                    "error": f"Triton returned status code {response.status_code}"
                }
    
    try:
        # Использовать retry для Triton timeout
        return await retry_triton_operation(
            _check_triton,
            timeout=30.0,
            max_attempts=3
        )
    except httpx.TimeoutException as e:
        # Timeout после всех попыток
        return {
            "status": "unhealthy",
            "endpoint": triton_endpoint,
            "error": f"Triton health check timeout after retries: {e}"
        }
    except Exception as e:
        # Другие ошибки
        if is_triton_timeout_error(e):
            # Это timeout, но retry не помог
            return {
                "status": "unhealthy",
                "endpoint": triton_endpoint,
                "error": f"Triton timeout after retries: {e}"
            }
        logger.warning(f"Triton health check failed: {e}")
        return {
            "status": "unhealthy",
            "endpoint": triton_endpoint,
            "error": str(e)
        }


def get_total_runs_today() -> int:
    """
    Получить общее количество run'ов за сегодня.
    
    В MVP использует in-memory счетчик.
    В production должно быть в Redis или БД.
    
    Returns:
        Количество run'ов за сегодня
    """
    today = date.today().isoformat()
    return _total_runs_today.get(today, 0)


def increment_total_runs_today() -> None:
    """
    Увеличить счетчик run'ов за сегодня.
    
    Вызывается при создании нового run'а.
    """
    today = date.today().isoformat()
    _total_runs_today[today] = _total_runs_today.get(today, 0) + 1


async def check_storage_health(storage) -> Dict[str, Any]:
    """
    Проверить здоровье Storage.
    
    Выполняет тестовую запись/чтение для проверки доступности.
    Использует retry логику для transient errors (exponential backoff: 1s, 2s, 4s, 8s).
    
    Должна вызываться только из async-контекста (FastAPI): нельзя использовать asyncio.run()
    внутри уже запущенного event loop uvicorn — это давало HTTP 500 на GET /health.
    
    Args:
        storage: Экземпляр Storage
        
    Returns:
        Словарь с информацией о статусе Storage
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2242)
    """
    try:
        # Создать уникальный тестовый ключ
        test_key = f"__health_check__/{uuid.uuid4().hex[:8]}.txt"
        test_data = f"health_check_{time.time()}".encode("utf-8")
        
        # Проверка записи с retry
        await retry_storage_operation(
            storage.atomic_write_bytes,
            test_key, test_data, content_type="text/plain"
        )
        
        # Проверка чтения с retry
        read_data = await retry_storage_operation(
            storage.read_bytes,
            test_key
        )
        if read_data != test_data:
            raise ValueError("Read data doesn't match written data")
        
        # Проверка exists с retry
        exists = await retry_storage_operation(
            storage.exists,
            test_key
        )
        if not exists:
            raise ValueError("File doesn't exist after write")
        
        # Попытка удаления (если метод доступен)
        storage_info = {
            "status": "healthy",
            "type": getattr(storage, "__class__", {}).__name__ if hasattr(storage, "__class__") else "unknown"
        }
        
        # Добавить информацию о типе storage если доступно
        storage_class_name = storage.__class__.__name__ if hasattr(storage, "__class__") else "unknown"
        if "FileSystem" in storage_class_name or hasattr(storage, "fs_root"):
            storage_info["type"] = "fs"
            if hasattr(storage, "fs_root"):
                storage_info["base_path"] = getattr(storage, "fs_root", "unknown")
        elif "S3" in storage_class_name or hasattr(storage, "bucket"):
            storage_info["type"] = "s3"
            if hasattr(storage, "bucket"):
                storage_info["bucket"] = getattr(storage, "bucket", "unknown")
        else:
            storage_info["type"] = storage_class_name
        
        return storage_info
        
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get(
    "",
    response_model=HealthResponse,
    summary="Проверка здоровья API",
    description="""
    Проверяет состояние API сервера и его зависимостей.
    
    ## Проверяемые компоненты
    
    * **API**: Базовое состояние сервера
    * **Storage**: Доступность хранилища (FS/S3) через тестовую запись/чтение
    * **Redis**: Доступность Redis (если настроен)
    * **Triton**: Доступность Triton Inference Server (если настроен)
    
    ## Статусы
    
    * `healthy` - Все критичные сервисы работают
    * `degraded` - API работает, но некритичные сервисы недоступны
    * `unhealthy` - Критичные сервисы недоступны (возвращается 503)
    
    ## Метрики
    
    Включает метрики:
    * `active_runs` - Количество активных run'ов
    * `queue_length` - Длина очереди
    * `total_runs_today` - Общее количество run'ов за сегодня
    * `max_concurrent_runs` - Максимальное количество параллельных run'ов
    
    ## Примеры ответов
    
    ### Здоровый сервис (200 OK)
    ```json
    {
        "status": "healthy",
        "api": "healthy",
        "storage": "healthy",
        "version": "0.1.0",
        "uptime_seconds": 3600.0,
        "dependencies": {
            "storage": {"status": "healthy", "type": "fs"},
            "redis": {"status": "healthy"},
            "triton": {"status": "not_configured"}
        },
        "metrics": {
            "active_runs": 2,
            "queue_length": 5,
            "total_runs_today": 100,
            "max_concurrent_runs": 4
        },
        "timestamp": "2024-01-01T12:00:00Z"
    }
    ```
    
    ### Недоступный сервис (503 Service Unavailable)
    ```json
    {
        "status": "unhealthy",
        "api": "healthy",
        "storage": "unhealthy",
        "version": "0.1.0",
        "uptime_seconds": 3600.0,
        "dependencies": {
            "storage": {"status": "unhealthy", "error": "Storage read/write test failed"}
        },
        "metrics": {},
        "timestamp": "2024-01-01T12:00:00Z"
    }
    ```
    """,
    responses={
        200: {
            "description": "Сервис здоров",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "api": "healthy",
                        "storage": "healthy",
                        "version": "0.1.0",
                        "uptime_seconds": 3600.0,
                        "dependencies": {},
                        "metrics": {},
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        },
        503: {
            "description": "Сервис недоступен",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "api": "healthy",
                        "storage": "unhealthy",
                        "version": "0.1.0",
                        "uptime_seconds": 3600.0,
                        "dependencies": {},
                        "metrics": {},
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                }
            }
        }
    },
    tags=["health"]
)
async def health_check(
    storage: StorageDep,
    task_manager: TaskManagerDep
):
    """
    Проверка здоровья API сервера.
    
    Проверяет:
    - Работает ли API
    - Доступен ли Storage (запись/чтение тестового файла)
    - Доступен ли Redis (если настроен)
    - Доступен ли Triton (если настроен)
    - Метрики (активные run'ы, длина очереди, общее количество run'ов за сегодня)
    
    Args:
        storage: Storage dependency
        task_manager: TaskManager dependency
        
    Returns:
        HealthResponse с информацией о здоровье сервиса
        Или JSONResponse с 503 статусом если unhealthy
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1287-1328, 2433-2484)
    """
    try:
        # Базовая проверка API
        api_status = "healthy"
        
        # Проверка Storage
        storage_info = await check_storage_health(storage)
        storage_status = storage_info.get("status", "unknown")
        
        # Проверка Redis (опционально)
        redis_health = await check_redis_health()
        
        # Проверка Triton (опционально)
        triton_health = await check_triton_health()
        
        # Активные run: по Redis run:state (worker в другом процессе обновляет статус там).
        active_runs_count = await get_effective_active_runs_count(task_manager)
        queue_length_total = await get_total_queue_length()
        total_runs_today = get_total_runs_today()
        
        # Обновить метрику активных run'ов в Prometheus
        try:
            from api.services.metrics import active_runs as active_runs_metric
            active_runs_metric.set(active_runs_count)
        except Exception as e:
            logger.debug(f"Failed to update active_runs metric: {e}")
        
        # Определить общий статус
        # Критичные сервисы: API и Storage
        # Некритичные сервисы: Redis и Triton (если настроены)
        overall_status = "healthy"
        
        if api_status != "healthy" or storage_status == "unhealthy":
            overall_status = "unhealthy"
        elif storage_status == "unknown":
            overall_status = "degraded"
        
        # Если Redis настроен но недоступен, статус degraded (но не unhealthy)
        if redis_health.get("status") == "unhealthy" and redis_health.get("status") != "not_configured":
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Если Triton настроен но недоступен, статус degraded (но не unhealthy)
        if triton_health.get("status") == "unhealthy" and triton_health.get("status") != "not_configured":
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Получить uptime
        uptime_seconds = get_uptime_seconds()
        
        # Формировать ответ
        dependencies = {
            "storage": storage_info,
            "redis": redis_health,
            "triton": triton_health
        }
        
        metrics = {
            "active_runs": active_runs_count,
            "queue_length": queue_length_total,
            "total_runs_today": total_runs_today,
            "max_concurrent_runs": config.max_concurrent_runs
        }
        
        health_response = HealthResponse(
            status=overall_status,
            api=api_status,
            storage=storage_status,
            version=config.api_version if hasattr(config, "api_version") else "0.1.0",
            uptime_seconds=uptime_seconds,
            dependencies=dependencies,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        # Возврат 503 если unhealthy
        if overall_status == "unhealthy":
            return JSONResponse(
                status_code=503,
                content=health_response.model_dump()
            )
        
        return health_response
        
    except Exception as e:
        logger.exception(f"Unexpected error in health_check: {e}")
        error_response = HealthResponse(
            status="unhealthy",
            api="unknown",
            storage="unknown",
            version=config.api_version if hasattr(config, "api_version") else "0.1.0",
            uptime_seconds=get_uptime_seconds(),
            dependencies={},
            metrics={},
            timestamp=datetime.now()
        )
        return JSONResponse(
            status_code=503,
            content=error_response.model_dump()
        )

