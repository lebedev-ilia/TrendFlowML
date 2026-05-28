"""
Worker Entry Point - точка входа для отдельного worker процесса

Этот модуль запускает worker процесс, который читает задачи из Redis Streams
и обрабатывает их через subprocess isolation.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2619-2622, 686-728)

Запуск:
    python -m api.worker
    или
    python api/worker.py

В Docker:
    CMD ["python", "-m", "api.worker"]
"""

import asyncio
import logging
import signal
import sys
import os
from typing import Optional

from api.config import config
from api.services.worker import Worker
from api.services.redis_client import init_redis_client, close_redis_client
from api.dependencies import get_storage, get_key_layout
from storage.base import Storage
from storage.paths import KeyLayout

# Настройка логирования
def setup_logging():
    """
    Настройка логирования для worker процесса.
    
    Использует те же настройки что и API (из config).
    """
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    if config.log_format == "json":
        # JSON формат для production
        try:
            from pythonjsonlogger import jsonlogger
            
            log_handler = logging.StreamHandler()
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
            log_handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            root_logger.addHandler(log_handler)
            
            # Удаляем дефолтный handler если есть
            if root_logger.handlers:
                root_logger.handlers = [log_handler]
        except ImportError:
            # Fallback на обычный формат если python-json-logger не установлен
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        # Текстовый формат для development
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


# Инициализация логирования
setup_logging()
from api.utils.logging import get_logger
logger = get_logger(__name__)

# Глобальная переменная для worker
_worker: Optional[Worker] = None


def get_worker_id() -> str:
    """
    Получить уникальный ID worker'а.
    
    Использует переменную окружения WORKER_ID из config или hostname + PID.
    """
    from api.config import config
    if config.worker_id:
        return config.worker_id
    
    import socket
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


def _maybe_start_worker_metrics_http() -> None:
    """
    Опциональный HTTP /metrics (prometheus_client) в процессе worker.
    Гистограммы/счётчики (processing_time, failures) обновляются в worker; при scrape только
    с API они не видны. См. monitoring/README.md §«Пилот 15 (15 видео)».
    """
    port_raw = (os.environ.get("DP_WORKER_METRICS_PORT") or "").strip()
    if not port_raw:
        return
    try:
        port = int(port_raw, 10)
    except ValueError:
        logger.warning("DP_WORKER_METRICS_PORT invalid, ignoring: %r", port_raw)
        return
    if port < 1 or port > 65535:
        logger.warning("DP_WORKER_METRICS_PORT out of range, ignoring: %s", port)
        return
    from prometheus_client import start_http_server

    start_http_server(port)
    logger.info("Worker Prometheus metrics: http://0.0.0.0:%s/metrics", port)


async def run_worker():
    """
    Запустить worker loop.
    
    Инициализирует Redis клиент, создает Worker и запускает его.
    """
    global _worker, _shutdown_event

    _maybe_start_worker_metrics_http()
    
    # Создать shutdown event
    _shutdown_event = asyncio.Event()
    
    worker_id = get_worker_id()
    logger.info(f"Starting worker process: {worker_id}")
    logger.info(f"Redis: {config.redis_host or config.redis_url}")
    logger.info(f"Storage: {config.storage_type}")
    
    try:
        # Инициализировать Redis клиент
        await init_redis_client()
        logger.info("Redis client initialized")
        
        # Получить Storage и KeyLayout для checkpoint support
        # Используем функции из dependencies, но без FastAPI Depends
        from storage.settings import load_storage_settings
        from storage.fs import FileSystemStorage
        from storage.paths import KeyLayout as KeyLayoutImpl
        
        storage_settings = load_storage_settings()
        if storage_settings.backend == "s3":
            from storage.s3 import S3Storage
            storage = S3Storage(
                endpoint_url=storage_settings.s3_endpoint,
                bucket=storage_settings.s3_bucket,
                region=storage_settings.aws_region
            )
        else:
            storage = FileSystemStorage(root_dir=storage_settings.fs_root)
        
        prefix = storage_settings.s3_prefix if storage_settings.backend == "s3" else ""
        key_layout = KeyLayoutImpl(prefix=prefix)
        
        # Создать Worker с storage и key_layout для checkpoint support
        _worker = Worker(
            worker_id=worker_id,
            storage=storage,
            key_layout=key_layout
        )
        
        # Запустить worker loop (блокирующий вызов)
        await _worker.start()
        
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user", worker_id=worker_id)
    except (ConnectionError, TimeoutError) as e:
        from api.services.redis_client import RedisConnectionError
        logger.exception(
            "Redis connection error during worker startup",
            worker_id=worker_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error during worker startup",
            worker_id=worker_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    finally:
        # Остановить worker с graceful shutdown
        if _worker:
            logger.info("Stopping worker...")
            await _worker.stop()
        
        # Закрыть Redis клиент
        await close_redis_client()
        logger.info("Worker stopped")


# Глобальная переменная для shutdown event
_shutdown_event: Optional[asyncio.Event] = None


def signal_handler(signum, frame):
    """
    Обработчик сигналов для graceful shutdown.
    
    Обрабатывает SIGTERM и SIGINT для корректного завершения worker'а.
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2278-2285)
    """
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Установить shutdown event
    global _shutdown_event
    if _shutdown_event:
        _shutdown_event.set()
    
    # Остановить worker асинхронно
    if _worker:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Создать задачу для остановки worker'а
            asyncio.create_task(_worker.stop())
        else:
            # Если loop не запущен, запустить его для остановки
            loop.run_until_complete(_worker.stop())
    
    # Закрыть Redis клиент
    if loop.is_running():
        asyncio.create_task(close_redis_client())
    else:
        loop.run_until_complete(close_redis_client())
    
    logger.info("Worker shutdown complete")
    sys.exit(0)


def main():
    """
    Главная функция для запуска worker процесса.
    
    Настраивает обработчики сигналов и запускает worker loop.
    """
    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Запуск worker loop
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    except (ConnectionError, TimeoutError) as e:
        # Используем logger.error, чтобы не передавать конфликтующие kwargs в стандартный logging.exception
        logger.error(
            "Fatal Redis connection error",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Fatal error",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

