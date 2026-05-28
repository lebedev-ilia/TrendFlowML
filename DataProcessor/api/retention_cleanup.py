"""
Retention Cleanup Entry Point - точка входа для запуска очистки retention policy

Этот модуль запускает очистку старых данных согласно retention policy:
- Удаление Redis state старше 1 дня
- Удаление storage старше 7 дней

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2349-2376)

Запуск:
    python -m api.retention_cleanup
    или
    python api/retention_cleanup.py

В Docker (cron job):
    CMD ["python", "-m", "api.retention_cleanup"]
"""

import asyncio
import logging
import sys
import os

from api.config import config
from api.services.retention import run_retention_cleanup
from api.services.redis_client import init_redis_client, close_redis_client
from storage.settings import load_storage_settings
from storage.fs import FileSystemStorage
from storage.paths import KeyLayout


# Настройка логирования
def setup_logging():
    """
    Настройка логирования для retention cleanup процесса.
    
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
logger = logging.getLogger(__name__)


async def run_cleanup():
    """
    Запустить retention cleanup.
    
    Инициализирует Redis клиент, Storage и KeyLayout, затем запускает очистку.
    """
    logger.info("Starting retention policy cleanup")
    logger.info(f"Redis: {config.redis_host or config.redis_url}")
    logger.info(f"Storage: {config.storage_type}")
    
    try:
        # Инициализировать Redis клиент
        await init_redis_client()
        logger.info("Redis client initialized")
        
        # Получить Storage и KeyLayout
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
        key_layout = KeyLayout(prefix=prefix)
        
        # Запустить очистку
        results = await run_retention_cleanup(storage, key_layout)
        
        # Вывести результаты
        logger.info("Retention cleanup completed successfully")
        logger.info(f"Redis cleanup: {results['redis']}")
        logger.info(f"Storage cleanup: {results['storage']}")
        logger.info(f"Total elapsed time: {results['elapsed_seconds']:.2f}s")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error during retention cleanup: {e}")
        raise
    finally:
        # Закрыть Redis клиент
        await close_redis_client()
        logger.info("Retention cleanup process finished")


def main():
    """
    Главная функция для запуска retention cleanup процесса.
    """
    try:
        results = asyncio.run(run_cleanup())
        
        # Проверить наличие ошибок
        total_errors = results.get("redis", {}).get("errors", 0) + results.get("storage", {}).get("errors", 0)
        
        if total_errors > 0:
            logger.warning(f"Retention cleanup completed with {total_errors} errors")
            sys.exit(1)
        else:
            logger.info("Retention cleanup completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Retention cleanup interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

