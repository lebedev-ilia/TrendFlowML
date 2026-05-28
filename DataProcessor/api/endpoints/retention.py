"""
Retention Policy Endpoint

POST /api/v1/admin/retention/cleanup - запуск очистки retention policy вручную

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2349-2376)
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from api.dependencies import StorageDep, KeyLayoutDep
from api.services.retention import run_retention_cleanup
from api.security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/retention", tags=["admin", "retention"])


@router.post(
    "/cleanup",
    summary="Запустить очистку retention policy",
    description="""
    Запускает очистку старых данных согласно retention policy:
    - Удаление Redis state старше 1 дня
    - Удаление storage старше 7 дней
    
    ## Когда использовать
    
    Этот endpoint предназначен для:
    - Ручного запуска очистки (для тестирования или экстренных случаев)
    - Мониторинга результатов очистки
    
    ## Автоматический запуск
    
    В production рекомендуется использовать cron job или scheduler для автоматического запуска:
    - Ежедневно в 2:00 UTC (или другое время с низкой нагрузкой)
    - Через отдельный сервис в docker-compose или через системный cron
    
    ## Результаты
    
    Возвращает детальную информацию о результатах очистки:
    - Количество проверенных ключей/run'ов
    - Количество удалённых ключей/run'ов
    - Количество ошибок
    - Время выполнения
    
    ## Пример ответа
    
    ```json
    {
        "redis": {
            "checked": 100,
            "deleted": 5,
            "errors": 0
        },
        "storage": {
            "checked": 50,
            "deleted": 2,
            "errors": 0
        },
        "timestamp": 1704067200.0,
        "elapsed_seconds": 12.34
    }
    ```
    
    ## Ошибки
    
    - 500: Внутренняя ошибка при выполнении очистки
    - 503: Redis или Storage недоступны
    """,
    responses={
        200: {
            "description": "Очистка выполнена успешно",
            "content": {
                "application/json": {
                    "example": {
                        "redis": {"checked": 100, "deleted": 5, "errors": 0},
                        "storage": {"checked": 50, "deleted": 2, "errors": 0},
                        "timestamp": 1704067200.0,
                        "elapsed_seconds": 12.34
                    }
                }
            }
        },
        500: {
            "description": "Ошибка при выполнении очистки"
        },
        503: {
            "description": "Redis или Storage недоступны"
        }
    }
)
async def cleanup_retention(
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Запустить очистку retention policy.
    
    Выполняет очистку:
    - Redis state старше 1 дня
    - Storage старше 7 дней
    
    Args:
        storage: Storage dependency
        key_layout: KeyLayout dependency
        api_key: API key для аутентификации
        
    Returns:
        Словарь с результатами очистки
        
    Raises:
        HTTPException: Если произошла ошибка при очистке
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2349-2376)
    """
    try:
        logger.info("Starting retention cleanup via API endpoint")
        
        # Запустить очистку
        results = await run_retention_cleanup(storage, key_layout)
        
        # Проверить наличие ошибок
        total_errors = (
            results.get("redis", {}).get("errors", 0) +
            results.get("storage", {}).get("errors", 0)
        )
        
        if total_errors > 0:
            logger.warning(f"Retention cleanup completed with {total_errors} errors")
            # Возвращаем результаты даже при ошибках, но с предупреждением
        
        return results
        
    except Exception as e:
        logger.exception(f"Error during retention cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run retention cleanup: {str(e)}"
        )

