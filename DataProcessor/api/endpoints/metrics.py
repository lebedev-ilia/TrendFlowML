"""
Prometheus Metrics Endpoint

GET /api/v1/metrics - метрики Prometheus для мониторинга

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1330-1350, 2167-2217)
"""

import logging
from fastapi import APIRouter
from fastapi.responses import Response

from api.services.metrics import get_metrics, get_metrics_content_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "",
    summary="Prometheus метрики",
    description="""
    Возвращает метрики в формате Prometheus text format.
    
    ## Формат ответа
    
    Content-Type: `text/plain; version=0.0.4; charset=utf-8`
    
    Метрики в формате Prometheus:
    ```
    # HELP dataprocessor_queue_length Current queue length
    # TYPE dataprocessor_queue_length gauge
    dataprocessor_queue_length{priority="high"} 5.0
    dataprocessor_queue_length{priority="normal"} 10.0
    
    # HELP dataprocessor_processing_seconds Processing time per run
    # TYPE dataprocessor_processing_seconds histogram
    dataprocessor_processing_seconds_bucket{processor="visual",component="core_clip",le="60.0"} 10.0
    ```
    
    ## Доступные метрики
    
    * `dataprocessor_queue_length` - Длина очереди по приоритетам
    * `dataprocessor_queue_wait_seconds` - Время ожидания в очереди
    * `dataprocessor_processing_seconds` - Время обработки по процессорам и компонентам
    * `dataprocessor_failures_total` - Общее количество ошибок
    * `dataprocessor_memory_bytes` - Использование памяти по run_id
    * `dataprocessor_active_runs` - Количество активных run'ов
    * `dataprocessor_crashed_runs_total` - Количество упавших run'ов
    
    ## Примеры ответов
    
    ### Успешный ответ (200 OK)
    Content-Type: `text/plain; version=0.0.4; charset=utf-8`
    Тело: метрики в формате Prometheus
    
    ### Ошибка (200 OK с пустыми метриками)
    Content-Type: `text/plain; version=0.0.4; charset=utf-8`
    Тело: `# Error generating metrics\n`
    """,
    responses={
        200: {
            "description": "Метрики Prometheus",
            "content": {
                "text/plain; version=0.0.4; charset=utf-8": {
                    "example": "# HELP dataprocessor_queue_length Current queue length\n# TYPE dataprocessor_queue_length gauge\ndataprocessor_queue_length{priority=\"high\"} 5.0\n"
                }
            }
        }
    },
    tags=["metrics"]
)
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Возвращает метрики в формате Prometheus text format.
    
    Returns:
        Response с метриками в формате text/plain
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1330-1350)
    """
    try:
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type=get_metrics_content_type()
        )
    except Exception as e:
        logger.exception(f"Error generating metrics: {e}")
        # Возвращаем пустые метрики при ошибке
        return Response(
            content=b"# Error generating metrics\n",
            media_type=get_metrics_content_type()
        )

