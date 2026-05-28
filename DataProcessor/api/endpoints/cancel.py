"""
Endpoint для отмены обработки run'а

POST /api/v1/runs/{run_id}/cancel - отменить обработку run'а

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2720-2756)
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Annotated

from api.schemas.responses import CancelResponse
from api.schemas.state import RunStatus
from api.dependencies import StateReaderDep, TaskManagerDep
from api.utils.errors import RunNotFoundError
from api.services.redis_schema import set_cancel_flag, save_run_state
from api.services.state_machine import validate_transition, parse_status
from api.security import verify_api_key
from api.services.audit import audit_log
from api.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "/{run_id}/cancel",
    response_model=CancelResponse,
    summary="Отменить обработку run'а",
    description="""
    Отменяет обработку активного run'а.
    
    Устанавливает флаг отмены в Redis, который worker периодически проверяет
    и мягко завершает обработку. Run'ы со статусами `success`, `error` или `cancelled`
    не могут быть отменены.
    
    ## Параметры запроса
    
    * `run_id` - UUID run'а для отмены
    
    ## Поведение
    
    После установки флага отмены worker проверяет его каждые 5 секунд и:
    1. Отменяет текущую задачу обработки
    2. Обновляет статус на `cancelled`
    3. Сохраняет событие `processing_cancelled`
    4. Очищает флаг отмены
    
    ## Примеры ответов
    
    ### Успешная отмена (200 OK)
    ```json
    {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "cancelled",
        "message": "Run 550e8400-e29b-41d4-a716-446655440000 has been cancelled"
    }
    ```
    
    ### Ошибка: Run уже завершён (400 Bad Request)
    ```json
    {
        "detail": "Run 550e8400-e29b-41d4-a716-446655440000 is already success and cannot be cancelled"
    }
    ```
    
    ### Ошибка: Run не найден (404 Not Found)
    ```json
    {
        "detail": "Run not found: 550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """,
    responses={
        200: {
            "description": "Run успешно отменён",
            "content": {
                "application/json": {
                    "example": {
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "cancelled",
                        "message": "Run 550e8400-e29b-41d4-a716-446655440000 has been cancelled"
                    }
                }
            }
        },
        400: {
            "description": "Run уже завершён и не может быть отменён",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Run 550e8400-e29b-41d4-a716-446655440000 is already success and cannot be cancelled"
                    }
                }
            }
        },
        404: {
            "description": "Run не найден",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Run not found: 550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        }
    },
    tags=["runs"]
)
async def cancel_run(
    run_id: str,
    request: Request,
    state_reader: StateReaderDep,
    task_manager: TaskManagerDep,
    api_key: str = Depends(verify_api_key)
):
    """
    Отменить обработку run'а.
    
    Устанавливает флаг отмены в Redis и обновляет статус run'а на "cancelled".
    Worker периодически проверяет этот флаг и мягко завершает обработку.
    
    Args:
        run_id: UUID run'а для отмены
        request: Request объект для получения request_id и IP
        state_reader: StateReader dependency
        task_manager: TaskManager dependency
        
    Returns:
        CancelResponse с информацией об отмене
        
    Raises:
        HTTPException 404: Run не найден
        HTTPException 400: Run уже завершён (success, error, cancelled)
        HTTPException 500: Ошибка при отмене
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2724-2738)
    """
    try:
        # Получить текущий статус run'а
        try:
            run_status = await state_reader.get_run_status(run_id)
        except RunNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found: {run_id}"
            )
        
        current_status_str = run_status.get("status", "unknown")
        
        # Парсить статус
        try:
            current_status = parse_status(current_status_str)
        except ValueError:
            logger.warning(f"Invalid status '{current_status_str}' for run {run_id}")
            current_status = None
        
        # Проверить, можно ли отменить run
        if current_status in (RunStatus.SUCCESS, RunStatus.ERROR, RunStatus.CANCELLED):
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_id} is already {current_status_str} and cannot be cancelled"
            )
        
        # Валидировать переход к CANCELLED
        try:
            validate_transition(current_status, RunStatus.CANCELLED, run_id)
        except ValueError as e:
            logger.warning(f"Invalid transition to cancelled for run {run_id}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel run {run_id}: {str(e)}"
            )
        
        # Установить флаг отмены в Redis
        cancel_set = await set_cancel_flag(run_id)
        if not cancel_set:
            logger.warning(f"Failed to set cancel flag for run {run_id}, but continuing with status update")
        
        # Обновить статус на "cancelled"
        await save_run_state(run_id, {
            "status": RunStatus.CANCELLED.value,
            "updated_at": time.time(),
            "cancelled_at": time.time()
        })
        
        # Обновить статус в TaskManager если run активен
        if task_manager and task_manager.is_run_active(run_id):
            task_manager.update_run_status(run_id, RunStatus.CANCELLED)
        
        # Audit log: отмена run'а
        request_id = getattr(request.state, 'request_id', None)
        client_ip = request.client.host if request.client else None
        await audit_log(
            action="run_cancelled",
            run_id=run_id,
            details={
                "previous_status": current_status_str,
                "video_id": run_status.get("video_id"),
                "platform_id": run_status.get("platform_id")
            },
            request_id=request_id,
            ip_address=client_ip
        )
        
        logger.info(
            "Run cancelled",
            run_id=run_id,
            previous_status=current_status_str,
            request_id=request_id
        )
        
        return CancelResponse(
            run_id=run_id,
            status=RunStatus.CANCELLED.value,
            message=f"Run {run_id} has been cancelled"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error cancelling run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while cancelling run: {str(e)}"
        )

