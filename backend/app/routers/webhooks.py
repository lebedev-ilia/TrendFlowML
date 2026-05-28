"""
Webhook endpoints для получения уведомлений от внешних сервисов.

POST /api/webhooks/dataprocessor - webhook для получения уведомлений от DataProcessor API
"""

from __future__ import annotations

import hmac
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel

from ..config import Settings
from ..db import session_scope
from ..dbv2 import enums
from ..dbv2 import models as v2_models
from ..services.events import publish_run_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

settings = Settings()


class WebhookPayload(BaseModel):
    """Payload для webhook от DataProcessor API."""
    run_id: str
    status: str  # success, error, empty, skipped, cancelled
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    stage: Optional[str] = None
    component: Optional[str] = None
    timestamp: Optional[str] = None


def verify_webhook_signature(
    payload_body: bytes,
    signature: Optional[str] = Header(None, alias="X-Webhook-Signature")
) -> bool:
    """
    Валидация webhook signature.
    
    Для MVP используем простую проверку через API Key.
    В production можно использовать HMAC-SHA256.
    
    Args:
        payload_body: Тело запроса
        signature: Подпись из заголовка X-Webhook-Signature
        
    Returns:
        True если подпись валидна
    """
    if not signature:
        # Если подпись не требуется (development mode)
        if not settings.dataprocessor_api_key:
            logger.warning("Webhook signature not provided and API key not configured")
            return True  # Разрешить в development mode
        
        return False
    
    if not settings.dataprocessor_api_key:
        logger.warning("Webhook signature provided but API key not configured")
        return False
    
    # Простая проверка: signature должен совпадать с API key
    # В production использовать HMAC-SHA256
    expected_signature = settings.dataprocessor_api_key
    return hmac.compare_digest(signature, expected_signature)


@router.post(
    "/dataprocessor",
    summary="Webhook для получения уведомлений от DataProcessor API",
    description="""
    Endpoint для получения уведомлений о статусе обработки run'а от DataProcessor API.
    
    ## Аутентификация
    
    Webhook должен содержать заголовок `X-Webhook-Signature` с подписью запроса.
    Для MVP используется простая проверка через API Key.
    
    ## Payload
    
    Webhook отправляет JSON payload с информацией о статусе обработки:
    - `run_id`: UUID run'а
    - `status`: Финальный статус (success, error, empty, skipped, cancelled)
    - `progress`: Информация о прогрессе (опционально)
    - `error`: Сообщение об ошибке (опционально)
    - `error_code`: Код ошибки (опционально)
    - `stage`: Текущая стадия (опционально)
    - `component`: Текущий компонент (опционально)
    - `timestamp`: Временная метка события (опционально)
    
    ## Обработка
    
    При получении webhook:
    1. Валидируется подпись
    2. Обновляется статус AnalysisJob в БД
    3. Отправляется WebSocket событие через Redis pubsub
    """,
    status_code=status.HTTP_200_OK
)
async def dataprocessor_webhook(
    payload: WebhookPayload,
    request: Request,
    signature: Optional[str] = Header(None, alias="X-Webhook-Signature")
) -> Dict[str, Any]:
    """
    Webhook endpoint для получения уведомлений от DataProcessor API.
    
    Args:
        payload: Payload webhook'а
        request: FastAPI Request объект
        signature: Подпись из заголовка X-Webhook-Signature
        
    Returns:
        Подтверждение получения webhook'а
        
    Raises:
        HTTPException 401: Если подпись невалидна
        HTTPException 404: Если AnalysisJob не найден
        HTTPException 500: При ошибке обработки
    """
    # Получить тело запроса для проверки подписи
    body = await request.body()
    
    # Валидация подписи
    if not verify_webhook_signature(body, signature):
        logger.warning(
            f"Invalid webhook signature for run_id={payload.run_id}",
            extra={"run_id": payload.run_id, "signature": signature}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature"
        )
    
    logger.info(
        "Received webhook for run",
        extra={
            "run_id": payload.run_id,
            "status": payload.status,
            "error": payload.error,
            "error_code": payload.error_code
        }
    )
    
    # Обновить статус в БД
    try:
        with session_scope() as db:
            # Найти AnalysisJob по run_id
            analysis_job = (
                db.query(v2_models.AnalysisJob)
                .filter(v2_models.AnalysisJob.id == UUID(payload.run_id))
                .first()
            )
            
            if not analysis_job:
                logger.warning(
                    f"AnalysisJob not found for run_id={payload.run_id}",
                    extra={"run_id": payload.run_id}
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"AnalysisJob not found for run_id={payload.run_id}"
                )
            
            # Обновить статус в зависимости от финального статуса
            if payload.status == "success":
                analysis_job.status = enums.AnalysisStatus.completed
            elif payload.status == "error":
                analysis_job.status = enums.AnalysisStatus.failed
                if payload.error:
                    analysis_job.error_message = payload.error
            elif payload.status == "cancelled":
                analysis_job.status = enums.AnalysisStatus.canceled
            # Другие статусы (empty, skipped) также считаем completed
            
            # Обновить completed_at если статус финальный
            if payload.status in ["success", "error", "empty", "skipped", "cancelled"]:
                analysis_job.completed_at = datetime.utcnow()
            
            db.flush()
            
            logger.info(
                "Updated AnalysisJob status",
                extra={
                    "analysis_job_id": str(analysis_job.id),
                    "run_id": payload.run_id,
                    "status": payload.status
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Error processing webhook for run_id={payload.run_id}",
            extra={"run_id": payload.run_id, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing webhook: {str(e)}"
        )
    
    # Отправить WebSocket событие через Redis pubsub
    try:
        event_payload = {
            "type": "run.status_changed",
            "run_id": payload.run_id,
            "ts": payload.timestamp or datetime.utcnow().isoformat() + "Z",
            "payload": {
                "status": payload.status,
                "error": payload.error,
                "error_code": payload.error_code,
                "stage": payload.stage,
                "component": payload.component,
                "progress": payload.progress
            }
        }
        await publish_run_event(payload.run_id, event_payload)
        
        logger.debug(
            "Published WebSocket event for run",
            extra={"run_id": payload.run_id, "status": payload.status}
        )
    except Exception as e:
        # Не критично, если не удалось отправить событие
        logger.warning(
            f"Failed to publish WebSocket event for run_id={payload.run_id}",
            extra={"run_id": payload.run_id, "error": str(e)}
        )
    
    return {
        "status": "ok",
        "message": "Webhook processed successfully",
        "run_id": payload.run_id
    }

