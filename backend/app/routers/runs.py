"""
Runs ингестиции по URL (YouTube и др.): Backend ↔ Fetcher, Phase 1–2.

POST /api/runs — создание run по source_url, запись в БД и вызов Fetcher API.
GET /api/runs — список run'ов пользователя.
GET /api/runs/{run_id} — детали run'а.
POST /api/runs/{run_id}/trigger-processing — запуск обработки DataProcessor (вызывается Fetcher после finalize, Phase 2).

Контракт: docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md, backend/docs/FETCHER_INTEGRATION.md.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.orm import Session

from ..auth import decode_token
from ..config import Settings
from ..dbv2.models import IngestionRun, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import CreateRunRequest, IngestionRunOut
from ..services.events import subscribe_run_events
from ..services.fetcher_client import create_run_async as fetcher_create_run_async
from ..tasks import process_ingestion_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _verify_trigger_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """Проверка API key для trigger-processing (вызов от Fetcher). Если run_trigger_api_key задан — ключ обязателен."""
    settings = Settings()
    if not settings.run_trigger_api_key:
        return
    if not x_api_key or x_api_key != settings.run_trigger_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key for trigger-processing",
        )


def _require_workspace_member(
    db: Session, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Проверить, что пользователь — участник workspace."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found"
        )
    m = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
    )
    if not m:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied"
        )


def _run_to_out(r: IngestionRun, message: Optional[str] = None) -> IngestionRunOut:
    return IngestionRunOut(
        run_id=r.run_id,
        source_url=r.source_url,
        workspace_id=r.workspace_id,
        ingestion_status=r.ingestion_status,
        created_at=r.created_at,
        updated_at=r.updated_at,
        message=message,
        fetcher_stage=getattr(r, "fetcher_stage", None),
        fetcher_error_code=getattr(r, "fetcher_error_code", None),
        fetcher_error_message=getattr(r, "fetcher_error_message", None),
    )


@router.post("", response_model=IngestionRunOut, status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: CreateRunRequest,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> IngestionRunOut:
    """
    Создать run ингестиции по URL (YouTube и др.).

    Backend генерирует run_id (UUID), сохраняет запись в БД и передаёт задачу в Fetcher
    (POST /api/v1/runs). Fetcher асинхронно выполняет сбор метаданных, комментариев и
    скачивание видео.

    - **source_url**: URL видео (обязательно).
    - **workspace_id**: опционально; при указании проверяется доступ пользователя к workspace.
    - **Idempotency-Key**: опциональный заголовок; при повторном запросе с тем же ключом
      возвращается существующий run (201 с тем же run_id).
    """
    settings = Settings()

    # Идемпотентность: если передан Idempotency-Key и run с таким ключом уже есть — вернуть его
    if idempotency_key:
        existing = (
            db.query(IngestionRun)
            .filter(
                IngestionRun.user_id == user.id,
                IngestionRun.idempotency_key == idempotency_key,
            )
            .first()
        )
        if existing:
            logger.info(
                "Idempotency key matched existing run",
                extra={"run_id": str(existing.run_id), "idempotency_key": idempotency_key},
            )
            return _run_to_out(
                existing,
                message="Run already exists (idempotency key matched)",
            )

    # Проверка доступа к workspace при указании
    if payload.workspace_id is not None:
        _require_workspace_member(db, payload.workspace_id, user.id)

    run_id = uuid.uuid4()
    ingestion = IngestionRun(
        run_id=run_id,
        user_id=user.id,
        source_url=payload.source_url,
        workspace_id=payload.workspace_id,
        ingestion_status="pending",
        idempotency_key=idempotency_key,
    )
    db.add(ingestion)
    db.commit()
    db.refresh(ingestion)

    # Вызов Fetcher API (асинхронно, без блокировки event loop)
    try:
        fetcher_resp = await fetcher_create_run_async(
            run_id=run_id,
            source_url=payload.source_url,
            platform=None,
            webhook_url=None,
            idempotency_key=idempotency_key,
            settings=settings,
        )
        # Обновить статус по ответу Fetcher.
        # Контракт: во внешнем API Backend поле ingestion_status принимает только
        # "pending" | "running" | "completed" | "failed".
        # Fetcher возвращает более детальные статусы (PENDING, CHECKING_CACHE, ...),
        # поэтому здесь не копируем их напрямую, а маппим "не PENDING" в "running".
        fetcher_status = fetcher_resp.get("status", "")
        if fetcher_status and fetcher_status != "PENDING":
            ingestion.ingestion_status = "running"
            db.commit()
            db.refresh(ingestion)
    except Exception as e:
        logger.exception("Fetcher create_run failed for run_id=%s", run_id)
        ingestion.ingestion_status = "failed"
        ingestion.fetcher_error_message = str(e)[:2000]  # Phase 5: сохранить для GET /api/runs/{run_id}
        code = None
        if hasattr(e, "response") and e.response is not None:
            try:
                code = getattr(e.response, "status_code", None)
            except Exception:
                pass
        if code is not None:
            ingestion.fetcher_error_code = f"HTTP_{code}"
        elif type(e).__name__ in ("ConnectTimeout", "ReadTimeout", "TimeoutException"):
            ingestion.fetcher_error_code = "FETCHER_TIMEOUT"
        else:
            ingestion.fetcher_error_code = "FETCHER_UNAVAILABLE"
        db.commit()
        db.refresh(ingestion)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Fetcher service error: {e!s}",
        ) from e

    return _run_to_out(ingestion, message=fetcher_resp.get("message"))


@router.get("", response_model=List[IngestionRunOut])
def list_runs(
    workspace_id: Optional[uuid.UUID] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
) -> List[IngestionRunOut]:
    """
    Список run'ов ингестиции текущего пользователя.

    - **workspace_id**: опциональный фильтр по workspace (проверяется доступ).
    - **limit**: максимум записей (по умолчанию 50).
    """
    if workspace_id is not None:
        _require_workspace_member(db, workspace_id, user.id)

    q = db.query(IngestionRun).filter(IngestionRun.user_id == user.id)
    if workspace_id is not None:
        q = q.filter(IngestionRun.workspace_id == workspace_id)
    rows = q.order_by(IngestionRun.created_at.desc()).limit(limit).all()
    return [_run_to_out(r) for r in rows]


@router.get("/{run_id}", response_model=IngestionRunOut)
def get_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
) -> IngestionRunOut:
    """Получить run ингестиции по run_id. Доступ только к своим run'ам."""
    run = (
        db.query(IngestionRun)
        .filter(IngestionRun.run_id == run_id, IngestionRun.user_id == user.id)
        .first()
    )
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
    return _run_to_out(run)


@router.post(
    "/{run_id}/trigger-processing",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить обработку DataProcessor по run_id (Phase 2)",
)
def trigger_processing(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    _: None = Depends(_verify_trigger_api_key),
) -> dict:
    """
    Запуск обработки DataProcessor после успешного finalize в Fetcher.

    Вызывается **Fetcher** (сервером), когда run перешёл в COMPLETED: Backend ставит
    задачу `process_ingestion_run(run_id)` в очередь. Задача забирает manifest и
    артефакты из Fetcher, скачивает видео и запускает DataProcessor.

    - Требуется заголовок **X-API-Key** совпадающий с `TF_BACKEND_RUN_TRIGGER_API_KEY`,
      если тот задан в конфиге Backend.
    - Возвращает 202 Accepted и `{run_id, status: "accepted", message}`.
    - 404 если run не найден в БД Backend.
    - Phase 5.4 (idempotency): если задача DataProcessor уже запущена (ingestion_status=processing),
      повторный вызов возвращает 202 без повторной постановки задачи.
    """
    run = db.query(IngestionRun).filter(IngestionRun.run_id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
    if run.ingestion_status == "processing":
        logger.info("Trigger-processing idempotent: run_id=%s already processing", run_id)
        return {
            "run_id": str(run_id),
            "status": "accepted",
            "message": "Processing already triggered",
        }
    process_ingestion_run.delay(str(run_id))
    logger.info("Triggered processing for run_id=%s", run_id)
    return {
        "run_id": str(run_id),
        "status": "accepted",
        "message": "Processing triggered",
    }


# -----------------------------------------------------------------------------
# Phase 4: WebSocket — поток событий run (run.status_changed, run.stage_changed)
# -----------------------------------------------------------------------------


@router.websocket("/{run_id}/events")
async def ws_run_events(
    websocket: WebSocket,
    run_id: UUID,
    db: Session = Depends(get_db),
) -> None:
    """
    WebSocket поток событий по run (Phase 4).

    Подписывается на канал Redis `run:{run_id}` и пересылает клиенту события
    (run.status_changed, run.stage_changed и др.), публикуемые при синхронизации
    статуса из Fetcher и при обработке DataProcessor.

    Авторизация: обязателен query-параметр `token` (JWT). Проверяется доступ к run:
    только владелец run (user_id) может подписаться на события.
    """
    run_id_str = str(run_id)
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    user_id = decode_token(token)
    if not user_id:
        await websocket.close(code=1008)
        return
    user = db.query(CoreUser).filter(CoreUser.id == user_id).first()
    if not user:
        await websocket.close(code=1008)
        return
    run = (
        db.query(IngestionRun)
        .filter(IngestionRun.run_id == run_id, IngestionRun.user_id == user.id)
        .first()
    )
    if not run:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        async for payload in subscribe_run_events(run_id_str):
            await websocket.send_text(
                json.dumps(payload, ensure_ascii=False),
            )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket run events error run_id=%s: %s", run_id_str, e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
