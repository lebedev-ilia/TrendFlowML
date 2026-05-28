"""
Endpoints для работы с run'ами

GET /api/v1/runs/{run_id} - получить метаданные run'а
GET /api/v1/runs/{run_id}/status - получить детальный статус обработки
GET /api/v1/runs/{run_id}/events - получить события (SSE)

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1105-1220)
"""

import json
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated, Optional

from api.schemas.responses import RunMetadataResponse, RunStatusResponse, ManifestResponse
from api.dependencies import StorageDep, KeyLayoutDep, StateReaderDep
from api.utils.errors import RunNotFoundError
from api.services.sse_service import stream_run_events
from api.services.state_reader import StateReader
from api.services.redis_client import get_redis_client
from api.security import verify_api_key
from api.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}", response_model=RunMetadataResponse)
async def get_run_metadata(
    run_id: str,
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    state_reader: StateReaderDep,
    api_key: str = Depends(verify_api_key)
):
    """
    Получить метаданные run'а.
    
    Args:
        run_id: UUID run'а
        storage: Storage dependency
        key_layout: KeyLayout dependency
        state_reader: StateReader dependency
        
    Returns:
        RunMetadataResponse с метаданными run'а
        
    Raises:
        HTTPException 404: Run не найден
    """
    try:
        # TODO: Реализовать получение метаданных через StateReader
        # metadata = await state_reader.get_run_metadata(run_id)
        # return RunMetadataResponse(**metadata)
        
        raise NotImplementedError("Will be implemented in MVP")
        
    except RunNotFoundError as e:
        logger.warning(f"Run not found: {run_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in get_run_metadata: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{run_id}/status",
    response_model=RunStatusResponse,
    summary="Получить статус обработки",
    description="""
    Получает детальный статус обработки run'а, включая прогресс по компонентам.
    
    ## Параметры запроса
    
    * `run_id` - UUID run'а
    * `include_components` - Включить детальную информацию о компонентах (по умолчанию: true)
    * `include_events` - Включить последние события (по умолчанию: false)
    
    ## Статусы run'а
    
    * `pending` - Run создан, но ещё не поставлен в очередь
    * `queued` - Run в очереди, ожидает обработки
    * `running` - Run обрабатывается
    * `recovering` - Run восстанавливается после сбоя
    * `success` - Обработка завершена успешно
    * `error` - Обработка завершена с ошибкой
    * `cancelled` - Обработка отменена
    
    ## Примеры ответов
    
    ### Успешный ответ (200 OK)
    ```json
    {
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "video_id": "dQw4w9WgXcQ",
        "platform_id": "youtube",
        "status": "running",
        "stage": "visual",
        "progress": {
            "overall": 0.75,
            "current_processor": "visual",
            "current_component": "core_clip",
            "components": {
                "segmenter": {
                    "status": "success",
                    "progress": 1.0
                },
                "visual": {
                    "status": "running",
                    "progress": 0.5
                }
            }
        },
        "started_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:05:00Z"
    }
    ```
    
    ### Ошибка: Run не найден (404 Not Found)
    ```json
    {
        "error": "Run not found",
        "run_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """,
    responses={
        200: {
            "description": "Статус успешно получен",
            "content": {
                "application/json": {
                    "example": {
                        "run_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_id": "dQw4w9WgXcQ",
                        "platform_id": "youtube",
                        "status": "running",
                        "stage": "visual",
                        "progress": {
                            "overall": 0.75,
                            "current_processor": "visual",
                            "current_component": "core_clip",
                            "components": {}
                        },
                        "started_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:05:00Z"
                    }
                }
            }
        },
        404: {
            "description": "Run не найден",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Run not found",
                        "run_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        }
    },
    tags=["runs"]
)
async def get_run_status(
    run_id: str,
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    state_reader: StateReaderDep,
    include_components: bool = Query(True, description="Включить детальную информацию о компонентах"),
    include_events: bool = Query(False, description="Включить последние события")
):
    """
    Получить детальный статус обработки run'а.
    
    Возвращает полную информацию о статусе обработки, включая прогресс по компонентам,
    текущую стадию, последние события и метаданные.
    
    Args:
        run_id: UUID run'а (обязательно, path parameter)
        storage: Storage dependency для доступа к хранилищу
        key_layout: KeyLayout dependency для работы с путями
        state_reader: StateReader dependency для чтения состояния
        include_components: Включить детальную информацию о компонентах (по умолчанию True)
            Если True, возвращает информацию о каждом компоненте (статус, прогресс, ошибки)
        include_events: Включить последние события (по умолчанию False)
            Если True, возвращает последние события из event stream
            
    Returns:
        RunStatusResponse: Детальный статус обработки
            - run_id: UUID run'а
            - status: Текущий статус (queued, running, success, error, cancelled)
            - progress: Общий прогресс (0.0 - 1.0)
            - current_stage: Текущая стадия обработки
            - current_component: Текущий компонент
            - components: Детальная информация о компонентах (если include_components=True)
            - events: Последние события (если include_events=True)
            - started_at: Время начала обработки
            - finished_at: Время завершения обработки (если завершено)
            - error: Сообщение об ошибке (если status=error)
        
    Raises:
        HTTPException 404: Run не найден (RunNotFoundError)
        HTTPException 500: Внутренняя ошибка сервера
        
    Example:
        ```python
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Получить статус с компонентами
            response = await client.get(
                "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/status",
                params={"include_components": True, "include_events": True},
                headers={"X-API-Key": "your-api-key"}
            )
            assert response.status_code == 200
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Progress: {data['progress']}")
            print(f"Current component: {data['current_component']}")
        ```
        
    Note:
        - Использует hot path (Redis cache) если доступен для быстрого доступа
        - Автоматически обновляет cache при чтении из Storage
        - Поддерживает фильтрацию событий по компонентам
    """
    try:
        # Получить статус через StateReader
        # StateReader автоматически найдет platform_id и video_id если они не указаны
        status_data = await state_reader.get_run_status(
            run_id=run_id,
            include_components=include_components,
            include_events=include_events
        )
        
        # Конвертировать строковые даты в datetime если нужно
        from datetime import datetime
        
        def parse_datetime(value):
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    # Парсинг ISO 8601 формата
                    # Формат: "2024-01-01T12:00:00Z" или "2024-01-01T12:00:00"
                    if value.endswith("Z"):
                        value = value[:-1] + "+00:00"
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    try:
                        # Альтернативный формат
                        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
                    except Exception:
                        return None
            return value
        
        # Преобразовать ProgressInfo
        progress_data = status_data.get("progress", {})
        from api.schemas.responses import ProgressInfo, ComponentProgress
        
        # Конвертировать компоненты в ComponentProgress объекты
        components_dict = {}
        for comp_name, comp_data in progress_data.get("components", {}).items():
            if isinstance(comp_data, dict):
                # Парсить даты в comp_data
                comp_data_parsed = dict(comp_data)
                if "started_at" in comp_data_parsed:
                    comp_data_parsed["started_at"] = parse_datetime(comp_data_parsed["started_at"])
                if "finished_at" in comp_data_parsed:
                    comp_data_parsed["finished_at"] = parse_datetime(comp_data_parsed["finished_at"])
                
                try:
                    # Убедиться что progress есть, если нет - рассчитать или установить 0.0
                    if "progress" not in comp_data_parsed or comp_data_parsed.get("progress") is None:
                        done = comp_data_parsed.get("done")
                        total = comp_data_parsed.get("total")
                        if done is not None and total is not None and total > 0:
                            comp_data_parsed["progress"] = float(done) / float(total)
                        else:
                            comp_data_parsed["progress"] = 0.0
                    
                    components_dict[comp_name] = ComponentProgress(**comp_data_parsed)
                except Exception as e:
                    logger.warning(f"Failed to create ComponentProgress for {comp_name}: {e}, data: {comp_data_parsed}")
                    # Fallback: создать минимальный ComponentProgress
                    components_dict[comp_name] = ComponentProgress(
                        status=comp_data_parsed.get("status", "unknown"),
                        progress=comp_data_parsed.get("progress", 0.0)
                    )
        
        progress_info = ProgressInfo(
            overall=progress_data.get("overall", 0.0),
            current_processor=progress_data.get("current_processor"),
            current_component=progress_data.get("current_component"),
            components=components_dict
        )
        
        # Создать ответ
        response = RunStatusResponse(
            run_id=status_data["run_id"],
            video_id=status_data["video_id"],
            platform_id=status_data["platform_id"],
            status=status_data["status"],
            stage=status_data.get("stage"),
            progress=progress_info,
            started_at=parse_datetime(status_data.get("started_at")),
            updated_at=parse_datetime(status_data.get("updated_at")) or datetime.now(),
            estimated_finish=None,  # TODO: Рассчитать на основе прогресса
            error=status_data.get("error"),
            error_code=status_data.get("error_code")
        )
        
        return response
        
    except RunNotFoundError as e:
        logger.warning(f"Run not found: {run_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in get_run_status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{run_id}/events")
async def stream_run_events_endpoint(
    run_id: str,
    state_reader: StateReaderDep,
    api_key: str = Depends(verify_api_key),
    since: Optional[str] = Query(
        None,
        description="ISO 8601 timestamp для фильтрации событий (например, '2024-01-01T12:00:00Z')",
    ),
    component: Optional[str] = Query(
        None,
        description="Фильтр по компоненту (например, 'core_clip')",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Максимальное количество событий для возврата (pagination)",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Смещение для pagination",
    ),
):
    """
    Server-Sent Events (SSE) endpoint для стриминга событий прогресса в реальном времени.
    
    Открывает SSE соединение и стримит события обработки в реальном времени.
    Поддерживает фильтрацию по времени и компоненту.
    
    Args:
        run_id: UUID run'а (обязательно, path parameter)
        since: ISO 8601 timestamp для фильтрации событий (опционально)
            Пример: "2024-01-01T12:00:00Z"
            Если указан, возвращаются только события после этого времени
        component: Имя компонента для фильтрации (опционально)
            Пример: "core_clip", "visual_processor"
            Если указан, возвращаются только события для этого компонента
        state_reader: StateReader dependency для проверки существования run'а
        api_key: API ключ из заголовка X-API-Key (проверяется через verify_api_key)
        
    Returns:
        StreamingResponse: SSE stream с событиями (Content-Type: text/event-stream)
            Формат событий:
            - event: progress, stage, component_start, component_complete, complete, error
            - data: JSON объект с данными события
            
    Raises:
        HTTPException 404: Run не найден (RunNotFoundError)
        HTTPException 410: Run завершён, события больше не доступны
        HTTPException 503: Превышен лимит SSE соединений для run_id (максимум 10)
        
    Example:
        ```python
        import httpx
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET",
                "http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000/events",
                params={"since": "2024-01-01T12:00:00Z", "component": "core_clip"},
                headers={"X-API-Key": "your-api-key"}
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        event_data = json.loads(line[5:])
                        print(f"Event: {event_data}")
        ```
        
    Note:
        - Максимум 10 одновременных SSE соединений на run_id
        - Keepalive сообщения отправляются каждые 30 секунд
        - Соединение закрывается автоматически при завершении run'а
        - Использует Redis Streams для real-time стриминга событий
    """
    try:
        # Проверить существование run'а
        try:
            status_data = await state_reader.get_run_status(run_id=run_id)
            run_status = status_data.get("status")
            
            # Если run завершён, проверить доступность событий
            if run_status in ("success", "error", "cancelled"):
                # Проверить наличие событий в Redis или Storage
                redis_client = get_redis_client()
                if redis_client:
                    from api.services.redis_schema import KEY_PREFIX_EVENTS
                    stream_name = f"{KEY_PREFIX_EVENTS}{run_id}"
                    stream_exists = await redis_client.exists(stream_name)
                    if not stream_exists:
                        # События больше не доступны
                        raise HTTPException(
                            status_code=410,
                            detail=f"Run {run_id} is completed and events are no longer available"
                        )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        # Проверить доступность Redis
        redis_client = get_redis_client()
        if not redis_client:
            raise HTTPException(
                status_code=503,
                detail="Redis not available, cannot stream events"
            )
        
        # Создать SSE stream
        async def event_generator():
            try:
                # Для SSE streaming pagination не применяется напрямую,
                # но можно использовать для начальной загрузки исторических событий
                async for event_line in stream_run_events(run_id, since=since, component=component):
                    yield event_line
            except ValueError as e:
                # Превышен лимит соединений
                logger.warning(f"SSE connection limit exceeded for run {run_id}: {e}")
                yield f"event: error\n"
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            except RuntimeError as e:
                # Redis не доступен
                logger.error(f"Redis not available for SSE streaming: {e}")
                yield f"event: error\n"
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            except Exception as e:
                logger.exception(f"Unexpected error in SSE stream for run {run_id}: {e}")
                yield f"event: error\n"
                yield f"data: {{\"error\": \"Unexpected error: {str(e)}\"}}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in stream_run_events_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{run_id}/manifest", response_model=ManifestResponse)
async def get_run_manifest(
    run_id: str,
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    state_reader: StateReaderDep,
    api_key: str = Depends(verify_api_key)
):
    """
    Получить manifest.json run'а.
    
    Args:
        run_id: UUID run'а
        storage: Storage dependency
        key_layout: KeyLayout dependency
        state_reader: StateReader dependency
        
    Returns:
        ManifestResponse с данными manifest.json
        
    Raises:
        HTTPException 404: Run не найден или manifest.json не существует
        HTTPException 410: Run завершён и manifest.json больше не доступен
    """
    try:
        # Получить статус run'а для проверки существования
        try:
            status_data = await state_reader.get_run_status(run_id=run_id)
            platform_id = status_data.get("platform_id")
            video_id = status_data.get("video_id")
            run_status = status_data.get("status")
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        if not platform_id or not video_id:
            raise HTTPException(
                status_code=404,
                detail=f"Could not determine platform_id and video_id for run {run_id}"
            )
        
        # Путь к manifest.json в Storage
        manifest_path = f"{key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        
        # Проверить существование manifest.json
        try:
            from api.utils.retry import retry_storage_operation
            
            exists = await retry_storage_operation(
                storage.exists,
                manifest_path
            )
            if not exists:
                # Если run завершён, вернуть 410 Gone
                if run_status in ("success", "error", "cancelled"):
                    raise HTTPException(
                        status_code=410,
                        detail=f"Run {run_id} is completed and manifest.json is no longer available"
                    )
                # Иначе 404 Not Found
                raise HTTPException(
                    status_code=404,
                    detail=f"manifest.json not found for run {run_id}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking manifest existence: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        
        # Читать manifest.json из Storage с кэшированием через StateReader
        try:
            manifest_data = await state_reader.get_manifest(run_id, platform_id, video_id)
            if not manifest_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"manifest.json not found for run {run_id}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error reading manifest.json for run {run_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        
        # Извлечь данные из manifest
        run_meta = manifest_data.get("run", {})
        components_list = manifest_data.get("components", [])
        
        # Построить словарь компонентов
        components_dict = {}
        for comp_data in components_list:
            if not isinstance(comp_data, dict):
                continue
            
            comp_name = comp_data.get("name")
            if not comp_name:
                continue
            
            # Создать ManifestComponent из данных
            from api.schemas.responses import ManifestComponent
            components_dict[comp_name] = ManifestComponent(
                name=comp_name,
                kind=comp_data.get("kind"),
                status=comp_data.get("status", "error"),
                empty_reason=comp_data.get("empty_reason"),
                started_at=comp_data.get("started_at"),
                finished_at=comp_data.get("finished_at"),
                duration_ms=comp_data.get("duration_ms"),
                artifacts=comp_data.get("artifacts", []),
                error=comp_data.get("error"),
                error_code=comp_data.get("error_code"),
                notes=comp_data.get("notes"),
                warnings=comp_data.get("warnings"),
                producer_version=comp_data.get("producer_version"),
                schema_version=comp_data.get("schema_version"),
                device_used=comp_data.get("device_used"),
            )
        
        # Создать ответ
        response = ManifestResponse(
            schema_version=manifest_data.get("schema_version"),
            run_id=run_meta.get("run_id") or run_id,
            video_id=run_meta.get("video_id") or video_id,
            platform_id=run_meta.get("platform_id") or platform_id,
            config_hash=run_meta.get("config_hash"),
            sampling_policy_version=run_meta.get("sampling_policy_version"),
            dataprocessor_version=run_meta.get("dataprocessor_version"),
            created_at=run_meta.get("created_at"),
            finished_at=run_meta.get("finished_at"),
            updated_at=run_meta.get("updated_at"),
            components=components_dict
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_run_manifest: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

