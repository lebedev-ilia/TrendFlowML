"""
Endpoints для получения артефактов обработки

GET /api/v1/runs/{run_id}/artifacts/{component} - получить артефакты компонента

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1256-1286)
"""

import json
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response
from typing import Annotated, Optional

from api.dependencies import StorageDep, KeyLayoutDep, StateReaderDep
from api.utils.errors import RunNotFoundError
from api.schemas.responses import ManifestArtifact
from api.security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["artifacts"])


@router.get(
    "/{run_id}/artifacts/{component}",
    summary="Получить артефакты компонента",
    description="""
    Получает артефакты (NPZ файлы) компонента обработки.
    
    ## Параметры запроса
    
    * `run_id` - UUID run'а
    * `component` - Имя компонента (например, 'core_clip', 'audio', 'visual')
    * `format` - Формат ответа:
      - `raw` - Бинарный NPZ файл (по умолчанию)
      - `info` - JSON метаданные об артефакте
      - `url` - Presigned URL для прямого доступа к артефакту (TTL 1 час, только для S3)
    
    ## Форматы ответа
    
    ### format='raw'
    Возвращает бинарный NPZ файл с типом `application/octet-stream`.
    Файл содержит результаты обработки компонента в формате NumPy.
    
    ### format='info'
    Возвращает JSON с метаданными об артефакте:
    ```json
    {
        "component": "core_clip",
        "path": "result_store/youtube/dQw4w9WgXcQ/run_id/core_clip/features.npz",
        "size_bytes": 1024000,
        "schema_version": "v1"
    }
    ```
    
    ### format='url'
    Возвращает JSON с presigned URL для прямого доступа к артефакту (только для S3):
    ```json
    {
        "component": "core_clip",
        "artifact_path": "result_store/youtube/dQw4w9WgXcQ/run_id/core_clip/features.npz",
        "url": "https://s3.amazonaws.com/bucket/path?X-Amz-Algorithm=...",
        "expires_in": 3600,
        "size_bytes": 1024000,
        "schema_version": "v1"
    }
    ```
    Presigned URL действителен в течение 1 часа (3600 секунд).
    
    ## Примеры ответов
    
    ### Успешный ответ (200 OK) - raw формат
    Content-Type: `application/octet-stream`
    Тело: бинарный NPZ файл
    
    ### Успешный ответ (200 OK) - info формат
    ```json
    {
        "component": "core_clip",
        "path": "result_store/youtube/dQw4w9WgXcQ/run_id/core_clip/features.npz",
        "size_bytes": 1024000,
        "schema_version": "v1"
    }
    ```
    
    ### Ошибка: Run не найден (404 Not Found)
    ```json
    {
        "detail": "Run not found: 550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    
    ### Ошибка: Артефакт недоступен (410 Gone)
    ```json
    {
        "detail": "Artifact for component 'core_clip' in run 550e8400-e29b-41d4-a716-446655440000 is no longer available"
    }
    ```
    """,
    responses={
        200: {
            "description": "Артефакт успешно получен",
            "content": {
                "application/octet-stream": {
                    "example": "Binary NPZ file content"
                },
                "application/json": {
                    "example": {
                        "component": "core_clip",
                        "path": "result_store/youtube/dQw4w9WgXcQ/run_id/core_clip/features.npz",
                        "size_bytes": 1024000,
                        "schema_version": "v1"
                    }
                }
            }
        },
        404: {
            "description": "Run или артефакт не найден",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Artifact not found: result_store/youtube/dQw4w9WgXcQ/run_id/core_clip/features.npz"
                    }
                }
            }
        },
        410: {
            "description": "Артефакт больше не доступен (run завершён)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Artifact for component 'core_clip' in run 550e8400-e29b-41d4-a716-446655440000 is no longer available"
                    }
                }
            }
        }
    },
    tags=["artifacts"]
)
async def get_artifact(
    run_id: str,
    component: str,
    storage: StorageDep,
    key_layout: KeyLayoutDep,
    state_reader: StateReaderDep,
    format: str = Query("raw", description="Формат ответа: 'raw' (NPZ), 'info' (JSON метаданные) или 'url' (presigned URL для S3)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Получить артефакты компонента.
    
    Поддерживает два формата:
    - raw: бинарный NPZ файл
    - info: JSON метаданные об артефакте
    
    Args:
        run_id: UUID run'а
        component: Имя компонента (например, 'core_clip', 'audio', 'visual')
        storage: Storage dependency
        key_layout: KeyLayout dependency
        state_reader: StateReader dependency
        format: Формат ответа ('raw' или 'info')
        
    Returns:
        - Для format='raw': бинарный NPZ файл (application/octet-stream)
        - Для format='info': JSON метаданные об артефакте
        
    Raises:
        HTTPException 404: Run или артефакт не найден
        HTTPException 410: Run завершён, артефакты больше не доступны
        HTTPException 400: Невалидный формат
    """
    try:
        # Валидация формата
        if format not in ("raw", "info"):
            raise HTTPException(status_code=400, detail=f"Invalid format: {format}. Must be 'raw' or 'info'")
        
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
        
        # Читать manifest.json для получения информации об артефактах
        manifest_path = f"{key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        
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
                        detail=f"Run {run_id} is completed and artifacts are no longer available"
                    )
                # Иначе 404 Not Found
                raise HTTPException(
                    status_code=404,
                    detail=f"manifest.json not found for run {run_id}"
                )
            
            manifest_bytes = await retry_storage_operation(
                storage.read_bytes,
                manifest_path
            )
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        except HTTPException:
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in manifest.json for run {run_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"manifest.json for run {run_id} contains invalid JSON"
            )
        except Exception as e:
            logger.error(f"Error reading manifest.json for run {run_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        
        # Найти компонент в manifest
        components_list = manifest_data.get("components", [])
        component_data = None
        
        for comp_data in components_list:
            if not isinstance(comp_data, dict):
                continue
            if comp_data.get("name") == component:
                component_data = comp_data
                break
        
        if not component_data:
            raise HTTPException(
                status_code=404,
                detail=f"Component '{component}' not found in manifest for run {run_id}"
            )
        
        # Получить список артефактов компонента
        artifacts = component_data.get("artifacts", [])
        if not artifacts:
            raise HTTPException(
                status_code=404,
                detail=f"No artifacts found for component '{component}' in run {run_id}"
            )
        
        # Использовать первый артефакт (обычно компонент имеет один основной NPZ файл)
        # В будущем можно добавить параметр для выбора конкретного артефакта
        artifact_info = artifacts[0]
        artifact_path = artifact_info.get("path")
        
        if not artifact_path:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact path not found for component '{component}' in run {run_id}"
            )
        
        # Построить полный путь к артефакту
        # artifact_path может быть относительным (например, "core_clip/core_clip_npz_v2.npz")
        # или абсолютным от result_store_run_prefix
        if artifact_path.startswith("/"):
            # Абсолютный путь - использовать как есть
            full_artifact_path = artifact_path.lstrip("/")
        else:
            # Относительный путь - добавить к result_store_run_prefix
            run_prefix = key_layout.result_store_run_prefix(platform_id, video_id, run_id)
            full_artifact_path = f"{run_prefix}/{artifact_path}"
        
        # Проверить существование артефакта с retry
        from api.utils.retry import retry_storage_operation
        
        exists = await retry_storage_operation(
            storage.exists,
            full_artifact_path
        )
        if not exists:
            # Если run завершён, вернуть 410 Gone
            if run_status in ("success", "error", "cancelled"):
                raise HTTPException(
                    status_code=410,
                    detail=f"Artifact for component '{component}' in run {run_id} is no longer available"
                )
            # Иначе 404 Not Found
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {full_artifact_path}"
            )
        
        # Обработка формата ответа
        if format == "raw":
            # Чтение бинарного NPZ файла
            try:
                artifact_bytes = await retry_storage_operation(
                    storage.read_bytes,
                    full_artifact_path
                )
                return Response(
                    content=artifact_bytes,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f'attachment; filename="{component}_{artifact_path.split("/")[-1]}"'
                    }
                )
            except Exception as e:
                logger.error(f"Error reading artifact {full_artifact_path}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        elif format == "url":
            # Генерация presigned URL для прямого доступа к артефакту
            # Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2340-2346)
            try:
                if hasattr(storage, "generate_presigned_url"):
                    # S3Storage поддерживает presigned URL
                    presigned_url = storage.generate_presigned_url(
                        key=full_artifact_path,
                        expiration=3600,  # 1 час TTL
                        http_method="GET"
                    )
                    return {
                        "component": component,
                        "artifact_path": artifact_path,
                        "url": presigned_url,
                        "expires_in": 3600,
                        "size_bytes": artifact_info.get("size_bytes"),
                        "schema_version": artifact_info.get("schema_version") or component_data.get("schema_version"),
                    }
                else:
                    # FileSystemStorage не поддерживает presigned URL
                    # Возвращаем относительный путь (для development)
                    return {
                        "component": component,
                        "artifact_path": artifact_path,
                        "url": f"/api/v1/runs/{run_id}/artifacts/{component}?format=raw",
                        "expires_in": None,
                        "size_bytes": artifact_info.get("size_bytes"),
                        "schema_version": artifact_info.get("schema_version") or component_data.get("schema_version"),
                        "note": "FileSystemStorage does not support presigned URLs. Use format=raw for direct download."
                    }
            except Exception as e:
                logger.error(f"Error generating presigned URL for {full_artifact_path}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        else:  # format == "info"
            # Возврат JSON метаданных об артефакте
            artifact_info_response = {
                "component": component,
                "artifact_path": artifact_path,
                "size_bytes": artifact_info.get("size_bytes"),
                "schema_version": artifact_info.get("schema_version") or component_data.get("schema_version"),
                "created_at": component_data.get("started_at"),  # Используем started_at как created_at
            }
            
            # Добавить дополнительную информацию из component_data если доступна
            if component_data.get("producer_version"):
                artifact_info_response["producer_version"] = component_data.get("producer_version")
            if component_data.get("finished_at"):
                artifact_info_response["finished_at"] = component_data.get("finished_at")
            
            return artifact_info_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_artifact: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

