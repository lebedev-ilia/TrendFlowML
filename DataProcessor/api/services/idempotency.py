"""
Idempotency Service - проверка существующих результатов для идемпотентности

Этот модуль реализует проверку существующих результатов обработки для обеспечения
идемпотентности processors при повторных запусках.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2246-2269)
"""

import json
import logging
from typing import Optional, Dict, Any, List
from storage.base import Storage
from storage.paths import KeyLayout
from api.utils.retry import retry_storage_operation
from api.utils.logging import get_logger

logger = get_logger(__name__)


async def check_existing_result(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """
    Проверить существующий результат обработки run'а.
    
    Если run уже полностью обработан (manifest.json существует и все компоненты завершены),
    возвращает существующий результат для идемпотентности.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        
    Returns:
        Словарь с существующим результатом если run уже обработан, None иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2257-2268)
    """
    try:
        # Путь к manifest.json
        manifest_path = f"{key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        
        # Проверить существование manifest.json
        exists = await retry_storage_operation(
            storage.exists,
            manifest_path
        )
        
        if not exists:
            logger.debug(f"Manifest not found for run {run_id}, will process from scratch")
            return None
        
        # Читать manifest.json
        manifest_bytes = await retry_storage_operation(
            storage.read_bytes,
            manifest_path
        )
        
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        
        # Проверить статус run'а из manifest
        run_meta = manifest_data.get("run", {}) or {}
        run_status = run_meta.get("status")

        def _is_component_done(status: Any) -> bool:
            # VisualProcessor manifests use ok/empty/error; API uses success/skipped/error.
            s = str(status or "").lower()
            return s in ("success", "ok", "empty", "skipped")

        def _needs_artifacts(status: Any) -> bool:
            s = str(status or "").lower()
            # "empty"/"skipped" are valid terminal states without artifacts.
            return s in ("success", "ok")

        # If run_status is missing (older manifests), infer completion from components.
        components = manifest_data.get("components", [])
        inferred_complete = False
        if isinstance(components, list) and components:
            inferred_complete = all(_is_component_done((c or {}).get("status")) for c in components if isinstance(c, dict))

        # If run is completed (success), or completion can be inferred, return cached result (after artifact checks).
        if run_status == "success" or (run_status is None and inferred_complete):
            logger.info(
                "Run already completed, returning existing result (idempotency)",
                run_id=run_id,
                video_id=video_id,
                platform_id=platform_id
            )
            
            # Проверить наличие артефактов для всех компонентов
            all_components_complete = True
            
            for component in components:
                component_name = component.get("name")
                component_status = component.get("status")
                artifacts = component.get("artifacts", [])
                
                if not _is_component_done(component_status):
                    all_components_complete = False
                    logger.debug(
                        f"Component {component_name} not complete: status={component_status}, artifacts={len(artifacts)}"
                    )
                    break

                if _needs_artifacts(component_status) and not artifacts:
                    all_components_complete = False
                    logger.debug(
                        f"Component {component_name} has no artifacts: status={component_status}"
                    )
                    break
                
                # Проверить существование артефактов в Storage
                for artifact in artifacts:
                    artifact_path = artifact.get("path")
                    if artifact_path:
                        run_prefix = key_layout.result_store_run_prefix(platform_id, video_id, run_id)

                        # VisualProcessor manifests may contain:
                        # - run-local relative paths: "core_clip/embeddings.npz"  (preferred)
                        # - older absolute paths under dp_results (dev)
                        ap = str(artifact_path)
                        if ap.startswith("/"):
                            # Try to cut to run-local path: .../<platform>/<video>/<run>/<rel>
                            marker = f"/{platform_id}/{video_id}/{run_id}/"
                            if marker in ap:
                                rel = ap.split(marker, 1)[1].lstrip("/")
                                full_artifact_path = f"{run_prefix}/{rel}"
                            else:
                                # Fallback: treat as storage key without leading slash (legacy)
                                full_artifact_path = ap.lstrip("/")
                        else:
                            # If the artifact path is already a full dp_results key, keep it as-is.
                            if ap.startswith("dp_results/") or ap.startswith(f"dp_results/{platform_id}/"):
                                full_artifact_path = ap
                            else:
                                full_artifact_path = f"{run_prefix}/{ap}"
                        
                        artifact_exists = await retry_storage_operation(
                            storage.exists,
                            full_artifact_path
                        )
                        
                        if not artifact_exists:
                            logger.warning(
                                f"Artifact not found in Storage: {full_artifact_path}",
                                run_id=run_id,
                                component=component_name
                            )
                            all_components_complete = False
                            break
                
                if not all_components_complete:
                    break
            
            if all_components_complete:
                # Все компоненты завершены и артефакты существуют
                return {
                    "success": True,
                    "run_id": run_id,
                    "video_id": video_id,
                    "platform_id": platform_id,
                    "status": "success",
                    "manifest": manifest_data,
                    "from_cache": True,
                    "message": "Run already completed, using cached result"
                }
            else:
                # Manifest существует, но не все компоненты завершены
                logger.info(
                    "Manifest exists but not all components complete, will resume processing",
                    run_id=run_id
                )
                return None
        
        # Если run не завершен, вернуть None для продолжения обработки
        logger.debug(f"Run {run_id} status is {run_status}, will continue processing")
        return None
        
    except Exception as e:
        logger.error(
            f"Error checking existing result for run {run_id}: {e}",
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id
        )
        # В случае ошибки, не использовать кэш - обработать заново
        return None


async def check_component_result(
    storage: Storage,
    key_layout: KeyLayout,
    platform_id: str,
    video_id: str,
    run_id: str,
    component_name: str
) -> Optional[Dict[str, Any]]:
    """
    Проверить существующий результат конкретного компонента.
    
    Args:
        storage: Экземпляр Storage
        key_layout: KeyLayout для работы с путями
        platform_id: ID платформы
        video_id: ID видео
        run_id: UUID run'а
        component_name: Имя компонента (например, "core_clip", "asr_extractor")
        
    Returns:
        Словарь с информацией о компоненте если он уже обработан, None иначе
    """
    try:
        # Проверить manifest.json
        manifest_path = f"{key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        
        exists = await retry_storage_operation(
            storage.exists,
            manifest_path
        )
        
        if not exists:
            return None
        
        # Читать manifest.json
        manifest_bytes = await retry_storage_operation(
            storage.read_bytes,
            manifest_path
        )
        
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
        
        # Найти компонент в manifest
        components = manifest_data.get("components", [])
        for component in components:
            if component.get("name") == component_name:
                component_status = component.get("status")
                artifacts = component.get("artifacts", [])
                
                # Если компонент завершен и имеет артефакты
                if component_status == "success" and artifacts:
                    # Проверить существование артефактов
                    for artifact in artifacts:
                        artifact_path = artifact.get("path")
                        if artifact_path:
                            if not artifact_path.startswith("/"):
                                run_prefix = key_layout.result_store_run_prefix(platform_id, video_id, run_id)
                                full_artifact_path = f"{run_prefix}/{artifact_path}"
                            else:
                                full_artifact_path = artifact_path.lstrip("/")
                            
                            artifact_exists = await retry_storage_operation(
                                storage.exists,
                                full_artifact_path
                            )
                            
                            if not artifact_exists:
                                return None
                    
                    # Все артефакты существуют
                    return {
                        "component": component_name,
                        "status": component_status,
                        "artifacts": artifacts,
                        "from_cache": True
                    }
        
        return None
        
    except Exception as e:
        logger.error(
            f"Error checking component result for {component_name} in run {run_id}: {e}",
            run_id=run_id,
            component=component_name
        )
        return None

