from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import httpx
import yaml

from ..config import Settings
from .dataprocessor_adapter import resolve_dataprocessor_global_config_path

logger = logging.getLogger(__name__)


@dataclass
class RunPaths:
    run_rs_path: Path
    frames_dir: Optional[Path]
    manifest_path: Path
    state_events_path: Path


def build_profile_yaml(config_json: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_json, f, sort_keys=False, allow_unicode=True)


def request_dataprocessor_cancel(run_id: str, timeout: float = 30.0) -> bool:
    """
    Вызывает DataProcessor ``POST /api/v1/runs/{run_id}/cancel`` (флаг в Redis для worker'а DP).

    Возвращает True при ответе 2xx. Ошибки HTTP/сети логируются и дают False —
    клиенту всё равно можно вернуть «отмена запрошена», финальный статус придёт из poll/webhook.
    """
    settings = Settings()
    api_url = settings.dataprocessor_api_url.rstrip("/")
    url = f"{api_url}/api/v1/runs/{run_id}/cancel"
    headers: Dict[str, str] = {}
    if settings.dataprocessor_api_key:
        headers["X-API-Key"] = settings.dataprocessor_api_key
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers)
            if r.status_code >= 400:
                logger.warning(
                    "DataProcessor cancel HTTP %s for run_id=%s: %s",
                    r.status_code,
                    run_id,
                    (r.text or "")[:500],
                )
                return False
            return True
    except Exception as e:
        logger.warning("DataProcessor cancel failed run_id=%s: %s", run_id, e)
        return False


def resolve_run_paths(
    *,
    platform_id: str,
    video_id: str,
    run_id: str,
    result_store_base: Path,
) -> RunPaths:
    run_rs_path = result_store_base / platform_id / video_id / run_id
    manifest_path = run_rs_path / "manifest.json"
    runs_root = result_store_base.parent
    state_events_path = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
    return RunPaths(
        run_rs_path=run_rs_path,
        frames_dir=None,
        manifest_path=manifest_path,
        state_events_path=state_events_path,
    )


def run_dataprocessor(
    *,
    video_path: Path,
    platform_id: str,
    video_id: str,
    run_id: str,
    profile_config: Dict[str, Any],
    result_store_base: Path,
    frames_dir_base: Path,
    visual_cfg_default: Path,
) -> RunPaths:
    settings = Settings()
    paths = settings.resolve_paths()

    profile_dir = result_store_base.parent / "profiles_cache" / run_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.yaml"
    build_profile_yaml(profile_config, profile_path)

    dp_main = paths.dataproc_root / "main.py"
    if not dp_main.exists():
        raise FileNotFoundError(f"DataProcessor main not found: {dp_main}")

    cmd = [
        os.environ.get("PYTHON", "python3"),
        str(dp_main),
        "--video-path",
        str(video_path),
        "--output",
        str(frames_dir_base),
        "--chunk-size",
        "64",
        "--visual-cfg-path",
        str(visual_cfg_default),
        "--profile-path",
        str(profile_path),
        "--dag-path",
        str(paths.dataproc_root / "docs" / "reference" / "component_graph.yaml"),
        "--dag-stage",
        "baseline",
        "--platform-id",
        platform_id,
        f"--video-id={video_id}",
        "--run-id",
        run_id,
        "--sampling-policy-version",
        "v1",
        "--dataprocessor-version",
        "dev",
        "--rs-base",
        str(result_store_base),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    run_paths = resolve_run_paths(
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        result_store_base=result_store_base,
    )
    frames_dir = frames_dir_base / video_id / "video"
    run_paths.frames_dir = frames_dir if frames_dir.exists() else None
    return run_paths


async def run_dataprocessor_async(
    *,
    video_path: Optional[Path] = None,
    video_url: Optional[str] = None,
    platform_id: str,
    video_id: str,
    run_id: str,
    profile_config: Dict[str, Any],
    result_store_base: Path,
    frames_dir_base: Path,
    visual_cfg_default: Path,
) -> RunPaths:
    """
    Асинхронный запуск DataProcessor через HTTP API.

    Поддержка Phase 3: можно передать либо video_path (локальный файл), либо
    video_url (signed URL — DataProcessor скачает в кэш). См. docs/PHASE3_ARTIFACTS_CONTRACT.md.

    Args:
        video_path: Путь к видео файлу (обязателен, если не передан video_url).
        video_url: URL для скачивания видео (Phase 3: Fetcher signed URL).
        platform_id: ID платформы (youtube, upload)
        video_id: ID видео
        run_id: UUID run'а
        profile_config: Конфигурация профиля обработки
        result_store_base: Базовый путь к result_store
        frames_dir_base: Базовый путь к frames_dir
        visual_cfg_default: Путь к конфигурации visual процессора

    Returns:
        RunPaths с путями к результатам обработки

    Raises:
        ValueError: Если не передан ни video_path, ни video_url.
        httpx.HTTPError: При ошибках HTTP запроса
    """
    if not video_url and (not video_path or not str(video_path).strip()):
        raise ValueError("Either video_path or video_url must be provided")
    settings = Settings()
    paths = settings.resolve_paths()

    # Подготовить payload для API (Phase 3: video_url или video_path)
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "video_id": video_id,
        "platform_id": platform_id,
        "config_hash": profile_config.get("config_hash", ""),
        "profile_config": profile_config,
        "rs_base": str(result_store_base),
        "output": str(frames_dir_base),
        "visual_cfg_path": str(visual_cfg_default),
        "dag_path": str(paths.dataproc_root / "docs" / "reference" / "component_graph.yaml"),
        "dag_stage": "baseline",
        "sampling_policy_version": "v1",
        "dataprocessor_version": "dev",
        "chunk_size": 64,
    }
    if video_url:
        payload["video_url"] = video_url
    if video_path and str(video_path).strip():
        payload["video_path"] = str(Path(video_path).absolute())
    
    # Добавить опциональные поля версионирования если есть
    if "profile_version" in profile_config:
        payload["profile_version"] = profile_config.get("profile_version")
    if "feature_schema_version" in profile_config:
        payload["feature_schema_version"] = profile_config.get("feature_schema_version")
    if "pipeline_version" in profile_config:
        payload["pipeline_version"] = profile_config.get("pipeline_version")

    gc_path = resolve_dataprocessor_global_config_path(settings, paths)
    if gc_path:
        payload["global_config_path"] = str(gc_path)

    # Подготовить headers
    headers = {}
    if settings.dataprocessor_api_key:
        headers["X-API-Key"] = settings.dataprocessor_api_key
    
    # URL API
    api_url = settings.dataprocessor_api_url.rstrip("/")
    endpoint = f"{api_url}/api/v1/process"
    
    logger.info(
        "Sending processing request to DataProcessor API",
        extra={
            "run_id": run_id,
            "video_id": video_id,
            "platform_id": platform_id,
            "endpoint": endpoint
        }
    )
    
    try:
        enqueue_timeout = float(settings.dataprocessor_enqueue_timeout_seconds)
        max_retries = max(0, int(settings.dataprocessor_enqueue_max_retries))
        retry_cap = max(1, int(settings.dataprocessor_enqueue_retry_after_cap_seconds))
        async with httpx.AsyncClient(timeout=enqueue_timeout) as client:
            response: httpx.Response | None = None
            for attempt in range(max_retries + 1):
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 503 and attempt < max_retries:
                    ra_hdr = response.headers.get("Retry-After")
                    try:
                        wait_s = int(float(ra_hdr)) if ra_hdr else 60
                    except (TypeError, ValueError):
                        wait_s = 60
                    wait_s = max(1, min(wait_s, retry_cap))
                    logger.warning(
                        "DataProcessor POST /process 503 (backpressure?), retry %s/%s after %ss run_id=%s",
                        attempt + 1,
                        max_retries,
                        wait_s,
                        run_id,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                response.raise_for_status()
                break
            assert response is not None

            # Получить результат
            result = response.json()
            
            logger.info(
                "Processing request accepted by DataProcessor API",
                extra={
                    "run_id": run_id,
                    "dataprocessor_status": result.get("status"),
                    "dataprocessor_message": result.get("message"),
                },
            )
            
            # Вернуть пути (они будут доступны после завершения обработки)
            run_paths = resolve_run_paths(
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                result_store_base=result_store_base,
            )
            frames_dir = frames_dir_base / video_id / "video"
            run_paths.frames_dir = frames_dir if frames_dir.exists() else None
            
            return run_paths
            
    except httpx.HTTPStatusError as e:
        logger.error(
            "DataProcessor API returned error status",
            extra={
                "run_id": run_id,
                "status_code": e.response.status_code,
                "response": e.response.text
            }
        )
        raise
    except httpx.RequestError as e:
        logger.error(
            "Failed to connect to DataProcessor API",
            extra={
                "run_id": run_id,
                "endpoint": endpoint,
                "error": str(e)
            }
        )
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error calling DataProcessor API",
            extra={"run_id": run_id, "error": str(e)}
        )
        raise


async def poll_run_status(
    run_id: str,
    timeout_seconds: Optional[int] = None,
    poll_interval: Optional[int] = None
) -> Dict[str, Any]:
    """
    Polling статуса обработки run'а до завершения.
    
    Периодически запрашивает статус run'а через DataProcessor API до получения
    финального статуса (success, error, empty, skipped) или истечения timeout.
    
    Args:
        run_id: UUID run'а
        timeout_seconds: Максимальное время ожидания в секундах (по умолчанию из config)
        poll_interval: Интервал между запросами в секундах (по умолчанию из config)
        
    Returns:
        Словарь с финальным статусом run'а:
        {
            "run_id": str,
            "status": str,  # success, error, empty, skipped
            "progress": dict,
            "components": dict,
            ...
        }
        
    Raises:
        TimeoutError: Если обработка не завершилась за timeout_seconds
        ValueError: Если run не найден (404)
        httpx.HTTPError: При других ошибках HTTP
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1939-1969)
    """
    settings = Settings()
    
    # Использовать значения из config если не указаны
    if timeout_seconds is None:
        timeout_seconds = settings.dataprocessor_timeout_seconds
    if poll_interval is None:
        poll_interval = settings.dataprocessor_poll_interval
    
    api_url = settings.dataprocessor_api_url.rstrip("/")
    endpoint = f"{api_url}/api/v1/runs/{run_id}/status"
    
    # Подготовить headers
    headers = {}
    if settings.dataprocessor_api_key:
        headers["X-API-Key"] = settings.dataprocessor_api_key
    
    start_time = time.time()
    last_status = None
    
    logger.info(
        "Starting polling for run status",
        extra={
            "run_id": run_id,
            "timeout_seconds": timeout_seconds,
            "poll_interval": poll_interval
        }
    )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            # Проверить timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    "Polling timeout for run",
                    extra={
                        "run_id": run_id,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                        "last_status": last_status
                    }
                )
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout_seconds} seconds "
                    f"(elapsed: {elapsed:.1f}s, last_status: {last_status})"
                )
            
            try:
                # Запросить статус
                response = await client.get(
                    endpoint,
                    headers=headers
                )
                
                # Обработка 404
                if response.status_code == 404:
                    logger.warning(
                        "Run not found during polling",
                        extra={"run_id": run_id}
                    )
                    raise ValueError(f"Run {run_id} not found")
                
                # Проверить другие ошибки
                response.raise_for_status()
                
                # Получить статус
                status_data = response.json()
                current_status = status_data.get("status")
                last_status = current_status
                
                logger.debug(
                    "Polling status update",
                    extra={
                        "run_id": run_id,
                        "status": current_status,
                        "elapsed_seconds": elapsed,
                        "progress": status_data.get("progress", {}).get("overall", 0)
                    }
                )
                
                # Проверить финальный статус
                final_statuses = ["success", "error", "empty", "skipped", "cancelled"]
                if current_status in final_statuses:
                    logger.info(
                        "Run completed with final status",
                        extra={
                            "run_id": run_id,
                            "status": current_status,
                            "elapsed_seconds": elapsed
                        }
                    )
                    return status_data
                
                # Подождать перед следующим запросом
                await asyncio.sleep(poll_interval)
                
            except httpx.HTTPStatusError as e:
                # Ошибки статуса HTTP (кроме 404, уже обработано)
                logger.error(
                    "HTTP error during polling",
                    extra={
                        "run_id": run_id,
                        "status_code": e.response.status_code,
                        "response": e.response.text
                    }
                )
                raise
            except httpx.RequestError as e:
                # Ошибки соединения
                logger.error(
                    "Connection error during polling",
                    extra={
                        "run_id": run_id,
                        "endpoint": endpoint,
                        "error": str(e)
                    }
                )
                # Продолжить polling при transient ошибках
                # Подождать перед повтором
                await asyncio.sleep(poll_interval)
                continue
            except (ValueError, TimeoutError):
                # Эти ошибки нужно пробросить дальше
                raise
            except Exception as e:
                # Неожиданные ошибки
                logger.exception(
                    "Unexpected error during polling",
                    extra={"run_id": run_id, "error": str(e)}
                )
                # Продолжить polling при неожиданных ошибках
                await asyncio.sleep(poll_interval)
                continue


async def stream_run_events_sse(
    run_id: str,
    timeout_seconds: Optional[int] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    SSE listener для real-time обновлений статуса run'а.
    
    Подключается к SSE endpoint DataProcessor API и получает события в реальном времени.
    
    Args:
        run_id: UUID run'а
        timeout_seconds: Максимальное время ожидания (по умолчанию из config)
        
    Yields:
        События из SSE stream:
        {
            "type": str,  # progress, stage, component_start, component_complete, complete, error
            "data": dict,  # Данные события
            ...
        }
        
    Raises:
        httpx.HTTPError: При ошибках HTTP
        TimeoutError: Если stream не получил финальное событие за timeout
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1971-1990)
    """
    settings = Settings()
    
    if timeout_seconds is None:
        timeout_seconds = settings.dataprocessor_timeout_seconds
    
    api_url = settings.dataprocessor_api_url.rstrip("/")
    endpoint = f"{api_url}/api/v1/runs/{run_id}/events"
    
    # Подготовить headers
    headers = {}
    if settings.dataprocessor_api_key:
        headers["X-API-Key"] = settings.dataprocessor_api_key
    
    logger.info(
        "Starting SSE stream for run",
        extra={"run_id": run_id, "endpoint": endpoint}
    )
    
    start_time = time.time()
    event_type = None
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                endpoint,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                # Читать SSE stream
                async for line in response.aiter_lines():
                    # Проверить timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        logger.warning(
                            "SSE stream timeout for run",
                            extra={"run_id": run_id, "elapsed_seconds": elapsed}
                        )
                        raise TimeoutError(
                            f"SSE stream timeout for run {run_id} "
                            f"(elapsed: {elapsed:.1f}s, timeout: {timeout_seconds}s)"
                        )
                    
                    # Парсить SSE формат
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            event_data = json.loads(line[6:])
                            yield {
                                "type": event_type if event_type else "message",
                                "data": event_data
                            }
                            
                            # Проверить финальное событие
                            if event_type in ["complete", "error"]:
                                logger.info(
                                    "SSE stream completed for run",
                                    extra={"run_id": run_id, "event_type": event_type}
                                )
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Failed to parse SSE event data",
                                extra={"run_id": run_id, "line": line, "error": str(e)}
                            )
                            continue
                    
                    # Пропустить пустые строки и комментарии
                    if not line or line.startswith(":"):
                        continue
                        
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error in SSE stream",
            extra={
                "run_id": run_id,
                "status_code": e.response.status_code,
                "response": e.response.text
            }
        )
        raise
    except httpx.RequestError as e:
        logger.error(
            "Connection error in SSE stream",
            extra={"run_id": run_id, "endpoint": endpoint, "error": str(e)}
        )
        raise
    except TimeoutError:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error in SSE stream",
            extra={"run_id": run_id, "error": str(e)}
        )
        raise


async def _get_run_status_once(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Получить статус run'а один раз (для проверки финального статуса).
    
    Args:
        run_id: UUID run'а
        
    Returns:
        Словарь со статусом или None при ошибке
    """
    settings = Settings()
    api_url = settings.dataprocessor_api_url.rstrip("/")
    endpoint = f"{api_url}/api/v1/runs/{run_id}/status"
    
    headers = {}
    if settings.dataprocessor_api_key:
        headers["X-API-Key"] = settings.dataprocessor_api_key
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(
            "Failed to get run status",
            extra={"run_id": run_id, "error": str(e)}
        )
        return None


async def wait_for_run_completion_hybrid(
    run_id: str,
    webhook_timeout: int = 30,
    timeout_seconds: Optional[int] = None,
    poll_interval: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Hybrid подход для ожидания завершения обработки run'а.
    
    Использует комбинацию:
    1. SSE listener (для real-time обновлений)
    2. Polling fallback (если SSE не работает)
    
    Args:
        run_id: UUID run'а
        webhook_timeout: Время ожидания webhook в секундах (по умолчанию 30)
        timeout_seconds: Максимальное время ожидания (по умолчанию из config)
        poll_interval: Интервал polling (по умолчанию из config)
        progress_callback: Callback функция для обработки прогресса из SSE событий
        
    Returns:
        Словарь с финальным статусом run'а
        
    Raises:
        TimeoutError: Если обработка не завершилась за timeout_seconds
        ValueError: Если run не найден
        httpx.HTTPError: При ошибках HTTP
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1868-1990)
    """
    settings = Settings()
    
    if timeout_seconds is None:
        timeout_seconds = settings.dataprocessor_timeout_seconds
    if poll_interval is None:
        poll_interval = settings.dataprocessor_poll_interval
    
    logger.info(
        "Starting hybrid wait for run completion",
        extra={
            "run_id": run_id,
            "webhook_timeout": webhook_timeout,
            "timeout_seconds": timeout_seconds,
            "poll_interval": poll_interval
        }
    )
    
    start_time = time.time()
    
    # Попытка 1: SSE listener (real-time обновления)
    try:
        logger.debug("Attempting SSE stream for run", extra={"run_id": run_id})
        
        final_status = None
        async for event in stream_run_events_sse(run_id, timeout_seconds):
            event_type = event.get("type")
            event_data = event.get("data", {})
            
            logger.debug(
                "SSE event received",
                extra={
                    "run_id": run_id,
                    "event_type": event_type,
                    "data": event_data
                }
            )
            
            # Вызвать callback для обработки прогресса
            if progress_callback:
                try:
                    progress_callback(event)
                except Exception as e:
                    logger.warning(
                        "Error in progress callback",
                        extra={"run_id": run_id, "error": str(e)}
                    )
            
            # Проверить финальное событие
            if event_type in ["complete", "error"]:
                # Получить финальный статус через API
                final_status = await _get_run_status_once(run_id)
                if final_status:
                    logger.info(
                        "Run completed via SSE",
                        extra={
                            "run_id": run_id,
                            "status": final_status.get("status"),
                            "elapsed_seconds": time.time() - start_time
                        }
                    )
                    return final_status
        
        # Если SSE stream завершился без финального события, fallback на polling
        if not final_status:
            logger.info(
                "SSE stream ended without final event, falling back to polling",
                extra={"run_id": run_id}
            )
            return await poll_run_status(run_id, timeout_seconds, poll_interval)
            
    except (httpx.HTTPError, TimeoutError) as e:
        # Ошибка SSE, fallback на polling
        logger.warning(
            "SSE stream failed, falling back to polling",
            extra={"run_id": run_id, "error": str(e)}
        )
        return await poll_run_status(run_id, timeout_seconds, poll_interval)
    except Exception as e:
        # Неожиданная ошибка, fallback на polling
        logger.warning(
            "Unexpected error in SSE stream, falling back to polling",
            extra={"run_id": run_id, "error": str(e)}
        )
        return await poll_run_status(run_id, timeout_seconds, poll_interval)

