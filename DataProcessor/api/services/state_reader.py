"""
State Reader Service - чтение state из storage

Этот сервис отвечает за чтение состояния обработки из Storage.

Для MVP реализует cold path (чтение из Storage).
В будущем (Этап 2) будет добавлен hot path (кэширование в Redis).

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1382, 1488-1629)
"""

import copy
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from storage.base import Storage
from storage.paths import KeyLayout
from state.managers import RunStateManager, ProcessorStateManager
from state.enums import Status
from api.schemas.state import RunStatus
from api.services.redis_schema import (
    get_run_state as get_run_state_from_redis,
    save_run_state as save_run_state_to_redis,
    get_run_events as get_run_events_from_redis,
    get_run_metadata as get_run_metadata_from_redis
)
from api.services.redis_client import get_redis_client
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

PROCESSOR_NAMES = ("segmenter", "audio", "text", "visual")


class StateReader:
    """
    Сервис для чтения состояния обработки из Storage.
    
    Реализует двухуровневую архитектуру чтения состояния:
    - Hot path: Чтение из Redis cache (быстро, < 10ms)
    - Cold path: Чтение из Storage (медленнее, но всегда актуально)
    
    Автоматически обновляет cache при чтении из Storage для последующих запросов.
    
    Attributes:
        storage: Storage для чтения состояния из persistent storage
        key_layout: KeyLayout для работы с путями в Storage
        task_manager: TaskManager для проверки активных run'ов (опционально)
        redis_client: Redis клиент для кэширования (опционально)
        cache_ttl: TTL для кэша в Redis (по умолчанию 300 секунд)
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1488-1629)
    
    Example:
        ```python
        state_reader = StateReader(
            storage=storage,
            key_layout=key_layout,
            redis_client=redis_client
        )
        
        # Чтение статуса (использует cache если доступен)
        status = await state_reader.get_run_status("run-id-123")
        print(f"Status: {status['status']}")
        print(f"Progress: {status['progress']}")
        ```
    """
    
    def __init__(
        self,
        storage: Storage,
        key_layout: KeyLayout,
        task_manager=None,
        redis_client: Optional[Redis] = None
    ):
        """
        Инициализация StateReader.
        
        Args:
            storage: Storage для чтения состояния из persistent storage (обязательно)
            key_layout: KeyLayout для работы с путями в Storage (обязательно)
            task_manager: TaskManager для проверки активных run'ов (опционально)
            redis_client: Redis клиент для кэширования (опционально)
                Если не предоставлен, используется только cold path (Storage)
        """
        """
        Инициализация StateReader.
        
        Args:
            storage: Экземпляр Storage для чтения данных
            key_layout: KeyLayout для работы с путями
            task_manager: TaskManager для получения метаданных активных run'ов (опционально)
            redis_client: Redis клиент для кэширования (опционально, будет получен автоматически)
        """
        self.storage = storage
        self.key_layout = key_layout
        self.task_manager = task_manager
        self.redis_client = redis_client or get_redis_client()
        self.cache_ttl = 300  # 5 минут для активных run'ов
        
        # Кэш для lazy loading компонентов и manifest
        self._component_cache: Dict[str, Dict[str, Any]] = {}  # {(run_id, processor_name): state}
        self._manifest_cache: Dict[str, Dict[str, Any]] = {}  # {run_id: manifest}
        self._cache_timestamps: Dict[str, float] = {}  # {cache_key: timestamp}
        self._cache_ttl_components = 300  # 5 минут для компонентов
        self._cache_ttl_manifest = 600  # 10 минут для manifest

    def _get_enabled_processors(self, metadata: Optional[Dict[str, Any]]) -> set[str]:
        """Return enabled processors from Redis metadata/profile config."""
        enabled = set(PROCESSOR_NAMES)
        if not isinstance(metadata, dict):
            return enabled
        profile_config = metadata.get("profile_config")
        if not isinstance(profile_config, dict):
            return enabled
        processors_cfg = profile_config.get("processors")
        if not isinstance(processors_cfg, dict):
            return enabled

        discovered = {
            name
            for name in PROCESSOR_NAMES
            if isinstance(processors_cfg.get(name), dict)
        }
        if discovered:
            enabled = discovered

        explicit_enabled = {
            name
            for name in PROCESSOR_NAMES
            if isinstance(processors_cfg.get(name), dict)
            and processors_cfg.get(name, {}).get("enabled") is True
        }
        explicit_disabled = {
            name
            for name in PROCESSOR_NAMES
            if isinstance(processors_cfg.get(name), dict)
            and processors_cfg.get(name, {}).get("enabled") is False
        }
        if explicit_enabled:
            enabled = explicit_enabled
        enabled -= explicit_disabled
        return enabled or {"segmenter"}

    def _processor_required_flags(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Читает processors.<name>.required из profile_config в metadata Redis.
        Если ключ отсутствует — считаем процессор обязательным (как раньше).
        """
        out: Dict[str, bool] = {name: True for name in PROCESSOR_NAMES}
        if not isinstance(metadata, dict):
            return out
        profile_config = metadata.get("profile_config")
        if not isinstance(profile_config, dict):
            return out
        processors_cfg = profile_config.get("processors")
        if not isinstance(processors_cfg, dict):
            return out
        for name in PROCESSOR_NAMES:
            block = processors_cfg.get(name)
            if isinstance(block, dict) and "required" in block:
                out[name] = bool(block["required"])
        return out

    def _snapshot_to_processor_state(
        self,
        processor_name: str,
        snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Convert compact run_state processor snapshot into state_<processor>.json-like structure."""
        if not isinstance(snapshot, dict):
            return None
        if isinstance(snapshot.get("processor"), dict):
            return snapshot
        return {
            "processor": {
                "name": processor_name,
                "status": snapshot.get("status", "waiting"),
                "started_at": snapshot.get("started_at"),
                "finished_at": snapshot.get("finished_at"),
                "duration_ms": snapshot.get("duration_ms"),
                "error": snapshot.get("error"),
                "error_code": snapshot.get("error_code"),
            },
            "components": snapshot.get("components", {}) if isinstance(snapshot.get("components"), dict) else {},
            "updated_at": snapshot.get("updated_at"),
        }

    def _effective_processor_status(
        self,
        processor_name: str,
        proc_state: Optional[Dict[str, Any]],
        enabled_processors: set[str],
    ) -> str:
        if not proc_state:
            return "waiting"
        processor_info = proc_state.get("processor", {})
        if isinstance(processor_info, dict):
            status = processor_info.get("status", proc_state.get("status", "waiting"))
            started_at = processor_info.get("started_at")
            finished_at = processor_info.get("finished_at")
            error = processor_info.get("error")
        else:
            status = proc_state.get("status", "waiting")
            started_at = proc_state.get("started_at")
            finished_at = proc_state.get("finished_at")
            error = proc_state.get("error")

        if (
            processor_name not in enabled_processors
            and status in ("waiting", "queued", None)
            and not started_at
            and not finished_at
            and not error
        ):
            return "skipped"
        return status or "waiting"

    def _effective_processor_progress(
        self,
        proc_state: Optional[Dict[str, Any]],
        effective_status: str,
    ) -> float:
        if not proc_state:
            return 0.0
        processor_info = proc_state.get("processor", {})
        if isinstance(processor_info, dict):
            progress = processor_info.get("progress")
        else:
            progress = proc_state.get("progress")

        if progress is None or progress == 0.0:
            done = proc_state.get("done") or (
                processor_info.get("done") if isinstance(processor_info, dict) else None
            )
            total = proc_state.get("total") or (
                processor_info.get("total") if isinstance(processor_info, dict) else None
            )
            if done is not None and total is not None and total > 0:
                progress = float(done) / float(total)
            elif effective_status in ("success", "skipped", "empty"):
                progress = 1.0
            elif effective_status == "running":
                progress = 0.5
            else:
                progress = 0.0
        return float(progress)

    def _derive_run_time_bounds(
        self,
        processors: Dict[str, Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        started_candidates: List[str] = []
        finished_candidates: List[str] = []
        for proc_state in processors.values():
            processor_info = proc_state.get("processor", {}) if isinstance(proc_state, dict) else {}
            if isinstance(processor_info, dict):
                started_at = processor_info.get("started_at")
                finished_at = processor_info.get("finished_at")
                if started_at:
                    started_candidates.append(started_at)
                if finished_at:
                    finished_candidates.append(finished_at)
        started_at = min(started_candidates) if started_candidates else None
        finished_at = max(finished_candidates) if finished_candidates else None
        return started_at, finished_at
    
    async def _find_run_metadata(self, run_id: str) -> Optional[tuple]:
        """
        Найти platform_id и video_id для run_id.
        
        Сначала проверяет TaskManager, затем пытается найти в storage.
        
        Args:
            run_id: UUID run'а
            
        Returns:
            Кортеж (platform_id, video_id) или None если не найдено
        """
        # Попробовать получить из TaskManager если доступен
        if self.task_manager:
            run_info = self.task_manager.get_run(run_id)
            if run_info:
                platform_id = run_info.get("platform_id")
                video_id = run_info.get("video_id")
                if platform_id and video_id:
                    return (platform_id, video_id)

        # Попробовать получить из Redis metadata (общий источник между API и worker)
        redis_meta = await get_run_metadata_from_redis(run_id)
        if redis_meta:
            platform_id = redis_meta.get("platform_id")
            video_id = redis_meta.get("video_id")
            if platform_id and video_id:
                return (platform_id, video_id)
        
        # Попробовать найти в storage через поиск
        # Ищем run_state.json содержащий run_id
        state_prefix = self.key_layout.state_prefix()
        
        try:
            # Перебираем возможные платформы
            for platform_id in ["youtube", "upload"]:
                platform_prefix = f"{state_prefix}/{platform_id}"
                
                # Получаем список видео с retry
                from api.utils.retry import retry_storage_operation

                obj_list = await retry_storage_operation(
                    self.storage.list,
                    platform_prefix
                )
                for obj_info in obj_list:
                    if not obj_info.key.endswith("/run_state.json"):
                        continue
                    
                    # Извлекаем video_id и run_id из пути
                    # Формат: state/{platform_id}/{video_id}/{run_id}/run_state.json
                    parts = obj_info.key.split("/")
                    if len(parts) >= 4:
                        video_id = parts[-3]
                        run_id_from_path = parts[-2]
                        
                        if run_id_from_path == run_id:
                            return (platform_id, video_id)
        except Exception as e:
            logger.debug(f"Error searching for run metadata: {e}")
        
        return None
    
    async def _load_run_state(
        self,
        platform_id: str,
        video_id: str,
        run_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Загрузить run_state.json из Storage.
        
        Args:
            platform_id: ID платформы
            video_id: ID видео
            run_id: UUID run'а
            
        Returns:
            Словарь с состоянием run'а или None если не найден
        """
        state_prefix = self.key_layout.state_run_prefix(platform_id, video_id, run_id)
        state_key = f"{state_prefix}/run_state.json"
        
        try:
            from api.utils.retry import retry_storage_operation

            # Проверка exists с retry
            exists = await retry_storage_operation(
                self.storage.exists,
                state_key
            )
            if not exists:
                logger.debug(f"Run state not found: {state_key}")
                return None
            
            # Чтение с retry
            data = await retry_storage_operation(
                self.storage.read_bytes,
                state_key
            )
            return json.loads(data.decode("utf-8"))
            
        except Exception as e:
            logger.error(f"Error loading run state from {state_key}: {e}")
            return None
    
    async def _load_processor_state(
        self,
        platform_id: str,
        video_id: str,
        run_id: str,
        processor_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Загрузить state_{processor}.json из Storage.
        
        Args:
            platform_id: ID платформы
            video_id: ID видео
            run_id: UUID run'а
            processor_name: Имя процессора (segmenter, audio, text, visual)
            
        Returns:
            Словарь с состоянием процессора или None если не найден
        """
        state_prefix = self.key_layout.state_run_prefix(platform_id, video_id, run_id)
        state_key = f"{state_prefix}/state_{processor_name}.json"
        
        try:
            from api.utils.retry import retry_storage_operation

            # Проверка exists с retry
            exists = await retry_storage_operation(
                self.storage.exists,
                state_key
            )
            if not exists:
                logger.debug(f"Processor state not found: {state_key}")
                return None
            
            # Чтение с retry
            data = await retry_storage_operation(
                self.storage.read_bytes,
                state_key
            )
            return json.loads(data.decode("utf-8"))
            
        except Exception as e:
            logger.error(f"Error loading processor state from {state_key}: {e}")
            return None

    async def _read_manifest_from_storage(
        self,
        platform_id: str,
        video_id: str,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read manifest.json for the run directly from storage (no long-lived cache).
        AudioProcessor writes per-extractor status here while state_audio.json stays coarse.
        """
        manifest_path = f"{self.key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        try:
            from api.utils.retry import retry_storage_operation

            exists = await retry_storage_operation(self.storage.exists, manifest_path)
            if not exists:
                return None
            data = await retry_storage_operation(self.storage.read_bytes, manifest_path)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.debug("Manifest read failed %s: %s", manifest_path, e)
            return None

    def _manifest_entry_is_audio_row(self, c: Dict[str, Any]) -> bool:
        """AudioProcessor rows use kind=audio. Do not treat visual cores (e.g. ocr_extractor) as audio."""
        kind = str(c.get("kind") or "").lower().strip()
        if kind == "audio":
            return True
        if kind in ("core", "module", "text"):
            return False
        name = str(c.get("name") or "").strip()
        return name.endswith("_extractor")

    def _manifest_status_to_api(self, raw: str) -> str:
        s = (raw or "").strip().lower()
        if s == "ok":
            return "success"
        if s == "empty":
            return "empty"
        if s == "skipped":
            return "skipped"
        if s == "error":
            return "error"
        if s == "running":
            return "running"
        return s or "waiting"

    def _apply_manifest_audio_components(
        self,
        audio_proc_state: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        base = audio_proc_state.get("components")
        merged: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
        for c in manifest.get("components") or []:
            if not isinstance(c, dict) or not self._manifest_entry_is_audio_row(c):
                continue
            name = c.get("name")
            if not name:
                continue
            key = str(name)
            st = self._manifest_status_to_api(str(c.get("status") or ""))
            merged[key] = {
                "name": key,
                "status": st,
                "artifacts": list(c.get("artifacts") or []),
                "error": c.get("error"),
                "error_code": c.get("error_code"),
                "notes": c.get("notes"),
                "started_at": c.get("started_at"),
                "finished_at": c.get("finished_at"),
                "duration_ms": c.get("duration_ms"),
                "device_used": c.get("device_used"),
            }
        audio_proc_state["components"] = merged

    def _apply_manifest_text_components(
        self,
        text_proc_state: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        base = text_proc_state.get("components")
        merged: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
        for c in manifest.get("components") or []:
            if not isinstance(c, dict) or str(c.get("kind") or "").lower().strip() != "text":
                continue
            name = c.get("name")
            if not name:
                continue
            key = str(name)
            st = self._manifest_status_to_api(str(c.get("status") or ""))
            merged[key] = {
                "name": key,
                "status": st,
                "artifacts": list(c.get("artifacts") or []),
                "error": c.get("error"),
                "error_code": c.get("error_code"),
                "notes": c.get("notes"),
                "started_at": c.get("started_at"),
                "finished_at": c.get("finished_at"),
                "duration_ms": c.get("duration_ms"),
                "device_used": c.get("device_used"),
            }
        text_proc_state["components"] = merged

    def _apply_manifest_visual_components(
        self,
        visual_proc_state: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        """Подкомпоненты visual из manifest: kind module и core (CLIP, depth, …), без audio/text."""
        visual_rows: List[Dict[str, Any]] = []
        for c in manifest.get("components") or []:
            if not isinstance(c, dict):
                continue
            k = str(c.get("kind") or "").lower().strip()
            if k in ("module", "core"):
                visual_rows.append(c)
        if not visual_rows:
            return
        merged: Dict[str, Any] = {}
        for c in visual_rows:
            name = c.get("name")
            if not name:
                continue
            key = str(name)
            st = self._manifest_status_to_api(str(c.get("status") or ""))
            merged[key] = {
                "name": key,
                "status": st,
                "artifacts": list(c.get("artifacts") or []),
                "error": c.get("error"),
                "error_code": c.get("error_code"),
                "notes": c.get("notes"),
                "started_at": c.get("started_at"),
                "finished_at": c.get("finished_at"),
                "duration_ms": c.get("duration_ms"),
                "device_used": c.get("device_used"),
            }
        visual_proc_state["components"] = merged

    def _calculate_overall_progress(
        self,
        processors: Dict[str, Dict[str, Any]],
        enabled_processors: Optional[set[str]] = None,
    ) -> float:
        """
        Вычислить общий прогресс на основе состояний процессоров.
        
        Args:
            processors: Словарь с состояниями процессоров
            
        Returns:
            Прогресс от 0.0 до 1.0
        """
        if not processors:
            return 0.0
        
        total_progress = 0.0
        count = 0
        
        for proc_name, proc_state in processors.items():
            if proc_state:
                if enabled_processors is not None and proc_name not in enabled_processors:
                    continue
                status = self._effective_processor_status(proc_name, proc_state, enabled_processors or set(PROCESSOR_NAMES))
                progress = self._effective_processor_progress(proc_state, status)

                if status in ("success", "running", "error", "skipped", "empty"):
                    total_progress += progress
                    count += 1
        
        if count == 0:
            return 0.0
        
        return total_progress / count
    
    async def get_run_status(
        self,
        run_id: str,
        platform_id: Optional[str] = None,
        video_id: Optional[str] = None,
        include_components: bool = True,
        include_events: bool = False
    ) -> Dict[str, Any]:
        """
        Получить детальный статус run'а.
        
        Сначала проверяет Redis cache (hot path), затем Storage (cold path).
        
        Args:
            run_id: UUID run'а
            platform_id: ID платформы (опционально, будет найдено автоматически)
            video_id: ID видео (опционально, будет найдено автоматически)
            include_components: Включить детальную информацию о компонентах
            include_events: Включить последние события
            
        Returns:
            Словарь с детальным статусом
            
        Raises:
            RunNotFoundError: Если run не найден
        """
        # Попробовать получить из Redis cache (hot path)
        redis_state = await get_run_state_from_redis(run_id)
        redis_meta = await get_run_metadata_from_redis(run_id)
        if redis_state:
            logger.debug(f"Got run {run_id} state from Redis cache (hot path)")
            
            # Проверить heartbeat для running run'ов
            status = redis_state.get("status", "unknown")
            if status == "running":
                from api.services.recovery import check_and_recover_run
                recovered = await check_and_recover_run(run_id)
                if recovered:
                    # Обновить статус после recovery
                    redis_state = await get_run_state_from_redis(run_id)
                    if redis_state:
                        redis_state["status"] = "recovering"
            # TODO: Объединить данные из Redis и Storage для полной информации
            # Пока используем данные из Storage для детальной информации

        # Если platform_id и video_id не указаны, сначала попробовать Redis metadata/state
        if not platform_id or not video_id:
            if redis_meta:
                platform_id = platform_id or redis_meta.get("platform_id")
                video_id = video_id or redis_meta.get("video_id")
            if redis_state:
                platform_id = platform_id or redis_state.get("platform_id")
                video_id = video_id or redis_state.get("video_id")
        
        # Если platform_id и video_id не указаны, попробовать найти их
        if not platform_id or not video_id:
            metadata = await self._find_run_metadata(run_id)
            if metadata:
                platform_id, video_id = metadata
            else:
                from api.utils.errors import RunNotFoundError
                raise RunNotFoundError(f"Run not found: {run_id}")
        
        # Загрузить run_state.json
        run_state = await self._load_run_state(platform_id, video_id, run_id)
        
        if not run_state:
            if redis_state:
                result = {
                    "run_id": run_id,
                    "video_id": video_id,
                    "platform_id": platform_id,
                    "status": redis_state.get("status", "queued"),
                    "stage": redis_state.get("current_processor"),
                    "progress": {
                        "overall": redis_state.get("progress", 0.0) or 0.0,
                        "current_processor": redis_state.get("current_processor"),
                        "current_component": redis_state.get("current_component"),
                        "components": {},
                    },
                    "started_at": redis_state.get("started_at"),
                    "updated_at": redis_state.get("updated_at"),
                    "finished_at": redis_state.get("finished_at"),
                    "error": redis_state.get("error"),
                    "error_code": redis_state.get("error_code"),
                }
                if include_events:
                    result["events"] = await self.get_events(run_id, platform_id, video_id)
                return result

            from api.utils.errors import RunNotFoundError
            raise RunNotFoundError(f"Run not found: {run_id}")
        
        enabled_processors = self._get_enabled_processors(redis_meta)

        run_processors = run_state.get("processors", {}) if isinstance(run_state, dict) else {}

        # Загружаем состояния процессоров даже без include_components, чтобы корректно
        # агрегировать status/progress из run_state.json.
        processors: Dict[str, Dict[str, Any]] = {}
        for proc_name in PROCESSOR_NAMES:
            fallback_state = self._snapshot_to_processor_state(
                proc_name,
                run_processors.get(proc_name) if isinstance(run_processors, dict) else None,
            )
            proc_state = None
            if include_components:
                cache_key = f"{run_id}:{proc_name}"
                cached_state = self._component_cache.get(cache_key)
                cache_timestamp = self._cache_timestamps.get(cache_key, 0)
                if cached_state and (time.time() - cache_timestamp) < self._cache_ttl_components:
                    proc_state = cached_state
                    logger.debug(f"Using cached processor state for {cache_key}")
                else:
                    proc_state = await self._load_processor_state(platform_id, video_id, run_id, proc_name)
                    if proc_state:
                        self._component_cache[cache_key] = proc_state
                        self._cache_timestamps[cache_key] = time.time()
            if proc_state or fallback_state:
                processors[proc_name] = proc_state or fallback_state

        # Subprocessors often publish fine-grained status in manifest.json; merge into API view (fresh manifest read).
        if include_components and any(
            processors.get(p) for p in ("audio", "text", "visual")
        ):
            mf = await self._read_manifest_from_storage(platform_id, video_id, run_id)
            if mf:
                if processors.get("audio"):
                    audio_view = copy.deepcopy(processors["audio"])
                    self._apply_manifest_audio_components(audio_view, mf)
                    processors["audio"] = audio_view
                if processors.get("text"):
                    text_view = copy.deepcopy(processors["text"])
                    self._apply_manifest_text_components(text_view, mf)
                    processors["text"] = text_view
                if processors.get("visual"):
                    visual_view = copy.deepcopy(processors["visual"])
                    self._apply_manifest_visual_components(visual_view, mf)
                    processors["visual"] = visual_view

        # Вычислить общий прогресс
        overall_progress = self._calculate_overall_progress(processors, enabled_processors)
        
        # Определить текущую стадию и текущий компонент
        current_stage = None
        current_component = None
        
        for proc_name, proc_state in processors.items():
            proc_status = self._effective_processor_status(proc_name, proc_state, enabled_processors)
            if proc_status == "running":
                current_stage = proc_name
                # Попробовать найти текущий компонент внутри процессора
                components = proc_state.get("components", {})
                for comp_name, comp_data in components.items():
                    if isinstance(comp_data, dict) and comp_data.get("status") == "running":
                        current_component = comp_name
                        break
                break
        
        # Формировать компоненты для ответа
        components_dict = {}
        if include_components:
            for proc_name, proc_state in processors.items():
                proc_status = self._effective_processor_status(proc_name, proc_state, enabled_processors)
                proc_progress = self._effective_processor_progress(proc_state, proc_status)
                processor_info = proc_state.get("processor", {})
                pe = processor_info if isinstance(processor_info, dict) else {}
                proc_err = pe.get("error")
                proc_ec = pe.get("error_code")
                if proc_err is None and isinstance(proc_state, dict):
                    proc_err = proc_state.get("error")
                if proc_ec is None and isinstance(proc_state, dict):
                    proc_ec = proc_state.get("error_code")
                component_progress = {
                    "status": proc_status,
                    "progress": proc_progress,
                    "started_at": pe.get("started_at") if isinstance(processor_info, dict) else proc_state.get("started_at"),
                    "finished_at": pe.get("finished_at") if isinstance(processor_info, dict) else proc_state.get("finished_at"),
                    "duration_ms": pe.get("duration_ms") if isinstance(processor_info, dict) else proc_state.get("duration_ms"),
                    "error": proc_err,
                    "error_code": proc_ec,
                    "current_component": None,
                    "components": proc_state.get("components", {}) if include_components else None,
                }
                
                # Добавить done/total если есть
                if "done" in proc_state:
                    component_progress["done"] = proc_state["done"]
                if "total" in proc_state:
                    component_progress["total"] = proc_state["total"]
                
                components_dict[proc_name] = component_progress
        
        # Извлечь статус из run_state
        run_info = run_state.get("run", {})
        if isinstance(run_info, dict):
            status = run_info.get("status")
            if not status:
                required_flags = self._processor_required_flags(redis_meta)
                proc_statuses_any = [
                    self._effective_processor_status(name, proc_state, enabled_processors)
                    for name, proc_state in processors.items()
                    if name in enabled_processors and proc_state
                ]
                req_names = [n for n in enabled_processors if required_flags.get(n, True)]
                if not req_names:
                    req_names = list(enabled_processors)
                proc_statuses_required = [
                    self._effective_processor_status(name, processors[name], enabled_processors)
                    for name in req_names
                    if name in processors and processors[name]
                ]
                if proc_statuses_required and all(
                    s in ("success", "skipped", "empty") for s in proc_statuses_required if s
                ):
                    status = RunStatus.SUCCESS.value
                elif any(s == "running" for s in proc_statuses_any if s):
                    status = RunStatus.RUNNING.value
                elif any(s == "error" for s in proc_statuses_required if s):
                    status = RunStatus.ERROR.value
                else:
                    status = RunStatus.PENDING.value
        else:
            status = run_state.get("status", "unknown")

        derived_started_at, derived_finished_at = self._derive_run_time_bounds(processors)
        started_at = (
            run_info.get("started_at") if isinstance(run_info, dict) else None
        ) or derived_started_at or (redis_state.get("started_at") if redis_state else None)
        finished_at = (
            run_info.get("finished_at") if isinstance(run_info, dict) else None
        ) or derived_finished_at or (redis_state.get("finished_at") if redis_state else None)
        
        result = {
            "run_id": run_id,
            "video_id": video_id,
            "platform_id": platform_id,
            "status": status,
            "stage": current_stage,
            "progress": {
                "overall": overall_progress,
                "current_processor": current_stage,
                "current_component": current_component,
                "components": components_dict
            },
            "started_at": started_at,
            "updated_at": run_state.get("updated_at"),
            "finished_at": finished_at,
            "error": run_info.get("error") if isinstance(run_info, dict) else run_state.get("error"),
            "error_code": run_info.get("error_code") if isinstance(run_info, dict) else run_state.get("error_code")
        }

        try:
            await save_run_state_to_redis(
                run_id,
                {
                    "status": status,
                    "platform_id": platform_id,
                    "video_id": video_id,
                    "progress": overall_progress,
                    "current_processor": current_stage,
                    "current_component": current_component,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "updated_at": result["updated_at"],
                    "error": result["error"],
                    "error_code": result["error_code"],
                },
                validate_status_transition=False,
            )
        except Exception as e:
            logger.debug(f"Failed to refresh Redis cache for run {run_id}: {e}")
        
        # Добавить события если нужно
        if include_events:
            events = await self.get_events(run_id, platform_id, video_id)
            result["events"] = events
        
        return result
    
    async def get_events(
        self,
        run_id: str,
        platform_id: Optional[str] = None,
        video_id: Optional[str] = None,
        since: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Получить события для run'а с поддержкой pagination.
        
        Сначала проверяет Redis Streams (hot path), затем Storage (cold path).
        Поддерживает фильтрацию по времени и компоненту, а также pagination.
        
        Args:
            run_id: UUID run'а
            platform_id: ID платформы (опционально, будет найдено автоматически)
            video_id: ID видео (опционально, будет найдено автоматически)
            since: Временная метка для фильтрации событий (опционально, ISO 8601)
            component: Фильтр по компоненту (опционально)
            limit: Максимальное количество событий для возврата (по умолчанию 100)
            offset: Смещение для pagination (по умолчанию 0)
            
        Returns:
            List[Dict[str, Any]]: Список событий с учетом pagination и фильтров
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1579-1628)
        
        Example:
            ```python
            # Получить первые 50 событий
            events = await state_reader.get_events("run-id", limit=50, offset=0)
            
            # Получить следующие 50 событий
            events = await state_reader.get_events("run-id", limit=50, offset=50)
            
            # Получить события с фильтрацией
            events = await state_reader.get_events(
                "run-id",
                since="2024-01-01T12:00:00Z",
                component="core_clip",
                limit=100
            )
            ```
        """
        # 1. Попробовать получить из Redis Streams (hot path)
        if self.redis_client:
            try:
                # Для pagination нужно получить больше событий, чем limit
                # чтобы учесть offset и фильтры
                fetch_count = limit + offset + 100  # Дополнительный буфер для фильтрации
                redis_events = await get_run_events_from_redis(run_id, count=fetch_count)
                if redis_events:
                    logger.debug(f"Got {len(redis_events)} events for run {run_id} from Redis")
                    
                    # Применить фильтры
                    filtered_events = []
                    for event in redis_events:
                        # Фильтр по времени
                        if since:
                            event_ts = event.get("timestamp") or event.get("ts")
                            if event_ts:
                                # Конвертировать timestamp в строку для сравнения
                                if isinstance(event_ts, (int, float)):
                                    from datetime import datetime
                                    event_ts_str = datetime.fromtimestamp(event_ts).isoformat()
                                else:
                                    event_ts_str = str(event_ts)
                                if event_ts_str < since:
                                    continue
                        
                        # Фильтр по компоненту
                        if component:
                            event_data = event.get("data", {})
                            if isinstance(event_data, str):
                                try:
                                    event_data = json.loads(event_data)
                                except:
                                    pass
                            event_component = event_data.get("component") if isinstance(event_data, dict) else None
                            if event_component != component:
                                continue
                        
                        filtered_events.append(event)
                    
                    # Применить pagination
                    paginated_events = filtered_events[offset:offset + limit]
                    logger.debug(f"Returning {len(paginated_events)} events after pagination (offset={offset}, limit={limit})")
                    return paginated_events
                    
            except Exception as e:
                logger.warning(f"Failed to get events from Redis for run {run_id}: {e}")
        
        # 2. Fallback: читать из Storage (cold path)
        if not platform_id or not video_id:
            # Попробовать найти из Redis metadata
            redis_meta = await get_run_metadata_from_redis(run_id)
            if redis_meta:
                platform_id = redis_meta.get("platform_id")
                video_id = redis_meta.get("video_id")
            
            if not platform_id or not video_id:
                metadata = await self._find_run_metadata(run_id)
                if metadata:
                    platform_id, video_id = metadata
                else:
                    logger.warning(f"Could not find metadata for run {run_id}, returning empty events")
                    return []
        
        # Читать из Storage
        events_key = self.key_layout.state_run_prefix(
            platform_id, video_id, run_id
        ) + "/state_events.jsonl"
        
        try:
            from api.utils.retry import retry_storage_operation
            import asyncio
            
            # Проверка exists с retry
            exists = await retry_storage_operation(
                self.storage.exists,
                events_key
            )
            if not exists:
                return []
            
            # Использовать streaming чтение JSONL (не читать весь файл в память)
            # Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2330-2337)
            events = []
            
            # Проверить, поддерживает ли storage streaming
            if hasattr(self.storage, "stream_jsonl"):
                # Async streaming чтение JSONL с поддержкой pagination
                skipped = 0
                async for event in self.storage.stream_jsonl(events_key):
                    # Фильтр по времени
                    if since:
                        event_ts = event.get("ts") or event.get("timestamp")
                        if event_ts:
                            if isinstance(event_ts, str):
                                from datetime import datetime
                                try:
                                    event_ts_dt = datetime.fromisoformat(event_ts.replace("Z", "+00:00"))
                                    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                                    if event_ts_dt < since_dt:
                                        continue
                                except:
                                    pass
                            elif isinstance(event_ts, (int, float)):
                                from datetime import datetime
                                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                                event_ts_dt = datetime.fromtimestamp(event_ts)
                                if event_ts_dt < since_dt:
                                    continue
                    
                    # Фильтр по компоненту
                    if component:
                        event_component = event.get("component")
                        if event_component != component:
                            continue
                    
                    # Применить offset (skip первые offset событий)
                    if skipped < offset:
                        skipped += 1
                        continue
                    
                    events.append(event)
                    
                    # Ограничение количества
                    if len(events) >= limit:
                        break
            else:
                # Fallback: чтение всего файла (для совместимости)
                events_data = await retry_storage_operation(
                    self.storage.read_bytes,
                    events_key
                )
                
                for line in events_data.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        
                        # Фильтр по времени
                        if since:
                            event_ts = event.get("ts") or event.get("timestamp")
                            if event_ts and event_ts < since:
                                continue
                        
                        events.append(event)
                        
                        # Ограничение количества
                        if len(events) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse event line: {line[:100]}")
                        continue
            
            logger.debug(f"Got {len(events)} events for run {run_id} from Storage")
            return events
            
        except Exception as e:
            logger.error(f"Failed to read events from Storage for run {run_id}: {e}")
            return []
    
    async def get_manifest(
        self,
        run_id: str,
        platform_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Получить manifest.json для run'а с кэшированием.
        
        Кэширует manifest после первого чтения для последующих запросов.
        
        Args:
            run_id: UUID run'а
            platform_id: ID платформы (опционально, будет найдено автоматически)
            video_id: ID видео (опционально, будет найдено автоматически)
            
        Returns:
            Dict[str, Any]: Данные manifest.json или None если не найден
            
        Example:
            ```python
            manifest = await state_reader.get_manifest("run-id-123")
            if manifest:
                print(f"Schema version: {manifest.get('schema_version')}")
            ```
        """
        # Проверить кэш
        cached_manifest = self._manifest_cache.get(run_id)
        cache_timestamp = self._cache_timestamps.get(f"manifest:{run_id}", 0)
        
        if cached_manifest and (time.time() - cache_timestamp) < self._cache_ttl_manifest:
            logger.debug(f"Using cached manifest for run {run_id}")
            return cached_manifest
        
        # Если platform_id и video_id не указаны, найти их
        if not platform_id or not video_id:
            metadata = await self._find_run_metadata(run_id)
            if metadata:
                platform_id, video_id = metadata
            else:
                logger.warning(f"Could not find metadata for run {run_id}")
                return None
        
        # Путь к manifest.json
        manifest_path = f"{self.key_layout.result_store_run_prefix(platform_id, video_id, run_id)}/manifest.json"
        
        try:
            from api.utils.retry import retry_storage_operation
            import asyncio
            
            # Проверка exists с retry
            exists = await retry_storage_operation(
                self.storage.exists,
                manifest_path
            )
            if not exists:
                logger.debug(f"Manifest not found: {manifest_path}")
                return None
            
            # Чтение с retry
            manifest_bytes = await retry_storage_operation(
                self.storage.read_bytes,
                manifest_path
            )
            manifest_data = json.loads(manifest_bytes.decode("utf-8"))
            
            # Сохранить в кэш
            self._manifest_cache[run_id] = manifest_data
            self._cache_timestamps[f"manifest:{run_id}"] = time.time()
            
            logger.debug(f"Loaded and cached manifest for run {run_id}")
            return manifest_data
            
        except Exception as e:
            logger.error(f"Error loading manifest from {manifest_path}: {e}")
            return None

