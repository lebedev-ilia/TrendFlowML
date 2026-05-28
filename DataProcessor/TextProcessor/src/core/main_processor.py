from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib
import importlib.util
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.base_extractor import BaseExtractor
from src.schemas.models import VideoDocument, video_document_from_dict


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _step_uses_cuda(step_device: Optional[str]) -> bool:
    if not step_device:
        return False
    s = str(step_device).lower()
    return "cuda" in s or "gpu" in s


def _text_processor_memory_after_step(
    *,
    step_device: Optional[str],
    logger: Optional[logging.Logger],
    sync_model_manager: bool = True,
) -> None:
    """
    Best-effort RAM/VRAM hygiene between extractors.

    sync_model_manager: set False from worker threads (e.g. CPU parallel batch path) to avoid
    touching CUDA / the process-wide ModelManager off the main thread.

    Env:
    - DP_TEXT_SKIP_CUDA_EMPTY_CACHE: skip torch.cuda.empty_cache / ipc_collect for this hook
    - DP_TEXT_EVICT_MM_AFTER_EXTRACTOR: evict all ModelManager LRU entries after every step
    - DP_TEXT_EVICT_MM_CUDA_AFTER_EXTRACTOR: evict only CUDA-tagged LRU entries after CUDA steps
    """
    import gc

    gc.collect()
    if not sync_model_manager:
        return
    cuda_step = _step_uses_cuda(step_device)
    skip_empty = _env_flag("DP_TEXT_SKIP_CUDA_EMPTY_CACHE")

    if cuda_step and not skip_empty:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc):
                    ipc()
        except Exception:
            pass

    evict_all = _env_flag("DP_TEXT_EVICT_MM_AFTER_EXTRACTOR")
    evict_cuda = _env_flag("DP_TEXT_EVICT_MM_CUDA_AFTER_EXTRACTOR")
    if evict_all:
        try:
            from dp_models import get_global_model_manager  # type: ignore

            n = get_global_model_manager().evict_cached_models(device_prefix=None)
            if logger and n:
                logger.debug("TextProcessor: ModelManager evicted %d cached model(s) (all devices)", n)
        except Exception:
            pass
    elif evict_cuda and cuda_step:
        try:
            from dp_models import get_global_model_manager  # type: ignore

            n = get_global_model_manager().evict_cached_models(device_prefix="cuda")
            if logger and n:
                logger.debug("TextProcessor: ModelManager evicted %d cached CUDA model(s)", n)
        except Exception:
            pass

    gc.collect()
    if cuda_step and not skip_empty:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _text_processor_memory_run_end(logger: Optional[logging.Logger]) -> None:
    """
    Called once when MainProcessor.run() finishes (same OS process as other processors).

    Env:
    - DP_TEXT_EVICT_MM_ON_RUN_END: evict entire ModelManager LRU
    - DP_TEXT_SKIP_FINAL_CUDA_EMPTY: do not call empty_cache at run end
    """
    import gc

    gc.collect()
    if _env_flag("DP_TEXT_EVICT_MM_ON_RUN_END"):
        try:
            from dp_models import get_global_model_manager  # type: ignore

            n = get_global_model_manager().evict_cached_models(device_prefix=None)
            if logger and n:
                logger.debug("TextProcessor: ModelManager evicted %d cached model(s) at run end", n)
        except Exception:
            pass
    if _env_flag("DP_TEXT_SKIP_FINAL_CUDA_EMPTY"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc):
                ipc()
    except Exception:
        pass


def _build_dependency_levels(extractor_specs: List[Tuple[str, str, Dict[str, Any]]]) -> List[List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Группирует extractors по уровням зависимостей (топологическая сортировка).
    
    Returns:
        Список уровней, где каждый уровень - список extractor specs, которые могут выполняться параллельно.
    """
    # Определение зависимостей между extractors (на основе документации и анализа кода)
    DEPENDENCIES: Dict[str, List[str]] = {
        # Уровень 0: Tags первым — мутация doc (очистка хэштегов) до lexical/ASR proxy и embedder'ов title/description.
        "TagsExtractor": [],
        "LexicalStatsExtractor": ["TagsExtractor"],
        "ASRTextProxyExtractor": ["TagsExtractor"],
        # Уровень 1: зависят от уровня 0
        "TitleEmbedder": ["TagsExtractor"],
        "DescriptionEmbedder": ["TagsExtractor"],
        "HashtagEmbedder": ["TagsExtractor"],
        "TranscriptChunkEmbedder": [],  # Зависит только от transcripts в doc
        "CommentsEmbedder": [],  # Зависит только от comments в doc
        "SpeakerTurnEmbeddingsAggregatorExtractor": [],
        # Уровень 2: зависят от уровня 1
        "TranscriptAggregatorExtractor": ["TranscriptChunkEmbedder"],
        "CommentsAggregationExtractor": ["CommentsEmbedder"],
        "QAEmbeddingPairsExtractor": ["TranscriptChunkEmbedder"],
        "EmbeddingPairTopKExtractor": ["TitleEmbedder", "DescriptionEmbedder"],
        "SemanticTopicExtractor": ["TranscriptChunkEmbedder"],
        # Уровень 3: зависят от уровня 2
        "EmbeddingStatsExtractor": ["TranscriptChunkEmbedder"],
        "CosineMetricsExtractor": ["TitleEmbedder", "DescriptionEmbedder", "TranscriptAggregatorExtractor", "CommentsEmbedder"],
        "TitleEmbeddingClusterEntropyExtractor": ["TitleEmbedder"],
        "TitleToHashtagCosineExtractor": ["TitleEmbedder", "HashtagEmbedder"],
        "SemanticClusterExtractor": ["TitleEmbedder", "DescriptionEmbedder", "HashtagEmbedder"],
        "TopKSimilarCorpusTitlesExtractor": ["TitleEmbedder"],
        "EmbeddingShiftIndicatorExtractor": ["TranscriptChunkEmbedder"],
        "EmbeddingSourceIdExtractor": ["TitleEmbedder", "DescriptionEmbedder", "TranscriptAggregatorExtractor"],
    }
    
    # Создаем словарь: extractor_name -> spec
    extractor_map: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    for spec in extractor_specs:
        name, _, _ = spec
        extractor_map[name] = spec
    
    # Строим граф зависимостей только для присутствующих extractors
    present_names = set(extractor_map.keys())
    in_degree: Dict[str, int] = {name: 0 for name in present_names}
    graph: Dict[str, List[str]] = {name: [] for name in present_names}
    
    for name in present_names:
        deps = DEPENDENCIES.get(name, [])
        for dep in deps:
            if dep in present_names:
                in_degree[name] += 1
                graph[dep].append(name)
    
    # Топологическая сортировка (Kahn's algorithm)
    levels: List[List[Tuple[str, str, Dict[str, Any]]]] = []
    queue: List[str] = [name for name, deg in in_degree.items() if deg == 0]
    
    while queue:
        current_level: List[Tuple[str, str, Dict[str, Any]]] = []
        next_queue: List[str] = []
        
        for name in queue:
            if name in extractor_map:
                current_level.append(extractor_map[name])
            
            # Уменьшаем in_degree для зависимых extractors
            for dependent in graph.get(name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_queue.append(dependent)
        
        if current_level:
            levels.append(current_level)
        queue = next_queue
    
    # Добавляем оставшиеся extractors (если есть циклы или неопределенные зависимости)
    remaining = [extractor_map[name] for name in present_names if name not in {n for level in levels for n, _, _ in level}]
    if remaining:
        levels.append(remaining)
    
    return levels


# CPU extractors safe to run concurrently on the same VideoDocument (read-only; no tp_artifacts / doc mutations).
_PARALLEL_SAFE_CPU_EXTRACTORS = frozenset({"LexicalStatsExtractor", "ASRTextProxyExtractor"})


def _effective_extractor_device(device: str, params: Dict[str, Any]) -> str:
    """
    Device used for CUDA memory hooks and run_batch GPU vs CPU grouping.

    devices_config keys like cpu2 still map to tuple device \"cpu\", but extractor_params may set device=cuda
    (e.g. aggregators). Those must be treated as GPU for correct batch routing and cache hygiene.
    """
    p_dev = params.get("device") if isinstance(params, dict) else None
    if p_dev is not None and "cuda" in str(p_dev).lower():
        return "cuda"
    if "gpu" in str(p_dev).lower():
        return "cuda"
    dk = str(device or "").lower()
    if dk in ("cuda", "gpu") or "cuda" in dk:
        return "cuda"
    return "cpu"


def _bundle_specs_lexical_asr_parallel(
    specs: List[Tuple[str, str, Dict[str, Any]]],
    enable: bool,
) -> List[List[Tuple[str, str, Dict[str, Any]]]]:
    """Group consecutive lexical + ASR proxy extractors for parallel CPU execution."""
    if not enable or not specs:
        return [[s] for s in specs]
    out: List[List[Tuple[str, str, Dict[str, Any]]]] = []
    i = 0
    n = len(specs)
    while i < n:
        name, device, params = specs[i]
        if name in _PARALLEL_SAFE_CPU_EXTRACTORS and _effective_extractor_device(str(device), params) == "cpu":
            group: List[Tuple[str, str, Dict[str, Any]]] = []
            while i < n:
                n2, d2, p2 = specs[i]
                if n2 in _PARALLEL_SAFE_CPU_EXTRACTORS and _effective_extractor_device(str(d2), p2) == "cpu":
                    group.append(specs[i])
                    i += 1
                else:
                    break
            out.append(group)
        else:
            out.append([specs[i]])
            i += 1
    return out


class MainProcessor:
    """
    Класс-оркестратор: хранит список экстракторов и последовательно применяет их к документу.

    Инициализация поддерживает конфиг вида:
    {"cpu": "ExtractorName" | ["ExtractorName", ...], "gpu": "..." | ["..."]}
    В зависимости от ключа создаются экземпляры на указанном устройстве.
    """

    def __init__(
        self,
        extractors: List[BaseExtractor] | None = None,
        devices_config: Dict[str, Union[str, List[str]]] | None = None,
        extractor_params: Dict[str, Dict[str, Any]] | None = None,
        strict: bool = True,
        artifacts_dir: str | None = None,
        required_extractors: List[str] | None = None,
        logger: Optional[logging.Logger] = None,
        # Batch processing parameters (Stage 4)
        batch_max_workers: int | None = None,
        batch_enable_gpu_batching: bool = True,
        batch_enable_cpu_parallel: bool = True,
    ) -> None:
        if extractors is not None:
            # Legacy mode: если переданы готовые экстракторы, используем их напрямую
            self._extractor_configs: List[Tuple[str, str, Dict[str, Any]]] | None = None
            self.extractors: List[BaseExtractor] = extractors
            self.required_extractors: List[str] = []
            self.logger = logger or logging.getLogger(__name__)
            return

        self._extractor_params = extractor_params or {}
        self.strict = bool(strict)
        # Per-run artifacts directory (sub-artifacts like *.npy). If set, passed into extractors that accept `artifacts_dir`.
        self.artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self.extractors: List[BaseExtractor] = []  # экстракторы не создаются в __init__
        self.required_extractors: List[str] = [str(x) for x in (required_extractors or [])]
        self.logger = logger or logging.getLogger(__name__)
        
        # Batch processing parameters (Stage 4) - stored for use in run_batch()
        self._batch_max_workers = batch_max_workers
        self._batch_enable_gpu_batching = batch_enable_gpu_batching
        self._batch_enable_cpu_parallel = batch_enable_cpu_parallel

        if devices_config:
            # Сохраняем конфигурацию для ленивой инициализации
            self._extractor_configs = self._build_extractor_configs(devices_config)
        else:
            # по умолчанию — один TitleEmbedder с авто-выбором устройства
            self._extractor_configs = [("TitleEmbedder", "cuda", {})]

    def _build_extractor_configs(self, config: Dict[str, Union[str, List[str]]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Строит список конфигураций экстракторов (name, device, params) без их создания.
        """
        def to_list(x: Union[str, List[str]]) -> List[str]:
            return x if isinstance(x, list) else [x]

        configs: List[Tuple[str, str, Dict[str, Any]]] = []
        for device_key, names in config.items():
            device = "cuda" if device_key.lower() in ("gpu", "cuda") else "cpu"
            for name in to_list(names):
                params = dict(self._extractor_params.get(name, {}))
                configs.append((name, device, params))
        return configs

    def _get_registry_entry(self, name: str) -> Tuple[List[str], str, str] | None:
        """
        Возвращает ([module_paths], class_name, relative_file_path) для зарегистрированного экстрактора.
        Здесь перечисляем доступные экстракторы, но импорт происходит лениво.
        """
        registry: Dict[str, Tuple[List[str], str, str]] = {
            "TagsExtractor": (["src.extractors.tags_extractor.main"], "TagsExtractor", os.path.join("src", "extractors", "tags_extractor", "main.py")),
            "TitleEmbedder": (["src.extractors.title_embedder.main"], "TitleEmbedder", os.path.join("src", "extractors", "title_embedder", "main.py")),
            "DescriptionEmbedder": (["src.extractors.description_embedder.main"], "DescriptionEmbedder", os.path.join("src", "extractors", "description_embedder", "main.py")),
            "TranscriptChunkEmbedder": (["src.extractors.transcript_chunk_embedder.main"], "TranscriptChunkEmbedder", os.path.join("src", "extractors", "transcript_chunk_embedder", "main.py")),
            "TranscriptAggregatorExtractor": (["src.extractors.transcript_aggregator.main"], "TranscriptAggregatorExtractor", os.path.join("src", "extractors", "transcript_aggregator", "main.py")),
            "CommentsEmbedder": (["src.extractors.comments_embedder.main"], "CommentsEmbedder", os.path.join("src", "extractors", "comments_embedder", "main.py")),
            "CommentsAggregationExtractor": (["src.extractors.comments_aggregator.main"], "CommentsAggregationExtractor", os.path.join("src", "extractors", "comments_aggregator", "main.py")),
            "HashtagEmbedder": (["src.extractors.hashtag_embedder.main"], "HashtagEmbedder", os.path.join("src", "extractors", "hashtag_embedder", "main.py")),
            "CosineMetricsExtractor": (["src.extractors.cosine_metrics_extractor.main"], "CosineMetricsExtractor", os.path.join("src", "extractors", "cosine_metrics_extractor", "main.py")),
            "EmbeddingPairTopKExtractor": (["src.extractors.embedding_pair_topk_extractor.main"], "EmbeddingPairTopKExtractor", os.path.join("src", "extractors", "embedding_pair_topk_extractor", "main.py")),
            "SemanticClusterExtractor": (["src.extractors.semantic_cluster_extractor.main"], "SemanticClusterExtractor", os.path.join("src", "extractors", "semantic_cluster_extractor", "main.py")),
            "EmbeddingStatsExtractor": (["src.extractors.embedding_stats_extractor.main"], "EmbeddingStatsExtractor", os.path.join("src", "extractors", "embedding_stats_extractor", "main.py")),
            "TitleToHashtagCosineExtractor": (["src.extractors.title_to_hashtag_cosine_extractor.main"], "TitleToHashtagCosineExtractor", os.path.join("src", "extractors", "title_to_hashtag_cosine_extractor", "main.py")),
            "TopKSimilarCorpusTitlesExtractor": (["src.extractors.topk_similar_titles_extractor.main"], "TopKSimilarCorpusTitlesExtractor", os.path.join("src", "extractors", "topk_similar_titles_extractor", "main.py")),
            "SpeakerTurnEmbeddingsAggregatorExtractor": (["src.extractors.speaker_turn_embeddings_aggregator.main"], "SpeakerTurnEmbeddingsAggregatorExtractor", os.path.join("src", "extractors", "speaker_turn_embeddings_aggregator", "main.py")),
            "QAEmbeddingPairsExtractor": (["src.extractors.qa_embedding_pairs_extractor.main"], "QAEmbeddingPairsExtractor", os.path.join("src", "extractors", "qa_embedding_pairs_extractor", "main.py")),
            "EmbeddingShiftIndicatorExtractor": (["src.extractors.embedding_shift_indicator_extractor.main"], "EmbeddingShiftIndicatorExtractor", os.path.join("src", "extractors", "embedding_shift_indicator_extractor", "main.py")),
            "TitleEmbeddingClusterEntropyExtractor": (["src.extractors.title_embedding_cluster_entropy_extractor.main"], "TitleEmbeddingClusterEntropyExtractor", os.path.join("src", "extractors", "title_embedding_cluster_entropy_extractor", "main.py")),
            "EmbeddingSourceIdExtractor": (["src.extractors.embedding_source_id_extractor.main"], "EmbeddingSourceIdExtractor", os.path.join("src", "extractors", "embedding_source_id_extractor", "main.py")),
            "LexicalStatsExtractor": (["src.extractors.lexico_static_features.main"], "LexicalStatsExtractor", os.path.join("src", "extractors", "lexico_static_features", "main.py")),
            "ASRTextProxyExtractor": (["src.extractors.asr_text_proxy_audio_features.main"], "ASRTextProxyExtractor", os.path.join("src", "extractors", "asr_text_proxy_audio_features", "main.py")),
            "SemanticTopicExtractor": (["src.extractors.semantics_topics_keyphrases.main"], "SemanticTopicExtractor", os.path.join("src", "extractors", "semantics_topics_keyphrases", "main.py")),
        }
        return registry.get(name)

    def _instantiate_extractor_by_name(self, name: str, device: str | None, *, artifacts_dir_override: str | None = None) -> BaseExtractor | None:
        entry = self._get_registry_entry(name)
        if not entry:
            return None
        module_paths, class_name, rel_file = entry
        cls = None
        last_err: Exception | None = None
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                break
            except Exception as e:
                last_err = e
                continue
        if cls is None:
            # Fallback: import by absolute file path
            try:
                # project root = two levels up from this file (src/core → project)
                this_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
                file_path = os.path.join(project_root, rel_file)
                if os.path.exists(file_path):
                    spec = importlib.util.spec_from_file_location(f"dyn_{class_name}", file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        cls = getattr(module, class_name)
            except Exception:
                cls = None
            if cls is None:
                if self.strict:
                    raise RuntimeError(f"TextProcessor failed to load extractor {name}: {last_err}") from last_err
                return None
        # собрать kwargs: из конфигурации + возможно device
        params_local = dict(self._extractor_params.get(name, {}))
        if device is not None:
            params_local.setdefault("device", device)
        artifacts_dir = artifacts_dir_override if artifacts_dir_override is not None else self.artifacts_dir
        if artifacts_dir is not None:
            params_local.setdefault("artifacts_dir", artifacts_dir)
        try:
            return cls(**params_local)
        except TypeError:
            # конструктор без аргументов device
            try:
                return cls()
            except Exception:
                if self.strict:
                    raise
                return None
    
    def _instantiate_extractor_by_name_with_params(
        self,
        name: str,
        device: str | None,
        params: Dict[str, Any],
        *,
        artifacts_dir_override: str | None = None,
    ) -> BaseExtractor | None:
        """
        Вспомогательный метод для создания экстрактора с явными параметрами.
        """
        entry = self._get_registry_entry(name)
        if not entry:
            return None
        module_paths, class_name, rel_file = entry
        cls = None
        last_err: Exception | None = None
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                break
            except Exception as e:
                last_err = e
                continue
        if cls is None:
            # Fallback: import by absolute file path
            try:
                this_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
                file_path = os.path.join(project_root, rel_file)
                if os.path.exists(file_path):
                    spec = importlib.util.spec_from_file_location(f"dyn_{class_name}", file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        cls = getattr(module, class_name)
            except Exception:
                cls = None
            if cls is None:
                if self.strict:
                    raise RuntimeError(f"TextProcessor failed to load extractor {name}: {last_err}") from last_err
                return None
        
        # Get the signature of __init__ to see what parameters it accepts
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        # Используем переданные params + device/artifacts_dir (только если extractor их принимает)
        params_final = dict(params)
        if device is not None and "device" in valid_params:
            params_final.setdefault("device", device)
        artifacts_dir = artifacts_dir_override if artifacts_dir_override is not None else self.artifacts_dir
        if artifacts_dir is not None and "artifacts_dir" in valid_params:
            params_final.setdefault("artifacts_dir", artifacts_dir)
        
        # Filter params to only include valid ones
        params_final = {k: v for k, v in params_final.items() if k in valid_params}
        
        # Log parameters for debugging
        if self.logger:
            self.logger.debug(f"Instantiating {name} with params: {list(params_final.keys())}")
        
        # Try to instantiate with filtered params
        try:
            instance = cls(**params_final)
            # Validate that we got an instance, not a class
            if not isinstance(instance, BaseExtractor):
                error_msg = f"Instantiation of {name} returned {type(instance).__name__}, expected BaseExtractor instance"
                if self.logger:
                    self.logger.error(error_msg)
                if self.strict:
                    raise RuntimeError(error_msg)
                return None
            if self.logger:
                self.logger.debug(f"Successfully instantiated {name} with params")
            return instance
        except (TypeError, RuntimeError) as e:
            # If TypeError or RuntimeError (e.g., missing dependencies), log and re-raise if strict
            if self.logger:
                self.logger.warning(f"Failed to instantiate {name} with params {list(params_final.keys())}: {e}")
            if self.strict:
                raise RuntimeError(f"Failed to instantiate {name}: {e}") from e
            return None

    def _tp_run_lazy_spec_thread(
        self,
        spec: Tuple[str, str, Dict[str, Any]],
        current_doc: VideoDocument,
        artifacts_dir_override: str | None,
        idx: int,
        total_extractors: int,
        parallel_group: bool,
    ) -> Dict[str, Any]:
        """Instantiate + extract one lazy-config spec (thread-safe: no CUDA cache on worker thread)."""
        name, device, params = spec
        ext_name = str(name)
        params = dict(params)
        step_device_effective = _effective_extractor_device(str(device), params)
        tuple_device = str(device)
        t_init_start = time.perf_counter()
        ext: BaseExtractor | None = None
        try:
            if self.logger and params:
                self.logger.debug(f"Creating {ext_name} with params: {list(params.keys())}")
            if self.logger:
                self.logger.debug(
                    f"TextProcessor: [{idx}/{total_extractors}] instantiating {ext_name} (device={tuple_device})..."
                )
            ext = self._instantiate_extractor_by_name_with_params(
                name,
                device=device,
                params=params,
                artifacts_dir_override=artifacts_dir_override,
            )
            init_s = round(time.perf_counter() - t_init_start, 3)
            if self.logger and init_s > 0.1:
                self.logger.debug(
                    f"TextProcessor: [{idx}/{total_extractors}] {ext_name} instantiation took {init_s}s"
                )
            if ext is None:
                return {
                    "idx": idx,
                    "ext_name": ext_name,
                    "ext_class_name": ext_name,
                    "part": {},
                    "init_s": init_s,
                    "step_device": step_device_effective,
                    "tuple_device": tuple_device,
                    "instantiation_failed": True,
                    "not_an_instance": False,
                    "extract_exception": None,
                    "extract_time": 0.0,
                    "parallel_group": parallel_group,
                }
            if not isinstance(ext, BaseExtractor):
                return {
                    "idx": idx,
                    "ext_name": ext_name,
                    "ext_class_name": ext_name,
                    "part": {},
                    "init_s": init_s,
                    "step_device": step_device_effective,
                    "tuple_device": tuple_device,
                    "instantiation_failed": False,
                    "not_an_instance": True,
                    "extract_exception": None,
                    "extract_time": 0.0,
                    "parallel_group": parallel_group,
                }
            tag = " [parallel lexical/asr]" if parallel_group else ""
            if self.logger:
                self.logger.info(
                    f"TextProcessor: [{idx}/{total_extractors}] running {ext_name} (device={tuple_device}){tag}"
                )
            t_extract_start = time.perf_counter()
            try:
                if self.logger:
                    self.logger.debug(
                        f"TextProcessor: [{idx}/{total_extractors}] {ext.__class__.__name__} calling extract()..."
                    )
                part = ext.extract(current_doc) or {}
                extract_exception = None
                if self.logger:
                    self.logger.debug(
                        f"TextProcessor: [{idx}/{total_extractors}] {ext.__class__.__name__} extract() returned: "
                        f"status={part.get('status', 'ok')}, has_result={'result' in part}, "
                        f"has_timings={'timings_s' in part}"
                    )
            except Exception as e:
                part = {}
                extract_exception = e
            extract_time = round(time.perf_counter() - t_extract_start, 3)
            return {
                "idx": idx,
                "ext_name": ext_name,
                "ext_class_name": ext.__class__.__name__,
                "part": part,
                "init_s": init_s,
                "step_device": step_device_effective,
                "tuple_device": tuple_device,
                "instantiation_failed": False,
                "not_an_instance": False,
                "extract_exception": extract_exception,
                "extract_time": extract_time,
                "parallel_group": parallel_group,
            }
        finally:
            try:
                if ext is not None:
                    del ext
            except Exception:
                pass

    def _apply_lazy_spec_thread_result(
        self,
        res: Dict[str, Any],
        *,
        features: Dict[str, Any],
        current_doc: VideoDocument,
        status_by_extractor: Dict[str, str],
        errors_by_extractor: Dict[str, str],
        empty_reasons_by_extractor: Dict[str, str],
        models_used_all: List[Dict[str, Any]],
        features_flat_conflicts: List[str],
        counters: Dict[str, int],
        total_extractors: int,
    ) -> None:
        """Merge one lazy-spec thread result into run state (mirrors MainProcessor.run sequential path)."""
        idx = int(res["idx"])
        ext_name = str(res["ext_name"])
        ext_class_name = str(res["ext_class_name"])
        part: Dict[str, Any] = res.get("part") or {}
        init_s = float(res.get("init_s") or 0.0)
        step_device: str | None = str(res.get("step_device") or "cpu")
        extract_time = float(res.get("extract_time") or 0.0)

        if res.get("instantiation_failed"):
            if ext_name in self.required_extractors:
                raise RuntimeError(f"TextProcessor failed to create required extractor: {ext_name}")
            if self.strict:
                raise RuntimeError(f"TextProcessor failed to create extractor: {ext_name}")
            if self.logger:
                self.logger.warning(
                    f"TextProcessor: [{idx}/{total_extractors}] {ext_name} failed to instantiate (strict={self.strict})"
                )
            status_by_extractor[ext_name] = "error"
            errors_by_extractor[ext_name] = "instantiation_failed"
            counters["failed_count"] = int(counters.get("failed_count", 0)) + 1
            _text_processor_memory_after_step(step_device=step_device, logger=self.logger)
            return

        if res.get("not_an_instance"):
            error_msg = (
                f"TextProcessor: {ext_name} instantiation returned non-BaseExtractor, expected BaseExtractor instance"
            )
            if self.strict:
                raise RuntimeError(error_msg)
            if self.logger:
                self.logger.error(f"TextProcessor: [{idx}/{total_extractors}] {ext_name}: {error_msg}")
            status_by_extractor[ext_name] = "error"
            errors_by_extractor[ext_name] = "not_an_instance"
            counters["failed_count"] = int(counters.get("failed_count", 0)) + 1
            _text_processor_memory_after_step(step_device=step_device, logger=self.logger)
            return

        ex = res.get("extract_exception")
        if ex is not None:
            err_msg = str(ex)
            status_by_extractor[ext_class_name] = "error"
            errors_by_extractor[ext_class_name] = err_msg
            counters["failed_count"] = int(counters.get("failed_count", 0)) + 1
            if ext_class_name in self.required_extractors:
                if self.logger:
                    self.logger.error(
                        f"TextProcessor: [{idx}/{total_extractors}] REQUIRED extractor {ext_class_name} FAILED "
                        f"after {extract_time}s",
                        exc_info=True,
                    )
                raise RuntimeError(
                    f"TextProcessor: required extractor {ext_class_name} raised exception: {err_msg}"
                ) from ex
            if self.logger:
                import traceback

                tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                self.logger.error(
                    f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} raised exception after {extract_time}s:\n"
                    f"  Error: {err_msg}\n"
                    f"  Type: {type(ex).__name__}\n"
                    f"  Traceback:\n{tb_str}",
                    exc_info=False,
                )
            _text_processor_memory_after_step(step_device=step_device, logger=self.logger)
            return

        part_status = part.get("status", "ok")
        if isinstance(part_status, str):
            status_by_extractor[ext_class_name] = part_status
            if part_status == "error":
                counters["failed_count"] = int(counters.get("failed_count", 0)) + 1
                err = part.get("error") or errors_by_extractor.get(ext_class_name) or "unknown_error"
                errors_by_extractor[ext_class_name] = str(err)
                if self.logger:
                    error_details = part.get("error_details") or {}
                    self.logger.error(
                        f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} FAILED after {extract_time}s:\n"
                        f"  Error: {err}\n"
                        f"  Details: {error_details if error_details else 'none'}"
                    )
            elif part_status == "empty":
                counters["empty_count"] = int(counters.get("empty_count", 0)) + 1
                empty_reason = part.get("empty_reason")
                if empty_reason:
                    empty_reasons_by_extractor[ext_class_name] = str(empty_reason)
                if self.logger:
                    self.logger.info(
                        f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} completed "
                        f"(empty: {empty_reason}) ({extract_time}s)"
                    )
            else:
                counters["successful_count"] = int(counters.get("successful_count", 0)) + 1
                if self.logger:
                    result = part.get("result", {})
                    features_flat = result.get("features_flat", {}) if isinstance(result, dict) else {}
                    features_count = len(features_flat) if isinstance(features_flat, dict) else 0
                    models_used = result.get("meta", {}).get("models_used", []) if isinstance(result, dict) else []
                    models_str = (
                        ", ".join([m.get("name", "unknown") for m in models_used if isinstance(m, dict)])
                        if models_used
                        else "none"
                    )
                    details = []
                    if features_count > 0:
                        details.append(f"{features_count} features")
                    if models_str != "none":
                        details.append(f"models: {models_str}")
                    details_str = f" ({', '.join(details)})" if details else ""
                    self.logger.info(
                        f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} completed (ok) "
                        f"({extract_time}s){details_str}"
                    )
        else:
            counters["successful_count"] = int(counters.get("successful_count", 0)) + 1
            if self.logger:
                self.logger.info(
                    f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} completed (ok) ({extract_time}s)"
                )

        if "system" in part and isinstance(part["system"], dict):
            features.setdefault("systems_by_extractor", {})
            features["systems_by_extractor"][ext_class_name] = part["system"]

        if "result" in part and isinstance(part["result"], dict):
            features.setdefault("results_by_extractor", {})
            features["results_by_extractor"][ext_class_name] = part["result"]
            ff = part["result"].get("features_flat") if isinstance(part["result"].get("features_flat"), dict) else None
            if isinstance(ff, dict) and ff:
                features.setdefault("features_flat", {})
                for key in ff.keys():
                    if key in features["features_flat"]:
                        conflict_msg = f"{key} (from {ext_class_name}, previously from another extractor)"
                        features_flat_conflicts.append(conflict_msg)
                        if self.logger:
                            self.logger.warning(f"TextProcessor: features_flat conflict: {conflict_msg}")
                features["features_flat"].update(ff)

            result_meta = part["result"].get("meta") if isinstance(part["result"].get("meta"), dict) else {}
            models_from_result = result_meta.get("models_used") if isinstance(result_meta.get("models_used"), list) else []
            if models_from_result:
                for m in models_from_result:
                    if isinstance(m, dict):
                        models_used_all.append(dict(m))

            muts = part.get("mutations") if isinstance(part.get("mutations"), dict) else None
            if muts:
                cleaned = muts.get("cleaned_texts") if isinstance(muts.get("cleaned_texts"), dict) else None
                if cleaned:
                    try:
                        if "title" in cleaned:
                            setattr(current_doc, "title", cleaned["title"])
                        if "description" in cleaned:
                            setattr(current_doc, "description", cleaned["description"])
                    except Exception:
                        pass
                tags = muts.get("hashtags") if isinstance(muts.get("hashtags"), list) else None
                if tags is not None:
                    try:
                        setattr(current_doc, "hashtags", tags)
                    except Exception:
                        pass

        if "timings_s" in part and isinstance(part["timings_s"], dict):
            features.setdefault("timings_by_extractor", {})
            tdict = dict(part["timings_s"])
            tdict.setdefault("init", init_s)
            features["timings_by_extractor"][ext_class_name] = tdict
        else:
            features.setdefault("timings_by_extractor", {})
            features["timings_by_extractor"][ext_class_name] = {"init": init_s}

        if part.get("device"):
            features["device"] = part.get("device")
        if part.get("version"):
            features["version"] = part.get("version")

        if part.get("error"):
            features["error"] = part.get("error")

        _text_processor_memory_after_step(step_device=step_device, logger=self.logger)

    def run(self, document: VideoDocument, *, artifacts_dir_override: str | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        features: Dict[str, Any] = {}
        
        # Track orchestrator-level state
        status_by_extractor: Dict[str, str] = {}
        errors_by_extractor: Dict[str, str] = {}
        empty_reasons_by_extractor: Dict[str, str] = {}
        models_used_all: List[Dict[str, Any]] = []
        features_flat_conflicts: List[str] = []
        successful_count = 0
        failed_count = 0
        empty_count = 0
        
        # Определяем какой список использовать: готовые экстракторы или конфигурации
        if self._extractor_configs is not None:
            # Ленивая инициализация: создаём экстрактор → используем → очищаем → удаляем
            extractor_specs = self._extractor_configs
        else:
            # Legacy mode: используем уже созданные экстракторы
            extractor_specs = [(ext.__class__.__name__, getattr(ext, "device", "cpu"), {}) for ext in self.extractors]
        
        total_extractors = len(extractor_specs)
        
        if self.logger and total_extractors > 0:
            self.logger.info(f"TextProcessor: starting {total_extractors} extractor(s)")
        
        # mutable document to allow earlier extractors to influence later ones (e.g., cleaned texts, hashtags)
        current_doc = document
        if self._extractor_configs is not None:
            cnt = {"successful_count": 0, "failed_count": 0, "empty_count": 0}
            bundles = _bundle_specs_lexical_asr_parallel(
                extractor_specs, self._batch_enable_cpu_parallel
            )
            idx_acc = 0
            for bundle in bundles:
                if len(bundle) > 1:
                    with ThreadPoolExecutor(max_workers=len(bundle)) as ex:
                        futs = [
                            ex.submit(
                                self._tp_run_lazy_spec_thread,
                                bundle[i],
                                current_doc,
                                artifacts_dir_override,
                                idx_acc + i + 1,
                                total_extractors,
                                True,
                            )
                            for i in range(len(bundle))
                        ]
                        results = [f.result() for f in futs]
                else:
                    results = [
                        self._tp_run_lazy_spec_thread(
                            bundle[0],
                            current_doc,
                            artifacts_dir_override,
                            idx_acc + 1,
                            total_extractors,
                            False,
                        )
                    ]
                idx_acc += len(bundle)
                for res in results:
                    self._apply_lazy_spec_thread_result(
                        res,
                        features=features,
                        current_doc=current_doc,
                        status_by_extractor=status_by_extractor,
                        errors_by_extractor=errors_by_extractor,
                        empty_reasons_by_extractor=empty_reasons_by_extractor,
                        models_used_all=models_used_all,
                        features_flat_conflicts=features_flat_conflicts,
                        counters=cnt,
                        total_extractors=total_extractors,
                    )
            successful_count = cnt["successful_count"]
            failed_count = cnt["failed_count"]
            empty_count = cnt["empty_count"]
        else:
            for idx, spec in enumerate(extractor_specs, 1):
                ext = None
                ext_name = None
                step_device: str | None = None
                try:
                    # Legacy mode: находим экстрактор по имени
                    ext_name = spec[0] if isinstance(spec, tuple) else spec.__class__.__name__
                    if isinstance(spec, tuple) and len(spec) > 1:
                        step_device = str(spec[1])
                    ext = next((e for e in self.extractors if e.__class__.__name__ == ext_name), None)
                    if ext is None:
                        continue
                    init_s = 0.0
                    if self.logger:
                        device_attr = getattr(ext, "device", "cpu")
                        self.logger.info(f"TextProcessor: [{idx}/{total_extractors}] running {ext_name} (device={device_attr})")

                    # Выполняем извлечение с обработкой ошибок
                    part: Dict[str, Any] = {}
                    t_extract_start = time.perf_counter()
                    try:
                        if self.logger:
                            self.logger.debug(f"TextProcessor: [{idx}/{total_extractors}] {ext.__class__.__name__} calling extract()...")
                        part = ext.extract(current_doc) or {}
                        if self.logger:
                            self.logger.debug(f"TextProcessor: [{idx}/{total_extractors}] {ext.__class__.__name__} extract() returned: status={part.get('status', 'ok')}, has_result={'result' in part}, has_timings={'timings_s' in part}")
                    except Exception as e:
                        ext_class_name = ext.__class__.__name__
                        err_msg = str(e)
                        extract_time = round(time.perf_counter() - t_extract_start, 3)
                        status_by_extractor[ext_class_name] = "error"
                        errors_by_extractor[ext_class_name] = err_msg
                        failed_count += 1
                        if ext_class_name in self.required_extractors:
                            if self.logger:
                                self.logger.error(f"TextProcessor: [{idx}/{total_extractors}] REQUIRED extractor {ext_class_name} FAILED after {extract_time}s", exc_info=True)
                            raise RuntimeError(f"TextProcessor: required extractor {ext_class_name} raised exception: {err_msg}") from e
                        if self.logger:
                            import traceback
                            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                            self.logger.error(
                                f"TextProcessor: [{idx}/{total_extractors}] {ext_class_name} raised exception after {extract_time}s:\n"
                                f"  Error: {err_msg}\n"
                                f"  Type: {type(e).__name__}\n"
                                f"  Traceback:\n{tb_str}",
                                exc_info=False  # уже вывели traceback вручную
                            )
                        continue

                    # Extract status from part (if available)
                    extract_time = round(time.perf_counter() - t_extract_start, 3)
                    part_status = part.get("status", "ok")
                    ext_name = ext.__class__.__name__

                    if isinstance(part_status, str):
                        status_by_extractor[ext_name] = part_status
                        if part_status == "error":
                            failed_count += 1
                            err = part.get("error") or errors_by_extractor.get(ext_name) or "unknown_error"
                            errors_by_extractor[ext_name] = str(err)
                            if self.logger:
                                error_details = part.get("error_details") or {}
                                self.logger.error(
                                    f"TextProcessor: [{idx}/{total_extractors}] {ext_name} FAILED after {extract_time}s:\n"
                                    f"  Error: {err}\n"
                                    f"  Details: {error_details if error_details else 'none'}"
                                )
                        elif part_status == "empty":
                            empty_count += 1
                            empty_reason = part.get("empty_reason")
                            if empty_reason:
                                empty_reasons_by_extractor[ext_name] = str(empty_reason)
                            if self.logger:
                                self.logger.info(f"TextProcessor: [{idx}/{total_extractors}] {ext_name} completed (empty: {empty_reason}) ({extract_time}s)")
                        else:
                            successful_count += 1
                            # Логируем детали успешного результата
                            if self.logger:
                                result = part.get("result", {})
                                features_flat = result.get("features_flat", {}) if isinstance(result, dict) else {}
                                features_count = len(features_flat) if isinstance(features_flat, dict) else 0
                                models_used = result.get("meta", {}).get("models_used", []) if isinstance(result, dict) else []
                                models_str = ", ".join([m.get("name", "unknown") for m in models_used if isinstance(m, dict)]) if models_used else "none"
                                details = []
                                if features_count > 0:
                                    details.append(f"{features_count} features")
                                if models_str != "none":
                                    details.append(f"models: {models_str}")
                                details_str = f" ({', '.join(details)})" if details else ""
                                self.logger.info(f"TextProcessor: [{idx}/{total_extractors}] {ext_name} completed (ok) ({extract_time}s){details_str}")
                    else:
                        successful_count += 1
                        if self.logger:
                            self.logger.info(f"TextProcessor: [{idx}/{total_extractors}] {ext_name} completed (ok) ({extract_time}s)")

                    # keep per-extractor system snapshots
                    if "system" in part and isinstance(part["system"], dict):
                        features.setdefault("systems_by_extractor", {})
                        features["systems_by_extractor"][ext.__class__.__name__] = part["system"]

                    # keep per-extractor results (separated by extractor)
                    if "result" in part and isinstance(part["result"], dict):
                        features.setdefault("results_by_extractor", {})
                        features["results_by_extractor"][ext.__class__.__name__] = part["result"]
                        # Optional: merge flat scalar features into a single stable dict (preferred for NPZ export).
                        ff = part["result"].get("features_flat") if isinstance(part["result"].get("features_flat"), dict) else None
                        if isinstance(ff, dict) and ff:
                            features.setdefault("features_flat", {})
                            # Detect conflicts (same key from different extractors)
                            for key in ff.keys():
                                if key in features["features_flat"]:
                                    conflict_msg = f"{key} (from {ext.__class__.__name__}, previously from another extractor)"
                                    features_flat_conflicts.append(conflict_msg)
                                    self.logger.warning(f"TextProcessor: features_flat conflict: {conflict_msg}")
                            # last-wins; keys should be unique across extractors by convention
                            features["features_flat"].update(ff)

                        # Collect models_used from extractor result
                        result_meta = part["result"].get("meta") if isinstance(part["result"].get("meta"), dict) else {}
                        models_from_result = result_meta.get("models_used") if isinstance(result_meta.get("models_used"), list) else []
                        if models_from_result:
                            for m in models_from_result:
                                if isinstance(m, dict):
                                    models_used_all.append(dict(m))

                        # propagate mutations to subsequent extractors (privacy-safe: mutations are not persisted in `result` by default)
                        muts = part.get("mutations") if isinstance(part.get("mutations"), dict) else None
                        if muts:
                            cleaned = muts.get("cleaned_texts") if isinstance(muts.get("cleaned_texts"), dict) else None
                            if cleaned:
                                try:
                                    if "title" in cleaned:
                                        setattr(current_doc, "title", cleaned["title"])
                                    if "description" in cleaned:
                                        setattr(current_doc, "description", cleaned["description"])
                                except Exception:
                                    pass
                            tags = muts.get("hashtags") if isinstance(muts.get("hashtags"), list) else None
                            if tags is not None:
                                try:
                                    setattr(current_doc, "hashtags", tags)
                                except Exception:
                                    pass

                    # keep per-extractor timings (seconds)
                    if "timings_s" in part and isinstance(part["timings_s"], dict):
                        features.setdefault("timings_by_extractor", {})
                        tdict = dict(part["timings_s"])  # copy
                        # добавить время инициализации экстрактора
                        tdict.setdefault("init", init_s)
                        features["timings_by_extractor"][ext.__class__.__name__] = tdict
                    else:
                        # нет таймингов от экстрактора — всё равно зафиксируем init
                        features.setdefault("timings_by_extractor", {})
                        features["timings_by_extractor"][ext.__class__.__name__] = {"init": init_s}

                    # device/version: keep last non-empty
                    if part.get("device"):
                        features["device"] = part.get("device")
                    if part.get("version"):
                        features["version"] = part.get("version")

                    # error: collect last non-empty (for backward compat); detailed errors can be found per extractor inside result if needed
                    if part.get("error"):
                        features["error"] = part.get("error")

                finally:
                    # Явно удаляем ссылку; затем gc / CUDA cache и опционально сброс глобального ModelManager.
                    if ext is not None:
                        try:
                            del ext
                        except Exception:
                            pass
                    _text_processor_memory_after_step(step_device=step_device, logger=self.logger)

        # Check required extractors
        for req_name in self.required_extractors:
            if req_name not in status_by_extractor or status_by_extractor[req_name] == "error":
                raise RuntimeError(f"TextProcessor: required extractor {req_name} failed or was not executed")

        # Aggregate orchestrator status
        orchestrator_status = "ok"
        orchestrator_empty_reason: Optional[str] = None
        if failed_count > 0:
            # If any required extractor failed, status is error (already raised above)
            # If only optional extractors failed, check if any succeeded
            if successful_count == 0 and empty_count == total_extractors:
                orchestrator_status = "empty"
                orchestrator_empty_reason = "all_extractors_empty"
            elif successful_count == 0:
                orchestrator_status = "error"
        elif successful_count == 0 and empty_count == total_extractors:
            orchestrator_status = "empty"
            orchestrator_empty_reason = "all_extractors_empty"
        elif empty_count == total_extractors and successful_count == 0:
            orchestrator_status = "empty"
            orchestrator_empty_reason = "all_extractors_empty"

        # Add orchestrator-level metadata
        features["status"] = orchestrator_status
        if orchestrator_empty_reason:
            features["empty_reason"] = orchestrator_empty_reason
        features["status_by_extractor"] = status_by_extractor
        if errors_by_extractor:
            features["errors_by_extractor"] = errors_by_extractor
        if empty_reasons_by_extractor:
            features["empty_reasons_by_extractor"] = empty_reasons_by_extractor
        if models_used_all:
            features["models_used"] = models_used_all

        # Add orchestrator metrics to features_flat
        total_duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        features.setdefault("features_flat", {})
        features["features_flat"]["tp_orchestrator_total_extractors"] = float(total_extractors)
        features["features_flat"]["tp_orchestrator_successful_count"] = float(successful_count)
        features["features_flat"]["tp_orchestrator_failed_count"] = float(failed_count)
        features["features_flat"]["tp_orchestrator_empty_count"] = float(empty_count)
        features["features_flat"]["tp_orchestrator_total_duration_ms"] = float(total_duration_ms)
        features["features_flat"]["tp_orchestrator_feature_conflicts_count"] = float(len(features_flat_conflicts))
        if features_flat_conflicts:
            features["features_flat_conflicts"] = features_flat_conflicts

        _text_processor_memory_run_end(self.logger)
        return features

    def run_batch(
        self,
        documents: List[VideoDocument],
        max_workers: int | None = None,
        enable_gpu_batching: bool | None = None,
        enable_cpu_parallel: bool | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Обработать список документов с оптимизациями.

        Args:
            documents: Список документов для обработки
            max_workers: Количество воркеров для CPU extractors (None = auto, обычно os.cpu_count())
                Если None, используется значение из __init__ (batch_max_workers).
            enable_gpu_batching: Использовать extract_batch() для GPU extractors с supports_batch=True
                Если None, используется значение из __init__ (batch_enable_gpu_batching).
            enable_cpu_parallel: Распараллеливать CPU extractors по документам
                Если None, используется значение из __init__ (batch_enable_cpu_parallel).

        Stage-4: GPU batching + CPU parallelism для независимых extractors.
        
        Для обратной совместимости: если оптимизации отключены, использует последовательный run() для каждого документа.
        """
        docs = list(documents or [])
        if not docs:
            return []
        
        # Use instance defaults if parameters not provided
        if max_workers is None:
            max_workers = self._batch_max_workers
        if enable_gpu_batching is None:
            enable_gpu_batching = self._batch_enable_gpu_batching
        if enable_cpu_parallel is None:
            enable_cpu_parallel = self._batch_enable_cpu_parallel

        # Prepare per-doc artifacts directories
        base_artifacts_dir: Path | None = None
        try:
            if self.artifacts_dir:
                base_artifacts_dir = Path(self.artifacts_dir).expanduser().resolve()
                base_artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_artifacts_dir = None

        # Stage-1: Setup per-doc isolation
        for i, doc in enumerate(docs):
            try:
                setattr(doc, "tp_artifacts", {})
            except Exception:
                pass
            art_override = None
            if base_artifacts_dir is not None:
                per_doc_dir = base_artifacts_dir / f"doc_{i:05d}"
                per_doc_dir.mkdir(parents=True, exist_ok=True)
                art_override = str(per_doc_dir)
            try:
                setattr(doc, "_tp_artifacts_dir", art_override)
            except Exception:
                pass

        # Fallback: if optimizations disabled, use sequential run() for each document
        if not enable_gpu_batching and not enable_cpu_parallel:
            results: List[Dict[str, Any]] = []
            for i, doc in enumerate(docs):
                try:
                    art_override = getattr(doc, "_tp_artifacts_dir", None)
                    results.append(self.run(doc, artifacts_dir_override=art_override) or {})
                except Exception as e:
                    msg = str(e)
                    if self.logger:
                        self.logger.error(f"TextProcessor: run_batch doc[{i}] failed: {msg}", exc_info=True)
                    results.append({"status": "error", "error": msg})
            return results

        # Get extractor configs
        if self._extractor_configs is not None:
            extractor_specs = self._extractor_configs
        else:
            extractor_specs = [(ext.__class__.__name__, getattr(ext, "device", "cpu"), {}) for ext in self.extractors]

        if not extractor_specs:
            # No extractors configured, return empty results
            return [{"status": "empty", "empty_reason": "no_extractors_configured"} for _ in docs]

        # Build dependency levels (topological sort)
        levels = _build_dependency_levels(extractor_specs)
        
        if self.logger:
            self.logger.debug(f"TextProcessor: run_batch grouped {len(extractor_specs)} extractors into {len(levels)} dependency levels")

        # Initialize results: one dict per document
        results: List[Dict[str, Any]] = [{} for _ in docs]
        
        # Process each level sequentially (respecting dependencies)
        for level_idx, level_specs in enumerate(levels):
            if self.logger:
                self.logger.debug(f"TextProcessor: run_batch processing level {level_idx + 1}/{len(levels)} with {len(level_specs)} extractors")
            
            # Group extractors in this level by device and batch support
            gpu_batch_specs: List[Tuple[str, str, Dict[str, Any]]] = []
            gpu_legacy_specs: List[Tuple[str, str, Dict[str, Any]]] = []
            cpu_specs: List[Tuple[str, str, Dict[str, Any]]] = []
            
            for spec in level_specs:
                name, device, params = spec
                eff = _effective_extractor_device(str(device), params)
                if eff == "cuda":
                    if enable_gpu_batching:
                        # Check if extractor supports batch
                        ext_probe: BaseExtractor | None = None
                        try:
                            ext_probe = self._instantiate_extractor_by_name_with_params(
                                name, device, params, artifacts_dir_override=None
                            )
                            if ext_probe and hasattr(ext_probe, "supports_batch") and ext_probe.supports_batch:
                                gpu_batch_specs.append(spec)
                            else:
                                gpu_legacy_specs.append(spec)
                        except Exception:
                            gpu_legacy_specs.append(spec)
                        finally:
                            try:
                                if ext_probe is not None:
                                    del ext_probe
                            except Exception:
                                pass
                            _text_processor_memory_after_step(
                                step_device=eff, logger=self.logger, sync_model_manager=True
                            )
                    else:
                        gpu_legacy_specs.append(spec)
                else:
                    cpu_specs.append(spec)
            
            # Process GPU batch extractors (all documents at once)
            for ext_name, device, params in gpu_batch_specs:
                ext = None
                try:
                    ext = self._instantiate_extractor_by_name_with_params(ext_name, device, params, artifacts_dir_override=None)
                    if ext is None:
                        continue
                    
                    batch_results = ext.extract_batch(docs)
                    # Merge results per document
                    for doc_idx, batch_result in enumerate(batch_results):
                        if doc_idx < len(results) and isinstance(batch_result, dict):
                            # Merge features_flat
                            if "result" in batch_result and "features_flat" in batch_result["result"]:
                                results[doc_idx].setdefault("features_flat", {}).update(batch_result["result"]["features_flat"])
                            # Merge timings
                            if "timings_s" in batch_result:
                                results[doc_idx].setdefault("timings_by_extractor", {})[ext_name] = batch_result["timings_s"]
                            # Track errors
                            if "error" in batch_result and batch_result["error"]:
                                results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = batch_result["error"]
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"TextProcessor: GPU batch extractor {ext_name} failed: {e}", exc_info=True)
                    for doc_idx in range(len(results)):
                        results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = str(e)
                finally:
                    try:
                        if ext is not None:
                            del ext
                    except Exception:
                        pass
                    _text_processor_memory_after_step(
                        step_device=_effective_extractor_device(str(device), params),
                        logger=self.logger,
                        sync_model_manager=True,
                    )
            
            # Process GPU legacy extractors (sequential per document)
            for ext_name, device, params in gpu_legacy_specs:
                for doc_idx, doc in enumerate(docs):
                    ext = None
                    try:
                        ext = self._instantiate_extractor_by_name_with_params(
                            ext_name, device, params,
                            artifacts_dir_override=getattr(doc, "_tp_artifacts_dir", None)
                        )
                        if ext is None:
                            continue
                        part = ext.extract(doc) or {}
                        if isinstance(part, dict):
                            if "result" in part and "features_flat" in part["result"]:
                                results[doc_idx].setdefault("features_flat", {}).update(part["result"]["features_flat"])
                            if "timings_s" in part:
                                results[doc_idx].setdefault("timings_by_extractor", {})[ext_name] = part["timings_s"]
                            if "error" in part and part["error"]:
                                results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = part["error"]
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"TextProcessor: GPU extractor {ext_name} failed for doc[{doc_idx}]: {e}")
                        results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = str(e)
                    finally:
                        try:
                            if ext is not None:
                                del ext
                        except Exception:
                            pass
                        _text_processor_memory_after_step(
                            step_device=str(device), logger=self.logger, sync_model_manager=True
                        )
            
            # Process CPU extractors in parallel (if enabled)
            if cpu_specs:
                if enable_cpu_parallel:
                    max_w = max_workers if max_workers is not None else min(len(docs), os.cpu_count() or 4)
                    
                    def process_doc_with_cpu_extractors(doc_idx: int, doc: VideoDocument) -> Tuple[int, Dict[str, Any]]:
                        """Process one document with all CPU extractors in this level."""
                        doc_result: Dict[str, Any] = {}
                        for ext_name, device, params in cpu_specs:
                            ext = None
                            try:
                                ext = self._instantiate_extractor_by_name_with_params(
                                    ext_name, device, params,
                                    artifacts_dir_override=getattr(doc, "_tp_artifacts_dir", None)
                                )
                                if ext is None:
                                    continue
                                part = ext.extract(doc) or {}
                                if isinstance(part, dict):
                                    if "result" in part and "features_flat" in part["result"]:
                                        doc_result.setdefault("features_flat", {}).update(part["result"]["features_flat"])
                                    if "timings_s" in part:
                                        doc_result.setdefault("timings_by_extractor", {})[ext_name] = part["timings_s"]
                                    if "error" in part and part["error"]:
                                        doc_result.setdefault("errors_by_extractor", {})[ext_name] = part["error"]
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"TextProcessor: CPU extractor {ext_name} failed for doc[{doc_idx}]: {e}")
                                doc_result.setdefault("errors_by_extractor", {})[ext_name] = str(e)
                            finally:
                                try:
                                    del ext
                                    import gc
                                    gc.collect()
                                except Exception:
                                    pass
                        return (doc_idx, doc_result)
                    
                    # Process documents in parallel
                    with ThreadPoolExecutor(max_workers=max_w) as executor:
                        futures = {
                            executor.submit(process_doc_with_cpu_extractors, i, doc): i
                            for i, doc in enumerate(docs)
                        }
                        for future in as_completed(futures):
                            try:
                                doc_idx, doc_result = future.result()
                                if doc_idx < len(results):
                                    if "features_flat" in doc_result:
                                        results[doc_idx].setdefault("features_flat", {}).update(doc_result["features_flat"])
                                    if "timings_by_extractor" in doc_result:
                                        results[doc_idx].setdefault("timings_by_extractor", {}).update(doc_result["timings_by_extractor"])
                                    if "errors_by_extractor" in doc_result:
                                        results[doc_idx].setdefault("errors_by_extractor", {}).update(doc_result["errors_by_extractor"])
                            except Exception as e:
                                doc_idx = futures.get(future, -1)
                                if self.logger:
                                    self.logger.error(f"TextProcessor: Parallel CPU processing failed for doc[{doc_idx}]: {e}", exc_info=True)
                                if doc_idx >= 0 and doc_idx < len(results):
                                    results[doc_idx].setdefault("errors_by_extractor", {})["parallel_cpu"] = str(e)
                else:
                    # Sequential processing for CPU extractors
                    for doc_idx, doc in enumerate(docs):
                        for ext_name, device, params in cpu_specs:
                            ext = None
                            try:
                                ext = self._instantiate_extractor_by_name_with_params(
                                    ext_name, device, params,
                                    artifacts_dir_override=getattr(doc, "_tp_artifacts_dir", None)
                                )
                                if ext is None:
                                    continue
                                part = ext.extract(doc) or {}
                                if isinstance(part, dict):
                                    if "result" in part and "features_flat" in part["result"]:
                                        results[doc_idx].setdefault("features_flat", {}).update(part["result"]["features_flat"])
                                    if "timings_s" in part:
                                        results[doc_idx].setdefault("timings_by_extractor", {})[ext_name] = part["timings_s"]
                                    if "error" in part and part["error"]:
                                        results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = part["error"]
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"TextProcessor: CPU extractor {ext_name} failed for doc[{doc_idx}]: {e}")
                                results[doc_idx].setdefault("errors_by_extractor", {})[ext_name] = str(e)
                            finally:
                                try:
                                    if ext is not None:
                                        del ext
                                except Exception:
                                    pass
                                _text_processor_memory_after_step(
                                    step_device=_effective_extractor_device(str(device), params),
                                    logger=self.logger,
                                    sync_model_manager=True,
                                )
        
        _text_processor_memory_run_end(self.logger)

        # Finalize results: ensure status, error, empty_reason fields (matching run() output structure)
        for i, res in enumerate(results):
            if not res:
                res["status"] = "error"
                res["error"] = "no_extractors_executed"
            elif "status" not in res:
                # Determine status from errors
                errors = res.get("errors_by_extractor", {})
                if errors:
                    res["status"] = "error"
                    # Aggregate error message
                    error_msgs = [f"{k}: {v}" for k, v in errors.items()]
                    res["error"] = "; ".join(error_msgs)
                else:
                    res["status"] = "ok"
        
        return results


def load_document_from_json(path: str) -> VideoDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return video_document_from_dict(data)
