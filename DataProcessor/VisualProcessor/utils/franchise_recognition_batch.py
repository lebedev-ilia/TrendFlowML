"""
Batch processing utilities for franchise_recognition component.

Stage 3: GPU batching для franchise_recognition с гибридным подходом:
- Сбор кадров из всех видео (из FrameManager)
- Batch search через Embedding Service
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz
from utils.frame_manager import FrameManager

# Import franchise_recognition functions
_franchise_recognition_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "franchise_recognition"
sys.path.insert(0, str(_franchise_recognition_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.franchise_recognition_batch")

# Import from franchise_recognition/main.py
_franchise_recognition_main = _franchise_recognition_path / "main.py"
if _franchise_recognition_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("franchise_recognition_main", str(_franchise_recognition_main))
    if spec and spec.loader:
        franchise_recognition_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(franchise_recognition_module)
        
        # Import functions
        _load_npz = getattr(franchise_recognition_module, "_load_npz", None)
        _require_frame_indices = getattr(franchise_recognition_module, "_require_frame_indices", None)
        _norm_text = getattr(franchise_recognition_module, "_norm_text", None)
        _load_ocr_npz = getattr(franchise_recognition_module, "_load_ocr_npz", None)
        _auto_find_ocr_npz = getattr(franchise_recognition_module, "_auto_find_ocr_npz", None)
        NAME = getattr(franchise_recognition_module, "NAME", "franchise_recognition")
        VERSION = getattr(franchise_recognition_module, "VERSION", "0.1")
        SCHEMA_VERSION = getattr(franchise_recognition_module, "SCHEMA_VERSION", "franchise_recognition_npz_v2")
        ARTIFACT_FILENAME = getattr(franchise_recognition_module, "ARTIFACT_FILENAME", "franchise_recognition.npz")
        FRANCHISE_CATEGORY = getattr(franchise_recognition_module, "FRANCHISE_CATEGORY", "franchise")
        TOP_K = getattr(franchise_recognition_module, "TOP_K", 5)
        
        # Import EmbeddingServiceClient
        try:
            from utils.embedding_service_client import EmbeddingServiceClient
        except ImportError:
            try:
                from embedding_service_client import EmbeddingServiceClient
            except ImportError:
                # Try utils directory first, then root
                _embedding_client_path = _franchise_recognition_path / "utils" / "embedding_service_client.py"
                if not _embedding_client_path.exists():
                    _embedding_client_path = _franchise_recognition_path / "embedding_service_client.py"
                
                if _embedding_client_path.exists():
                    spec_client = importlib.util.spec_from_file_location("embedding_service_client", str(_embedding_client_path))
                    if spec_client and spec_client.loader:
                        embedding_client_module = importlib.util.module_from_spec(spec_client)
                        spec_client.loader.exec_module(embedding_client_module)
                        EmbeddingServiceClient = embedding_client_module.EmbeddingServiceClient
                    else:
                        raise ImportError("Failed to load EmbeddingServiceClient")
                else:
                    raise ImportError(f"embedding_service_client.py not found. Checked: {_franchise_recognition_path / 'utils' / 'embedding_service_client.py'}, {_franchise_recognition_path / 'embedding_service_client.py'}")
    else:
        raise ImportError("Failed to load franchise_recognition module")
else:
    raise ImportError(f"franchise_recognition/main.py not found at {_franchise_recognition_main}")


def _atomic_save_npz(out_path: str, **kwargs) -> None:
    """Atomic NPZ save."""
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".",
        suffix=".npz",
        dir=out_dir,
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def process_franchise_recognition_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для franchise_recognition с гибридным подходом.
    
    Stage 3: GPU batching для franchise_recognition.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация franchise_recognition
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"{NAME} | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch})"
    )
    
    start_time = time.perf_counter()
    
    # Параметры конфигурации
    embedding_service_url = config.get("embedding_service_url") or os.environ.get("EMBEDDING_SERVICE_URL")
    topk = int(config.get("topk", TOP_K))
    similarity_threshold = float(config.get("similarity_threshold", 0.0))
    threshold_global = float(config.get("threshold_global", 0.23))
    ocr_npz_str = config.get("ocr_npz") or None
    # Обрабатываем "null" строку как None
    ocr_npz = None if (ocr_npz_str is None or str(ocr_npz_str).lower() == "null" or str(ocr_npz_str).strip() == "") else str(ocr_npz_str)
    ocr_min_confidence = float(config.get("ocr_min_confidence", 0.4))
    ocr_max_events = int(config.get("ocr_max_events", 5000))
    use_ocr_filtering = bool(config.get("use_ocr_filtering", False))
    max_franchises_for_full_search = int(config.get("max_full_labels", 500))
    batch_size = int(config.get("batch_size", 16))
    
    if topk != 5:
        raise RuntimeError(f"{NAME} | batch | topk must be 5 (contract), got {topk}")
    
    # Initialize Embedding Service client (fail-fast if unavailable)
    try:
        embedding_client = EmbeddingServiceClient(base_url=embedding_service_url)
        embedding_client._ensure_url()
    except Exception as e:
        raise RuntimeError(
            f"{NAME} | batch | Embedding Service unavailable (fail-fast): {e}. "
            "Ensure Embedding Service is running and accessible."
        )
    
    # Этап 1: Сбор всех кадров из всех видео с привязкой к видео
    frames_by_video: List[Dict[str, Any]] = []
    all_frames: List[Tuple[int, int, np.ndarray]] = []  # (video_idx, frame_idx, frame_image)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Получаем frame_indices
            try:
                frame_indices = _require_frame_indices(metadata)
            except Exception as e:
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": str(e),
                })
                continue
            
            if not frame_indices:
                logger.warning(f"{NAME} | batch | video {video_ctx.video_id} has no frame_indices")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "status": "empty",
                })
                continue
            
            # Получаем timestamps
            uts = metadata.get("union_timestamps_sec")
            if uts is None:
                raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            if fi_np.size == 0:
                raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
            if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
                raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Загружаем кадры через FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=int(metadata.get("chunk_size", 32)),
                cache_size=int(metadata.get("cache_size", 2)),
            )
            
            # Сохраняем кадры в общий батч
            video_frame_start_idx = len(all_frames)
            for frame_idx_global in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx_global)
                    all_frames.append((video_idx, frame_idx_global, frame))
                except Exception as e:
                    logger.warning(f"{NAME} | batch | video {video_ctx.video_id} failed to load frame {frame_idx_global}: {e}")
                    # Продолжаем с остальными кадрами
            video_frame_end_idx = len(all_frames)
            
            # Загружаем метаданные из core_clip для provenance
            core_clip_path = os.path.join(str(video_ctx.rs_path), "core_clip", "embeddings.npz")
            upstream_models_used: List[Dict[str, Any]] = []
            upstream_model_signature: Any = None
            if os.path.isfile(core_clip_path):
                try:
                    clip_npz = _load_npz(core_clip_path)
                    clip_meta = clip_npz.get("meta")
                    if isinstance(clip_meta, dict):
                        if isinstance(clip_meta.get("models_used"), list):
                            upstream_models_used = clip_meta.get("models_used") or []
                        upstream_model_signature = clip_meta.get("model_signature")
                except Exception as e:
                    logger.warning(f"{NAME} | batch | video {video_ctx.video_id} failed to load core_clip meta: {e}")
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "times_s": times_s,
                "frame_start_idx": video_frame_start_idx,
                "frame_end_idx": video_frame_end_idx,
                "upstream_models_used": upstream_models_used,
                "upstream_model_signature": upstream_model_signature,
                "metadata": metadata,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"{NAME} | batch | video {video_ctx.video_id} failed to prepare: {e}")
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_frames:
        logger.error(f"{NAME} | batch | no frames collected from any video")
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no frames collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"{NAME} | batch | collected {len(all_frames)} frames from {len(frames_by_video)} videos")
    
    # Этап 2: Batch search через Embedding Service
    logger.info(f"{NAME} | batch | searching franchises for {len(all_frames)} frames")
    
    # Используем batch search если доступен, иначе fallback к индивидуальным запросам
    all_results: List[List[Dict[str, Any]]] = []
    
    # Группируем кадры в батчи
    if max_frames_per_batch:
        batches = []
        for i in range(0, len(all_frames), max_frames_per_batch):
            batches.append(all_frames[i:i + max_frames_per_batch])
    else:
        batches = [all_frames]
    
    for batch_idx, batch in enumerate(batches):
        batch_images = [frame for _, _, frame in batch]
        
        try:
            # Используем search_batch если доступен
            batch_results = embedding_client.search_batch(
                category=FRANCHISE_CATEGORY,
                images=batch_images,
                top_k=topk,
                similarity_threshold=similarity_threshold,
                max_retries=3,
                retry_delay=1.0,
            )
            all_results.extend(batch_results)
        except Exception as e:
            logger.warning(f"{NAME} | batch | batch {batch_idx} search failed: {e}, falling back to individual requests")
            # Fallback к индивидуальным запросам
            for _, _, frame in batch:
                try:
                    result = embedding_client.search(
                        category=FRANCHISE_CATEGORY,
                        image=frame,
                        top_k=topk,
                        similarity_threshold=similarity_threshold,
                        max_retries=3,
                        retry_delay=1.0,
                    )
                    all_results.append(result if result else [])
                except Exception as e2:
                    logger.warning(f"{NAME} | batch | individual search failed: {e2}")
                    all_results.append([])
    
    # Этап 3: Распределение результатов обратно по видео
    logger.info(f"{NAME} | batch | distributing results back to videos")
    
    results = []
    for video_info in frames_by_video:
        video_idx = video_info["video_idx"]
        video_ctx = video_contexts[video_idx]
        
        if video_info["status"] != "ok":
            results.append({
                "video_id": video_ctx.video_id,
                "status": video_info["status"],
                "error": video_info.get("error"),
            })
            continue
        
        # Извлекаем результаты для этого видео
        video_frame_indices = video_info["frame_indices"]
        frame_start_idx = video_info.get("frame_start_idx", 0)
        frame_end_idx = video_info.get("frame_end_idx", len(all_results))
        
        if frame_start_idx >= frame_end_idx or frame_start_idx >= len(all_results):
            results.append({
                "video_id": video_ctx.video_id,
                "status": "empty",
                "empty_reason": "no frames processed",
            })
            continue
        
        # Извлекаем результаты для этого видео
        video_results = all_results[frame_start_idx:frame_end_idx]
        video_times_s = video_info["times_s"]
        
        # Проверяем соответствие размеров
        if len(video_results) != len(video_frame_indices) or len(video_results) != len(video_times_s):
            logger.warning(
                f"{NAME} | batch | video {video_ctx.video_id} size mismatch: "
                f"results={len(video_results)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
            )
            # Используем минимальный размер
            min_size = min(len(video_results), len(video_frame_indices), len(video_times_s))
            video_results = video_results[:min_size]
            video_frame_indices = video_frame_indices[:min_size]
            video_times_s = video_times_s[:min_size]
        
        # Build output arrays
        # Collect all unique franchise names and create label mapping
        all_franchise_names: Dict[str, int] = {}  # franchise_name -> label_id
        label_id_counter = 0
        
        # First pass: collect all franchise names
        for results_list in video_results:
            for result in results_list:
                franchise_name = result.get("name", "unknown")
                if franchise_name not in all_franchise_names:
                    all_franchise_names[franchise_name] = label_id_counter
                    label_id_counter += 1
        
        # Build semantic_label_names
        franchise_names_list = sorted(all_franchise_names.keys(), key=lambda x: all_franchise_names[x])
        semantic_label_names = np.asarray(
            [f"{all_franchise_names[name]}:{name}" for name in franchise_names_list], dtype="U"
        )
        
        # Build frame-level arrays
        n_frames = len(video_frame_indices)
        frame_topk_ids = np.full((n_frames, topk), -1, dtype=np.int32)
        frame_topk_scores = np.full((n_frames, topk), np.nan, dtype=np.float32)
        
        for frame_idx, results_list in enumerate(video_results):
            for k, result in enumerate(results_list[:topk]):
                franchise_name = result.get("name", "unknown")
                similarity = float(result.get("similarity", 0.0))
                if franchise_name in all_franchise_names:
                    label_id = all_franchise_names[franchise_name]
                    frame_topk_ids[frame_idx, k] = label_id
                    frame_topk_scores[frame_idx, k] = similarity
        
        # Compute is_confident flags
        frame_is_confident_top1 = np.zeros((n_frames,), dtype=np.bool_)
        for i in range(n_frames):
            if frame_topk_ids[i, 0] >= 0 and np.isfinite(frame_topk_scores[i, 0]):
                frame_is_confident_top1[i] = bool(frame_topk_scores[i, 0] >= threshold_global)
        
        # Video-level aggregate: max over time per franchise
        n_franchises = len(all_franchise_names)
        if n_franchises > 0:
            max_scores = np.full((n_franchises,), np.nan, dtype=np.float32)
            for franchise_name, label_id in all_franchise_names.items():
                scores_for_franchise = []
                for frame_idx, results_list in enumerate(video_results):
                    for result in results_list:
                        if result.get("name") == franchise_name:
                            scores_for_franchise.append(float(result.get("similarity", 0.0)))
                if scores_for_franchise:
                    max_scores[label_id] = float(max(scores_for_franchise))
            
            # Top-K franchises for video
            valid_indices = np.where(np.isfinite(max_scores))[0]
            if valid_indices.size > 0:
                top_vid = valid_indices[np.argsort(-max_scores[valid_indices])[:topk]]
                track_topk_ids = np.asarray(top_vid, dtype=np.int32).reshape(1, topk)
                track_topk_scores = np.asarray(max_scores[top_vid], dtype=np.float32).reshape(1, topk)
            else:
                track_topk_ids = np.full((1, topk), -1, dtype=np.int32)
                track_topk_scores = np.full((1, topk), np.nan, dtype=np.float32)
        else:
            track_topk_ids = np.full((1, topk), -1, dtype=np.int32)
            track_topk_scores = np.full((1, topk), np.nan, dtype=np.float32)
        
        # Evidence frames for top-K franchises
        track_topk_evidence_frame_indices = np.full((1, topk), -1, dtype=np.int32)
        for j in range(topk):
            if track_topk_ids[0, j] >= 0:
                label_id = int(track_topk_ids[0, j])
                franchise_name = franchise_names_list[label_id]
                # Find frame with max similarity for this franchise
                best_frame_idx = -1
                best_score = -1.0
                for frame_idx, results_list in enumerate(video_results):
                    for result in results_list:
                        if result.get("name") == franchise_name:
                            score = float(result.get("similarity", 0.0))
                            if score > best_score:
                                best_score = score
                                best_frame_idx = frame_idx
                if best_frame_idx >= 0:
                    track_topk_evidence_frame_indices[0, j] = int(video_frame_indices[best_frame_idx])
        
        # Track-level confidence
        top1_lid = int(track_topk_ids[0, 0]) if track_topk_ids[0, 0] >= 0 else -1
        top1_sc = float(track_topk_scores[0, 0]) if np.isfinite(track_topk_scores[0, 0]) else np.nan
        track_is_confident_top1 = np.asarray(
            [bool(top1_lid >= 0 and np.isfinite(top1_sc) and top1_sc >= threshold_global)], dtype=np.bool_
        )
        track_ids = np.asarray([0], dtype=np.int32)
        track_present_mask = np.asarray([True], dtype=np.bool_)
        
        # Threshold per label (not available from Embedding Service, use global)
        threshold_per_label_arr = np.full((n_franchises,), np.nan, dtype=np.float32)
        
        # Build metadata
        metadata = video_info["metadata"]
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
        
        output_meta: Dict[str, Any] = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "ok",
            "empty_reason": None,
            "embedding_service_url": embedding_client.base_url,
            "franchise_category": FRANCHISE_CATEGORY,
            "topk": topk,
            "similarity_threshold": similarity_threshold,
            "threshold_global": threshold_global,
            "num_franchises": n_franchises,
            "num_frames": n_frames,
            # Provenance chaining
            "core_clip_model_signature": video_info.get("upstream_model_signature"),
        }
        
        # Required run identity fields
        for k in required_run_keys:
            output_meta[k] = metadata.get(k)
        
        # Required by contract (baseline may use "unknown")
        output_meta["dataprocessor_version"] = str(metadata.get("dataprocessor_version") or "unknown")
        
        # Add models_used
        models_used_list = [
            model_used(
                model_name="embedding_service",
                model_version="v1",
                runtime="http",
                engine="http",
                precision="fp32",
                device="cpu",  # Embedding Service runs on server
            )
        ]
        models_used_list.extend(video_info.get("upstream_models_used", []))
        output_meta = apply_models_meta(output_meta, models_used=models_used_list)
        
        # Stage timings
        stage_timings_ms: Dict[str, float] = {
            "initialization": 0.0,
            "load_deps": 0.0,
            "process_frames": 0.0,
            "saving": 0.0,
            "total": (time.perf_counter() - start_time) * 1000.0,
        }
        output_meta["stage_timings_ms"] = stage_timings_ms
        
        # Сохраняем результаты в per-video rs_path
        component_dir = video_ctx.get_component_rs_path(NAME)
        npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
        os.makedirs(component_dir, exist_ok=True)
        
        # Сохранение NPZ
        npz_dict = {
            "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
            "times_s": video_times_s,
            "semantic_label_names": semantic_label_names,
            "threshold_per_label_arr": threshold_per_label_arr,
            "track_ids": track_ids,
            "track_present_mask": track_present_mask,
            "track_topk_ids": track_topk_ids,
            "track_topk_scores": track_topk_scores,
            "track_is_confident_top1": track_is_confident_top1,
            "track_topk_evidence_frame_indices": track_topk_evidence_frame_indices,
            "frame_topk_ids": frame_topk_ids,
            "frame_topk_scores": frame_topk_scores,
            "frame_is_confident_top1": frame_is_confident_top1,
            "meta": np.asarray(output_meta, dtype=object),
        }
        
        _atomic_save_npz(npz_path, **npz_dict)
        
        # Валидация NPZ
        ok, issues, _ = validate_npz(npz_path)
        if not ok:
            try:
                if os.path.exists(npz_path):
                    os.remove(npz_path)
            except Exception:
                pass
            raise RuntimeError(
                f"{NAME} | batch | saved artifact failed validation: "
                + "; ".join([f"{i.level}:{i.message}" for i in issues])
            )
        
        results.append({
            "video_id": video_ctx.video_id,
            "status": "ok",
            "saved_path": npz_path,
        })
    
    duration = time.perf_counter() - start_time
    logger.info(
        f"{NAME} | batch | completed in {duration:.2f}s "
        f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
    )
    
    return results

