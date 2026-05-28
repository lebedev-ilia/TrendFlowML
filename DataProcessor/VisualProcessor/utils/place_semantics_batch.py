"""
Batch processing utilities for place_semantics component.

Stage 3: Batch processing для place_semantics с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи для batch search через Embedding Service
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

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

logger = get_logger("VisualProcessor.place_semantics_batch")

# Import place_semantics functions
_place_semantics_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "place_semantics"
sys.path.insert(0, str(_place_semantics_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.place_semantics_batch")

# Add place_semantics directory to path for imports
sys.path.insert(0, str(_place_semantics_path))

# Import EmbeddingServiceClient (try full path first, then fallback to direct import)
try:
    from core.model_process.core_identity.place_semantics.utils.embedding_service_client import EmbeddingServiceClient
except ImportError:
    # Fallback: try direct import from utils directory
    try:
        _embedding_client_path = _place_semantics_path / "utils" / "embedding_service_client.py"
        if _embedding_client_path.exists():
            import importlib.util
            spec_client = importlib.util.spec_from_file_location("embedding_service_client", str(_embedding_client_path))
            if spec_client and spec_client.loader:
                embedding_client_module = importlib.util.module_from_spec(spec_client)
                spec_client.loader.exec_module(embedding_client_module)
                EmbeddingServiceClient = embedding_client_module.EmbeddingServiceClient
            else:
                raise ImportError("Failed to load EmbeddingServiceClient")
        else:
            # Last fallback: try direct import (should work now since we added path)
            from embedding_service_client import EmbeddingServiceClient
    except ImportError:
        raise ImportError(
            f"place_semantics_batch | embedding_service_client not found. "
            f"Expected at: {_place_semantics_path / 'utils' / 'embedding_service_client.py'}"
        )

# Import place_semantics functions
_place_semantics_main = _place_semantics_path / "main.py"
if _place_semantics_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("place_semantics_main", str(_place_semantics_main))
    if spec and spec.loader:
        place_semantics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(place_semantics_module)
        
        # Import functions
        _require_frame_indices = getattr(place_semantics_module, "_require_frame_indices", None)
        _group_frames_by_place = getattr(place_semantics_module, "_group_frames_by_place", None)
        NAME = getattr(place_semantics_module, "NAME", "place_semantics")
        VERSION = getattr(place_semantics_module, "VERSION", "0.1")
        SCHEMA_VERSION = getattr(place_semantics_module, "SCHEMA_VERSION", "place_semantics_npz_v1")
        ARTIFACT_FILENAME = getattr(place_semantics_module, "ARTIFACT_FILENAME", "place_semantics.npz")
        PLACE_CATEGORY = getattr(place_semantics_module, "PLACE_CATEGORY", "place")
        TOP_K = getattr(place_semantics_module, "TOP_K", 5)
    else:
        raise ImportError("Failed to load place_semantics module")
else:
    raise ImportError(f"place_semantics/main.py not found at {_place_semantics_main}")


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


def _require_union_timestamps_sec(metadata: Dict[str, Any], total_frames: int) -> np.ndarray:
    """Extract union_timestamps_sec from metadata."""
    uts = (
        metadata.get("union_timestamps_sec")
        or metadata.get("union_timestamps_s")
        or metadata.get("times_s")
    )
    if uts is None:
        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    if uts_arr.shape[0] != total_frames:
        raise RuntimeError(
            f"{NAME} | union_timestamps_sec length {uts_arr.shape[0]} != total_frames {total_frames}"
        )
    return uts_arr


def process_place_semantics_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для place_semantics с гибридным подходом.
    
    Stage 3: Batch processing для place_semantics.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация place_semantics
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для Embedding Service (если поддерживается batch API)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"place_semantics | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size})"
    )
    
    start_time = time.perf_counter()
    
    # Инициализация Embedding Service клиента
    embedding_service_url = config.get("embedding_service_url") or os.environ.get("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        embedding_service_url = "http://localhost:8001"
    
    embedding_client = EmbeddingServiceClient(base_url=embedding_service_url)
    
    # Параметры
    topk = int(config.get("topk", TOP_K))
    similarity_threshold = float(config.get("similarity_threshold", 0.0))
    min_track_length = int(config.get("min_track_length", 3))
    max_gap_sec = float(config.get("max_gap_sec", 5.0))
    
    # Этап 1: Сбор всех кадров с привязкой к видео
    frames_by_video: List[Dict[str, Any]] = []
    all_frames: List[Tuple[int, int, np.ndarray]] = []  # (video_idx, frame_idx, frame)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))
            
            # Получаем frame_indices
            try:
                frame_indices = _require_frame_indices(metadata)
            except Exception as e:
                logger.error(f"place_semantics | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "error",
                    "error": str(e),
                })
                continue
            
            if not frame_indices:
                logger.warning(f"place_semantics | batch | video {video_ctx.video_id} has no frame_indices")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "empty",
                })
                continue
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=metadata.get("chunk_size", 32),
                cache_size=metadata.get("cache_size", 2),
            )
            
            # Получаем timestamps
            union_timestamps_sec = _require_union_timestamps_sec(metadata, total_frames=total_frames)
            times_s = union_timestamps_sec[np.asarray(frame_indices, dtype=np.int32)].astype(np.float32)
            
            # Загружаем кадры и сохраняем маппинг
            video_frame_start_idx = len(all_frames)  # Начальный индекс в общем батче для этого видео
            for frame_idx in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx)
                    # Сохраняем кадр в общий батч
                    all_frames.append((video_idx, frame_idx, frame))
                except Exception as e:
                    logger.warning(
                        f"place_semantics | batch | video {video_ctx.video_id} failed to load frame {frame_idx}: {e}"
                    )
                    continue
            
            # Сохраняем информацию о диапазоне индексов для этого видео
            video_frame_end_idx = len(all_frames)
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "frame_manager": frame_manager,
                "times_s": times_s,
                "frame_start_idx": video_frame_start_idx,
                "frame_end_idx": video_frame_end_idx,
                "metadata": metadata,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"place_semantics | batch | video {video_ctx.video_id} failed to prepare: {e}")
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "frame_manager": None,
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_frames:
        logger.error("place_semantics | batch | no frames collected from any video")
        # Закрываем FrameManager
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no frames collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"place_semantics | batch | collected {len(all_frames)} frames from {len(frames_by_video)} videos")
    
    # Этап 2: Batch search через Embedding Service
    try:
        n_frames = len(all_frames)
        all_frame_results: List[List[Dict[str, Any]]] = []
        
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"place_semantics | batch | processing {n_frames} frames in batches of {effective_batch_size}")
        
        start = 0
        while start < n_frames:
            batch_end = min(start + effective_batch_size, n_frames)
            batch_frames = all_frames[start:batch_end]
            
            # Извлекаем кадры из батча
            batch_images = [frame for _, _, frame in batch_frames]
            
            # Batch search через Embedding Service
            try:
                batch_results = embedding_client.search_batch(
                    category=PLACE_CATEGORY,
                    images=batch_images,
                    top_k=topk,
                    similarity_threshold=similarity_threshold,
                    max_retries=3,
                    retry_delay=1.0,
                )
                all_frame_results.extend(batch_results)
            except Exception as e:
                logger.error(f"place_semantics | batch | Embedding Service batch search failed: {e}")
                # Fallback: individual requests
                for _, _, frame in batch_frames:
                    try:
                        result = embedding_client.search(
                            category=PLACE_CATEGORY,
                            image=frame,
                            top_k=topk,
                            similarity_threshold=similarity_threshold,
                            max_retries=3,
                            retry_delay=1.0,
                        )
                        all_frame_results.append(result)
                    except Exception as e2:
                        logger.warning(f"place_semantics | batch | individual search failed: {e2}")
                        all_frame_results.append([])
            
            start = batch_end
        
        logger.info(f"place_semantics | batch | completed search for {len(all_frame_results)} frames")
        
    except Exception as e:
        logger.exception(f"place_semantics | batch | failed during batch search: {e}")
        # Закрываем FrameManager
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": f"batch search failed: {e}",
            }
            for ctx in video_contexts
        ]
    
    # Этап 3: Распределение результатов обратно по видео
    results = []
    
    for video_info in frames_by_video:
        if video_info["status"] != "ok":
            results.append({
                "video_id": video_info["video_id"],
                "status": video_info["status"],
                "error": video_info.get("error"),
            })
            continue
        
        try:
            video_idx = video_info["video_idx"]
            frame_indices = video_info["frame_indices"]
            times_s = video_info["times_s"]
            frame_start_idx = video_info["frame_start_idx"]
            frame_end_idx = video_info["frame_end_idx"]
            metadata = video_info["metadata"]
            
            # Извлекаем результаты для этого видео
            video_frame_results = all_frame_results[frame_start_idx:frame_end_idx]
            
            # Группируем кадры по местам в tracks
            tracks = _group_frames_by_place(
                video_frame_results,
                frame_indices,
                times_s,
                min_track_length=min_track_length,
                max_gap_sec=max_gap_sec,
            )
            
            # Build semantic_label_names from results
            all_place_ids: Dict[str, int] = {}  # place_name -> label_id
            label_id_counter = 0
            
            for results_list in video_frame_results:
                for result in results_list[:topk]:
                    place_name = result.get("name", "unknown")
                    if place_name not in all_place_ids:
                        all_place_ids[place_name] = label_id_counter
                        label_id_counter += 1
            
            # Build semantic_label_names array
            place_names_list = sorted(all_place_ids.keys(), key=lambda x: all_place_ids[x])
            semantic_label_names = np.asarray(
                [f"{all_place_ids[name]}:{name}" for name in place_names_list], dtype="U"
            )
            
            # Build output arrays
            unique_track_ids = sorted(tracks.keys())
            n_frames_video = len(frame_indices)
            n_tracks = len(unique_track_ids)
            
            # Track-level arrays
            track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)  # (T,)
            track_topk_ids = np.zeros((n_tracks, topk), dtype=np.int32)  # (T, K)
            track_topk_scores = np.zeros((n_tracks, topk), dtype=np.float32)  # (T, K)
            track_present_mask = np.ones((n_tracks,), dtype=np.bool_)  # (T,)
            track_is_confident_top1 = np.zeros((n_tracks,), dtype=np.bool_)  # (T,)
            
            # Frame-level arrays
            frame_topk_ids = np.full((n_frames_video, topk), -1, dtype=np.int32)  # (N, K)
            frame_topk_scores = np.full((n_frames_video, topk), np.nan, dtype=np.float32)  # (N, K)
            frame_is_confident_top1 = np.zeros((n_frames_video,), dtype=np.bool_)  # (N,)
            
            # Fill track-level arrays
            for track_idx, track_id in enumerate(unique_track_ids):
                track_frame_indices = tracks[track_id]
                
                # Aggregate results for this track
                track_place_scores: Dict[str, float] = {}  # place_name -> max_similarity
                
                for frame_idx_local, frame_idx_global in enumerate(frame_indices):
                    if frame_idx_global not in track_frame_indices:
                        continue
                    
                    results_list = video_frame_results[frame_idx_local]
                    for result in results_list[:topk]:
                        place_name = result.get("name", "unknown")
                        similarity = float(result.get("similarity", 0.0))
                        
                        if place_name in all_place_ids:
                            # Deduplicate: take best similarity for each place
                            if place_name not in track_place_scores or similarity > track_place_scores[place_name]:
                                track_place_scores[place_name] = similarity
                
                # Sort by similarity and take top-K
                track_results_sorted = [
                    (similarity, all_place_ids[place_name])
                    for place_name, similarity in track_place_scores.items()
                ]
                track_results_sorted.sort(key=lambda x: x[0], reverse=True)
                
                for k, (similarity, label_id) in enumerate(track_results_sorted[:topk]):
                    track_topk_ids[track_idx, k] = label_id
                    track_topk_scores[track_idx, k] = similarity
                
                # Track confidence flag (top-1)
                if track_results_sorted:
                    top1_score = track_results_sorted[0][0]
                    track_is_confident_top1[track_idx] = bool(
                        np.isfinite(top1_score) and top1_score >= similarity_threshold
                    )
            
            # Fill frame-level arrays
            for frame_idx_local, results_list in enumerate(video_frame_results):
                frame_place_scores: Dict[str, float] = {}  # place_name -> max_similarity
                
                for result in results_list[:topk]:
                    place_name = result.get("name", "unknown")
                    similarity = float(result.get("similarity", 0.0))
                    
                    if place_name in all_place_ids:
                        # Deduplicate: take best similarity for each place
                        if place_name not in frame_place_scores or similarity > frame_place_scores[place_name]:
                            frame_place_scores[place_name] = similarity
                
                # Sort by similarity and take top-K
                frame_results_sorted = [
                    (similarity, all_place_ids[place_name])
                    for place_name, similarity in frame_place_scores.items()
                ]
                frame_results_sorted.sort(key=lambda x: x[0], reverse=True)
                
                for k, (similarity, label_id) in enumerate(frame_results_sorted[:topk]):
                    frame_topk_ids[frame_idx_local, k] = label_id
                    frame_topk_scores[frame_idx_local, k] = similarity
                
                # Frame confidence flag (top-1)
                if frame_results_sorted:
                    top1_score = frame_results_sorted[0][0]
                    frame_is_confident_top1[frame_idx_local] = bool(
                        np.isfinite(top1_score) and top1_score >= similarity_threshold
                    )
            
            # Build metadata
            required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
            missing = [k for k in required_run_keys if not metadata.get(k)]
            if missing:
                raise RuntimeError(
                    f"{NAME} | frames metadata missing required run identity keys: {missing}"
                )
            
            output_meta = {
                "producer": NAME,
                "producer_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "status": "ok",
                "empty_reason": None,
                "embedding_service_url": embedding_client.base_url,
                "place_category": PLACE_CATEGORY,
                "topk": topk,
                "similarity_threshold": similarity_threshold,
                "min_track_length": min_track_length,
                "max_gap_sec": max_gap_sec,
                "num_tracks": n_tracks,
                "num_places": len(all_place_ids),
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
            output_meta = apply_models_meta(output_meta, models_used=models_used_list)
            
            # Stage timings
            elapsed = time.perf_counter() - start_time
            output_meta["stage_timings_ms"] = {
                "total": float(elapsed) * 1000.0,
            }
            
            # Threshold arrays (aligned with semantic_label_names)
            threshold_per_label_arr = np.full((len(semantic_label_names),), np.nan, dtype=np.float32)
            
            # Save output
            output_dir = os.path.join(str(video_ctx.rs_path), NAME)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, ARTIFACT_FILENAME)
            
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            
            _atomic_save_npz(
                output_path,
                frame_indices=fi_np,
                times_s=times_s,
                track_ids=track_ids_arr,
                track_topk_ids=track_topk_ids,
                track_topk_scores=track_topk_scores,
                track_present_mask=track_present_mask,
                track_is_confident_top1=track_is_confident_top1,
                frame_topk_ids=frame_topk_ids,
                frame_topk_scores=frame_topk_scores,
                frame_is_confident_top1=frame_is_confident_top1,
                semantic_label_names=semantic_label_names,
                threshold_per_label_arr=threshold_per_label_arr,
                meta=np.asarray(output_meta, dtype=object),
            )
            
            # Validate NPZ
            try:
                validate_npz(output_path)
            except Exception as e:
                logger.warning(f"place_semantics | batch | NPZ validation warning for {video_info['video_id']}: {e}")
            
            results.append({
                "video_id": video_info["video_id"],
                "status": "ok",
                "output_path": output_path,
            })
            
        except Exception as e:
            logger.exception(f"place_semantics | batch | video {video_info['video_id']} failed to save: {e}")
            results.append({
                "video_id": video_info["video_id"],
                "status": "error",
                "error": str(e),
            })
        finally:
            # Закрываем FrameManager
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
    
    elapsed_total = time.perf_counter() - start_time
    logger.info(
        f"place_semantics | batch | completed processing {len(video_contexts)} videos "
        f"in {elapsed_total:.2f}s"
    )
    
    return results

