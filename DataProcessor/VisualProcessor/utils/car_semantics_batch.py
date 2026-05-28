"""
Batch processing utilities for car_semantics component.

Stage 3: GPU batching для car_semantics с гибридным подходом:
- Сбор кадров из всех видео
- Группировка треков по видео
- Batch поиск через Embedding Service
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
from collections import defaultdict

import numpy as np
import cv2

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

# Import car_semantics functions
_car_semantics_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "car_semantics"
sys.path.insert(0, str(_car_semantics_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.car_semantics_batch")

# Import from car_semantics
try:
    from core.model_process.core_identity.car_semantics.utils.embedding_service_client import EmbeddingServiceClient
    from core.model_process.core_identity.car_semantics.utils.crop_utils import crop_with_padding, select_best_crop_for_track
    from core.model_process.core_identity.car_semantics.main import _extract_car_metadata
except ImportError:
    # Fallback: try utils directory, then direct import
    try:
        sys.path.insert(0, str(_car_semantics_path / "utils"))
        from embedding_service_client import EmbeddingServiceClient
        from crop_utils import crop_with_padding, select_best_crop_for_track
        sys.path.insert(0, str(_car_semantics_path))
        from main import _extract_car_metadata
    except ImportError:
        # Last fallback: direct import from root
        sys.path.insert(0, str(_car_semantics_path))
        from embedding_service_client import EmbeddingServiceClient
        from crop_utils import crop_with_padding, select_best_crop_for_track
        from main import _extract_car_metadata

NAME = "car_semantics"
VERSION = "0.1"
SCHEMA_VERSION = "car_semantics_npz_v1"
ARTIFACT_FILENAME = "car_semantics.npz"
CAR_CATEGORY = "car"
TOP_K = 3


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


def _load_npz(path: str) -> Dict[str, Any]:
    """Load NPZ file and handle object arrays properly."""
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | required artifact not found: {path}")
    z = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _get_class_id_to_name(class_names: np.ndarray) -> Dict[int, str]:
    """Parse class_names array into id->name mapping."""
    result: Dict[int, str] = {}
    for item in class_names:
        item_str = str(item)
        if ":" in item_str:
            try:
                class_id_str, class_name = item_str.split(":", 1)
                result[int(class_id_str)] = class_name
            except Exception:
                continue
    return result


def process_car_semantics_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для car_semantics с гибридным подходом.
    
    Stage 3: GPU batching для car_semantics.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация car_semantics
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для Embedding Service (если поддерживается)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"car_semantics | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size})"
    )
    
    start_time = time.perf_counter()
    
    # Инициализация Embedding Service клиента
    embedding_service_url = (
        config.get("embedding_service_url") 
        or os.environ.get("EMBEDDING_SERVICE_URL") 
        or "http://localhost:8001"
    )
    
    embedding_client = EmbeddingServiceClient(base_url=embedding_service_url)
    
    # Параметры конфигурации
    topk = int(config.get("topk", TOP_K))
    similarity_threshold = float(config.get("similarity_threshold", 0.0))
    max_tracks = config.get("max_tracks")
    if max_tracks == "" or max_tracks is None:
        max_tracks = None
    else:
        max_tracks = int(max_tracks)
    
    max_dets_per_frame = config.get("max_dets_per_frame")
    if max_dets_per_frame == "" or max_dets_per_frame is None:
        max_dets_per_frame = None
    else:
        max_dets_per_frame = int(max_dets_per_frame)
    
    pad_ratio = float(config.get("pad_ratio", 0.15))
    use_sharpness = bool(config.get("use_sharpness", False))
    
    # Этап 1: Сбор всех треков с привязкой к видео
    tracks_by_video: List[Dict[str, Any]] = []
    all_tracks: List[Tuple[int, int, np.ndarray, Dict[str, Any]]] = []  # (video_idx, track_id, crop, metadata)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Получаем frame_indices из core_object_detections
            block = metadata.get("core_object_detections")
            if not isinstance(block, dict) or "frame_indices" not in block:
                logger.error(f"car_semantics | batch | video {video_ctx.video_id} missing core_object_detections.frame_indices")
                tracks_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "missing core_object_detections.frame_indices",
                })
                continue
            
            frame_indices = block.get("frame_indices")
            if not isinstance(frame_indices, list) or not frame_indices:
                logger.error(f"car_semantics | batch | video {video_ctx.video_id} empty frame_indices")
                tracks_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "empty frame_indices",
                })
                continue
            
            frame_indices = [int(x) for x in frame_indices]
            
            # Timestamps
            uts = (
                metadata.get("union_timestamps_sec")
                or metadata.get("union_timestamps_s")
                or metadata.get("times_s")
            )
            if uts is None:
                logger.error(f"car_semantics | batch | video {video_ctx.video_id} missing union_timestamps_sec")
                tracks_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "missing union_timestamps_sec",
                })
                continue
            
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
                logger.error(f"car_semantics | batch | video {video_ctx.video_id} frame_indices out of range")
                tracks_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "frame_indices out of range",
                })
                continue
            
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Загружаем detections
            detections_path = os.path.join(
                str(video_ctx.get_component_rs_path("core_object_detections")), "detections.npz"
            )
            if not os.path.isfile(detections_path):
                logger.error(f"car_semantics | batch | video {video_ctx.video_id} detections.npz not found")
                tracks_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "tracks": {},
                    "frame_indices": [],
                    "times_s": None,
                    "status": "error",
                    "error": "detections.npz not found",
                })
                continue
            
            detections = _load_npz(detections_path)
            
            # Parse detections
            boxes = np.asarray(detections.get("boxes"), dtype=np.float32)  # (N, MAX, 4)
            scores = np.asarray(detections.get("scores"), dtype=np.float32)  # (N, MAX)
            class_ids = np.asarray(detections.get("class_ids"), dtype=np.int32)  # (N, MAX)
            valid_mask = np.asarray(detections.get("valid_mask"), dtype=bool)  # (N, MAX)
            class_names = np.asarray(detections.get("class_names"), dtype="U")  # (M,)
            class_id_to_name = _get_class_id_to_name(class_names)
            
            # Find car class ID
            car_class_id = None
            for cid, cname in class_id_to_name.items():
                if cname in ["car", "vehicle", "automobile"]:
                    car_class_id = cid
                    break
            
            # Get tracks
            if "tracks" in detections:
                tracks = np.asarray(detections.get("tracks"), dtype=np.int32)  # (N, MAX)
            else:
                # Generate per-detection track IDs
                track_counter = 0
                tracks = np.full_like(class_ids, -1, dtype=np.int32)
                for i in range(boxes.shape[0]):
                    for j in range(boxes.shape[1]):
                        if valid_mask[i, j]:
                            tracks[i, j] = track_counter
                            track_counter += 1
            
            # Create FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=int(metadata.get("chunk_size", 32)),
                cache_size=int(metadata.get("cache_size", 2)),
            )
            
            # Group detections by track
            track_detections: Dict[int, List[Tuple[int, int, float, np.ndarray]]] = defaultdict(list)
            
            for frame_idx, frame_idx_global in enumerate(frame_indices):
                for det_idx in range(boxes.shape[1]):
                    if not valid_mask[frame_idx, det_idx]:
                        continue
                    
                    # Filter by class if car_class_id specified
                    if car_class_id is not None:
                        if class_ids[frame_idx, det_idx] != car_class_id:
                            continue
                    
                    track_id = int(tracks[frame_idx, det_idx])
                    if track_id < 0:
                        continue
                    
                    score = float(scores[frame_idx, det_idx])
                    bbox = boxes[frame_idx, det_idx].copy()  # (x1, y1, x2, y2)
                    
                    track_detections[track_id].append(
                        (frame_idx, det_idx, score, bbox)
                    )
            
            # Apply cost control: max_tracks
            if max_tracks and len(track_detections) > max_tracks:
                sorted_tracks = sorted(
                    track_detections.items(), key=lambda x: len(x[1]), reverse=True
                )
                track_detections = dict(sorted_tracks[:max_tracks])
            
            # Process each track to get best crop
            video_tracks: Dict[int, Tuple[np.ndarray, Dict[str, Any]]] = {}  # track_id -> (crop, metadata)
            
            for track_id, detections_list in track_detections.items():
                # Apply cost control: max_dets_per_frame
                if max_dets_per_frame and len(detections_list) > max_dets_per_frame:
                    detections_list = sorted(
                        detections_list, key=lambda x: x[2], reverse=True
                    )[:max_dets_per_frame]
                
                # Select best crop for track
                crops = []
                scores_list = []
                areas_list = []
                
                for frame_idx, det_idx, score, bbox in detections_list:
                    frame_idx_global = frame_indices[frame_idx]
                    try:
                        frame = frame_manager.get(frame_idx_global)
                    except Exception as e:
                        logger.warning(f"car_semantics | batch | video {video_ctx.video_id} failed to load frame {frame_idx_global}: {e}")
                        continue
                    
                    crop = crop_with_padding(frame, bbox, pad_ratio=pad_ratio)
                    crops.append(crop)
                    
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    areas_list.append(area)
                    scores_list.append(score)
                
                if not crops:
                    continue
                
                # Select best crop
                try:
                    best_idx, best_crop = select_best_crop_for_track(
                        crops, scores_list, areas_list, use_sharpness=use_sharpness
                    )
                    video_tracks[track_id] = (best_crop, {
                        "detections_list": detections_list,
                        "frame_indices": frame_indices,
                    })
                except Exception as e:
                    logger.warning(f"car_semantics | batch | video {video_ctx.video_id} failed to select best crop for track {track_id}: {e}")
                    continue
            
            # Сохраняем информацию о треках для этого видео
            tracks_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "tracks": video_tracks,
                "frame_indices": frame_indices,
                "times_s": times_s,
                "frame_manager": frame_manager,
                "track_detections": track_detections,
                "status": "ok",
            })
            
            # Добавляем треки в общий список для batch поиска
            for track_id, (crop, metadata) in video_tracks.items():
                all_tracks.append((video_idx, track_id, crop, metadata))
            
        except Exception as e:
            logger.exception(f"car_semantics | batch | video {video_ctx.video_id} failed to prepare: {e}")
            tracks_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "tracks": {},
                "frame_indices": [],
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_tracks:
        logger.error("car_semantics | batch | no tracks collected from any video")
        # Закрываем FrameManager
        for video_info in tracks_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": "no tracks collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"car_semantics | batch | collected {len(all_tracks)} tracks from {len(tracks_by_video)} videos")
    
    # Этап 2: Batch поиск через Embedding Service
    try:
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"car_semantics | batch | processing {len(all_tracks)} tracks in batches of {effective_batch_size}")
        
        # Batch поиск (используем search_batch если доступен, иначе fallback)
        track_results: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}  # (video_idx, track_id) -> results
        
        start = 0
        while start < len(all_tracks):
            batch_end = min(start + effective_batch_size, len(all_tracks))
            batch_tracks = all_tracks[start:batch_end]
            
            # Извлекаем crops из батча
            batch_images = [crop for _, _, crop, _ in batch_tracks]
            
            # Batch search через Embedding Service
            try:
                batch_results = embedding_client.search_batch(
                    category=CAR_CATEGORY,
                    images=batch_images,
                    top_k=topk,
                    similarity_threshold=similarity_threshold,
                    max_retries=3,
                    retry_delay=1.0,
                )
                
                # Сохраняем результаты
                for i, (video_idx, track_id, _, _) in enumerate(batch_tracks):
                    track_results[(video_idx, track_id)] = batch_results[i] if i < len(batch_results) else []
                    
            except Exception as e:
                logger.warning(f"car_semantics | batch | batch search failed, falling back to individual requests: {e}")
                # Fallback: individual requests
                for video_idx, track_id, crop, _ in batch_tracks:
                    try:
                        results = embedding_client.search(
                            category=CAR_CATEGORY,
                            image=crop,
                            top_k=topk,
                            similarity_threshold=similarity_threshold,
                            max_retries=3,
                            retry_delay=1.0,
                        )
                        track_results[(video_idx, track_id)] = results
                    except Exception as e2:
                        logger.warning(f"car_semantics | batch | search failed for track {track_id}: {e2}")
                        track_results[(video_idx, track_id)] = []
            
            if start % (effective_batch_size * 10) == 0:
                logger.info(f"car_semantics | batch | processed {batch_end}/{len(all_tracks)} tracks")
            
            start = batch_end
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("car_semantics | batch | distributing results back to videos")
        
        results = []
        for video_info in tracks_by_video:
            video_idx = video_info["video_idx"]
            video_ctx = video_contexts[video_idx]
            
            if video_info["status"] != "ok":
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": video_info["status"],
                    "error": video_info.get("error"),
                })
                continue
            
            # Собираем результаты для этого видео
            video_tracks = video_info["tracks"]
            video_frame_indices = video_info["frame_indices"]
            video_times_s = video_info["times_s"]
            video_track_detections = video_info["track_detections"]
            
            # Build output arrays
            unique_track_ids = sorted([tid for tid in video_tracks.keys() if (video_idx, tid) in track_results])
            n_frames = len(video_frame_indices)
            n_tracks = len(unique_track_ids)
            
            # Track-level arrays
            track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)  # (T,)
            track_topk_ids = np.zeros((n_tracks, topk), dtype=np.int32)  # (T, K)
            track_topk_scores = np.zeros((n_tracks, topk), dtype=np.float32)  # (T, K)
            track_topk_makes = np.zeros((n_tracks, topk), dtype="U256")  # (T, K)
            track_topk_models = np.zeros((n_tracks, topk), dtype="U256")  # (T, K)
            track_topk_segments = np.zeros((n_tracks, topk), dtype="U256")  # (T, K)
            
            # Frame-level arrays
            frame_topk_ids = np.zeros((n_frames, topk), dtype=np.int32)  # (N, K)
            frame_topk_scores = np.zeros((n_frames, topk), dtype=np.float32)  # (N, K)
            
            # Build semantic_label_names from results
            all_car_ids: Dict[str, int] = {}
            label_id_counter = 0
            
            for track_idx, track_id in enumerate(unique_track_ids):
                results_list = track_results.get((video_idx, track_id), [])
                
                # Track-level top-K
                for k, result in enumerate(results_list[:topk]):
                    car_name = result.get("name", "unknown")
                    similarity = float(result.get("similarity", 0.0))
                    metadata = result.get("metadata", {})
                    
                    # Map car name to label ID
                    if car_name not in all_car_ids:
                        all_car_ids[car_name] = label_id_counter
                        label_id_counter += 1
                    
                    label_id = all_car_ids[car_name]
                    track_topk_ids[track_idx, k] = label_id
                    track_topk_scores[track_idx, k] = similarity
                    
                    # Extract make, model, segment
                    make, model, segment = _extract_car_metadata(metadata)
                    track_topk_makes[track_idx, k] = make
                    track_topk_models[track_idx, k] = model
                    track_topk_segments[track_idx, k] = segment
            
            # Build semantic_label_names array
            car_names_list = sorted(all_car_ids.keys(), key=lambda x: all_car_ids[x])
            semantic_label_names = np.asarray(
                [f"{all_car_ids[name]}:{name}" for name in car_names_list], dtype="U"
            )
            
            # Frame-level aggregation
            for frame_idx in range(n_frames):
                car_scores: Dict[str, float] = {}
                
                for track_id, detections_list in video_track_detections.items():
                    if track_id not in video_tracks:
                        continue
                    
                    # Check if track has detection on this frame
                    has_detection = any(
                        det_frame_idx == frame_idx for det_frame_idx, _, _, _ in detections_list
                    )
                    
                    if not has_detection:
                        continue
                    
                    results_list = track_results.get((video_idx, track_id), [])
                    for result in results_list[:topk]:
                        car_name = result.get("name", "unknown")
                        similarity = float(result.get("similarity", 0.0))
                        
                        if car_name in all_car_ids:
                            if car_name not in car_scores or similarity > car_scores[car_name]:
                                car_scores[car_name] = similarity
                
                # Sort by similarity and take top-K
                frame_results = [
                    (similarity, all_car_ids[car_name])
                    for car_name, similarity in car_scores.items()
                ]
                frame_results.sort(key=lambda x: x[0], reverse=True)
                for k, (similarity, label_id) in enumerate(frame_results[:topk]):
                    frame_topk_ids[frame_idx, k] = label_id
                    frame_topk_scores[frame_idx, k] = similarity
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            
            save_metadata = {
                "producer": NAME,
                "producer_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "platform_id": video_ctx.platform_id or metadata.get("platform_id"),
                "video_id": video_ctx.video_id,
                "run_id": video_ctx.run_id or metadata.get("run_id"),
                "sampling_policy_version": video_ctx.sampling_policy_version or metadata.get("sampling_policy_version"),
                "config_hash": video_ctx.config_hash or metadata.get("config_hash"),
                "dataprocessor_version": video_ctx.dataprocessor_version or metadata.get("dataprocessor_version") or "unknown",
                "status": "ok",
                "empty_reason": None,
                "embedding_service_url": embedding_service_url,
                "car_category": CAR_CATEGORY,
                "topk": topk,
                "similarity_threshold": similarity_threshold,
                "pad_ratio": pad_ratio,
                "use_sharpness": use_sharpness,
                "num_tracks": n_tracks,
                "num_cars": len(all_car_ids),
            }
            
            # Models used
            models_used = [
                model_used(
                    model_name="embedding_service",
                    model_version="v1",
                    runtime="http",
                    engine="http",
                    precision="fp32",
                    device="cpu",
                )
            ]
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Сохранение NPZ
            npz_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "track_ids": track_ids_arr,
                "track_topk_ids": track_topk_ids,
                "track_topk_scores": track_topk_scores,
                "track_topk_makes": track_topk_makes,
                "track_topk_models": track_topk_models,
                "track_topk_segments": track_topk_segments,
                "frame_topk_ids": frame_topk_ids,
                "frame_topk_scores": frame_topk_scores,
                "semantic_label_names": semantic_label_names,
                "meta": np.asarray(save_metadata, dtype=object),
            }
            
            _atomic_save_npz(npz_path, **npz_dict)
            
            # Валидация NPZ
            ok, issues, _ = validate_npz(
                npz_path,
                required_meta_keys=[
                    "producer",
                    "producer_version",
                    "schema_version",
                    "created_at",
                    "platform_id",
                    "video_id",
                    "run_id",
                    "config_hash",
                    "sampling_policy_version",
                    "dataprocessor_version",
                    "status",
                    "empty_reason",
                    "models_used",
                    "model_signature",
                ],
            )
            if not ok:
                try:
                    if os.path.exists(npz_path):
                        os.remove(npz_path)
                except Exception:
                    pass
                raise RuntimeError(
                    f"car_semantics | batch | saved artifact failed validation: "
                    + "; ".join([f"{i.level}:{i.message}" for i in issues])
                )
            
            results.append({
                "video_id": video_ctx.video_id,
                "status": "ok",
                "saved_path": npz_path,
            })
        
        # Закрываем FrameManager для всех видео
        for video_info in tracks_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        duration = time.perf_counter() - start_time
        logger.info(
            f"car_semantics | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"car_semantics | batch | error: {e}")
        # Закрываем FrameManager в случае ошибки
        for video_info in tracks_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        # Возвращаем ошибки для всех видео
        return [
            {
                "video_id": ctx.video_id,
                "status": "error",
                "error": str(e),
            }
            for ctx in video_contexts
        ]

