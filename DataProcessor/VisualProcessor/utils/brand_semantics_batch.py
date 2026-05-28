"""
Batch processing utilities for brand_semantics component.

Stage 3: GPU batching для brand_semantics с гибридным подходом:
- Сбор кадров из всех видео
- Группировка треков по видео
- Batch поиск через Embedding Service
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
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

# Import brand_semantics functions
_brand_semantics_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "brand_semantics"
sys.path.insert(0, str(_brand_semantics_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.brand_semantics_batch")

# Import from brand_semantics
try:
    from core.model_process.core_identity.brand_semantics.utils.embedding_service_client import EmbeddingServiceClient
    from core.model_process.core_identity.brand_semantics.utils.crop_utils import crop_with_padding, select_best_crop_for_track
except ImportError:
    # Fallback: try utils directory, then direct import
    try:
        sys.path.insert(0, str(_brand_semantics_path / "utils"))
        from embedding_service_client import EmbeddingServiceClient
        from crop_utils import crop_with_padding, select_best_crop_for_track
    except ImportError:
        # Last fallback: direct import from root
        sys.path.insert(0, str(_brand_semantics_path))
        from embedding_service_client import EmbeddingServiceClient
        from crop_utils import crop_with_padding, select_best_crop_for_track

NAME = "brand_semantics"
VERSION = "0.2"
SCHEMA_VERSION = "brand_semantics_npz_v2"
ARTIFACT_FILENAME = "brand_semantics.npz"
BRAND_CATEGORY = "brand"
TOP_K = 5


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


def process_brand_semantics_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для brand_semantics с гибридным подходом.
    
    Stage 3: GPU batching для brand_semantics.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация brand_semantics
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для Embedding Service (если поддерживается)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"brand_semantics | batch | processing {len(video_contexts)} videos "
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

    # Fail-fast + deterministic label-space (UUID->int32 mapping)
    try:
        embedding_client._ensure_url()
    except Exception as e:
        raise RuntimeError(
            f"{NAME} | batch | Embedding Service unavailable at {embedding_client.base_url}: {e} (fail-fast)"
        ) from e

    labels = embedding_client.get_labels(category=BRAND_CATEGORY)
    if not labels:
        raise RuntimeError(
            f"{NAME} | batch | Embedding Service category '{BRAND_CATEGORY}' has 0 labels (fail-fast)"
        )

    def _canon_label_row(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(r.get("id") or ""),
            "name": str(r.get("name") or ""),
            "embedding_model": str(r.get("embedding_model") or ""),
            "embedding_dim": int(r.get("embedding_dim") or 0),
            "updated_at": str(r.get("updated_at") or ""),
        }

    labels_canon = [_canon_label_row(r) for r in labels]
    labels_canon = [r for r in labels_canon if r["id"]]
    labels_canon.sort(key=lambda r: r["id"])
    db_digest = hashlib.sha256(
        json.dumps(labels_canon, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    uuid_to_int: Dict[str, int] = {r["id"]: i for i, r in enumerate(labels_canon)}
    semantic_object_ids = np.asarray([r["id"] for r in labels_canon], dtype="U")
    semantic_label_names = np.asarray(
        [f"{i}:{labels_canon[i]['name']}" for i in range(len(labels_canon))],
        dtype="U",
    )
    threshold_per_label_arr = np.full((len(labels_canon),), np.nan, dtype=np.float32)
    embedding_models = sorted({r["embedding_model"] for r in labels_canon if r["embedding_model"]})
    embedding_model = embedding_models[0] if len(embedding_models) == 1 else ""
    
    # Параметры конфигурации
    topk = int(config.get("topk", TOP_K))
    if topk != TOP_K:
        raise RuntimeError(
            f"{NAME} | batch | topk must be fixed to {TOP_K} by contract; got {topk}"
        )

    # Contract: MUST NOT gate top-K by thresholds.
    # Keep config key name for backward compatibility, use it ONLY for confident flags.
    confidence_threshold_top1 = float(
        config.get("confidence_threshold_top1", config.get("similarity_threshold", 0.0))
    )
    if not (0.0 <= confidence_threshold_top1 <= 1.0):
        raise RuntimeError(
            f"{NAME} | batch | confidence_threshold_top1 out of range [0,1]: {confidence_threshold_top1}"
        )

    proposal_classes = config.get("proposal_classes", "logo_region,text_region")
    if isinstance(proposal_classes, list):
        proposal_classes_list = [str(x).strip() for x in proposal_classes if str(x).strip()]
    else:
        proposal_classes_list = [s.strip() for s in str(proposal_classes).split(",") if s.strip()]
    if not proposal_classes_list:
        raise RuntimeError(f"{NAME} | batch | proposal_classes is empty (contract)")
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
                logger.error(f"brand_semantics | batch | video {video_ctx.video_id} missing core_object_detections.frame_indices")
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
                logger.error(f"brand_semantics | batch | video {video_ctx.video_id} empty frame_indices")
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
                logger.error(f"brand_semantics | batch | video {video_ctx.video_id} missing union_timestamps_sec")
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
                logger.error(f"brand_semantics | batch | video {video_ctx.video_id} frame_indices out of range")
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
                logger.error(f"brand_semantics | batch | video {video_ctx.video_id} detections.npz not found")
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
            
            # Resolve proposal classes to class_ids (strict)
            available_class_names = set(class_id_to_name.values())
            missing = [c for c in proposal_classes_list if c not in available_class_names]
            if missing:
                raise RuntimeError(
                    f"{NAME} | batch | video {video_ctx.video_id} proposal_classes not found in taxonomy: {missing}"
                )
            allowed_class_ids = {
                int(cid) for cid, cname in class_id_to_name.items() if cname in set(proposal_classes_list)
            }
            if not allowed_class_ids:
                raise RuntimeError(
                    f"{NAME} | batch | video {video_ctx.video_id} resolved proposal_classes produced empty class_id set"
                )
            
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
                    
                    # Filter by allowed proposal classes
                    if int(class_ids[frame_idx, det_idx]) not in allowed_class_ids:
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
                crop_meta = []  # (frame_pos, det_idx, det_score, bbox, class_id)
                
                for frame_idx, det_idx, score, bbox in detections_list:
                    frame_idx_global = frame_indices[frame_idx]
                    try:
                        frame = frame_manager.get(frame_idx_global)
                    except Exception as e:
                        logger.warning(f"brand_semantics | batch | video {video_ctx.video_id} failed to load frame {frame_idx_global}: {e}")
                        continue
                    
                    crop = crop_with_padding(frame, bbox, pad_ratio=pad_ratio)
                    crops.append(crop)
                    
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    areas_list.append(area)
                    scores_list.append(score)
                    crop_meta.append(
                        (
                            int(frame_idx),
                            int(det_idx),
                            float(score),
                            bbox.astype(np.float32),
                            int(class_ids[frame_idx, det_idx]),
                        )
                    )
                
                if not crops:
                    continue
                
                # Select best crop
                try:
                    best_idx, best_crop = select_best_crop_for_track(
                        crops, scores_list, areas_list, use_sharpness=use_sharpness
                    )
                    best_frame_pos, best_det_idx, best_det_score, best_bbox, best_class_id = crop_meta[int(best_idx)]
                    video_tracks[track_id] = (best_crop, {
                        "detections_list": detections_list,
                        "frame_indices": frame_indices,
                        "best_frame_pos": best_frame_pos,
                        "best_det_idx": best_det_idx,
                        "best_bbox_xyxy": best_bbox,
                        "best_det_score": best_det_score,
                        "best_class_id": best_class_id,
                    })
                except Exception as e:
                    logger.warning(f"brand_semantics | batch | video {video_ctx.video_id} failed to select best crop for track {track_id}: {e}")
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
                "max_dets": int(boxes.shape[1]),
                "status": "ok",
            })
            
            # Добавляем треки в общий список для batch поиска
            for track_id, (crop, metadata) in video_tracks.items():
                all_tracks.append((video_idx, track_id, crop, metadata))
            
        except Exception as e:
            logger.exception(f"brand_semantics | batch | video {video_ctx.video_id} failed to prepare: {e}")
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
        logger.error("brand_semantics | batch | no tracks collected from any video")
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
    
    logger.info(f"brand_semantics | batch | collected {len(all_tracks)} tracks from {len(tracks_by_video)} videos")
    
    # Этап 2: Batch поиск через Embedding Service
    try:
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"brand_semantics | batch | processing {len(all_tracks)} tracks in batches of {effective_batch_size}")
        
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
                    category=BRAND_CATEGORY,
                    images=batch_images,
                    top_k=topk,
                    similarity_threshold=0.0,  # Contract: do NOT gate top-K
                    max_retries=3,
                    retry_delay=1.0,
                )
                
                # Сохраняем результаты
                for i, (video_idx, track_id, _, _) in enumerate(batch_tracks):
                    track_results[(video_idx, track_id)] = batch_results[i] if i < len(batch_results) else []
                    
            except Exception as e:
                logger.warning(f"brand_semantics | batch | batch search failed, falling back to individual requests: {e}")
                # Fallback: individual requests
                for video_idx, track_id, crop, _ in batch_tracks:
                    try:
                        results = embedding_client.search(
                            category=BRAND_CATEGORY,
                            image=crop,
                            top_k=topk,
                            similarity_threshold=0.0,  # Contract: do NOT gate top-K
                            max_retries=3,
                            retry_delay=1.0,
                        )
                        track_results[(video_idx, track_id)] = results
                    except Exception as e2:
                        logger.warning(f"brand_semantics | batch | search failed for track {track_id}: {e2}")
                        track_results[(video_idx, track_id)] = []
            
            if start % (effective_batch_size * 10) == 0:
                logger.info(f"brand_semantics | batch | processed {batch_end}/{len(all_tracks)} tracks")
            
            start = batch_end
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("brand_semantics | batch | distributing results back to videos")
        
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
            
            # Build output arrays (align with brand_semantics_npz_v2)
            unique_track_ids = sorted([tid for tid in video_tracks.keys() if (video_idx, tid) in track_results])
            n_frames = int(len(video_frame_indices))
            n_tracks = int(len(unique_track_ids))
            max_dets = int(video_info.get("max_dets") or 0)

            track_ids_arr = np.asarray(unique_track_ids, dtype=np.int32)
            track_present_mask = np.ones((n_tracks,), dtype=bool)
            track_topk_ids = np.full((n_tracks, TOP_K), -1, dtype=np.int32)
            track_topk_scores = np.full((n_tracks, TOP_K), np.nan, dtype=np.float32)
            track_is_confident_top1 = np.zeros((n_tracks,), dtype=bool)

            track_best_frame_pos = np.full((n_tracks,), -1, dtype=np.int32)
            track_best_det_idx = np.full((n_tracks,), -1, dtype=np.int32)
            track_best_bbox_xyxy = np.full((n_tracks, 4), np.nan, dtype=np.float32)
            track_best_det_score = np.full((n_tracks,), np.nan, dtype=np.float32)
            track_best_class_id = np.full((n_tracks,), -1, dtype=np.int32)

            det_present_mask = np.zeros((n_frames, max_dets), dtype=bool) if max_dets > 0 else np.zeros((n_frames, 0), dtype=bool)
            det_topk_ids = np.full((n_frames, max_dets, TOP_K), -1, dtype=np.int32) if max_dets > 0 else np.full((n_frames, 0, TOP_K), -1, dtype=np.int32)
            det_topk_scores = np.full((n_frames, max_dets, TOP_K), np.nan, dtype=np.float32) if max_dets > 0 else np.full((n_frames, 0, TOP_K), np.nan, dtype=np.float32)
            det_is_confident_top1 = np.zeros((n_frames, max_dets), dtype=bool) if max_dets > 0 else np.zeros((n_frames, 0), dtype=bool)

            frame_topk_ids = np.full((n_frames, TOP_K), -1, dtype=np.int32)
            frame_topk_scores = np.full((n_frames, TOP_K), np.nan, dtype=np.float32)
            frame_is_confident_top1 = np.zeros((n_frames,), dtype=bool)

            # Fill track arrays + per-detection arrays from search results
            for track_pos, track_id in enumerate(unique_track_ids):
                results_list = track_results.get((video_idx, track_id), []) or []

                track_meta = video_tracks[track_id][1] if track_id in video_tracks else {}
                track_best_frame_pos[track_pos] = int(track_meta.get("best_frame_pos", -1))
                track_best_det_idx[track_pos] = int(track_meta.get("best_det_idx", -1))
                try:
                    track_best_bbox_xyxy[track_pos, :] = np.asarray(track_meta.get("best_bbox_xyxy"), dtype=np.float32).reshape(4)
                except Exception:
                    pass
                track_best_det_score[track_pos] = float(track_meta.get("best_det_score", np.nan))
                track_best_class_id[track_pos] = int(track_meta.get("best_class_id", -1))

                for k, r in enumerate(results_list[:TOP_K]):
                    oid = str(r.get("id") or "")
                    if oid not in uuid_to_int:
                        continue
                    track_topk_ids[track_pos, k] = int(uuid_to_int[oid])
                    track_topk_scores[track_pos, k] = float(r.get("similarity", np.nan))

                top1 = float(track_topk_scores[track_pos, 0]) if np.isfinite(track_topk_scores[track_pos, 0]) else float("nan")
                if np.isfinite(top1) and top1 >= confidence_threshold_top1:
                    track_is_confident_top1[track_pos] = True

                # per-detection fill for kept detections list (post cost-control)
                kept_dets = track_meta.get("detections_list") or []
                for frame_pos, det_idx, _score, _bbox in kept_dets:
                    if max_dets <= 0:
                        continue
                    if int(det_idx) < 0 or int(det_idx) >= max_dets:
                        continue
                    det_present_mask[int(frame_pos), int(det_idx)] = True
                    det_topk_ids[int(frame_pos), int(det_idx), :] = track_topk_ids[track_pos, :]
                    det_topk_scores[int(frame_pos), int(det_idx), :] = track_topk_scores[track_pos, :]
                    if track_is_confident_top1[track_pos]:
                        det_is_confident_top1[int(frame_pos), int(det_idx)] = True

            # Frame-level aggregation (dedup by label_id)
            for frame_pos in range(n_frames):
                best_by_label: Dict[int, float] = {}
                for det_idx in range(max_dets):
                    if not det_present_mask[frame_pos, det_idx]:
                        continue
                    for k in range(TOP_K):
                        lid = int(det_topk_ids[frame_pos, det_idx, k])
                        sc = float(det_topk_scores[frame_pos, det_idx, k])
                        if lid < 0 or np.isnan(sc):
                            continue
                        prev = best_by_label.get(lid)
                        if prev is None or sc > prev:
                            best_by_label[lid] = sc
                if not best_by_label:
                    continue
                items = sorted(best_by_label.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
                for k, (lid, sc) in enumerate(items):
                    frame_topk_ids[frame_pos, k] = int(lid)
                    frame_topk_scores[frame_pos, k] = float(sc)
                top1 = float(frame_topk_scores[frame_pos, 0])
                if np.isfinite(top1) and top1 >= confidence_threshold_top1:
                    frame_is_confident_top1[frame_pos] = True
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            
            dets_present_cnt = int(np.sum(det_present_mask)) if det_present_mask is not None else 0
            if n_tracks <= 0 or dets_present_cnt <= 0:
                status = "empty"
                empty_reason = "no_logo_proposals"
            else:
                status = "ok"
                empty_reason = None

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
                "status": status,
                "empty_reason": empty_reason,
                # DB provenance
                "db_name": "embedding_service",
                "db_version": "v1",
                "db_digest": db_digest,
                "db_path": f"{embedding_client.base_url}/categories/{BRAND_CATEGORY}",
                "embedding_service_url": embedding_service_url,
                "brand_category": BRAND_CATEGORY,
                "embedding_model": embedding_model,
                "topk": TOP_K,
                "confidence_threshold_top1": confidence_threshold_top1,
                "proposal_classes": proposal_classes_list,
                "pad_ratio": pad_ratio,
                "use_sharpness": use_sharpness,
                "max_tracks": max_tracks,
                "max_dets_per_track": max_dets_per_frame,
                "tracks_total": int(n_tracks),
                "tracks_present": int(np.sum(track_present_mask)),
                "dets_present": dets_present_cnt,
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
                # label space (shared for all videos)
                "semantic_label_names": semantic_label_names,
                "semantic_object_ids": semantic_object_ids,
                "threshold_per_label_arr": threshold_per_label_arr,
                "track_ids": track_ids_arr,
                "track_present_mask": track_present_mask,
                "track_topk_ids": track_topk_ids,
                "track_topk_scores": track_topk_scores,
                "track_is_confident_top1": track_is_confident_top1,
                "frame_topk_ids": frame_topk_ids,
                "frame_topk_scores": frame_topk_scores,
                "frame_is_confident_top1": frame_is_confident_top1,
                "det_present_mask": det_present_mask,
                "det_topk_ids": det_topk_ids,
                "det_topk_scores": det_topk_scores,
                "det_is_confident_top1": det_is_confident_top1,
                "track_best_frame_pos": track_best_frame_pos,
                "track_best_det_idx": track_best_det_idx,
                "track_best_bbox_xyxy": track_best_bbox_xyxy,
                "track_best_det_score": track_best_det_score,
                "track_best_class_id": track_best_class_id,
                "meta": np.asarray(save_metadata, dtype=object),
                "meta_json": np.asarray(
                    json.dumps(save_metadata, ensure_ascii=False, sort_keys=True),
                    dtype="U",
                ),
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
                    f"brand_semantics | batch | saved artifact failed validation: "
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
            f"brand_semantics | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"brand_semantics | batch | error: {e}")
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

