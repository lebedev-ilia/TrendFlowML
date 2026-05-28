"""
Batch processing utilities for face_identity component.

Stage 3: GPU batching для face_identity с гибридным подходом:
- Сбор кадров из всех видео
- Группировка лиц по видео
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

# Import face_identity functions
_face_identity_path = _visual_processor_path / "core" / "model_process" / "core_identity" / "face_identity"
sys.path.insert(0, str(_face_identity_path.parent.parent.parent.parent))

logger = get_logger("VisualProcessor.face_identity_batch")

# Import from face_identity
try:
    from core.model_process.core_identity.face_identity.utils.embedding_service_client import EmbeddingServiceClient
    from core.model_process.core_identity.face_identity.main import (
        _extract_face_bbox_from_landmarks,
        _crop_face,
    )
except ImportError:
    # Fallback: try utils directory, then direct import
    try:
        sys.path.insert(0, str(_face_identity_path / "utils"))
        from embedding_service_client import EmbeddingServiceClient
        sys.path.insert(0, str(_face_identity_path))
        from main import _extract_face_bbox_from_landmarks, _crop_face
    except ImportError:
        # Last fallback: direct import from root
        sys.path.insert(0, str(_face_identity_path))
        from embedding_service_client import EmbeddingServiceClient
        from main import _extract_face_bbox_from_landmarks, _crop_face

NAME = "core_face_identity"
VERSION = "0.1"
SCHEMA_VERSION = "core_face_identity_npz_v1"
ARTIFACT_FILENAME = "face_identity.npz"
FACE_CATEGORY = "face"
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


def _require_frame_indices_from_landmarks(landmarks_npz: Dict[str, Any]) -> List[int]:
    """Extract frame indices from landmarks, filtering by face_present."""
    frame_indices_all = landmarks_npz.get("frame_indices")
    face_present_all = landmarks_npz.get("face_present")
    
    if frame_indices_all is None:
        raise RuntimeError(f"{NAME} | landmarks.npz missing frame_indices")
    if face_present_all is None:
        raise RuntimeError(f"{NAME} | landmarks.npz missing face_present")
    
    frame_indices_all = np.asarray(frame_indices_all, dtype=np.int32).reshape(-1)
    face_present_all = np.asarray(face_present_all, dtype=bool)
    
    # Filter: only frames where at least one face is present
    has_face = np.any(face_present_all, axis=1) if face_present_all.ndim > 1 else face_present_all
    frame_indices = frame_indices_all[has_face].tolist()
    
    return frame_indices


def process_face_identity_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для face_identity с гибридным подходом.
    
    Stage 3: GPU batching для face_identity.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация face_identity
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для Embedding Service (если поддерживается)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"face_identity | batch | processing {len(video_contexts)} videos "
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
    
    # Fail-fast: проверка доступности Embedding Service
    try:
        embedding_client._ensure_url()
    except Exception as e:
        raise RuntimeError(f"{NAME} | Embedding Service unavailable at {embedding_service_url}: {e}")
    
    # Параметры конфигурации
    topk = int(config.get("topk", TOP_K))
    similarity_threshold = float(config.get("similarity_threshold", 0.0))
    
    # Этап 1: Сбор всех лиц с привязкой к видео
    faces_by_video: List[Dict[str, Any]] = []
    all_faces: List[Tuple[int, int, int, np.ndarray, Dict[str, Any]]] = []  # (video_idx, frame_idx, face_idx, crop, metadata)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Загружаем landmarks
            landmarks_path = os.path.join(video_ctx.rs_path, "core_face_landmarks", "landmarks.npz")
            landmarks_npz = _load_npz(landmarks_path)
            
            # Извлекаем frame_indices (только кадры с лицами)
            frame_indices = _require_frame_indices_from_landmarks(landmarks_npz)
            
            if not frame_indices:
                # Valid empty: no faces in video
                faces_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "faces": [],
                    "status": "empty",
                    "empty_reason": "no_faces_in_video",
                })
                continue
            
            # Timestamps
            uts = (
                metadata.get("union_timestamps_sec")
                or metadata.get("union_timestamps_s")
                or metadata.get("times_s")
            )
            if uts is None:
                logger.error(f"face_identity | batch | video {video_ctx.video_id} missing union_timestamps_sec")
                faces_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "times_s": None,
                    "faces": [],
                    "status": "error",
                    "error": "missing union_timestamps_sec",
                })
                continue
            
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Извлекаем landmarks
            landmarks_fi_all = np.asarray(landmarks_npz.get("frame_indices"), dtype=np.int32).reshape(-1)
            face_landmarks_all = np.asarray(landmarks_npz.get("face_landmarks"), dtype=np.float32)
            face_present_all = np.asarray(landmarks_npz.get("face_present"), dtype=bool)
            
            # Создаем маппинг
            landmarks_fi_map: Dict[int, int] = {int(fi): int(i) for i, fi in enumerate(landmarks_fi_all)}
            
            # Фильтруем landmarks по frame_indices
            landmark_positions = [landmarks_fi_map[fi] for fi in frame_indices]
            face_landmarks = face_landmarks_all[landmark_positions]
            face_present = face_present_all[landmark_positions]
            
            max_faces = face_landmarks.shape[1] if len(face_landmarks.shape) > 1 else 1
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=int(metadata.get("chunk_size", 32)),
                cache_size=int(metadata.get("cache_size", 2)),
            )
            
            # Собираем все лица из этого видео
            video_faces: List[Tuple[int, int, np.ndarray, Dict[str, Any]]] = []
            
            for frame_idx_local, frame_idx_global in enumerate(frame_indices):
                try:
                    frame = frame_manager.get(frame_idx_global)
                except Exception as e:
                    logger.warning(f"face_identity | batch | video {video_ctx.video_id} failed to load frame {frame_idx_global}: {e}")
                    continue
                
                for face_idx in range(max_faces):
                    if face_idx >= face_present.shape[1] or not face_present[frame_idx_local, face_idx]:
                        continue
                    
                    # Извлекаем bbox из landmarks
                    landmarks = face_landmarks[frame_idx_local, face_idx]
                    bbox = _extract_face_bbox_from_landmarks(landmarks)
                    if bbox is None:
                        continue
                    
                    # Кропаем лицо
                    face_crop = _crop_face(frame, bbox)
                    if face_crop is None:
                        continue
                    
                    # Сохраняем для batch поиска
                    video_faces.append((
                        frame_idx_local,
                        face_idx,
                        face_crop,
                        {
                            "frame_idx_global": frame_idx_global,
                            "time_sec": float(times_s[frame_idx_local]),
                        }
                    ))
            
            frame_manager.close()
            
            faces_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "times_s": times_s.tolist(),
                "faces": video_faces,
                "status": "ok",
            })
            
            # Добавляем в общий список
            for frame_idx_local, face_idx, crop, metadata in video_faces:
                all_faces.append((video_idx, frame_idx_local, face_idx, crop, metadata))
                
        except Exception as e:
            logger.error(f"face_identity | batch | video {video_ctx.video_id} failed: {e}", exc_info=True)
            faces_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "times_s": None,
                "faces": [],
                "status": "error",
                "error": str(e),
            })
    
    if not all_faces:
        logger.warning("face_identity | batch | no faces found in any video")
        # Возвращаем empty результаты для всех видео
        results = []
        for video_data in faces_by_video:
            results.append({
                "video_id": video_data["video_id"],
                "status": video_data.get("status", "empty"),
                "empty_reason": video_data.get("empty_reason", "no_faces_in_video"),
            })
        return results
    
    # Этап 2: Batch поиск через Embedding Service
    logger.info(f"face_identity | batch | searching {len(all_faces)} faces via Embedding Service")
    
    # Группируем лица в батчи
    face_batches: List[List[Tuple[int, int, int, np.ndarray, Dict[str, Any]]]] = []
    current_batch: List[Tuple[int, int, int, np.ndarray, Dict[str, Any]]] = []
    
    for face_data in all_faces:
        current_batch.append(face_data)
        if len(current_batch) >= batch_size:
            face_batches.append(current_batch)
            current_batch = []
    
    if current_batch:
        face_batches.append(current_batch)
    
    # Выполняем batch поиск
    face_results_map: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    failed_faces = 0
    total_faces = len(all_faces)
    
    for batch_idx, batch in enumerate(face_batches):
        crops = [face_data[3] for face_data in batch]  # crop is at index 3
        
        try:
            # Используем search_batch если доступен
            if hasattr(embedding_client, "search_batch"):
                batch_results = embedding_client.search_batch(
                    category=FACE_CATEGORY,
                    images=crops,
                    top_k=topk,
                    similarity_threshold=similarity_threshold,
                )
            else:
                # Fallback: индивидуальные запросы
                batch_results = []
                for crop in crops:
                    try:
                        result = embedding_client.search(
                            category=FACE_CATEGORY,
                            image=crop,
                            top_k=topk,
                            similarity_threshold=similarity_threshold,
                        )
                        batch_results.append(result)
                    except Exception as e:
                        logger.warning(f"face_identity | batch | search failed for face: {e}")
                        batch_results.append([])
                        failed_faces += 1
            
            # Сохраняем результаты
            for face_data, results in zip(batch, batch_results):
                video_idx, frame_idx_local, face_idx, _, metadata = face_data
                face_results_map[(video_idx, frame_idx_local, face_idx)] = results
                
        except Exception as e:
            logger.error(f"face_identity | batch | batch search failed: {e}", exc_info=True)
            failed_faces += len(batch)
            for face_data in batch:
                video_idx, frame_idx_local, face_idx, _, _ = face_data
                face_results_map[(video_idx, frame_idx_local, face_idx)] = []
    
    # Fail-fast: если все лица упали с ошибками
    if total_faces > 0 and failed_faces == total_faces and not face_results_map:
        raise RuntimeError(
            f"{NAME} | All {total_faces} faces failed with Embedding Service errors. "
            "Service may be misconfigured or unavailable."
        )
    
    # Этап 3: Распределение результатов обратно по видео
    results = []
    
    for video_data in faces_by_video:
        if video_data["status"] != "ok":
            # Пропускаем видео с ошибками или empty
            results.append({
                "video_id": video_data["video_id"],
                "status": video_data.get("status", "empty"),
                "empty_reason": video_data.get("empty_reason", "no_faces_in_video"),
            })
            continue
        
        video_idx = video_data["video_idx"]
        frame_indices = video_data["frame_indices"]
        times_s = np.asarray(video_data["times_s"], dtype=np.float32)
        n_frames = len(frame_indices)
        
        # Подготавливаем выходные массивы
        face_ids = np.full((n_frames, topk), -1, dtype=np.int32)
        face_names = np.full((n_frames, topk), "", dtype="U256")
        face_similarities = np.full((n_frames, topk), 0.0, dtype=np.float32)
        
        # Группируем результаты по кадрам
        frame_results_map: Dict[int, List[Tuple[float, int, str]]] = {}  # frame_idx_local -> [(similarity, id, name), ...]
        
        for frame_idx_local, face_idx, _, metadata in video_data["faces"]:
            key = (video_idx, frame_idx_local, face_idx)
            search_results = face_results_map.get(key, [])
            
            for result in search_results:
                face_id = int(result.get("id", -1))
                face_name = str(result.get("name", ""))
                similarity = float(result.get("similarity", 0.0))
                
                if frame_idx_local not in frame_results_map:
                    frame_results_map[frame_idx_local] = []
                frame_results_map[frame_idx_local].append((similarity, face_id, face_name))
        
        # Дедуплицируем по имени и заполняем выходные массивы
        for frame_idx_local, frame_results in frame_results_map.items():
            # Сортируем по similarity
            frame_results.sort(key=lambda x: x[0], reverse=True)
            
            # Дедуплицируем по имени (берем лучший similarity для каждого имени)
            seen_names: Dict[str, Tuple[float, int, str]] = {}
            for similarity, face_id, face_name in frame_results:
                if face_name and face_name not in seen_names:
                    seen_names[face_name] = (similarity, face_id, face_name)
                elif face_name and similarity > seen_names[face_name][0]:
                    seen_names[face_name] = (similarity, face_id, face_name)
            
            # Заполняем выходные массивы
            deduplicated_results = sorted(seen_names.values(), key=lambda x: x[0], reverse=True)
            for k, (similarity, face_id, face_name) in enumerate(deduplicated_results[:topk]):
                face_ids[frame_idx_local, k] = face_id
                face_names[frame_idx_local, k] = face_name
                face_similarities[frame_idx_local, k] = similarity
        
        # Загружаем метаданные для сохранения
        video_ctx = video_contexts[video_idx]
        metadata = video_ctx.load_metadata()
        
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            logger.error(f"face_identity | batch | video {video_ctx.video_id} missing run identity keys: {missing}")
            results.append({
                "video_id": video_data["video_id"],
                "status": "error",
                "error": f"missing run identity keys: {missing}",
            })
            continue
        
        # Строим meta
        meta_out: Dict[str, Any] = {
            "producer": NAME,
            "producer_version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "ok",
            "empty_reason": None,
        }
        
        for k in required_run_keys:
            meta_out[k] = metadata.get(k)
        
        meta_out["dataprocessor_version"] = str(metadata.get("dataprocessor_version") or "unknown")
        meta_out["embedding_service_url"] = embedding_client.base_url
        meta_out["face_category"] = FACE_CATEGORY
        meta_out["topk"] = topk
        meta_out["similarity_threshold"] = similarity_threshold
        meta_out["total_faces_processed"] = len(video_data["faces"])
        meta_out["n_frames"] = n_frames
        
        # Models used
        models_used_list = [
            model_used(
                model_name="embedding_service",
                model_version="v1",
                runtime="http",
                engine="http",
                precision="fp32",
                device="cpu",
            )
        ]
        meta_out = apply_models_meta(meta_out, models_used=models_used_list)
        
        # Stage timings
        elapsed = time.perf_counter() - start_time
        meta_out["stage_timings_ms"] = {
            "initialization": 0.0,
            "load_deps": 0.0,
            "process_frames": float(elapsed * 1000.0),
            "saving": 0.0,
            "total": float(elapsed * 1000.0),
        }
        
        # Сохраняем NPZ
        out_dir = os.path.join(video_ctx.rs_path, NAME)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, ARTIFACT_FILENAME)
        
        _atomic_save_npz(
            out_path,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            times_s=times_s,
            face_ids=face_ids,
            face_names=face_names,
            face_similarities=face_similarities,
            meta=np.asarray(meta_out, dtype=object),
        )
        
        # Валидация
        try:
            validate_npz(out_path)
        except Exception as e:
            logger.warning(f"face_identity | batch | validation warning for {out_path}: {e}")
        
        results.append({
            "video_id": video_data["video_id"],
            "status": "ok",
            "artifact_path": out_path,
        })
    
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"face_identity | batch | completed in {elapsed:.2f}s "
        f"({len(video_contexts)} videos, {len(all_faces)} faces)"
    )
    
    return results

