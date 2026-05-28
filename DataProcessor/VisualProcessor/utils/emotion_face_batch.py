"""
Batch processing utilities for emotion_face component.

Stage 3: GPU batching для emotion_face с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи по max_frames_per_batch
- Последовательная обработка батчей через EmoNet
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import cv2
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.meta_builder import apply_models_meta, model_used
from utils.artifact_validator import validate_npz

logger = get_logger("VisualProcessor.emotion_face_batch")

# Import emotion_face functions
_emotion_face_path = _visual_processor_path / "modules" / "emotion_face"
sys.path.insert(0, str(_emotion_face_path))

# Import from emotion_face/_utils.py
_emotion_face_utils = _emotion_face_path / "_utils.py"
if _emotion_face_utils.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("emotion_face_utils", str(_emotion_face_utils))
    if spec and spec.loader:
        emotion_face_utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emotion_face_utils_module)
        
        # Import functions
        predict_emonet_batch = getattr(emotion_face_utils_module, "predict_emonet_batch", None)
    else:
        raise ImportError("Failed to load emotion_face utils module")
else:
    raise ImportError(f"emotion_face/_utils.py not found at {_emotion_face_utils}")

# Import from emotion_face/core/video_processor.py
_emotion_face_processor = _emotion_face_path / "core" / "video_processor.py"
if _emotion_face_processor.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("emotion_face_processor", str(_emotion_face_processor))
    if spec and spec.loader:
        emotion_face_processor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emotion_face_processor_module)
        
        # Import constants
        EMOTION_CLASSES = getattr(emotion_face_processor_module, "EMOTION_CLASSES", {
            0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
            4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
        })
    else:
        raise ImportError("Failed to load emotion_face processor module")
else:
    EMOTION_CLASSES = {
        0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
        4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
    }

# Module constants
MODULE_NAME = "emotion_face"
ARTIFACT_FILENAME = "emotion_face.npz"
SCHEMA_VERSION = "emotion_face_npz_v1"


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


def _get_frame_indices(metadata: Dict[str, Any], component_name: str) -> List[int]:
    """Получить frame_indices из метаданных."""
    block = metadata.get(component_name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{component_name} | metadata missing '{component_name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{component_name} | metadata '{component_name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices_raw]


def _load_emonet_model(config: Dict[str, Any]) -> Any:
    """Загрузить модель EmoNet из конфига."""
    emonet_model_spec = config.get("emonet_model_spec")
    emo_path = config.get("emo_path")
    device = config.get("device", "cuda")
    
    # Try ModelManager first
    if emonet_model_spec:
        try:
            from dp_models.manager import get_global_model_manager  # type: ignore
            mm = get_global_model_manager()
            resolved = mm.get(model_name=str(emonet_model_spec))
            return resolved.handle
        except Exception as e:
            logger.warning(f"emotion_face | batch | ModelManager failed for {emonet_model_spec}: {e}")
    
    # Fallback to explicit path
    if emo_path:
        import importlib.util
        # dp_models/emonet/emonet/models/emonet.py
        # Find DataProcessor root: utils/ -> VisualProcessor/ -> DataProcessor/
        utils_dir = os.path.dirname(__file__)
        data_processor_root = os.path.abspath(os.path.join(utils_dir, "..", ".."))
        emonet_py = os.path.join(data_processor_root, "dp_models", "emonet", "emonet", "models", "emonet.py")
        if not os.path.isfile(emonet_py):
            raise RuntimeError(f"EmoNet source file not found: {emonet_py}")
        spec = importlib.util.spec_from_file_location("_dp_vendor_emonet", emonet_py)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create import spec for EmoNet")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        EmoNet = getattr(mod, "EmoNet", None)
        if EmoNet is None:
            raise RuntimeError("EmoNet class not found in vendored emonet.py")
        
        state = torch.load(str(emo_path), map_location="cpu")
        if isinstance(state, dict):
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
        model = EmoNet(n_expression=8).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    
    raise RuntimeError("emotion_face | batch | no emonet_model_spec or emo_path provided")


def process_emotion_face_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Batch processing для emotion_face с гибридным подходом.
    
    Stage 3: GPU batching для emotion_face.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация emotion_face
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для inference
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"emotion_face | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size})"
    )
    
    start_time = time.perf_counter()
    
    # Параметры из конфига
    face_frame_stride = int(config.get("face_frame_stride", 4))
    max_frames = int(config.get("max_frames", 200))
    max_faces_per_frame = int(config.get("max_faces_per_frame", 2))
    face_bbox_margin = float(config.get("face_bbox_margin", 0.2))
    device = str(config.get("device", "cuda"))
    
    # Загружаем модель один раз для всех видео
    model = _load_emonet_model(config)
    
    # Этап 1: Сбор всех кадров с лицами из всех видео
    frames_by_video: List[Dict[str, Any]] = []
    all_crops: List[Tuple[int, int, int, np.ndarray]] = []  # (video_idx, frame_idx, face_slot, crop)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            
            # Загружаем core_face_landmarks
            try:
                from utils.results_store import ResultsStore
                rs = ResultsStore(video_ctx.rs_path)
                core_path = rs.get_component_path("core_face_landmarks", "landmarks.npz")
                if not core_path or not os.path.exists(core_path):
                    raise RuntimeError("core_face_landmarks/landmarks.npz not found")
                
                core_data = np.load(core_path, allow_pickle=True)
                face_fi = np.asarray(core_data["frame_indices"], dtype=np.int32).reshape(-1)
                face_present = np.asarray(core_data["face_present"], dtype=bool)
                face_landmarks = np.asarray(core_data["face_landmarks"], dtype=np.float32)
                
                if face_fi.size == 0:
                    raise RuntimeError("core_face_landmarks has empty frame_indices")
                if face_present.ndim != 2 or face_present.shape[0] != face_fi.shape[0]:
                    raise RuntimeError("core_face_landmarks face_present shape mismatch")
                if face_landmarks.ndim != 4 or face_landmarks.shape[0] != face_fi.shape[0]:
                    raise RuntimeError("core_face_landmarks face_landmarks shape mismatch")
                
            except Exception as e:
                logger.error(f"emotion_face | batch | video {video_ctx.video_id} failed to load core_face_landmarks: {e}")
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
            
            # Выбираем кадры с лицами
            any_face = np.any(face_present, axis=1)
            frames_with_face = face_fi[any_face]
            
            if frames_with_face.size == 0:
                logger.warning(f"emotion_face | batch | video {video_ctx.video_id} has no frames with faces")
                frames_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "empty",
                    "empty_reason": "no_faces_in_video",
                })
                continue
            
            # Применяем stride и max_frames
            stride = max(1, face_frame_stride)
            selected_fi = frames_with_face[::stride].astype(np.int32)
            if max_frames > 0 and selected_fi.size > max_frames:
                selected_fi = selected_fi[:max_frames]
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=metadata.get("chunk_size", 32),
                cache_size=metadata.get("cache_size", 2),
            )
            
            # Получаем timestamps
            uts = metadata.get("union_timestamps_sec") or metadata.get("times_s")
            if uts is None:
                raise RuntimeError("emotion_face | metadata.json missing union_timestamps_sec")
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            if np.any(selected_fi < 0) or np.any(selected_fi >= uts_arr.shape[0]):
                raise RuntimeError("emotion_face | selected frame_indices out of range")
            times_s = uts_arr[selected_fi].astype(np.float32)
            
            # Создаем маппинг frame_index -> position в core_face_landmarks
            fi_to_pos = {int(x): int(i) for i, x in enumerate(face_fi.tolist())}
            
            # Загружаем кадры и извлекаем crops лиц
            video_crop_start_idx = len(all_crops)
            face_count_per_frame = []
            
            for i, gidx in enumerate(selected_fi.tolist()):
                pos = fi_to_pos.get(int(gidx))
                if pos is None:
                    face_count_per_frame.append(0)
                    continue
                
                p = int(pos)
                present_row = face_present[p]
                if present_row.ndim != 1:
                    present_row = np.asarray(present_row, dtype=bool).reshape(-1)
                lm_row = face_landmarks[p]
                if lm_row.ndim != 3:
                    face_count_per_frame.append(0)
                    continue
                
                # Выбираем до max_faces_per_frame лиц (по размеру bbox)
                candidates: List[Tuple[float, int]] = []
                for f in range(int(present_row.shape[0])):
                    if not bool(present_row[f]):
                        continue
                    lm = lm_row[f]
                    if np.isnan(lm).all():
                        continue
                    xs = lm[:, 0]
                    ys = lm[:, 1]
                    area = float((np.nanmax(xs) - np.nanmin(xs)) * (np.nanmax(ys) - np.nanmin(ys)))
                    candidates.append((area, f))
                candidates.sort(reverse=True, key=lambda t: t[0])
                chosen = [f for _area, f in candidates[:max_faces_per_frame]]
                face_count_per_frame.append(len(chosen))
                
                if not chosen:
                    continue
                
                # Загружаем кадр
                frame = frame_manager.get(int(gidx))
                if frame is None:
                    continue
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                H, W = int(frame.shape[0]), int(frame.shape[1])
                
                # Извлекаем crops для каждого лица
                for slot, f in enumerate(chosen):
                    lm = lm_row[int(f)]
                    if np.isnan(lm).all():
                        continue
                    xs = np.clip(lm[:, 0], 0.0, 1.0) * W
                    ys = np.clip(lm[:, 1], 0.0, 1.0) * H
                    x0 = float(np.nanmin(xs)); x1 = float(np.nanmax(xs))
                    y0 = float(np.nanmin(ys)); y1 = float(np.nanmax(ys))
                    if not np.isfinite([x0, x1, y0, y1]).all():
                        continue
                    
                    # Расширяем bbox
                    mx = (x1 - x0) * face_bbox_margin
                    my = (y1 - y0) * face_bbox_margin
                    x0 = max(0.0, x0 - mx); x1 = min(float(W), x1 + mx)
                    y0 = max(0.0, y0 - my); y1 = min(float(H), y1 + my)
                    xi0, xi1 = int(x0), int(max(x0 + 1, x1))
                    yi0, yi1 = int(y0), int(max(y0 + 1, y1))
                    if xi1 <= xi0 or yi1 <= yi0:
                        continue
                    
                    crop = frame[yi0:yi1, xi0:xi1]
                    if crop.size == 0:
                        continue
                    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
                    all_crops.append((video_idx, int(gidx), slot, crop))
            
            video_crop_end_idx = len(all_crops)
            
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": selected_fi.tolist(),
                "frame_manager": frame_manager,
                "times_s": times_s,
                "face_count": face_count_per_frame,
                "crop_start_idx": video_crop_start_idx,
                "crop_end_idx": video_crop_end_idx,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"emotion_face | batch | video {video_ctx.video_id} failed to prepare: {e}")
            frames_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "frame_manager": None,
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_crops:
        logger.error("emotion_face | batch | no face crops collected from any video")
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
                "error": "no face crops collected",
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"emotion_face | batch | collected {len(all_crops)} face crops from {len(frames_by_video)} videos")
    
    # Этап 2: Батчинг и inference
    try:
        n_crops = len(all_crops)
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        logger.info(f"emotion_face | batch | processing {n_crops} face crops in batches of {effective_batch_size}")
        
        # Извлекаем crops
        crop_images = [crop for _, _, _, crop in all_crops]
        
        # Inference батчами
        all_results = []
        start = 0
        while start < n_crops:
            batch_end = min(start + effective_batch_size, n_crops)
            batch_crops = crop_images[start:batch_end]
            
            # Inference
            batch_results = predict_emonet_batch(
                batch_crops,
                model,
                batch_size=None,  # Use all crops in batch
                use_amp=True,
                temperature=1.0,
                face_confidence=None,
            )
            all_results.extend(batch_results)
            
            if start % (effective_batch_size * 10) == 0:
                logger.info(f"emotion_face | batch | processed {batch_end}/{n_crops} crops")
            
            start = batch_end
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("emotion_face | batch | distributing results back to videos")
        
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
            video_times_s = video_info["times_s"]
            video_face_count = video_info["face_count"]
            crop_start_idx = video_info.get("crop_start_idx", 0)
            crop_end_idx = video_info.get("crop_end_idx", len(all_results))
            
            if crop_start_idx >= crop_end_idx:
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "empty",
                    "empty_reason": "no crops processed",
                })
                continue
            
            # Группируем результаты по кадрам
            video_crops = all_crops[crop_start_idx:crop_end_idx]
            video_crop_results = all_results[crop_start_idx:crop_end_idx]
            
            # Создаем структуру данных для кадров
            max_faces = max_faces_per_frame
            N = len(video_frame_indices)
            valence_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
            arousal_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
            conf_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
            probs_faces = np.full((N, max_faces, 8), np.nan, dtype=np.float32)
            face_count_arr = np.array(video_face_count, dtype=np.int16)
            
            # Заполняем результаты
            crop_idx = 0
            for i, gidx in enumerate(video_frame_indices):
                for slot in range(video_face_count[i]):
                    if crop_idx >= len(video_crop_results):
                        break
                    r = video_crop_results[crop_idx]
                    valence_faces[i, slot] = float(r.get("valence", np.nan))
                    arousal_faces[i, slot] = float(r.get("arousal", np.nan))
                    conf_faces[i, slot] = float(r.get("emotion_confidence", np.nan))
                    probs = r.get("emotions") or {}
                    cls = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
                    probs_faces[i, slot, :] = np.asarray([float(probs.get(c, 0.0)) for c in cls], dtype=np.float32)
                    crop_idx += 1
            
            # Агрегируем по лицам (mean)
            valence = np.nanmean(valence_faces, axis=1).astype(np.float32)
            arousal = np.nanmean(arousal_faces, axis=1).astype(np.float32)
            emotion_confidence = np.nanmean(conf_faces, axis=1).astype(np.float32)
            emotion_probs = np.nanmean(probs_faces, axis=1).astype(np.float32)
            intensity = np.sqrt(np.square(valence) + np.square(arousal)).astype(np.float32)
            dominant_id = np.nanargmax(emotion_probs, axis=1).astype(np.int8) if emotion_probs.size else np.zeros((N,), dtype=np.int8)
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(MODULE_NAME)
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_ctx.load_metadata()
            
            save_metadata = {
                "producer": MODULE_NAME,
                "producer_version": "1.0",
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "platform_id": video_ctx.platform_id or metadata.get("platform_id"),
                "video_id": video_ctx.video_id,
                "run_id": video_ctx.run_id or metadata.get("run_id"),
                "sampling_policy_version": video_ctx.sampling_policy_version or metadata.get("sampling_policy_version"),
                "config_hash": video_ctx.config_hash or metadata.get("config_hash"),
                "dataprocessor_version": video_ctx.dataprocessor_version or metadata.get("dataprocessor_version") or "unknown",
                "status": "ok",
                "empty_reason": None,
                "total_frames": metadata.get("total_frames"),
                "processed_frames": N,
            }
            
            # Models used
            models_used = []
            emonet_model_spec = config.get("emonet_model_spec")
            if emonet_model_spec:
                try:
                    from dp_models.manager import get_global_model_manager
                    mm = get_global_model_manager()
                    resolved = mm.get(model_name=str(emonet_model_spec))
                    if resolved.models_used_entry:
                        models_used.append(resolved.models_used_entry)
                except Exception:
                    pass
            
            if not models_used:
                models_used.append(
                    model_used(
                        model_name=str(emonet_model_spec or "EmoNet"),
                        model_version="unknown",
                        weights_digest="unknown",
                        runtime="inprocess",
                        engine="torch",
                        precision="fp32",
                        device=device,
                    )
                )
            save_metadata["models_used"] = models_used
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Сохранение NPZ (плоская структура, как в BaseModule.save_results)
            # sequence_features сохраняем как вложенный dict через dtype=object
            sequence_features_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "valence": valence,
                "arousal": arousal,
                "intensity": intensity,
                "emotion_confidence": emotion_confidence,
                "emotion_probs": emotion_probs,
                "dominant_emotion_id": dominant_id,
                "face_count": face_count_arr,
                "valence_faces": valence_faces,
                "arousal_faces": arousal_faces,
                "emotion_confidence_faces": conf_faces,
                "emotion_probs_faces": probs_faces,
            }
            
            features_dict = {
                "valence_mean": float(np.nanmean(valence)) if np.isfinite(valence).any() else np.nan,
                "arousal_mean": float(np.nanmean(arousal)) if np.isfinite(arousal).any() else np.nan,
                "intensity_mean": float(np.nanmean(intensity)) if np.isfinite(intensity).any() else np.nan,
            }
            
            summary_dict = {
                "sequence_length": int(N),
                "faces_found_frames": int(np.sum(face_count_arr > 0)),
                "max_faces_per_frame": int(max_faces),
            }
            
            npz_dict = {
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "sequence_features": np.asarray(sequence_features_dict, dtype=object),
                "features": np.asarray(features_dict, dtype=object),
                "summary": np.asarray(summary_dict, dtype=object),
                "advanced_features": np.asarray({}, dtype=object),
                "keyframes": np.asarray([], dtype=object),
                "meta": np.asarray(save_metadata, dtype=object),
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
                    f"emotion_face | batch | saved artifact failed validation: "
                    + "; ".join([f"{i.level}:{i.message}" for i in issues])
                )
            
            results.append({
                "video_id": video_ctx.video_id,
                "status": "ok",
                "saved_path": npz_path,
            })
        
        # Закрываем FrameManager для всех видео
        for video_info in frames_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        
        duration = time.perf_counter() - start_time
        logger.info(
            f"emotion_face | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"emotion_face | batch | error: {e}")
        # Закрываем FrameManager в случае ошибки
        for video_info in frames_by_video:
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
    finally:
        # Очистка ресурсов
        if model is not None:
            del model
            torch.cuda.empty_cache()

