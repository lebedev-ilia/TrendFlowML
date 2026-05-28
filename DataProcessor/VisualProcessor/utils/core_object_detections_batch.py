"""
Batch processing utilities for core_object_detections component.

Stage 3: GPU batching для core_object_detections с гибридным подходом:
- Сбор кадров из всех видео
- Группировка в батчи по max_frames_per_batch
- Последовательная обработка батчей через Triton или ultralytics
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
import cv2
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
from utils.resource_probe import pick_device

# Import core_object_detections functions
_core_object_detections_path = _visual_processor_path / "core" / "model_process" / "core_object_detections"
sys.path.insert(0, str(_core_object_detections_path.parent.parent.parent))

logger = get_logger("VisualProcessor.core_object_detections_batch")

# Import from core_object_detections/main.py
_core_object_detections_main = _core_object_detections_path / "main.py"
if _core_object_detections_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("core_object_detections_main", str(_core_object_detections_main))
    if spec and spec.loader:
        core_object_detections_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_object_detections_module)
        
        # Import functions
        _prep_yolo_tensor_from_rgb_uint8 = getattr(core_object_detections_module, "_prep_yolo_tensor_from_rgb_uint8", None)
        _scale_boxes_back = getattr(core_object_detections_module, "_scale_boxes_back", None)
        _load_final_taxonomy_v1_classes = getattr(core_object_detections_module, "_load_final_taxonomy_v1_classes", None)
        _load_triton_spec_via_model_manager = getattr(core_object_detections_module, "_load_triton_spec_via_model_manager", None)
        MAX_DETECTIONS = getattr(core_object_detections_module, "MAX_DETECTIONS", 100)
        BBOX_DIMS = getattr(core_object_detections_module, "BBOX_DIMS", 4)
        NAME = getattr(core_object_detections_module, "NAME", "core_object_detections")
        VERSION = getattr(core_object_detections_module, "VERSION", "2.1")
        SCHEMA_VERSION = getattr(core_object_detections_module, "SCHEMA_VERSION", "core_object_detections_npz_v1")
        _PERSON_CLASS_ID = getattr(core_object_detections_module, "_PERSON_CLASS_ID", 0)
        _LOGO_REGION_CLASS_ID = getattr(core_object_detections_module, "_LOGO_REGION_CLASS_ID", 33)
        _TEXT_REGION_CLASS_ID = getattr(core_object_detections_module, "_TEXT_REGION_CLASS_ID", 34)
        ARTIFACT_FILENAME = "detections.npz"
    else:
        raise ImportError("Failed to load core_object_detections module")
else:
    raise ImportError(f"core_object_detections/main.py not found at {_core_object_detections_main}")


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


def process_core_object_detections_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Batch processing для core_object_detections с гибридным подходом.
    
    Stage 3: GPU batching для core_object_detections.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация core_object_detections
        max_frames_per_batch: Максимальное количество кадров в одном батче (None = без лимита)
        batch_size: Размер батча для inference
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    runtime = config.get("runtime", "ultralytics")
    logger.info(
        f"core_object_detections | batch | processing {len(video_contexts)} videos "
        f"(max_frames_per_batch={max_frames_per_batch}, batch_size={batch_size}, runtime={runtime})"
    )
    
    start_time = time.perf_counter()

    def _sha256_file(path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    
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
                frame_indices = _get_frame_indices(metadata, NAME)
            except Exception as e:
                logger.error(f"core_object_detections | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
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
                logger.warning(f"core_object_detections | batch | video {video_ctx.video_id} has no frame_indices")
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
            uts = metadata.get("union_timestamps_sec")
            if uts is None:
                raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
                raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Загружаем кадры и сохраняем маппинг
            video_frame_start_idx = len(all_frames)  # Начальный индекс в общем батче для этого видео
            for frame_idx in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx)
                    # Сохраняем кадр в общий батч
                    all_frames.append((video_idx, frame_idx, frame))
                except Exception as e:
                    logger.warning(
                        f"core_object_detections | batch | video {video_ctx.video_id} failed to load frame {frame_idx}: {e}"
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
            logger.exception(f"core_object_detections | batch | video {video_ctx.video_id} failed to prepare: {e}")
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
        logger.error("core_object_detections | batch | no frames collected from any video")
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
    
    logger.info(f"core_object_detections | batch | collected {len(all_frames)} frames from {len(frames_by_video)} videos")
    
    # Helper functions for Triton decoding
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """Convert xywh to xyxy format."""
        x = xywh[:, 0]
        y = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    
    def _nms_single_class(boxes_xyxy: np.ndarray, scores_: np.ndarray, iou_th: float, max_det: int) -> List[int]:
        """NMS for single class."""
        if boxes_xyxy.size == 0:
            return []
        order = scores_.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0 and len(keep) < int(max_det):
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            # IOU calculation
            ax1, ay1, ax2, ay2 = boxes_xyxy[i]
            bx1 = boxes_xyxy[rest, 0]
            by1 = boxes_xyxy[rest, 1]
            bx2 = boxes_xyxy[rest, 2]
            by2 = boxes_xyxy[rest, 3]
            inter_x1 = np.maximum(ax1, bx1)
            inter_y1 = np.maximum(ay1, by1)
            inter_x2 = np.minimum(ax2, bx2)
            inter_y2 = np.minimum(ay2, by2)
            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
            union = area_a + area_b - inter_area + 1e-12
            ious = inter_area / union
            order = rest[ious <= float(iou_th)]
        return keep
    
    def _decode_and_nms_yolo(out_b84n: np.ndarray, box_threshold: float, iou_threshold: float) -> np.ndarray:
        """
        Decode YOLO output and apply NMS.
        out_b84n: (1,84,N)
        returns det: (M,6) [x1,y1,x2,y2,conf,cls_id]
        """
        if out_b84n.ndim != 3 or out_b84n.shape[0] != 1 or out_b84n.shape[1] < 6:
            raise RuntimeError(f"{NAME} | unexpected YOLO output shape: {out_b84n.shape}")
        pred = out_b84n[0].T  # (N,84)
        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)  # (N,nc)
        cls_id = np.argmax(cls_scores, axis=1).astype(np.int32)
        conf = np.max(cls_scores, axis=1).astype(np.float32)
        m = conf >= float(box_threshold)
        if not np.any(m):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh[m])
        conf_m = conf[m]
        cls_m = cls_id[m]
        
        dets: List[np.ndarray] = []
        for c in np.unique(cls_m):
            idx = np.where(cls_m == c)[0]
            if idx.size == 0:
                continue
            keep = _nms_single_class(boxes_xyxy[idx], conf_m[idx], float(iou_threshold), int(MAX_DETECTIONS))
            if not keep:
                continue
            kk = idx[np.asarray(keep, dtype=np.int64)]
            cc = np.full((kk.size, 1), float(c), dtype=np.float32)
            dets.append(np.concatenate([boxes_xyxy[kk], conf_m[kk, None], cc], axis=1))
        
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        det = np.concatenate(dets, axis=0)
        # global top-k by conf
        order = det[:, 4].argsort()[::-1]
        det = det[order[: int(MAX_DETECTIONS)]]
        return det.astype(np.float32)
    
    # Этап 2: Инициализация модели и обработка батчей
    try:
        device = pick_device(config.get("device", "auto"))
        box_threshold = float(config.get("box_threshold", 0.6))
        iou_threshold = float(config.get("iou_threshold", 0.3))

        # Stable taxonomy (v1.0, 41 classes). We always emit full id->name mapping in NPZ.
        if callable(_load_final_taxonomy_v1_classes):
            base_taxonomy: Dict[int, str] = _load_final_taxonomy_v1_classes()
        else:
            base_taxonomy = {i: f"class_{i}" for i in range(41)}
        
        # Определяем размер батча
        effective_batch_size = max_frames_per_batch if max_frames_per_batch else batch_size
        
        n_frames = len(all_frames)
        boxes_out = np.zeros((n_frames, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
        scores_out = np.zeros((n_frames, MAX_DETECTIONS), dtype=np.float32)
        class_ids_out = np.zeros((n_frames, MAX_DETECTIONS), dtype=np.int32)
        valid_mask_out = np.zeros((n_frames, MAX_DETECTIONS), dtype=bool)
        # start from stable mapping, then allow model-provided names to override.
        class_names: Dict[int, str] = dict(base_taxonomy)
        raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in range(n_frames)]
        
        if runtime == "triton":
            # Triton runtime
            triton_http_url = config.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL")
            triton_model_spec = config.get("triton_model_spec")
            
            if triton_model_spec:
                mm_entry = _load_triton_spec_via_model_manager(str(triton_model_spec))
                client = mm_entry["client"]
                rp = mm_entry["rp"]
                triton_http_url = str(rp.get("triton_http_url") or triton_http_url or "")
                triton_model_name = str(rp.get("triton_model_name") or config.get("triton_model_name") or "")
                triton_model_version = str(rp.get("triton_model_version") or "") or None
                triton_input_name = str(rp.get("triton_input_name") or config.get("triton_input_name", "images"))
                triton_output_name = str(rp.get("triton_output_name") or config.get("triton_output_name", "output0"))
            else:
                if not triton_http_url:
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec")
                triton_model_name = str(config.get("triton_model_name") or "")
                if not triton_model_name:
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec")
                triton_model_version = config.get("triton_model_version")
                triton_input_name = str(config.get("triton_input_name", "images"))
                triton_output_name = str(config.get("triton_output_name", "output0"))
                from dp_triton import TritonHttpClient, TritonError
                client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=10.0)
                if not client.ready():
                    raise TritonError(f"{NAME} | Triton is not ready at {triton_http_url}", error_code="triton_unavailable")
            
            # Определяем input_size из preset
            preset = str(config.get("triton_preprocess_preset", "yolo11x_640")).strip().lower()
            if preset == "yolo11x_320":
                input_size = 320
            elif preset == "yolo11x_640":
                input_size = 640
            elif preset == "yolo11x_960":
                input_size = 960
            else:
                raise RuntimeError(f"{NAME} | unknown triton_preprocess_preset: {preset}")
            
            # Загружаем class names
            resolved_model_path = str(config.get("model", ""))
            if resolved_model_path and os.path.exists(resolved_model_path):
                try:
                    from ultralytics import YOLO
                    y = YOLO(str(resolved_model_path))
                    model_class_names = getattr(y, "names", {})
                    if model_class_names:
                        class_names = {int(k): str(v) for k, v in model_class_names.items()}
                    else:
                        class_names = _load_final_taxonomy_v1_classes()
                except Exception:
                    class_names = _load_final_taxonomy_v1_classes()
            else:
                class_names = _load_final_taxonomy_v1_classes()
            
            logger.info(f"core_object_detections | batch | processing {n_frames} frames in batches of {effective_batch_size} (Triton)")
            
            # Обработка батчей через Triton
            start = 0
            while start < n_frames:
                batch_end = min(start + effective_batch_size, n_frames)
                batch_frames = all_frames[start:batch_end]
                
                # Обрабатываем каждый кадр отдельно (Triton YOLO обычно batch=1)
                for i_local, (video_idx, frame_idx, frame) in enumerate(batch_frames):
                    global_idx = start + i_local
                    
                    # Preprocess
                    x, r, pad, orig_hw = _prep_yolo_tensor_from_rgb_uint8(frame, input_size=int(input_size))
                    
                    # Inference через Triton
                    try:
                        res = client.infer(
                            model_name=str(triton_model_name),
                            model_version=str(triton_model_version) if triton_model_version else None,
                            input_name=str(triton_input_name),
                            input_tensor=x,
                            output_name=str(triton_output_name),
                            datatype="FP32",
                        )
                    except Exception as e:
                        raise RuntimeError(f"{NAME} | batch | Triton infer failed: {e}") from e
                    
                    out = np.asarray(res.output, dtype=np.float32)  # expected (1,84,N)
                    det_np = _decode_and_nms_yolo(out, box_threshold, iou_threshold)
                    if det_np.size == 0:
                        continue
                    
                    # Scale boxes back to original
                    det_np[:, :4] = _scale_boxes_back(det_np[:, :4], r=float(r), pad=pad, orig_hw=orig_hw)
                    
                    # Сохраняем детекции
                    detections: List[List[float]] = []
                    for j in range(min(det_np.shape[0], MAX_DETECTIONS)):
                        xyxy = det_np[j, :4].astype(np.float32)
                        conf = float(det_np[j, 4])
                        cls_id = int(det_np[j, 5])
                        if conf < float(box_threshold):
                            continue
                        boxes_out[global_idx, j] = xyxy
                        scores_out[global_idx, j] = conf
                        class_ids_out[global_idx, j] = cls_id
                        valid_mask_out[global_idx, j] = True
                        detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])
                        
                        if cls_id not in class_names:
                            class_names[cls_id] = f"class_{cls_id}"
                    
                    raw_per_frame[global_idx] = np.asarray(detections, dtype=np.float32)
                
                if start % (effective_batch_size * 10) == 0:
                    logger.info(f"core_object_detections | batch | processed {batch_end}/{n_frames} frames")
                
                start = batch_end
        
        else:
            # Ultralytics runtime
            model_path = str(config.get("model", "visual/yolo/yolo11x_41_best.pt"))
            if model_path and not os.path.exists(model_path):
                mr = os.environ.get("DP_MODELS_ROOT")
                if mr and not os.path.isabs(model_path):
                    cand = os.path.join(str(mr), str(model_path))
                    if os.path.exists(cand):
                        model_path = cand
                        logger.info(f"{NAME} | batch | Resolved model path via DP_MODELS_ROOT: {model_path}")
            
            if not model_path or not os.path.exists(model_path):
                raise RuntimeError(
                    f"{NAME} | batch | Model file not found: {config.get('model')} (resolved: {model_path}). "
                    f"Please ensure DP_MODELS_ROOT is set or provide absolute path."
                )
            
            logger.info(f"{NAME} | batch | Using model: {model_path}")
            
            # ============================================================
            # ОПТИМИЗАЦИЯ 1: Загружаем модель ОДИН РАЗ перед всеми батчами
            # (критично для производительности - загрузка модели очень дорогая!)
            # ============================================================
            from ultralytics import YOLO
            logger.info(f"{NAME} | batch | Loading YOLO model (this may take a moment)...")
            model = YOLO(model_path)
            logger.info(f"{NAME} | batch | Model loaded successfully")
            
            logger.info(f"core_object_detections | batch | processing {n_frames} frames in batches of {effective_batch_size} (Ultralytics)")
            
            # ============================================================
            # ОПТИМИЗАЦИЯ 2: Используем уже загруженные кадры из all_frames
            # Конвертируем RGB->BGR для всех кадров заранее (batch conversion)
            # ============================================================
            logger.info(f"{NAME} | batch | Converting RGB->BGR for {n_frames} frames...")
            all_frames_bgr = []
            for video_idx, frame_idx, frame_rgb in all_frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                all_frames_bgr.append((video_idx, frame_idx, frame_bgr))
            
            # ============================================================
            # ОПТИМИЗАЦИЯ 3: Обрабатываем все кадры из всех видео одним большим батчем
            # Это более эффективно, чем обработка по видео отдельно
            # (лучше использует GPU, меньше overhead на переключение контекста)
            # ============================================================
            start = 0
            while start < n_frames:
                batch_end = min(start + effective_batch_size, n_frames)
                batch_frames_data = all_frames_bgr[start:batch_end]
                
                # Извлекаем только кадры для inference
                batch_frames = [frame for _, _, frame in batch_frames_data]
                
                # Inference через ultralytics (модель уже загружена)
                try:
                    results = model(batch_frames, device=device, verbose=False)
                except Exception as e:
                    raise RuntimeError(f"{NAME} | batch | Ultralytics inference failed: {e}") from e
                
                # ============================================================
                # ОПТИМИЗАЦИЯ 4: Векторизованная обработка результатов
                # Используем batch операции numpy вместо поэлементной обработки
                # ============================================================
                for i_local, res in enumerate(results):
                    global_idx = start + i_local
                    
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    
                    # Contract (Audit v3): threshold affects ONLY valid_mask and derived curves;
                    # boxes/scores/class_ids must keep top-MAX detections (after NMS), even if score<threshold.
                    try:
                        boxes_data = res.boxes.data.cpu().numpy()  # (N, 6) [x1,y1,x2,y2,conf,cls]
                        if boxes_data.shape[0] == 0:
                            continue
                        # Prefer sorting by confidence desc (stable for downstream/debug).
                        order = np.argsort(boxes_data[:, 4])[::-1]
                        boxes_sorted = boxes_data[order]
                        n_take = int(min(boxes_sorted.shape[0], MAX_DETECTIONS))
                        take = boxes_sorted[:n_take]

                        # Fill fixed tensors
                        boxes_out[global_idx, :n_take] = take[:, :4].astype(np.float32)
                        scores_out[global_idx, :n_take] = take[:, 4].astype(np.float32)
                        cls = take[:, 5].astype(np.int32) if take.shape[1] > 5 else np.zeros((n_take,), dtype=np.int32)
                        class_ids_out[global_idx, :n_take] = cls
                        valid_mask_out[global_idx, :n_take] = take[:, 4] >= float(box_threshold)

                        # raw_per_frame = only valid detections (above threshold)
                        vm = (take[:, 4] >= float(box_threshold))
                        raw_per_frame[global_idx] = take[vm][:, [0, 1, 2, 3, 4]].astype(np.float32) if np.any(vm) else np.zeros((0, 5), dtype=np.float32)

                        # Update class names from model if available
                        try:
                            names_map = getattr(res, "names", None)
                            if isinstance(names_map, dict) and names_map:
                                for cid in np.unique(cls).tolist():
                                    if int(cid) in names_map:
                                        class_names[int(cid)] = str(names_map[int(cid)])
                        except Exception:
                            pass
                    except Exception:
                        # Defensive fallback: per-element extraction
                        detections = []
                        for j in range(min(len(res.boxes), MAX_DETECTIONS)):
                            try:
                                conf = float(res.boxes.conf[j].item())
                                xyxy = res.boxes.xyxy[j].cpu().numpy().astype(np.float32)
                                cls_id = int(res.boxes.cls[j].item())
                            except Exception:
                                try:
                                    box = res.boxes.data[j].cpu().numpy()
                                    xyxy = box[:4].astype(np.float32)
                                    conf = float(box[4])
                                    cls_id = int(box[5]) if box.shape[0] > 5 else 0
                                except Exception:
                                    continue

                            boxes_out[global_idx, j] = xyxy
                            scores_out[global_idx, j] = conf
                            class_ids_out[global_idx, j] = int(cls_id)
                            valid_mask_out[global_idx, j] = bool(conf >= float(box_threshold))
                            if conf >= float(box_threshold):
                                detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])

                        raw_per_frame[global_idx] = np.asarray(detections, dtype=np.float32) if detections else np.zeros((0, 5), dtype=np.float32)
                
                if start % (effective_batch_size * 10) == 0:
                    logger.info(f"core_object_detections | batch | processed {batch_end}/{n_frames} frames")
                
                start = batch_end
            
            # Освобождаем память модели
            del model
            import gc
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        
        # Этап 3: Распределение результатов обратно по видео
        logger.info("core_object_detections | batch | distributing results back to videos")
        
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
            
            # Извлекаем детекции для этого видео используя сохраненные индексы
            video_frame_indices = video_info["frame_indices"]
            frame_start_idx = video_info.get("frame_start_idx", 0)
            frame_end_idx = video_info.get("frame_end_idx", len(boxes_out))
            
            if frame_start_idx >= frame_end_idx or frame_start_idx >= len(boxes_out):
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "error",
                    "error": "no frames processed (all frames failed to load)",
                })
                continue
            
            # Извлекаем детекции для этого видео
            video_boxes = boxes_out[frame_start_idx:frame_end_idx]
            video_scores = scores_out[frame_start_idx:frame_end_idx]
            video_class_ids = class_ids_out[frame_start_idx:frame_end_idx]
            video_valid_mask = valid_mask_out[frame_start_idx:frame_end_idx]
            video_times_s = video_info["times_s"]
            video_raw_per_frame = raw_per_frame[frame_start_idx:frame_end_idx]
            
            # Проверяем соответствие размеров
            n_video_frames = len(video_frame_indices)
            if len(video_boxes) != n_video_frames or len(video_times_s) != n_video_frames:
                logger.warning(
                    f"core_object_detections | batch | video {video_ctx.video_id} size mismatch: "
                    f"boxes={len(video_boxes)}, indices={len(video_frame_indices)}, times={len(video_times_s)}"
                )
                # Используем минимальный размер
                min_size = min(len(video_boxes), len(video_frame_indices), len(video_times_s))
                video_boxes = video_boxes[:min_size]
                video_scores = video_scores[:min_size]
                video_class_ids = video_class_ids[:min_size]
                video_valid_mask = video_valid_mask[:min_size]
                video_frame_indices = video_frame_indices[:min_size]
                video_times_s = video_times_s[:min_size]
                video_raw_per_frame = video_raw_per_frame[:min_size]
            
            # Сохраняем результаты в per-video rs_path
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, ARTIFACT_FILENAME)
            
            # Подготовка метаданных
            metadata = video_info["metadata"]
            total_frames = int(metadata.get("total_frames", 0))
            total_valid_detections = int(np.sum(video_valid_mask))
            
            if total_valid_detections == 0:
                status = "empty"
                empty_reason = "no_detections_above_threshold"
            else:
                status = "ok"
                empty_reason = None
            
            # Emit full stable mapping 0..40 even if only a subset of classes was seen.
            class_names_full = {i: str(class_names.get(i, base_taxonomy.get(i, f"class_{i}"))) for i in range(41)}
            class_names_arr = np.array([f"{k}:{v}" for k, v in sorted(class_names_full.items())], dtype="U")
            
            # Compute per-frame derived arrays (same contract as single-video provider).
            analysis_w = int(metadata.get("analysis_width") or 0)
            analysis_h = int(metadata.get("analysis_height") or 0)
            if analysis_w <= 0 or analysis_h <= 0:
                # Best-effort fallback: infer from first loaded frame for this video.
                try:
                    fr0 = all_frames[frame_start_idx][2]
                    analysis_h, analysis_w = int(fr0.shape[0]), int(fr0.shape[1])
                except Exception:
                    analysis_w, analysis_h = 1, 1

            denom_x = float(max(analysis_w - 1, 1))
            denom_y = float(max(analysis_h - 1, 1))
            boxes_norm = video_boxes.astype(np.float32).copy()
            boxes_norm[..., 0] /= denom_x
            boxes_norm[..., 2] /= denom_x
            boxes_norm[..., 1] /= denom_y
            boxes_norm[..., 3] /= denom_y
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0).astype(np.float32)

            centers_norm = np.zeros((video_boxes.shape[0], video_boxes.shape[1], 2), dtype=np.float32)
            centers_norm[..., 0] = ((video_boxes[..., 0] + video_boxes[..., 2]) / 2.0) / denom_x
            centers_norm[..., 1] = ((video_boxes[..., 1] + video_boxes[..., 3]) / 2.0) / denom_y
            centers_norm = np.clip(centers_norm, 0.0, 1.0).astype(np.float32)

            w_box = np.clip(video_boxes[..., 2] - video_boxes[..., 0], 0.0, None).astype(np.float32)
            h_box = np.clip(video_boxes[..., 3] - video_boxes[..., 1], 0.0, None).astype(np.float32)
            denom_area = float(max(int(analysis_w) * int(analysis_h), 1))
            areas_frac = (w_box * h_box / denom_area).astype(np.float32)

            det_count = np.sum(video_valid_mask, axis=1).astype(np.int32)
            person_mask = video_valid_mask & (video_class_ids == int(_PERSON_CLASS_ID))
            text_mask = video_valid_mask & (video_class_ids == int(_TEXT_REGION_CLASS_ID))
            logo_mask = video_valid_mask & (video_class_ids == int(_LOGO_REGION_CLASS_ID))

            person_count = np.sum(person_mask, axis=1).astype(np.int32)
            text_region_count = np.sum(text_mask, axis=1).astype(np.int32)
            logo_region_count = np.sum(logo_mask, axis=1).astype(np.int32)

            person_areas = np.where(person_mask, areas_frac, 0.0).astype(np.float32)
            text_areas = np.where(text_mask, areas_frac, 0.0).astype(np.float32)
            logo_areas = np.where(logo_mask, areas_frac, 0.0).astype(np.float32)

            sum_person_area_frac = np.sum(person_areas, axis=1).astype(np.float32)
            sum_text_area_frac = np.sum(text_areas, axis=1).astype(np.float32)
            sum_logo_area_frac = np.sum(logo_areas, axis=1).astype(np.float32)
            max_person_area_frac = np.max(person_areas, axis=1).astype(np.float32)
            max_text_area_frac = np.max(text_areas, axis=1).astype(np.float32)
            max_logo_area_frac = np.max(logo_areas, axis=1).astype(np.float32)

            impl = f"{runtime}:{config.get('triton_model_name', config.get('model', 'unknown'))}"
            if runtime == "triton":
                impl = f"triton:{config.get('triton_model_name', 'triton')}"
            else:
                impl = "yolo"

            save_metadata = {
                "producer": NAME,
                "producer_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "status": status,
                "empty_reason": empty_reason,
                "impl": impl,
                "model": config.get("model", ""),
                "box_threshold": box_threshold,
                "batch_size": int(batch_size),
                "device": str(device),
                "total_frames": total_frames,
                "total_detections": total_valid_detections,
                "platform_id": video_ctx.platform_id or metadata.get("platform_id"),
                "video_id": video_ctx.video_id,
                "run_id": video_ctx.run_id or metadata.get("run_id"),
                "sampling_policy_version": video_ctx.sampling_policy_version or metadata.get("sampling_policy_version"),
                "config_hash": video_ctx.config_hash or metadata.get("config_hash"),
                "dataprocessor_version": video_ctx.dataprocessor_version or metadata.get("dataprocessor_version") or "unknown",
            }
            
            # Models used
            if runtime == "triton":
                model_name = str(config.get("triton_model_name", "triton"))
                models_used = [
                    model_used(
                        model_name=model_name,
                        model_version=str(config.get("triton_model_version", "1")),
                        weights_digest="provided_by_deploy",
                        runtime="triton",
                        engine="triton",
                        precision="fp32",
                        device="cuda",
                    )
                ]
            else:
                model_name = str(config.get("model", ""))
                engine = "ultralytics"
                weights_digest = "unknown"
                try:
                    resolved = str(config.get("model", "") or "")
                    if resolved and not os.path.exists(resolved):
                        mr = os.environ.get("DP_MODELS_ROOT")
                        if mr and not os.path.isabs(resolved):
                            cand = os.path.join(str(mr), str(resolved))
                            if os.path.exists(cand):
                                resolved = cand
                    if resolved and os.path.exists(resolved):
                        weights_digest = _sha256_file(resolved)
                except Exception:
                    weights_digest = "unknown"
                models_used = [
                    model_used(
                        model_name=model_name,
                        model_version="unknown",
                        weights_digest=str(weights_digest),
                        runtime="inprocess",
                        engine=engine,
                        precision="fp32",
                        device=str(device),
                    )
                ]
            
            save_metadata["models_used"] = models_used
            save_metadata = apply_models_meta(save_metadata, models_used=models_used)
            
            # Stage timings
            timings = {
                "initialization": 0.0,
                "load_deps": 0.0,
                "process_frames": time.perf_counter() - start_time,
                "saving": 0.0,
                "total": time.perf_counter() - start_time,
            }
            stage_timings_ms = {k: float(v) * 1000.0 for k, v in timings.items()}
            save_metadata["stage_timings_ms"] = stage_timings_ms
            
            # Сохранение NPZ
            meta_json_str = json.dumps(save_metadata, ensure_ascii=False, default=str)
            npz_dict = {
                "meta": np.asarray(save_metadata, dtype=object),
                "meta_json": np.array(meta_json_str, dtype="U"),
                "frame_indices": np.asarray(video_frame_indices, dtype=np.int32),
                "times_s": video_times_s.astype(np.float32),
                "boxes": video_boxes.astype(np.float32),
                "boxes_norm": boxes_norm,
                "centers_norm": centers_norm,
                "areas_frac": areas_frac,
                "scores": video_scores.astype(np.float32),
                "class_ids": video_class_ids.astype(np.int32),
                "valid_mask": video_valid_mask,
                "class_names": class_names_arr,
                "det_count": det_count,
                "person_count": person_count,
                "text_region_count": text_region_count,
                "logo_region_count": logo_region_count,
                "sum_person_area_frac": sum_person_area_frac,
                "max_person_area_frac": max_person_area_frac,
                "sum_text_area_frac": sum_text_area_frac,
                "max_text_area_frac": max_text_area_frac,
                "sum_logo_area_frac": sum_logo_area_frac,
                "max_logo_area_frac": max_logo_area_frac,
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
                    f"core_object_detections | batch | saved artifact failed validation: "
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
            f"core_object_detections | batch | completed in {duration:.2f}s "
            f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
        )
        
        return results
        
    except Exception as e:
        logger.exception(f"core_object_detections | batch | error: {e}")
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

