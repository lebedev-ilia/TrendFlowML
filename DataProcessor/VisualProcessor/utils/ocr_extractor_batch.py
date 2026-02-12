"""
Batch processing utilities for ocr_extractor component.

Stage 3: CPU parallelism для ocr_extractor с гибридным подходом:
- Сбор всех bbox-кропов из всех видео
- Параллельная обработка через ThreadPoolExecutor
- Распределение результатов обратно по видео
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta
from utils.artifact_validator import validate_npz

# Import ocr_extractor functions
_ocr_extractor_path = _visual_processor_path / "core" / "model_process" / "ocr_extractor"
sys.path.insert(0, str(_ocr_extractor_path.parent.parent.parent))

logger = get_logger("VisualProcessor.ocr_extractor_batch")

# Import from ocr_extractor/main.py
_ocr_main = _ocr_extractor_path / "main.py"
if _ocr_main.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ocr_extractor_main", str(_ocr_main))
    if spec and spec.loader:
        ocr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ocr_module)
        
        # Import functions
        _load_npz = getattr(ocr_module, "_load_npz", None)
        _norm_text = getattr(ocr_module, "_norm_text", None)
        _require_frame_indices = getattr(ocr_module, "_require_frame_indices", None)
        _class_id_map = getattr(ocr_module, "_class_id_map", None)
        _crop_rgb = getattr(ocr_module, "_crop_rgb", None)
        _run_tesseract = getattr(ocr_module, "_run_tesseract", None)
        atomic_save_npz = getattr(ocr_module, "atomic_save_npz", None)
        NAME = getattr(ocr_module, "NAME", "ocr_extractor")
        VERSION = getattr(ocr_module, "VERSION", "0.1")
        SCHEMA_VERSION = getattr(ocr_module, "SCHEMA_VERSION", "ocr_extractor_npz_v1")
    else:
        raise ImportError("Failed to load ocr_extractor module")
else:
    raise ImportError(f"ocr_extractor/main.py not found at {_ocr_main}")


def _process_single_crop(
    crop: np.ndarray,
    video_idx: int,
    frame_idx: int,
    box_idx: int,
    lang: str,
    psm: int,
) -> Optional[Dict[str, Any]]:
    """Обработать один bbox-кроп через tesseract."""
    try:
        txt = _run_tesseract(crop, lang=lang, psm=psm)
        txt_raw = str(txt or "").strip()
        txt_norm = _norm_text(txt_raw) if txt_raw else ""
        if not txt_norm:
            return None
        return {
            "video_idx": video_idx,
            "frame_idx": frame_idx,
            "box_idx": box_idx,
            "text_raw": txt_raw,
            "text_norm": txt_norm,
        }
    except Exception as e:
        logger.warning(f"ocr_extractor | batch | failed to process crop: {e}")
        return None


def process_ocr_extractor_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для ocr_extractor с гибридным подходом.
    
    Stage 3: CPU parallelism для ocr_extractor.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация ocr_extractor
        max_workers: Максимальное количество потоков для параллельной обработки (None = auto)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    # Проверяем наличие tesseract
    if shutil.which("tesseract") is None:
        logger.warning("ocr_extractor | batch | tesseract not found; writing empty artifacts")
        results = []
        for video_ctx in video_contexts:
            try:
                metadata = video_ctx.load_metadata()
                frame_indices = _require_frame_indices(metadata)
                uts = metadata.get("union_timestamps_sec")
                if uts is None:
                    raise RuntimeError("metadata.json missing union_timestamps_sec")
                uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
                fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
                times_s = uts_arr[fi_np].astype(np.float32)
                
                out_dir = os.path.join(video_ctx.rs_path, NAME)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "ocr.npz")
                
                timings = {"initialization": 0.0, "load_deps": 0.0, "process_frames": 0.0, "saving": 0.0, "total": 0.0}
                stage_timings_ms = {k: v * 1000.0 for k, v in timings.items()}
                
                meta_out = {
                    "producer": NAME,
                    "producer_version": VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "status": "empty",
                    "empty_reason": "dependency_missing",
                    "platform_id": metadata.get("platform_id") or "unknown",
                    "video_id": metadata.get("video_id") or "unknown",
                    "run_id": metadata.get("run_id") or "unknown",
                    "config_hash": metadata.get("config_hash") or "unknown",
                    "sampling_policy_version": metadata.get("sampling_policy_version") or "unknown",
                    "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                    "engine": "tesseract",
                    "tesseract_lang": str(config.get("tesseract_lang", "eng+rus")),
                    "tesseract_psm": int(config.get("tesseract_psm", 6)),
                    "models_used": [],
                    "stage_timings_ms": stage_timings_ms,
                }
                meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
                
                atomic_save_npz(
                    out_path,
                    frame_indices=fi_np,
                    times_s=times_s,
                    ocr_raw=np.asarray([], dtype=object),
                    meta=np.asarray(meta_out, dtype=object),
                )
                
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "ok",
                    "out_path": out_path,
                })
            except Exception as e:
                logger.exception(f"ocr_extractor | batch | video {video_ctx.video_id} failed: {e}")
                results.append({
                    "video_id": video_ctx.video_id,
                    "status": "error",
                    "error": str(e),
                })
        return results
    
    logger.info(
        f"ocr_extractor | batch | processing {len(video_contexts)} videos "
        f"(max_workers={max_workers})"
    )
    
    start_time = time.perf_counter()
    
    # Параметры конфигурации
    proposal_class = str(config.get("proposal_class", "text_region"))
    min_det_score = float(config.get("min_det_score", 0.5))
    max_boxes_per_frame = int(config.get("max_boxes_per_frame", 5))
    max_total_boxes = int(config.get("max_total_boxes", 5000))
    crop_margin_frac = float(config.get("crop_margin_frac", 0.02))
    tesseract_lang = str(config.get("tesseract_lang", "eng+rus"))
    tesseract_psm = int(config.get("tesseract_psm", 6))
    
    # Этап 1: Сбор всех bbox-кропов с привязкой к видео
    crops_by_video: List[Dict[str, Any]] = []
    all_crops: List[Tuple[int, int, int, int, np.ndarray]] = []  # (video_idx, frame_idx, box_idx, frame_n_idx, crop)
    
    for video_idx, video_ctx in enumerate(video_contexts):
        try:
            # Загружаем метаданные
            metadata = video_ctx.load_metadata()
            total_frames = int(metadata.get("total_frames", 0))
            
            # Получаем frame_indices
            try:
                frame_indices = _require_frame_indices(metadata)
            except Exception as e:
                logger.error(f"ocr_extractor | batch | video {video_ctx.video_id} failed to get frame_indices: {e}")
                crops_by_video.append({
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
                logger.warning(f"ocr_extractor | batch | video {video_ctx.video_id} has no frame_indices")
                crops_by_video.append({
                    "video_idx": video_idx,
                    "video_id": video_ctx.video_id,
                    "frame_indices": [],
                    "frame_manager": None,
                    "times_s": None,
                    "status": "empty",
                })
                continue
            
            # Загружаем detections
            det_path = os.path.join(video_ctx.rs_path, "core_object_detections", "detections.npz")
            if not os.path.isfile(det_path):
                raise RuntimeError(f"core_object_detections/detections.npz not found for video {video_ctx.video_id}")
            
            det = _load_npz(det_path)
            fi_det = np.asarray(det.get("frame_indices"), dtype=np.int32).reshape(-1)
            if fi_det.size == 0:
                raise RuntimeError(f"core_object_detections.detections.npz missing frame_indices")
            if fi_det.shape[0] != len(frame_indices) or not np.all(fi_det == np.asarray(frame_indices, dtype=np.int32)):
                raise RuntimeError(f"frame_indices mismatch vs core_object_detections")
            
            boxes = np.asarray(det.get("boxes"), dtype=np.float32)  # (N,MAX,4)
            scores = np.asarray(det.get("scores"), dtype=np.float32)  # (N,MAX)
            class_ids = np.asarray(det.get("class_ids"), dtype=np.int32)  # (N,MAX)
            valid_mask = np.asarray(det.get("valid_mask"))  # (N,MAX)
            class_id_to_name = _class_id_map(det.get("class_names"))
            proposal_ids = {cid for cid, nm in class_id_to_name.items() if nm == proposal_class}
            
            if not proposal_ids and class_id_to_name:
                logger.warning(f"ocr_extractor | batch | video {video_ctx.video_id}: proposal class '{proposal_class}' not found")
            
            # Создаем FrameManager
            frame_manager = FrameManager(
                frames_dir=video_ctx.frames_dir,
                chunk_size=metadata.get("chunk_size", 32),
                cache_size=metadata.get("cache_size", 2),
            )
            
            # Получаем timestamps
            uts = metadata.get("union_timestamps_sec")
            if uts is None:
                raise RuntimeError("metadata.json missing union_timestamps_sec")
            uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
            fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
            times_s = uts_arr[fi_np].astype(np.float32)
            
            # Собираем bbox-кропы
            N, MAX = int(boxes.shape[0]), int(boxes.shape[1])
            total = 0
            video_crops = []
            
            for n_i in range(N):
                if total >= max_total_boxes:
                    break
                # select candidate boxes in this frame
                cand: List[Tuple[float, int]] = []  # (score_area, j)
                for j in range(MAX):
                    if not bool(valid_mask[n_i, j]):
                        continue
                    sc = float(scores[n_i, j])
                    if sc < min_det_score:
                        continue
                    if proposal_ids and int(class_ids[n_i, j]) not in proposal_ids:
                        continue
                    xyxy = boxes[n_i, j]
                    area = max(1.0, float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
                    cand.append((sc * area, j))
                if not cand:
                    continue
                cand.sort(reverse=True)
                cand = cand[:max_boxes_per_frame]
                fr_idx = int(frame_indices[n_i])
                frame_rgb = frame_manager.get(fr_idx)
                for _, j in cand:
                    if total >= max_total_boxes:
                        break
                    crop = _crop_rgb(frame_rgb, boxes[n_i, j], margin_frac=crop_margin_frac)
                    if crop is None:
                        continue
                    video_crops.append((video_idx, fr_idx, j, n_i, crop))
                    all_crops.append((video_idx, fr_idx, j, n_i, crop))
                    total += 1
            
            crops_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": frame_indices,
                "frame_manager": frame_manager,
                "times_s": times_s,
                "boxes": boxes,
                "scores": scores,
                "crops": video_crops,
                "status": "ok",
            })
            
        except Exception as e:
            logger.exception(f"ocr_extractor | batch | video {video_ctx.video_id} failed to prepare: {e}")
            crops_by_video.append({
                "video_idx": video_idx,
                "video_id": video_ctx.video_id,
                "frame_indices": [],
                "frame_manager": None,
                "times_s": None,
                "status": "error",
                "error": str(e),
            })
    
    if not all_crops:
        logger.warning("ocr_extractor | batch | no crops collected from any video")
        # Закрываем FrameManager
        for video_info in crops_by_video:
            if video_info.get("frame_manager"):
                try:
                    video_info["frame_manager"].close()
                except Exception:
                    pass
        return [
            {
                "video_id": ctx.video_id,
                "status": "ok" if ctx.video_id in [v.get("video_id") for v in crops_by_video if v.get("status") == "empty"] else "error",
                "out_path": None,
            }
            for ctx in video_contexts
        ]
    
    logger.info(f"ocr_extractor | batch | collected {len(all_crops)} crops from {len(crops_by_video)} videos")
    
    # Этап 2: Параллельная обработка всех кропов
    ocr_results: Dict[Tuple[int, int, int], Dict[str, Any]] = {}  # (video_idx, frame_idx, box_idx) -> result
    
    effective_max_workers = max_workers if max_workers else min(8, len(all_crops))
    
    logger.info(f"ocr_extractor | batch | processing {len(all_crops)} crops with {effective_max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        futures = {
            executor.submit(_process_single_crop, crop, video_idx, frame_idx, box_idx, tesseract_lang, tesseract_psm): (video_idx, frame_idx, box_idx, frame_n_idx)
            for video_idx, frame_idx, box_idx, frame_n_idx, crop in all_crops
        }
        
        for future in as_completed(futures):
            video_idx, frame_idx, box_idx, frame_n_idx = futures[future]
            try:
                result = future.result()
                if result:
                    ocr_results[(video_idx, frame_idx, box_idx)] = result
            except Exception as e:
                logger.warning(f"ocr_extractor | batch | failed to process crop for video_idx={video_idx}, frame_idx={frame_idx}, box_idx={box_idx}: {e}")
    
    logger.info(f"ocr_extractor | batch | processed {len(ocr_results)} crops successfully")
    
    # Этап 3: Распределение результатов обратно по видео и сохранение
    results = []
    
    for video_info in crops_by_video:
        if video_info.get("status") != "ok":
            results.append({
                "video_id": video_info["video_id"],
                "status": video_info.get("status", "error"),
                "error": video_info.get("error"),
            })
            continue
        
        video_idx = video_info["video_idx"]
        video_id = video_info["video_id"]
        frame_indices = video_info["frame_indices"]
        times_s = video_info["times_s"]
        boxes = video_info["boxes"]
        scores = video_info["scores"]
        crops = video_info["crops"]
        
        # Собираем OCR результаты для этого видео
        ocr_rows: List[Dict[str, Any]] = []
        for video_idx_crop, frame_idx, box_idx, frame_n_idx, crop in crops:
            if video_idx_crop != video_idx:
                continue
            key = (video_idx, frame_idx, box_idx)
            if key in ocr_results:
                result = ocr_results[key]
                frame_n_idx = next((i for i, (v, f, b, fn, _) in enumerate(crops) if v == video_idx and f == frame_idx and b == box_idx), 0)
                t = float(times_s[frame_n_idx]) if frame_n_idx < len(times_s) else 0.0
                ocr_rows.append({
                    "frame": frame_idx,
                    "time_s": t,
                    "bbox": [float(x) for x in boxes[frame_n_idx, box_idx].tolist()],
                    "text_raw": result["text_raw"],
                    "text_norm": result["text_norm"],
                    "det_confidence": float(scores[frame_n_idx, box_idx]),
                    "engine": "tesseract",
                    "lang": tesseract_lang,
                })
        
        # Сохраняем артефакт
        try:
            video_ctx = next(ctx for ctx in video_contexts if ctx.video_id == video_id)
            metadata = video_ctx.load_metadata()
            
            out_dir = os.path.join(video_ctx.rs_path, NAME)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "ocr.npz")
            
            timings = {
                "initialization": 0.0,
                "load_deps": 0.0,
                "process_frames": time.perf_counter() - start_time,
                "saving": 0.0,
                "total": time.perf_counter() - start_time,
            }
            stage_timings_ms = {k: v * 1000.0 for k, v in timings.items()}
            
            meta_out = {
                "producer": NAME,
                "producer_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "status": "ok" if ocr_rows else "empty",
                "empty_reason": None if ocr_rows else "no_text_available",
                "platform_id": metadata.get("platform_id") or "unknown",
                "video_id": metadata.get("video_id") or "unknown",
                "run_id": metadata.get("run_id") or "unknown",
                "config_hash": metadata.get("config_hash") or "unknown",
                "sampling_policy_version": metadata.get("sampling_policy_version") or "unknown",
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                "engine": "tesseract",
                "tesseract_lang": tesseract_lang,
                "tesseract_psm": tesseract_psm,
                "proposal_class": proposal_class,
                "models_used": [],
                "stage_timings_ms": stage_timings_ms,
            }
            meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
            
            t_save_start = time.perf_counter()
            atomic_save_npz(
                out_path,
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                times_s=times_s,
                ocr_raw=np.asarray(ocr_rows, dtype=object),
                meta=np.asarray(meta_out, dtype=object),
            )
            timings["saving"] = time.perf_counter() - t_save_start
            timings["total"] = time.perf_counter() - start_time
            
            # Update stage_timings_ms with final timings
            stage_timings_ms = {k: v * 1000.0 for k, v in timings.items()}
            meta_out["stage_timings_ms"] = stage_timings_ms
            
            # Validate artifact
            ok, issues, _ = validate_npz(out_path)
            if not ok:
                error_messages = [f"{i.level}: {i.message}" for i in issues if i.level == "error"]
                os.remove(out_path)
                raise RuntimeError(f"Artifact validation failed: {', '.join(error_messages)}")
            
            results.append({
                "video_id": video_id,
                "status": "ok",
                "out_path": out_path,
            })
            
        except Exception as e:
            logger.exception(f"ocr_extractor | batch | video {video_id} failed to save: {e}")
            results.append({
                "video_id": video_id,
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
    
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"ocr_extractor | batch | completed in {elapsed:.2f}s: "
        f"{len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful"
    )
    
    return results

