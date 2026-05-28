"""
Batch processing utilities for cut_detection module.

Stage 3: Batch processing для cut_detection с параллельной обработкой видео:
- Параллельная обработка каждого видео отдельно (CPU-bound операции)
- Изоляция артефактов между видео
- Корректная обработка метаданных для каждого видео
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.video_context import VideoContext
from utils.utilites import load_metadata

logger = get_logger("VisualProcessor.cut_detection_batch")


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


def _process_single_video_cut_detection(
    video_ctx: VideoContext,
    config: Dict[str, Any],
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Обработать одно видео через cut_detection.
    
    Args:
        video_ctx: VideoContext для видео
        config: Конфигурация cut_detection
        max_workers: Количество воркеров (не используется, для совместимости)
    
    Returns:
        Результат обработки с video_id и status
    """
    video_id = video_ctx.video_id
    frames_dir = video_ctx.frames_dir
    rs_path = video_ctx.rs_path
    
    try:
        # Загружаем метаданные
        metadata = video_ctx.load_metadata()
        if not metadata:
            return {
                "video_id": video_id,
                "status": "error",
                "error": "metadata.json not found or invalid",
            }
        
        # Получаем frame_indices для cut_detection
        try:
            frame_indices = _get_frame_indices(metadata, "cut_detection")
        except Exception as e:
            return {
                "video_id": video_id,
                "status": "error",
                "error": f"Failed to get frame_indices: {e}",
            }
        
        if not frame_indices:
            return {
                "video_id": video_id,
                "status": "error",
                "error": "cut_detection.frame_indices is empty",
            }
        
        # Подготавливаем аргументы для CLI
        cut_detection_main = _visual_processor_path / "modules" / "cut_detection" / "main.py"
        if not cut_detection_main.exists():
            return {
                "video_id": video_id,
                "status": "error",
                "error": f"cut_detection/main.py not found at {cut_detection_main}",
            }
        
        # Строим команду для subprocess
        cmd = [
            sys.executable,
            str(cut_detection_main),
            "--frames-dir", str(frames_dir),
            "--rs-path", str(rs_path),
        ]
        
        # Добавляем параметры из конфига
        if config.get("device"):
            cmd.extend(["--device", str(config["device"])])
        if config.get("use_clip"):
            cmd.append("--use-clip")
        if config.get("use_deep_features"):
            cmd.append("--use-deep-features")
        if config.get("use_adaptive_thresholds"):
            cmd.append("--use-adaptive-thresholds")
        if config.get("use_adaptive_thresholds") is False:
            cmd.append("--no-use-adaptive-thresholds")
        if config.get("use_semantic_clustering"):
            cmd.append("--use-semantic-clustering")
        if config.get("fade_threshold") is not None:
            cmd.extend(["--fade-threshold", str(config["fade_threshold"])])
        if config.get("min_duration_frames") is not None:
            cmd.extend(["--min-duration-frames", str(config["min_duration_frames"])])
        if config.get("use_flow_consistency"):
            cmd.append("--use-flow-consistency")
        if config.get("use_flow_consistency") is False:
            cmd.append("--no-use-flow-consistency")
        if config.get("ssim_max_side"):
            cmd.extend(["--ssim-max-side", str(config["ssim_max_side"])])
        if config.get("flow_max_side"):
            cmd.extend(["--flow-max-side", str(config["flow_max_side"])])
        if config.get("hard_cuts_preset"):
            cmd.extend(["--hard-cuts-preset", str(config["hard_cuts_preset"])])
        if config.get("hard_cuts_cascade"):
            cmd.append("--hard-cuts-cascade")
        if config.get("hard_cuts_cascade_keep_top_p"):
            cmd.extend(["--hard-cuts-cascade-keep-top-p", str(config["hard_cuts_cascade_keep_top_p"])])
        if config.get("hard_cuts_cascade_hist_margin"):
            cmd.extend(["--hard-cuts-cascade-hist-margin", str(config["hard_cuts_cascade_hist_margin"])])
        if config.get("max_sampling_gap_sec") is not None:
            cmd.extend(["--max-sampling-gap-sec", str(config["max_sampling_gap_sec"])])
        if config.get("prefer_core_optical_flow"):
            cmd.append("--prefer-core-optical-flow")
        if config.get("require_core_optical_flow"):
            cmd.append("--require-core-optical-flow")
        if config.get("no_require_core_optical_flow"):
            cmd.append("--no-require-core-optical-flow")
        if config.get("write_model_facing_npz"):
            cmd.append("--write-model-facing-npz")
        if config.get("require_model_facing_npz"):
            cmd.append("--require-model-facing-npz")
        if config.get("no_write_model_facing_npz"):
            cmd.append("--no-write-model-facing-npz")
        if config.get("clip_image_model_spec"):
            cmd.extend(["--clip-image-model-spec", str(config["clip_image_model_spec"])])
        if config.get("triton_http_url"):
            cmd.extend(["--triton-http-url", str(config["triton_http_url"])])
        if config.get("audio_path"):
            cmd.extend(["--audio-path", str(config["audio_path"])])
        
        # Запускаем subprocess
        logger.info(f"cut_detection | batch | processing video {video_id} ({len(frame_indices)} frames)")
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )
            
            elapsed = time.perf_counter() - start_time
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"cut_detection | batch | video {video_id} failed: {error_msg}")
                return {
                    "video_id": video_id,
                    "status": "error",
                    "error": error_msg[:500],  # Limit error message length
                    "elapsed_sec": elapsed,
                }
            
            logger.info(f"cut_detection | batch | video {video_id} completed in {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "ok",
                "elapsed_sec": elapsed,
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            logger.error(f"cut_detection | batch | video {video_id} timed out after {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "error",
                "error": "Timeout after 3600s",
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"cut_detection | batch | video {video_id} exception: {e}")
            return {
                "video_id": video_id,
                "status": "error",
                "error": str(e)[:500],
                "elapsed_sec": elapsed,
            }
            
    except Exception as e:
        logger.error(f"cut_detection | batch | video {video_id} failed: {e}")
        return {
            "video_id": video_id,
            "status": "error",
            "error": str(e)[:500],
        }


def process_cut_detection_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_video_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для cut_detection с параллельной обработкой видео.
    
    Stage 3: Batch processing для cut_detection.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация cut_detection
        max_video_workers: Максимальное количество параллельных воркеров для видео (None = последовательно)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"cut_detection | batch | processing {len(video_contexts)} videos "
        f"(max_video_workers={max_video_workers})"
    )
    
    start_time = time.perf_counter()
    
    # Параллельная обработка видео
    if max_video_workers and max_video_workers > 1:
        results = []
        with ThreadPoolExecutor(max_workers=max_video_workers) as executor:
            futures = {
                executor.submit(_process_single_video_cut_detection, video_ctx, config): video_ctx
                for video_ctx in video_contexts
            }
            
            for future in as_completed(futures):
                video_ctx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"cut_detection | batch | video {video_ctx.video_id} exception: {e}")
                    results.append({
                        "video_id": video_ctx.video_id,
                        "status": "error",
                        "error": str(e)[:500],
                    })
        
        # Сортируем результаты по порядку video_contexts
        results_dict = {r["video_id"]: r for r in results}
        results = [results_dict.get(ctx.video_id, {
            "video_id": ctx.video_id,
            "status": "error",
            "error": "Result not found",
        }) for ctx in video_contexts]
    else:
        # Последовательная обработка
        results = []
        for video_ctx in video_contexts:
            result = _process_single_video_cut_detection(video_ctx, config)
            results.append(result)
    
    elapsed = time.perf_counter() - start_time
    
    # Статистика
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = len(results) - ok_count
    
    logger.info(
        f"cut_detection | batch | completed {len(results)} videos "
        f"(ok={ok_count}, error={error_count}) in {elapsed:.2f}s"
    )
    
    return results

