"""
Batch processing utilities for video_pacing module.

Stage 3: Batch processing для video_pacing с параллельной обработкой видео:
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

logger = get_logger("VisualProcessor.video_pacing_batch")


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


def _process_single_video_video_pacing(
    video_ctx: VideoContext,
    config: Dict[str, Any],
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Обработать одно видео через video_pacing.
    
    Args:
        video_ctx: VideoContext для видео
        config: Конфигурация video_pacing
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
        
        # Получаем frame_indices для video_pacing
        try:
            frame_indices = _get_frame_indices(metadata, "video_pacing")
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
                "error": "video_pacing.frame_indices is empty",
            }
        
        # Подготавливаем аргументы для CLI
        video_pacing_main = _visual_processor_path / "modules" / "video_pacing" / "main.py"
        if not video_pacing_main.exists():
            return {
                "video_id": video_id,
                "status": "error",
                "error": f"video_pacing/main.py not found at {video_pacing_main}",
            }
        
        # Строим команду для subprocess
        cmd = [
            sys.executable,
            str(video_pacing_main),
            "--frames-dir", str(frames_dir),
            "--rs-path", str(rs_path),
        ]
        
        # Добавляем параметры из конфига
        if config.get("downscale_factor") is not None:
            cmd.extend(["--downscale-factor", str(config["downscale_factor"])])
        if config.get("min_shot_length_seconds") is not None:
            cmd.extend(["--min-shot-length-seconds", str(config["min_shot_length_seconds"])])
        if config.get("shot_detect_k") is not None:
            cmd.extend(["--shot-detect-k", str(config["shot_detect_k"])])
        if config.get("min_frames") is not None:
            cmd.extend(["--min-frames", str(config["min_frames"])])
        if config.get("enable_entropy_features"):
            cmd.append("--enable-entropy-features")
        if config.get("enable_histograms"):
            cmd.append("--enable-histograms")
        if config.get("enable_pace_curve_peaks"):
            cmd.append("--enable-pace-curve-peaks")
        if config.get("enable_periodicity"):
            cmd.append("--enable-periodicity")
        if config.get("enable_bursts"):
            cmd.append("--enable-bursts")
        
        # Запускаем subprocess
        logger.info(f"video_pacing | batch | processing video {video_id} ({len(frame_indices)} frames)")
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
                logger.error(f"video_pacing | batch | video {video_id} failed: {error_msg}")
                return {
                    "video_id": video_id,
                    "status": "error",
                    "error": error_msg[:500],  # Limit error message length
                    "elapsed_sec": elapsed,
                }
            
            logger.info(f"video_pacing | batch | video {video_id} completed in {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "ok",
                "elapsed_sec": elapsed,
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            logger.error(f"video_pacing | batch | video {video_id} timed out after {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "error",
                "error": "Timeout after 3600s",
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"video_pacing | batch | video {video_id} exception: {e}")
            return {
                "video_id": video_id,
                "status": "error",
                "error": str(e)[:500],
                "elapsed_sec": elapsed,
            }
            
    except Exception as e:
        logger.error(f"video_pacing | batch | video {video_id} failed: {e}")
        return {
            "video_id": video_id,
            "status": "error",
            "error": str(e)[:500],
        }


def process_video_pacing_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_video_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для video_pacing с параллельной обработкой видео.
    
    Stage 3: Batch processing для video_pacing.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация video_pacing
        max_video_workers: Максимальное количество параллельных воркеров для видео (None = последовательно)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"video_pacing | batch | processing {len(video_contexts)} videos "
        f"(max_video_workers={max_video_workers})"
    )
    
    start_time = time.perf_counter()
    
    # Параллельная обработка видео
    if max_video_workers and max_video_workers > 1:
        results = []
        with ThreadPoolExecutor(max_workers=max_video_workers) as executor:
            futures = {
                executor.submit(_process_single_video_video_pacing, video_ctx, config): video_ctx
                for video_ctx in video_contexts
            }
            
            for future in as_completed(futures):
                video_ctx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"video_pacing | batch | video {video_ctx.video_id} exception: {e}")
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
            result = _process_single_video_video_pacing(video_ctx, config)
            results.append(result)
    
    elapsed = time.perf_counter() - start_time
    
    # Статистика
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = len(results) - ok_count
    
    logger.info(
        f"video_pacing | batch | completed {len(results)} videos "
        f"(ok={ok_count}, error={error_count}) in {elapsed:.2f}s"
    )
    
    return results

