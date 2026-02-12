"""
Batch processing utilities for scene_classification module.

Stage 3: Batch processing для scene_classification с параллельной обработкой видео:
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

logger = get_logger("VisualProcessor.scene_classification_batch")


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


def _process_single_video_scene_classification(
    video_ctx: VideoContext,
    config: Dict[str, Any],
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Обработать одно видео через scene_classification.
    
    Args:
        video_ctx: VideoContext для видео
        config: Конфигурация scene_classification
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
        
        # Получаем frame_indices для scene_classification
        try:
            frame_indices = _get_frame_indices(metadata, "scene_classification")
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
                "error": "scene_classification.frame_indices is empty",
            }
        
        # Подготавливаем аргументы для CLI
        scene_classification_main = _visual_processor_path / "modules" / "scene_classification" / "main.py"
        if not scene_classification_main.exists():
            return {
                "video_id": video_id,
                "status": "error",
                "error": f"scene_classification/main.py not found at {scene_classification_main}",
            }
        
        # Строим команду для subprocess
        cmd = [
            sys.executable,
            str(scene_classification_main),
            "--frames-dir", str(frames_dir),
            "--rs-path", str(rs_path),
        ]
        
        # Добавляем параметры из конфига
        if config.get("runtime"):
            cmd.extend(["--runtime", str(config["runtime"])])
        if config.get("triton_model_spec"):
            cmd.extend(["--triton-model-spec", str(config["triton_model_spec"])])
        if config.get("model_arch"):
            cmd.extend(["--model-arch", str(config["model_arch"])])
        if config.get("use_timm"):
            cmd.append("--use-timm")
        if config.get("device"):
            cmd.extend(["--device", str(config["device"])])
        if config.get("batch_size"):
            cmd.extend(["--batch-size", str(config["batch_size"])])
        if config.get("input_size"):
            cmd.extend(["--input-size", str(config["input_size"])])
        if config.get("use_tta"):
            cmd.append("--use-tta")
        if config.get("use_multi_crop"):
            cmd.append("--use-multi-crop")
        if config.get("temporal_smoothing"):
            cmd.append("--temporal-smoothing")
        if config.get("smoothing_window"):
            cmd.extend(["--smoothing-window", str(config["smoothing_window"])])
        if config.get("enable_advanced_features"):
            cmd.append("--enable-advanced-features")
        if config.get("use_clip_for_semantics"):
            cmd.append("--use-clip-for-semantics")
        if config.get("label_fusion"):
            cmd.extend(["--label-fusion", str(config["label_fusion"])])
        if config.get("min_scene_length"):
            cmd.extend(["--min-scene-length", str(config["min_scene_length"])])
        if config.get("min_scene_seconds"):
            cmd.extend(["--min-scene-seconds", str(config["min_scene_seconds"])])
        if config.get("triton_http_url"):
            cmd.extend(["--triton-http-url", str(config["triton_http_url"])])
        
        # Запускаем subprocess
        logger.info(f"scene_classification | batch | processing video {video_id} ({len(frame_indices)} frames)")
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
                logger.error(f"scene_classification | batch | video {video_id} failed: {error_msg}")
                return {
                    "video_id": video_id,
                    "status": "error",
                    "error": error_msg[:500],  # Limit error message length
                    "elapsed_sec": elapsed,
                }
            
            logger.info(f"scene_classification | batch | video {video_id} completed in {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "ok",
                "elapsed_sec": elapsed,
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            logger.error(f"scene_classification | batch | video {video_id} timed out after {elapsed:.2f}s")
            return {
                "video_id": video_id,
                "status": "error",
                "error": "Timeout after 3600s",
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"scene_classification | batch | video {video_id} exception: {e}")
            return {
                "video_id": video_id,
                "status": "error",
                "error": str(e)[:500],
                "elapsed_sec": elapsed,
            }
            
    except Exception as e:
        logger.error(f"scene_classification | batch | video {video_id} failed: {e}")
        return {
            "video_id": video_id,
            "status": "error",
            "error": str(e)[:500],
        }


def process_scene_classification_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_video_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch processing для scene_classification с параллельной обработкой видео.
    
    Stage 3: Batch processing для scene_classification.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация scene_classification
        max_video_workers: Максимальное количество параллельных воркеров для видео (None = последовательно)
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"scene_classification | batch | processing {len(video_contexts)} videos "
        f"(max_video_workers={max_video_workers})"
    )
    
    start_time = time.perf_counter()
    
    # Параллельная обработка видео
    if max_video_workers and max_video_workers > 1:
        results = []
        with ThreadPoolExecutor(max_workers=max_video_workers) as executor:
            futures = {
                executor.submit(_process_single_video_scene_classification, video_ctx, config): video_ctx
                for video_ctx in video_contexts
            }
            
            for future in as_completed(futures):
                video_ctx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"scene_classification | batch | video {video_ctx.video_id} exception: {e}")
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
            result = _process_single_video_scene_classification(video_ctx, config)
            results.append(result)
    
    elapsed = time.perf_counter() - start_time
    
    # Статистика
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = len(results) - ok_count
    
    logger.info(
        f"scene_classification | batch | completed {len(results)} videos "
        f"(ok={ok_count}, error={error_count}) in {elapsed:.2f}s"
    )
    
    return results

