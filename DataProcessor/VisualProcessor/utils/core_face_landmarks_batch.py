"""
Batch processing utilities for core_face_landmarks component.

Stage 3: CPU parallelism для core_face_landmarks с гибридным подходом:
- Параллельная обработка видео через subprocess с использованием изолированной виртуальной среды
- Каждое видео обрабатывается отдельным subprocess с правильным Python из .core_face_landmarks_venv
- Распределение результатов обратно по видео

Примечание: core_face_landmarks требует изолированную виртуальную среду .core_face_landmarks_venv
из-за конфликтов зависимостей MediaPipe, поэтому используем subprocess вместо прямого импорта.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add VisualProcessor to path
_visual_processor_path = Path(__file__).parent.parent
sys.path.insert(0, str(_visual_processor_path))

from utils.logger import get_logger
from utils.video_context import VideoContext

logger = get_logger("VisualProcessor.core_face_landmarks_batch")

# Constants
NAME = "core_face_landmarks"


def _get_venv_python() -> str:
    """Получить путь к Python из .core_face_landmarks_venv."""
    vp_root = _visual_processor_path
    venv_path = vp_root / "core" / "model_process" / "core_face_landmarks" / ".core_face_landmarks_venv"
    python_exec = venv_path / "bin" / "python"
    
    if python_exec.exists():
        return str(python_exec)
    else:
        logger.warning(
            f"core_face_landmarks | batch | venv python not found at {python_exec}; "
            f"falling back to current interpreter: {sys.executable}"
        )
        return sys.executable


def _build_subprocess_cmd(video_ctx: VideoContext, config: Dict[str, Any]) -> List[str]:
    """Построить команду subprocess для обработки одного видео."""
    vp_root = _visual_processor_path
    python_exec = _get_venv_python()
    entry = vp_root / "core" / "model_process" / "core_face_landmarks" / "main.py"
    
    if not entry.exists():
        raise FileNotFoundError(f"Entry not found for {NAME}: {entry}")
    
    # Строим аргументы из конфигурации
    kwargs = []
    for k, v in config.items():
        # Пропускаем вложенные объекты и специальные ключи
        if k in ("venv_path", "sampling", "render"):
            continue
        if isinstance(v, dict):
            continue
        if v is None or v == "False" or v is False:
            continue
        # Skip empty strings (they cause issues with optional int arguments)
        if v == "" or v == "''" or v == '""':
            continue
        key = f"--{k.replace('_', '-')}"
        if v is True or v == "True":
            kwargs.append(key)
        else:
            kwargs.extend([key, str(v)])
    
    cmd = [
        python_exec,
        str(entry),
        *kwargs,
        "--frames-dir",
        video_ctx.frames_dir,
        "--rs-path",
        video_ctx.rs_path,
    ]
    
    return cmd


def process_core_face_landmarks_batch(
    video_contexts: List[VideoContext],
    config: Dict[str, Any],
    *,
    max_frames_per_batch: Optional[int] = None,
    num_workers: int = 2,
) -> List[Dict[str, Any]]:
    """
    Batch processing для core_face_landmarks с гибридным подходом.
    
    Stage 3: CPU parallelism для core_face_landmarks через subprocess.
    
    Args:
        video_contexts: Список VideoContext для каждого видео
        config: Конфигурация core_face_landmarks
        max_frames_per_batch: Максимальное количество кадров в одном батче (не используется, оставлено для совместимости)
        num_workers: Количество параллельных воркеров для обработки видео
    
    Returns:
        Список результатов для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"core_face_landmarks | batch | processing {len(video_contexts)} videos "
        f"(num_workers={num_workers})"
    )
    
    start_time = time.perf_counter()
    
    # Baseline policy checks
    if not config.get("use_face_mesh", True):
        raise RuntimeError(f"{NAME} | baseline requires use_face_mesh (no-fallback)")
    if not config.get("use_person_mask", True):
        raise RuntimeError(f"{NAME} | baseline requires use_person_mask (no-fallback)")
    
    results = []
    
    def process_single_video(video_ctx: VideoContext) -> Dict[str, Any]:
        """Обработать одно видео через subprocess."""
        try:
            # Строим команду subprocess
            cmd = _build_subprocess_cmd(video_ctx, config)
            
            logger.debug(f"{NAME} | batch | video {video_ctx.video_id} | running: {' '.join(cmd)}")
            
            # Suppress MediaPipe verbose logs
            env = os.environ.copy()
            env["GLOG_minloglevel"] = "2"  # Suppress INFO, WARNING (keep ERROR, FATAL)
            env["GLOG_stderrthreshold"] = "2"  # Only ERROR and FATAL to stderr
            
            # Запускаем subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env,
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(
                    f"{NAME} | batch | video {video_ctx.video_id} | subprocess failed "
                    f"(returncode={result.returncode}): {error_msg[:500]}"
                )
                return {
                    "video_id": video_ctx.video_id,
                    "status": "error",
                    "error": f"subprocess failed: {error_msg[:200]}",
                }
            
            # Проверяем, что артефакт создан
            component_dir = video_ctx.get_component_rs_path(NAME)
            npz_path = os.path.join(component_dir, "landmarks.npz")
            
            if not os.path.exists(npz_path):
                logger.error(f"{NAME} | batch | video {video_ctx.video_id} | artifact not created: {npz_path}")
                return {
                    "video_id": video_ctx.video_id,
                    "status": "error",
                    "error": "artifact not created",
                }
            
            logger.info(f"{NAME} | batch | video {video_ctx.video_id} completed successfully")
            return {
                "video_id": video_ctx.video_id,
                "status": "ok",
                "saved_path": npz_path,
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"{NAME} | batch | video {video_ctx.video_id} | subprocess timeout")
            return {
                "video_id": video_ctx.video_id,
                "status": "error",
                "error": "subprocess timeout",
            }
        except Exception as e:
            logger.exception(f"{NAME} | batch | video {video_ctx.video_id} failed: {e}")
            return {
                "video_id": video_ctx.video_id,
                "status": "error",
                "error": str(e),
            }
    
    # Обрабатываем видео параллельно
    if num_workers > 1 and len(video_contexts) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_video, ctx): ctx for ctx in video_contexts}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        # Последовательная обработка
        for video_ctx in video_contexts:
            result = process_single_video(video_ctx)
            results.append(result)
    
    duration = time.perf_counter() - start_time
    logger.info(
        f"core_face_landmarks | batch | completed in {duration:.2f}s "
        f"({len([r for r in results if r.get('status') == 'ok'])}/{len(results)} successful)"
    )
    
    return results
