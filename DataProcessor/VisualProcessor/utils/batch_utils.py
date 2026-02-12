"""
Утилиты для batch processing в VisualProcessor.
"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from utils.logger import get_logger
from utils.video_context import VideoContext

logger = get_logger("VisualProcessor.batch_utils")


def collect_video_contexts(
    video_input_dir: Optional[str] = None,
    video_input_list: Optional[str] = None,
    rs_base: str = None,
    platform_id: str = "youtube",
    run_id: Optional[str] = None,
) -> List[VideoContext]:
    """
    Собирает VideoContext для каждого видео из входной директории или списка.
    
    Args:
        video_input_dir: Директория с поддиректориями видео (frames_dir для каждого видео)
        video_input_list: Путь к JSON файлу со списком путей к frames_dir
        rs_base: Базовая директория result_store
        platform_id: ID платформы
        run_id: ID запуска (если None, будет извлечен из metadata.json)
    
    Returns:
        Список VideoContext для каждого видео
        
    Raises:
        ValueError: Если не указан ни video_input_dir, ни video_input_list
        RuntimeError: Если не найдено ни одного валидного видео
    """
    frames_dirs = []
    
    if video_input_dir:
        # Собираем все поддиректории в video_input_dir
        video_input_path = Path(video_input_dir)
        if not video_input_path.exists():
            raise ValueError(f"Video input directory not found: {video_input_dir}")
        
        for item in video_input_path.iterdir():
            if item.is_dir():
                # Проверяем наличие metadata.json
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    frames_dirs.append(str(item))
                else:
                    logger.warning(
                        f"VisualProcessor | skipping {item}: missing metadata.json"
                    )
    
    elif video_input_list:
        # Загружаем список из JSON файла
        if not os.path.exists(video_input_list):
            raise ValueError(f"Video input list file not found: {video_input_list}")
        
        try:
            with open(video_input_list, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Поддерживаем разные форматы:
            # 1. Список строк (путей)
            # 2. Словарь с ключом "frames_dirs" или "videos"
            if isinstance(data, list):
                frames_dirs = [str(Path(d).resolve()) for d in data]
            elif isinstance(data, dict):
                frames_dirs = data.get("frames_dirs") or data.get("videos") or []
                frames_dirs = [str(Path(d).resolve()) for d in frames_dirs]
            else:
                raise ValueError(f"Invalid format in video_input_list: expected list or dict")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in video_input_list: {e}") from e
    else:
        raise ValueError("Either video_input_dir or video_input_list must be provided")
    
    if not frames_dirs:
        raise RuntimeError("No valid video frames directories found for batch processing")
    
    # Создаем VideoContext для каждого frames_dir
    video_contexts = []
    for frames_dir_path in frames_dirs:
        frames_dir = str(Path(frames_dir_path).resolve())
        
        # Проверяем наличие metadata.json
        metadata_path = os.path.join(frames_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(
                f"VisualProcessor | skipping {frames_dir}: missing metadata.json"
            )
            continue
        
        # Загружаем metadata для извлечения video_id и run_id
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(
                f"VisualProcessor | skipping {frames_dir}: error loading metadata.json: {e}"
            )
            continue
        
        # Извлекаем video_id из metadata или имени директории
        video_id = metadata.get("video_id")
        if not video_id:
            video_id = os.path.basename(os.path.normpath(frames_dir))
        
        # Извлекаем run_id из metadata или используем переданный
        file_run_id = run_id or metadata.get("run_id")
        if not file_run_id:
            import uuid
            file_run_id = uuid.uuid4().hex[:12]
            logger.warning(
                f"VisualProcessor | no run_id in metadata for {video_id}, generated: {file_run_id}"
            )
        
        # Создаем per-video rs_path
        if rs_base:
            file_rs_path = os.path.join(
                os.path.abspath(rs_base),
                platform_id,
                video_id,
                file_run_id
            )
        else:
            # Если rs_base не указан, используем frames_dir/../result_store
            file_rs_path = os.path.join(
                os.path.dirname(frames_dir),
                "result_store",
                platform_id,
                video_id,
                file_run_id
            )
        
        # Создаем VideoContext
        video_ctx = VideoContext(
            video_id=video_id,
            frames_dir=frames_dir,
            rs_path=file_rs_path,
            metadata_path=metadata_path,
            platform_id=platform_id,
            run_id=file_run_id,
            config_hash=metadata.get("config_hash"),
            sampling_policy_version=metadata.get("sampling_policy_version"),
            dataprocessor_version=metadata.get("dataprocessor_version"),
        )
        video_contexts.append(video_ctx)
    
    if not video_contexts:
        raise RuntimeError("No valid video contexts created for batch processing")
    
    logger.info(
        f"VisualProcessor | batch mode: processing {len(video_contexts)} videos"
    )
    return video_contexts


def process_batch_results(
    batch_results: List[Dict[str, Any]],
    per_component_report: Dict[str, Any],
) -> tuple[int, int]:
    """
    Обрабатывает результаты batch processing и обновляет отчет.
    
    Args:
        batch_results: Список результатов для каждого видео
        (каждый элемент: {"video_id": str, "status": str, ...})
        per_component_report: Словарь для накопления статистики по компоненту
    
    Returns:
        Кортеж (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    for result in batch_results:
        video_id = result.get("video_id", "unknown")
        status = result.get("status", "unknown")
        
        if status == "ok":
            success_count += 1
        elif status == "error":
            error_count += 1
            error_msg = result.get("error", "unknown error")
            logger.error(
                f"VisualProcessor | batch | video {video_id} failed: {error_msg}"
            )
        elif status == "empty":
            # Empty results are considered success (no error, just no data)
            success_count += 1
            empty_reason = result.get("empty_reason", "unknown")
            logger.info(
                f"VisualProcessor | batch | video {video_id} empty: {empty_reason}"
            )
    
    # Обновляем отчет
    per_component_report["batch_success"] = success_count
    per_component_report["batch_errors"] = error_count
    per_component_report["batch_total"] = len(batch_results)
    
    return success_count, error_count

