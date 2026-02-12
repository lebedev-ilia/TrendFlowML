"""
Модуль для batch обработки множества аудио файлов.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from .audio_file_context import AudioFileContext

logger = logging.getLogger(__name__)


def collect_frames_dirs(
    audio_input_dir: Optional[str] = None,
    audio_input_list: Optional[str] = None,
) -> List[str]:
    """
    Собирает список путей frames_dir для batch обработки.
    
    Args:
        audio_input_dir: Директория с поддиректориями frames_dir
        audio_input_list: Путь к файлу со списком путей frames_dir
    
    Returns:
        Список путей к frames_dir
    
    Raises:
        RuntimeError: Если не найдено ни одного валидного frames_dir
    """
    frames_dirs = []
    
    if audio_input_dir:
        # Собираем из директории: каждая поддиректория должна быть frames_dir
        input_dir = os.path.abspath(audio_input_dir)
        if not os.path.isdir(input_dir):
            raise RuntimeError(f"AudioProcessor | --audio-input-dir is not a directory: {input_dir}")
        
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                # Проверяем наличие audio/audio.wav и audio/segments.json
                audio_path = os.path.join(item_path, "audio", "audio.wav")
                segments_path = os.path.join(item_path, "audio", "segments.json")
                if os.path.exists(audio_path) and os.path.exists(segments_path):
                    frames_dirs.append(item_path)
                else:
                    logger.warning(f"AudioProcessor | skipping {item_path}: missing audio/audio.wav or audio/segments.json")
    
    elif audio_input_list:
        # Собираем из файла: один путь на строку
        list_path = os.path.abspath(audio_input_list)
        if not os.path.exists(list_path):
            raise RuntimeError(f"AudioProcessor | --audio-input-list file not found: {list_path}")
        
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                frames_dir = os.path.abspath(line)
                if not os.path.isdir(frames_dir):
                    logger.warning(f"AudioProcessor | skipping {frames_dir}: not a directory")
                    continue
                # Проверяем наличие audio/audio.wav и audio/segments.json
                audio_path = os.path.join(frames_dir, "audio", "audio.wav")
                segments_path = os.path.join(frames_dir, "audio", "segments.json")
                if os.path.exists(audio_path) and os.path.exists(segments_path):
                    frames_dirs.append(frames_dir)
                else:
                    logger.warning(f"AudioProcessor | skipping {frames_dir}: missing audio/audio.wav or audio/segments.json")
    
    if not frames_dirs:
        raise RuntimeError("AudioProcessor | no valid frames_dirs found for batch processing")
    
    logger.info(f"AudioProcessor | batch mode: found {len(frames_dirs)} frames_dirs")
    return frames_dirs


def create_audio_file_contexts(
    frames_dirs: List[str],
    rs_base: str,
    platform_id: str,
    run_id: str,
) -> List[AudioFileContext]:
    """
    Создает AudioFileContext для каждого frames_dir.
    
    Args:
        frames_dirs: Список путей к frames_dir
        rs_base: Базовая директория result_store
        platform_id: ID платформы
        run_id: ID запуска
    
    Returns:
        Список AudioFileContext
    """
    audio_file_contexts = []
    
    for frames_dir_path in frames_dirs:
        # Получаем video_id из имени frames_dir
        file_video_id = os.path.basename(os.path.normpath(frames_dir_path))
        
        # Создаем per-file run_rs_path
        file_run_rs_path = os.path.join(os.path.abspath(rs_base), platform_id, file_video_id, run_id)
        os.makedirs(file_run_rs_path, exist_ok=True)
        
        # Создаем per-file tmp_dir
        file_tmp_dir = os.path.join(file_run_rs_path, "_tmp_audio")
        os.makedirs(file_tmp_dir, exist_ok=True)
        
        # Создаем per-file artifacts_dir
        file_artifacts_dir = os.path.join(file_run_rs_path, "audio_processor")
        os.makedirs(file_artifacts_dir, exist_ok=True)
        
        # Пути к аудио и segments
        audio_path = os.path.join(frames_dir_path, "audio", "audio.wav")
        segments_json_path = os.path.join(frames_dir_path, "audio", "segments.json")
        
        # Проверяем существование файлов
        if not os.path.exists(audio_path):
            logger.warning(f"AudioProcessor | skipping {frames_dir_path}: missing audio/audio.wav")
            continue
        if not os.path.exists(segments_json_path):
            logger.warning(f"AudioProcessor | skipping {frames_dir_path}: missing audio/segments.json")
            continue
        
        # Создаем AudioFileContext
        file_ctx = AudioFileContext(
            file_id=file_video_id,
            input_uri=audio_path,
            tmp_path=file_tmp_dir,
            artifacts_dir=file_artifacts_dir,
            segments_json_path=segments_json_path,
        )
        audio_file_contexts.append(file_ctx)
    
    if not audio_file_contexts:
        raise RuntimeError("AudioProcessor | no valid audio files found for batch processing")
    
    logger.info(f"AudioProcessor | batch mode: processing {len(audio_file_contexts)} files")
    return audio_file_contexts


def process_batch_results(
    batch_results: List[Dict[str, Any]],
    per_extractor_report: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Обрабатывает результаты batch обработки и валидирует NPZ файлы.
    
    Args:
        batch_results: Список результатов batch обработки
        per_extractor_report: Словарь для сохранения отчетов
    
    Returns:
        Tuple[successful_count, failed_count]
    """
    successful_count = sum(1 for r in batch_results if r.get("success", False))
    failed_count = len(batch_results) - successful_count
    
    logger.info(f"AudioProcessor | batch mode: completed {successful_count} successful, {failed_count} failed")
    
    # Валидация NPZ файлов и генерация render для каждого файла
    for result in batch_results:
        file_id = result.get("file_id", "unknown")
        output_dir = result.get("output_dir")
        if output_dir and os.path.exists(output_dir):
            # Находим NPZ файлы в output_dir и валидируем + генерируем render
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".npz"):
                        npz_path = os.path.join(root, file)
                        
                        # Валидация NPZ
                        try:
                            # Импорт из VisualProcessor/utils (через sys.path)
                            import sys
                            from pathlib import Path
                            # Добавляем путь к VisualProcessor если его нет
                            repo_root = Path(__file__).resolve().parent.parent.parent.parent
                            vp_root = repo_root / "VisualProcessor"
                            if str(vp_root) not in sys.path:
                                sys.path.insert(0, str(vp_root))
                            from utils.artifact_validator import validate_npz  # type: ignore
                            validate_npz(npz_path)
                        except Exception as e:
                            logger.warning(f"AudioProcessor | batch mode: NPZ validation failed for {file_id}/{file}: {e}")
                        
                        # Генерация render-context для этого компонента
                        try:
                            from .renderer import render_component  # type: ignore
                            
                            # Извлекаем имя компонента из пути: <component_name>/<component_name>_features.npz
                            component_dir = os.path.dirname(npz_path)
                            component_name = os.path.basename(component_dir)
                            
                            # По умолчанию включаем render для batch mode (можно добавить поддержку флагов из конфига)
                            render = render_component(
                                npz_path, 
                                component_name, 
                                component_dir,
                                enable_render=True,
                                enable_html_render=True,
                            )
                            render_path = os.path.join(component_dir, "_render", "render_context.json")
                            logger.info(f"AudioProcessor | batch mode: render-context saved for {file_id}/{component_name}")
                        except Exception as e:
                            # Best-effort: не падаем если render не удался
                            logger.warning(f"AudioProcessor | batch mode: failed to generate render-context for {file_id}/{component_name}: {e}")
    
    # Сохраняем summary batch результатов
    per_extractor_report["batch_summary"] = {
        "total_files": len(batch_results),
        "successful": successful_count,
        "failed": failed_count,
        "results": batch_results,
    }
    
    return successful_count, failed_count

