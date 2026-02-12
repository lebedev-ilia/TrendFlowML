"""
Модуль для запуска extractors и обработки результатов.
"""
import os
import sys
import time
import logging
import subprocess
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from ..utils.progress import emit_progress
from ..utils.retry import retry_with_backoff, run_clap_with_oom_fallback
from .dependency_resolver import REQUIRED_EXTRACTOR_DEPENDENCIES, OPTIONAL_EXTRACTOR_DEPENDENCIES

logger = logging.getLogger(__name__)

def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        # Try to log directly - handlers may exist but streams may be closed
        # We catch all exceptions to prevent crashes when logging infrastructure is shutting down
        logger_instance.warning(message, *args, **kwargs)
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        # This is expected behavior during cleanup/shutdown phases
        pass


def get_extractor_parallelism(
    extractor_key: str,
    setting: str,
    default: Any,
    extractor_parallelism_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Any:
    """
    Получить индивидуальную настройку parallelism для extractor'а.
    
    Args:
        extractor_key: Ключ extractor'а
        setting: Название настройки
        default: Значение по умолчанию
        extractor_parallelism_config: Конфигурация parallelism (опционально)
    
    Returns:
        Значение настройки
    """
    if extractor_parallelism_config and extractor_key in extractor_parallelism_config:
        return extractor_parallelism_config[extractor_key].get(setting, default)
    return default


def get_extractor_render_flags(
    extractor_key: str,
    extractor_config: Optional[Dict[str, Dict[str, Any]]] = None,
    default_enable_render: bool = True,
    default_enable_html_render: bool = True,
) -> Tuple[bool, bool]:
    """
    Получить флаги рендеринга для extractor'а.
    
    Args:
        extractor_key: Ключ extractor'а
        extractor_config: Полная конфигурация extractor'а из global_config.yaml (extractors секция)
        default_enable_render: Значение по умолчанию для enable_render
        default_enable_html_render: Значение по умолчанию для enable_html_render
    
    Returns:
        Tuple[enable_render, enable_html_render]
    """
    if extractor_config and extractor_key in extractor_config:
        render_cfg = extractor_config[extractor_key].get("render", {})
        enable_render = render_cfg.get("enable_render", default_enable_render) if render_cfg else default_enable_render
        enable_html_render = render_cfg.get("enable_html_render", default_enable_html_render) if render_cfg else default_enable_html_render
        return bool(enable_render), bool(enable_html_render)
    return default_enable_render, default_enable_html_render


def create_progress_callback(
    extractor_key: str,
    component_name: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    extractor_idx: int,
    total_extractors: int,
    callback_type: str = "segments",  # "segments", "batches", "generic"
    t_start: Optional[float] = None,  # Start time for total_elapsed_sec calculation
) -> Callable:
    """
    Создает callback для прогресса extractor'а.
    
    Args:
        extractor_key: Ключ extractor'а
        component_name: Имя компонента
        platform_id: ID платформы
        video_id: ID видео
        run_id: ID запуска
        extractor_idx: Индекс extractor'а (0-based)
        total_extractors: Общее количество extractors
        callback_type: Тип callback'а ("segments", "batches", "generic")
    
    Returns:
        Callback функция
    """
    if callback_type == "segments":
        def callback(seg_idx: int, total_segs: int, message: str = ""):
            # Обновляем только каждые 5% или на последнем сегменте
            if total_segs >= 10:
                pct = int((seg_idx / total_segs) * 100)
                # Обновляем только при значительных изменениях (каждые 5%)
                if seg_idx == 0 or seg_idx == total_segs - 1 or (pct % 5 == 0 and seg_idx > 0):
                    # Более точный расчет процентов: сначала вычисляем общий прогресс, потом округляем
                    if total_extractors > 0 and total_segs > 0:
                        base_progress = (extractor_idx / total_extractors) * 70
                        internal_progress = (seg_idx / total_segs) * (70 / total_extractors)
                        progress_pct = int(10 + base_progress + internal_progress)
                    else:
                        progress_pct = 10
                    total_elapsed_sec = None
                    if t_start is not None:
                        import time as time_module
                        total_elapsed_sec = time_module.time() - t_start
                    emit_progress(
                        platform_id=platform_id,
                        video_id=video_id,
                        run_id=run_id,
                        component=component_name,
                        stage_id="run_segments",
                        stage_name=f"Processing segments",
                        progress_pct=min(80, progress_pct),
                        extractor=extractor_key,
                        total_elapsed_sec=total_elapsed_sec,
                    )
        return callback
    
    elif callback_type == "batches":
        def callback(batch_idx: int, total_batches: int, message: str = ""):
            # Обновляем прогресс для всех случаев (включая малое количество батчей)
            if total_batches >= 10:
                # Для большого количества батчей - обновляем каждые 10%
                pct = int((batch_idx / total_batches) * 100)
                if batch_idx == 0 or batch_idx == total_batches - 1 or (pct % 10 == 0 and batch_idx > 0):
                    # Более точный расчет процентов: сначала вычисляем общий прогресс, потом округляем
                    if total_extractors > 0 and total_batches > 0:
                        base_progress = (extractor_idx / total_extractors) * 70
                        internal_progress = (batch_idx / total_batches) * (70 / total_extractors)
                        progress_pct = int(10 + base_progress + internal_progress)
                    else:
                        progress_pct = 10
                    total_elapsed_sec = None
                    if t_start is not None:
                        import time as time_module
                        total_elapsed_sec = time_module.time() - t_start
                    emit_progress(
                        platform_id=platform_id,
                        video_id=video_id,
                        run_id=run_id,
                        component=component_name,
                        stage_id="run_segments",
                        stage_name=f"Processing batches: {message}" if message else "Processing batches",
                        progress_pct=min(80, progress_pct),
                        extractor=extractor_key,
                        total_elapsed_sec=total_elapsed_sec,
                    )
            else:
                # Для малого количества батчей - обновляем на каждом батче
                # Более точный расчет процентов: сначала вычисляем общий прогресс, потом округляем
                if total_extractors > 0 and total_batches > 0:
                    base_progress = (extractor_idx / total_extractors) * 70
                    internal_progress = (batch_idx / total_batches) * (70 / total_extractors)
                    progress_pct = int(10 + base_progress + internal_progress)
                else:
                    progress_pct = 10
                total_elapsed_sec = None
                if t_start is not None:
                    import time as time_module
                    total_elapsed_sec = time_module.time() - t_start
                emit_progress(
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    component=component_name,
                    stage_id="run_segments",
                    stage_name=f"Processing batches: {message}" if message else "Processing batches",
                    progress_pct=min(80, progress_pct),
                    extractor=extractor_key,
                    total_elapsed_sec=total_elapsed_sec,
                )
        return callback
    
    else:  # generic
        _last_callback_pct = [-1]  # Используем список для замыкания
        _last_callback_time = [0.0]  # Время последнего обновления
        
        def callback(extractor_name: str, current: int, total: int, message: str = ""):
            if total > 0:
                import time as time_module
                current_time = time_module.time()
                pct = int((current / total) * 100)
                
                # Обновляем если:
                # 1. Начало (current == 0 или current == 1)
                # 2. Последний элемент
                # 3. Изменился процент на 10% или больше
                # 4. Прошло больше 1 секунды с последнего обновления (для долгих операций)
                should_update = False
                if current == 0 or current == 1 or current == total:
                    should_update = True
                elif pct % 10 == 0 and pct != _last_callback_pct[0]:
                    should_update = True
                elif current_time - _last_callback_time[0] >= 1.0:
                    should_update = True
                
                if should_update:
                    _last_callback_pct[0] = pct
                    _last_callback_time[0] = current_time
                    # Более точный расчет процентов: сначала вычисляем общий прогресс, потом округляем
                    if total_extractors > 0 and total > 0:
                        base_progress = (extractor_idx / total_extractors) * 70
                        internal_progress = (current / total) * (70 / total_extractors)
                        progress_pct = int(10 + base_progress + internal_progress)
                    else:
                        progress_pct = 10
                    stage_msg = message if message else "Processing"
                    # Calculate total elapsed time from t_start if available
                    total_elapsed_sec = None
                    if t_start is not None:
                        total_elapsed_sec = current_time - t_start
                    emit_progress(
                        platform_id=platform_id,
                        video_id=video_id,
                        run_id=run_id,
                        component=component_name,
                        stage_id="run_segments",
                        stage_name=stage_msg,
                        progress_pct=min(80, progress_pct),
                        extractor=extractor_key,
                        elapsed_sec=None,
                        total_elapsed_sec=total_elapsed_sec,
                    )
        return callback


def run_single_extractor(
    extractor_key: str,
    extractor: Any,
    audio_path: str,
    tmp_dir: str,
    segments_payload: Optional[Dict[str, Any]],
    run_rs_path: str,
    extractor_results: Dict[str, Any],
    extractor_idx: int,
    total_extractors: int,
    platform_id: str,
    video_id: str,
    run_id: str,
    segment_parallelism: int,
    max_inflight: int,
    clap_batch_size: int,
    extractor_parallelism_config: Optional[Dict[str, Dict[str, Any]]] = None,
    t_start: Optional[float] = None,  # Start time for total_elapsed_sec calculation
    # Segments для разных семейств
    primary_segments: Optional[List] = None,
    clap_segments: Optional[List] = None,
    tempo_segments: Optional[List] = None,
    asr_segments: Optional[List] = None,
    diar_segments: Optional[List] = None,
    emo_segments: Optional[List] = None,
    sep_segments: Optional[List] = None,
    pitch_segments: Optional[List] = None,
    spectral_segments: Optional[List] = None,
    quality_segments: Optional[List] = None,
    mfcc_segments: Optional[List] = None,
    mel_segments: Optional[List] = None,
    onset_segments: Optional[List] = None,
    chroma_segments: Optional[List] = None,
    rhythmic_segments: Optional[List] = None,
    voice_quality_segments: Optional[List] = None,
    hpss_segments: Optional[List] = None,
    key_segments: Optional[List] = None,
    band_energy_segments: Optional[List] = None,
    spectral_entropy_segments: Optional[List] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Запускает один extractor и возвращает результат.
    
    Returns:
        Tuple[ExtractorResult, effective_knobs]
    """
    t_e0 = time.time()
    effective = {"segment_parallelism": 1, "max_inflight": 1}
    
    # Получаем имя компонента
    key_to_component = {
        "clap": "clap_extractor",
        "tempo": "tempo_extractor",
        "loudness": "loudness_extractor",
        "asr": "asr_extractor",
        "speaker_diarization": "speaker_diarization_extractor",
        "emotion_diarization": "emotion_diarization_extractor",
        "source_separation": "source_separation_extractor",
        "speech_analysis": "speech_analysis_extractor",
        "pitch": "pitch_extractor",
        "spectral": "spectral_extractor",
        "quality": "quality_extractor",
        "mfcc": "mfcc_extractor",
        "mel": "mel_extractor",
        "onset": "onset_extractor",
        "chroma": "chroma_extractor",
        "rhythmic": "rhythmic_extractor",
        "voice_quality": "voice_quality_extractor",
        "hpss": "hpss_extractor",
        "key": "key_extractor",
        "band_energy": "band_energy_extractor",
        "spectral_entropy": "spectral_entropy_extractor",
    }
    component_name = key_to_component.get(extractor_key, f"{extractor_key}_extractor")
    
    if not segments_payload:
        # Fallback на run() если нет segments
        r = extractor.run(audio_path, tmp_dir)
        return r, effective
    
    # CLAP
    if extractor_key == "clap":
        # Создаем progress callback для CLAP extractor'а
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "generic", t_start=t_start
        )
        # Устанавливаем progress callback в extractor (CLAP использует self.progress_callback)
        extractor.progress_callback = callback
        
        clap_preprocess_workers = get_extractor_parallelism("clap", "preprocess_workers", 4, extractor_parallelism_config)
        clap_batch_size_individual = get_extractor_parallelism("clap", "batch_size", clap_batch_size, extractor_parallelism_config)
        try:
            r = run_clap_with_oom_fallback(
                extractor, audio_path, tmp_dir, clap_segments or [],
                clap_batch_size_individual, segment_parallelism=clap_preprocess_workers
            )
            effective = {"segment_parallelism": clap_preprocess_workers, "max_inflight": 1, "model_batch_size": clap_batch_size_individual}
        except TypeError:
            r = extractor.run_segments(audio_path, tmp_dir, clap_segments or [], segment_parallelism=clap_preprocess_workers)  # type: ignore
            effective = {"segment_parallelism": clap_preprocess_workers, "max_inflight": 1, "model_batch_size": None}
    
    # Loudness
    elif extractor_key == "loudness":
        # Создаем progress callback для loudness extractor'а (используем "generic" как в tempo)
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "generic", t_start=t_start
        )
        # Устанавливаем progress callback в extractor (loudness использует self.progress_callback)
        extractor.progress_callback = callback
        
        loudness_workers = get_extractor_parallelism("loudness", "segment_workers", segment_parallelism, extractor_parallelism_config)
        loudness_max_inflight = get_extractor_parallelism("loudness", "max_inflight", max_inflight if max_inflight else loudness_workers, extractor_parallelism_config)
        try:
            r = extractor.run_segments(  # type: ignore
                audio_path, tmp_dir, primary_segments or [],
                segment_parallelism=loudness_workers, max_inflight=loudness_max_inflight
            )
            effective = {"segment_parallelism": loudness_workers, "max_inflight": loudness_max_inflight}
        except TypeError:
            r = extractor.run_segments(audio_path, tmp_dir, primary_segments or [])  # type: ignore
            effective = {"segment_parallelism": 1, "max_inflight": 1}
    
    # Tempo
    elif extractor_key == "tempo":
        # Создаем progress callback для tempo extractor'а (используем "generic" как в clap)
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "generic", t_start=t_start
        )
        # Устанавливаем progress callback в extractor (tempo использует self.progress_callback)
        extractor.progress_callback = callback
        
        tempo_workers = get_extractor_parallelism("tempo", "segment_workers", segment_parallelism, extractor_parallelism_config)
        tempo_max_inflight = get_extractor_parallelism("tempo", "max_inflight", max_inflight if max_inflight else tempo_workers, extractor_parallelism_config)
        try:
            r = extractor.run_segments(  # type: ignore
                audio_path, tmp_dir, tempo_segments or [],
                segment_parallelism=tempo_workers, max_inflight=tempo_max_inflight
            )
            effective = {"segment_parallelism": tempo_workers, "max_inflight": tempo_max_inflight}
        except TypeError:
            r = extractor.run_segments(audio_path, tmp_dir, tempo_segments or [])  # type: ignore
            effective = {"segment_parallelism": 1, "max_inflight": 1}
    
    # ASR (inprocess Whisper)
    elif extractor_key == "asr":
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "segments", t_start=t_start
        )
        # Set progress callback on extractor instance (not passed as parameter)
        extractor.progress_callback = callback
        logger.info(f"ASR | run_single_extractor: audio_path={audio_path}, asr_segments_count={len(asr_segments) if asr_segments else 0}")
        def run_asr():
            result = extractor.run_segments(  # type: ignore
                audio_path, tmp_dir, asr_segments or []
            )
            logger.info(f"ASR | run_segments returned: success={result.success}, error={result.error}, payload_keys={list(result.payload.keys()) if result.payload else None}")
            return result
        r = retry_with_backoff(
            run_asr, max_attempts=2, backoff_base=1.0,
            retry_on=["503", "504", "timeout", "connection", "triton"],
        )
    
    # Speaker diarization (может запускаться в персональной venv через subprocess)
    elif extractor_key == "speaker_diarization":
        # Проверяем наличие персональной venv
        extractor_dir = Path(__file__).resolve().parent.parent / "extractors" / "speaker_diarization_extractor"
        venv_path = extractor_dir / ".speaker_diarization_venv"
        venv_python = venv_path / "bin" / "python"
        
        if venv_python.exists():
            # Запускаем через subprocess в персональной venv
            logger.info(f"SpeakerDiarization | Using isolated venv: {venv_python}")
            
            # Создаем временный файл для сегментов (если есть)
            segments_file = None
            if diar_segments:
                segments_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=tmp_dir)
                json.dump({"families": {"diarization": {"segments": diar_segments}}}, segments_file)
                segments_file.close()
                segments_file_path = segments_file.name
            else:
                segments_file_path = None
            
            # Создаем временный файл для результата
            result_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=tmp_dir)
            result_file.close()
            result_file_path = result_file.name
            
            # Собираем аргументы для wrapper скрипта
            wrapper_script = extractor_dir / "run_in_venv.py"
            cmd = [
                str(venv_python),
                str(wrapper_script),
                "--audio-path", audio_path,
                "--tmp-dir", tmp_dir,
                "--output-json", result_file_path,
                "--device", getattr(extractor, "device", "auto"),
                "--whisper-model-size", getattr(extractor, "whisper_model_size", "small"),
                "--sample-rate", str(getattr(extractor, "sample_rate", 16000)),
            ]
            
            # Добавляем HuggingFace token если есть
            hf_token = getattr(extractor, "huggingface_token", None) or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                cmd.extend(["--huggingface-token", hf_token])
            
            # Добавляем сегменты если есть
            if segments_file_path:
                cmd.extend(["--segments-json", segments_file_path])
            
            # Добавляем feature flags
            if getattr(extractor, "enable_speaker_segments", False):
                cmd.append("--enable-speaker-segments")
            if getattr(extractor, "enable_speaker_embeddings", False):
                cmd.append("--enable-speaker-embeddings")
            if getattr(extractor, "enable_speaker_stats", False):
                cmd.append("--enable-speaker-stats")
            if getattr(extractor, "enable_speaker_durations", False):
                cmd.append("--enable-speaker-durations")
            if getattr(extractor, "enable_transcript", False):
                cmd.append("--enable-transcript")
            if getattr(extractor, "enable_word_segments", False):
                cmd.append("--enable-word-segments")
            if not getattr(extractor, "enable_silence_detection", True):
                cmd.append("--disable-silence-detection")
            
            # Настраиваем окружение
            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            prev_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(repo_root) if not prev_pp else (str(repo_root) + os.pathsep + prev_pp)
            
            def run_diar_subprocess():
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    
                    # Загружаем результат из JSON файла
                    with open(result_file_path, "r") as f:
                        result_dict = json.load(f)
                    
                    # Восстанавливаем ExtractorResult
                    from src.core.base_extractor import ExtractorResult  # type: ignore
                    r = ExtractorResult(
                        success=result_dict.get("success", False),
                        error=result_dict.get("error"),
                        payload=result_dict.get("payload", {}),
                        processing_time=result_dict.get("processing_time", 0.0),
                    )
                    
                    logger.info(f"SpeakerDiarization | subprocess returned: success={r.success}, error={r.error}, payload_keys={list(r.payload.keys()) if r.payload else None}")
                    return r
                except subprocess.CalledProcessError as e:
                    error_msg = f"Subprocess failed (exit code {e.returncode}): {e.stderr if e.stderr else e.stdout if e.stdout else str(e)}"
                    logger.error(f"SpeakerDiarization | {error_msg}")
                    from src.core.base_extractor import ExtractorResult  # type: ignore
                    return ExtractorResult(
                        success=False,
                        error=error_msg,
                        payload={},
                        processing_time=0.0,
                    )
                except Exception as e:
                    error_msg = f"Unexpected error running subprocess: {str(e)}"
                    logger.error(f"SpeakerDiarization | {error_msg}", exc_info=True)
                    from src.core.base_extractor import ExtractorResult  # type: ignore
                    return ExtractorResult(
                        success=False,
                        error=error_msg,
                        payload={},
                        processing_time=0.0,
                    )
                finally:
                    # Очищаем временные файлы
                    try:
                        if segments_file_path and os.path.exists(segments_file_path):
                            os.unlink(segments_file_path)
                        if os.path.exists(result_file_path):
                            os.unlink(result_file_path)
                    except Exception:
                        pass
            
            r = retry_with_backoff(
                run_diar_subprocess, max_attempts=2, backoff_base=1.0,
                retry_on=["503", "504", "timeout", "connection", "triton"],
            )
        else:
            # Запускаем напрямую (как раньше)
            callback = create_progress_callback(
                extractor_key, component_name, platform_id, video_id, run_id,
                extractor_idx, total_extractors, "segments", t_start=t_start
            )
            # Set progress callback on extractor instance (not passed as parameter)
            extractor.progress_callback = callback
            logger.info(f"SpeakerDiarization | run_single_extractor: audio_path={audio_path}, diar_segments_count={len(diar_segments) if diar_segments else 0}")
            def run_diar():
                result = extractor.run_segments(  # type: ignore
                    audio_path, tmp_dir, diar_segments or []
                )
                logger.info(f"SpeakerDiarization | run_segments returned: success={result.success}, error={result.error}, payload_keys={list(result.payload.keys()) if result.payload else None}")
                return result
            r = retry_with_backoff(
                run_diar, max_attempts=2, backoff_base=1.0,
                retry_on=["503", "504", "timeout", "connection", "triton"],
            )
    
    # Emotion diarization (может запускаться в персональной venv через subprocess)
    elif extractor_key == "emotion_diarization":
        # Проверяем наличие персональной venv
        extractor_dir = Path(__file__).resolve().parent.parent / "extractors" / "emotion_diarization_extractor"
        venv_path = extractor_dir / ".emotion_diarization_venv"
        venv_python = venv_path / "bin" / "python"
        
        if venv_python.exists():
            # Запускаем через subprocess в персональной venv
            logger.info(f"EmotionDiarization | Using isolated venv: {venv_python}")
            
            # Создаем временный файл для сегментов (если есть)
            segments_file = None
            if emo_segments:
                segments_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=tmp_dir)
                json.dump({"families": {"emotion": {"segments": emo_segments}}}, segments_file)
                segments_file.close()
                segments_file_path = segments_file.name
            else:
                segments_file_path = None
            
            # Создаем временный файл для результата
            result_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=tmp_dir)
            result_file.close()
            result_file_path = result_file.name
            
            # Собираем аргументы для wrapper скрипта
            wrapper_script = extractor_dir / "run_in_venv.py"
            cmd = [
                str(venv_python),
                str(wrapper_script),
                "--audio-path", audio_path,
                "--tmp-dir", tmp_dir,
                "--output-json", result_file_path,
                "--device", getattr(extractor, "device", "auto"),
                "--model-size", getattr(extractor, "model_size", "small"),
                "--sample-rate", str(getattr(extractor, "sample_rate", 16000)),
                "--batch-size", str(getattr(extractor, "batch_size", 16)),
            ]
            
            # Добавляем сегменты если есть
            if segments_file_path:
                cmd.extend(["--segments-json", segments_file_path])
            
            # Добавляем process_full_audio флаг
            if getattr(extractor, "process_full_audio", False):
                cmd.append("--process-full-audio")
            
            # Добавляем feature flags
            if getattr(extractor, "enable_probs", False):
                cmd.append("--enable-probs")
            if getattr(extractor, "enable_ids", False):
                cmd.append("--enable-ids")
            if getattr(extractor, "enable_confidence", False):
                cmd.append("--enable-confidence")
            if getattr(extractor, "enable_mean_probs", False):
                cmd.append("--enable-mean-probs")
            if getattr(extractor, "enable_entropy", False):
                cmd.append("--enable-entropy")
            if getattr(extractor, "enable_dominant", False):
                cmd.append("--enable-dominant")
            if getattr(extractor, "enable_quality_metrics", False):
                cmd.append("--enable-quality-metrics")
            if not getattr(extractor, "enable_silence_detection", True):
                cmd.append("--disable-silence-detection")
            
            # Настраиваем окружение
            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            prev_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(repo_root) if not prev_pp else (str(repo_root) + os.pathsep + prev_pp)
            
            def run_emotion_subprocess():
                try:
                    result = subprocess.run(
                        cmd,
                        check=False,  # Не выбрасываем исключение, проверяем exit code вручную
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    
                    # Если exit code не 0, но файл результата существует, пытаемся загрузить его
                    # (возможно, это просто warnings в stderr)
                    if result.returncode != 0:
                        # Проверяем, есть ли файл результата (возможно, скрипт успешно выполнился, но были warnings)
                        if os.path.exists(result_file_path):
                            try:
                                with open(result_file_path, "r") as f:
                                    result_dict = json.load(f)
                                # Если результат успешный, игнорируем exit code (это были только warnings)
                                if result_dict.get("success", False):
                                    from src.core.base_extractor import ExtractorResult  # type: ignore
                                    r = ExtractorResult(
                                        name="emotion_diarization_extractor",
                                        version="3.0.0",
                                        success=True,
                                        error=None,
                                        payload=result_dict.get("payload", {}),
                                        processing_time=result_dict.get("processing_time", 0.0),
                                    )
                                    logger.info(f"EmotionDiarization | subprocess returned: success={r.success}, payload_keys={list(r.payload.keys()) if r.payload else None}")
                                    return r
                            except Exception:
                                pass
                        # Если файла нет или результат неуспешный, это реальная ошибка
                        error_msg = f"Subprocess failed (exit code {result.returncode})"
                        if result.stderr:
                            # Фильтруем warnings и DEBUG логи из stderr
                            stderr_lines = result.stderr.split('\n')
                            error_lines = [
                                line for line in stderr_lines 
                                if 'UserWarning' not in line 
                                and 'Warning' not in line 
                                and 'DEBUG:' not in line
                                and 'speechbrain.utils.checkpoints' not in line
                                and line.strip()
                                and not line.strip().startswith('DEBUG:')
                            ]
                            if error_lines:
                                # Берем только реальные ошибки (не DEBUG, не warnings)
                                real_errors = [line for line in error_lines if 'ERROR' in line or 'Traceback' in line or 'Exception' in line]
                                if real_errors:
                                    error_msg += f": {''.join(real_errors[:3])}"  # Берем первые 3 строки реальных ошибок
                                elif error_lines:
                                    error_msg += f": {error_lines[0]}"  # Берем первую строку, если нет явных ошибок
                        logger.error(f"EmotionDiarization | {error_msg}")
                        from src.core.base_extractor import ExtractorResult  # type: ignore
                        return ExtractorResult(
                            name="emotion_diarization_extractor",
                            version="3.0.0",
                            success=False,
                            error=error_msg,
                            processing_time=0.0
                        )
                    
                    # Загружаем результат из JSON файла
                    with open(result_file_path, "r") as f:
                        result_dict = json.load(f)
                    
                    # Восстанавливаем ExtractorResult
                    from src.core.base_extractor import ExtractorResult  # type: ignore
                    r = ExtractorResult(
                        name="emotion_diarization_extractor",
                        version="3.0.0",
                        success=result_dict.get("success", False),
                        error=result_dict.get("error"),
                        payload=result_dict.get("payload", {}),
                        processing_time=result_dict.get("processing_time", 0.0),
                    )
                    
                    logger.info(f"EmotionDiarization | subprocess returned: success={r.success}, error={r.error}, payload_keys={list(r.payload.keys()) if r.payload else None}")
                    return r
                except Exception as e:
                    error_msg = f"Subprocess exception: {str(e)}"
                    logger.error(f"EmotionDiarization | {error_msg}")
                    from src.core.base_extractor import ExtractorResult  # type: ignore
                    return ExtractorResult(
                        name="emotion_diarization_extractor",
                        version="3.0.0",
                        success=False,
                        error=error_msg,
                        processing_time=0.0
                    )
                finally:
                    # Cleanup temp files
                    if segments_file_path and os.path.exists(segments_file_path):
                        try:
                            os.remove(segments_file_path)
                        except Exception:
                            pass
                    if os.path.exists(result_file_path):
                        try:
                            os.remove(result_file_path)
                        except Exception:
                            pass
            
            # Для subprocess режима: добавляем промежуточные обновления прогресса на основе времени
            # (subprocess не может использовать callback напрямую)
            t_emotion_start = time.time()
            progress_interval = 1.5  # Обновляем каждые 1.5 секунды для более плавного прогресса
            
            def update_progress_periodically():
                """Периодически обновляет прогресс во время выполнения subprocess"""
                import threading
                stop_event = threading.Event()
                
                def progress_loop():
                    iteration = 0
                    while not stop_event.is_set():
                        time.sleep(progress_interval)
                        if stop_event.is_set():
                            break
                        iteration += 1
                        # Симулируем прогресс: 0% -> 50% -> 100% (на основе времени)
                        # Предполагаем, что inference занимает большую часть времени
                        elapsed = time.time() - t_emotion_start
                        # Оценочное время выполнения: ~30-40 секунд для inference
                        estimated_total = 40.0
                        # Прогресс внутри extractor: 0-70% (остальные 30% - это завершение и сохранение)
                        progress_pct_internal = min(70, int((elapsed / estimated_total) * 70))
                        # Базовый прогресс для extractor: 10% + (extractor_idx / total_extractors) * 70%
                        base_progress = 10 + int((extractor_idx / total_extractors) * 70)
                        progress_pct = base_progress + progress_pct_internal
                        
                        # Определяем этап на основе времени
                        if elapsed < 2.0:
                            stage_name = "Loading audio"
                        elif elapsed < 5.0:
                            stage_name = "Preprocessing"
                        else:
                            stage_name = f"Inference: {elapsed:.1f}s"
                        
                        from src.utils.progress import emit_progress
                        emit_progress(
                            platform_id=platform_id,
                            video_id=video_id,
                            run_id=run_id,
                            component=component_name,
                            stage_id="run_segments",
                            stage_name=stage_name,
                            progress_pct=min(80, progress_pct),
                            extractor=extractor_key,
                            elapsed_sec=elapsed,
                            total_elapsed_sec=time.time() - t_start if t_start is not None else None,
                        )
                
                thread = threading.Thread(target=progress_loop, daemon=True)
                thread.start()
                return stop_event
            
            # Запускаем периодические обновления прогресса
            stop_event = update_progress_periodically()
            try:
                r = run_emotion_subprocess()
            finally:
                stop_event.set()
        else:
            # Fallback: запуск напрямую (без изолированной venv)
            logger.info(f"EmotionDiarization | venv not found at {venv_python}, running directly")
            callback = create_progress_callback(
                extractor_key, component_name, platform_id, video_id, run_id,
                extractor_idx, total_extractors, "batches", t_start=t_start
            )
            extractor.progress_callback = callback
            r = extractor.run_segments(  # type: ignore
                audio_path, tmp_dir, emo_segments or []
            )
    
    # Source separation (inprocess PyTorch model)
    elif extractor_key == "source_separation":
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "batches", t_start=t_start
        )
        # Set progress callback on extractor instance (not passed as parameter)
        extractor.progress_callback = callback
        r = extractor.run_segments(  # type: ignore
            audio_path, tmp_dir, sep_segments or []
        )
    
    # Speech analysis (bundle API)
    elif extractor_key == "speech_analysis":
        callback = create_progress_callback(
            extractor_key, component_name, platform_id, video_id, run_id,
            extractor_idx, total_extractors, "generic", t_start=t_start
        )
        # Set progress callback on extractor instance (not passed as parameter)
        extractor.progress_callback = callback
        
        # Извлекаем результаты зависимых компонентов из extractor_results
        # (новая архитектура: компонент использует существующие результаты вместо запуска под-экстракторов)
        asr_result = extractor_results.get("asr")  # ExtractorResult от asr_extractor
        diarization_result = extractor_results.get("speaker_diarization")  # ExtractorResult от speaker_diarization_extractor
        pitch_result = extractor_results.get("pitch")  # ExtractorResult от pitch_extractor (опционально)
        
        r = extractor.run_bundle(  # type: ignore
            audio_path, tmp_dir,
            asr_segments=asr_segments or [],
            diar_segments=diar_segments or [],
            asr_result=asr_result,
            diarization_result=diarization_result,
            pitch_result=pitch_result,
        )
    
    # Остальные extractors с progress callback
    else:
        # Определяем segments для extractor'а
        segments_map = {
            "pitch": pitch_segments,
            "spectral": spectral_segments,
            "quality": quality_segments,
            "mfcc": mfcc_segments,
            "mel": mel_segments,
            "onset": onset_segments,
            "chroma": chroma_segments,
            "rhythmic": rhythmic_segments,
            "voice_quality": voice_quality_segments,
            "hpss": hpss_segments,
            "key": key_segments,
            "band_energy": band_energy_segments,
            "spectral_entropy": spectral_entropy_segments,
        }
        segments = segments_map.get(extractor_key)
        
        if segments is not None:
            callback = create_progress_callback(
                extractor_key, component_name, platform_id, video_id, run_id,
                extractor_idx, total_extractors, "generic", t_start=t_start
            )
            
            # Устанавливаем progress callback и artifacts_dir
            extractor.progress_callback = callback
            extractor.artifacts_dir = str(Path(run_rs_path) / component_name / "_artifacts")
            
            # Специальная обработка для некоторых extractors
            if extractor_key == "onset" and "tempo" in extractor_results:
                # Интеграция с tempo_extractor
                tempo_payload = extractor_results.get("tempo", {}).get("payload")
                if isinstance(tempo_payload, dict):
                    extractor.tempo_payload = tempo_payload
            
            elif extractor_key == "voice_quality":
                # Интеграция с pitch_extractor (если доступен)
                if "pitch" in extractor_results:
                    pitch_result = extractor_results.get("pitch")
                    pitch_payload = None
                    if pitch_result and pitch_result.get("success") and isinstance(pitch_result.get("payload"), dict):
                        pitch_payload = pitch_result.get("payload")
                    extractor.pitch_payload = pitch_payload
                
                # Оптимизация: используем segment_parallelism для voice_quality
                voice_quality_workers = get_extractor_parallelism("voice_quality", "segment_workers", segment_parallelism, extractor_parallelism_config)
                voice_quality_max_inflight = get_extractor_parallelism("voice_quality", "max_inflight", max_inflight if max_inflight else voice_quality_workers, extractor_parallelism_config)
                try:
                    r = extractor.run_segments(  # type: ignore
                        audio_path, tmp_dir, segments,
                        segment_parallelism=voice_quality_workers, max_inflight=voice_quality_max_inflight
                    )
                    effective = {"segment_parallelism": voice_quality_workers, "max_inflight": voice_quality_max_inflight}
                except TypeError:
                    r = extractor.run_segments(audio_path, tmp_dir, segments)  # type: ignore
                    effective = {"segment_parallelism": 1, "max_inflight": 1}
                return r, effective
            
            elif extractor_key == "hpss":
                # Оптимизация: используем segment_parallelism для hpss
                hpss_workers = get_extractor_parallelism("hpss", "segment_workers", segment_parallelism, extractor_parallelism_config)
                hpss_max_inflight = get_extractor_parallelism("hpss", "max_inflight", max_inflight if max_inflight else hpss_workers, extractor_parallelism_config)
                try:
                    r = extractor.run_segments(  # type: ignore
                        audio_path, tmp_dir, segments,
                        segment_parallelism=hpss_workers, max_inflight=hpss_max_inflight
                    )
                    effective = {"segment_parallelism": hpss_workers, "max_inflight": hpss_max_inflight}
                except TypeError:
                    r = extractor.run_segments(audio_path, tmp_dir, segments)  # type: ignore
                    effective = {"segment_parallelism": 1, "max_inflight": 1}
                return r, effective
            
            elif extractor_key == "key":
                # Переиспользование chroma матрицы (если доступен существующий результат)
                # key_extractor может использовать chroma из chroma_extractor, но не требует его запуска
                shared_features = None
                if "chroma" in extractor_results:
                    chroma_result = extractor_results.get("chroma")
                    if chroma_result and chroma_result.get("success"):
                        chroma_payload = chroma_result.get("payload", {})
                        if isinstance(chroma_payload, dict):
                            chroma_ts = chroma_payload.get("chroma")
                            if chroma_ts is None:
                                chroma_npy = chroma_payload.get("chroma_npy")
                                if isinstance(chroma_npy, str) and chroma_npy:
                                    npy_path = chroma_npy
                                    if not os.path.isabs(npy_path):
                                        npy_path = os.path.join(run_rs_path, "chroma_extractor", npy_path)
                                    try:
                                        if os.path.exists(npy_path):
                                            chroma_ts = np.load(npy_path)
                                    except Exception:
                                        chroma_ts = None
                            if chroma_ts is not None:
                                try:
                                    shared_features = {"chroma": np.asarray(chroma_ts, dtype=np.float32)}
                                except Exception:
                                    shared_features = None
                
                # Get parallelism settings for key extractor
                key_workers = get_extractor_parallelism(
                    "key", "segment_workers", segment_parallelism, extractor_parallelism_config
                )
                key_max_inflight = get_extractor_parallelism(
                    "key", "max_inflight", max_inflight, extractor_parallelism_config
                )
                
                try:
                    r = extractor.run_segments(
                        audio_path, tmp_dir, segments,
                        shared_features=shared_features,
                        segment_parallelism=key_workers,
                        max_inflight=key_max_inflight
                    )  # type: ignore
                    effective = {"segment_parallelism": key_workers, "max_inflight": key_max_inflight}
                except TypeError:
                    r = extractor.run_segments(audio_path, tmp_dir, segments, shared_features=shared_features)  # type: ignore
                    effective = {"segment_parallelism": 1, "max_inflight": 1}
                return r, effective
            
            r = extractor.run_segments(audio_path, tmp_dir, segments)  # type: ignore
        else:
            # Fallback на run()
            r = extractor.run(audio_path, tmp_dir)
    
    return r, effective


def run_extractors(
    processor: Any,
    extractor_keys: List[str],
    audio_path: str,
    tmp_dir: str,
    segments_payload: Optional[Dict[str, Any]],
    run_rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    segment_parallelism: int,
    max_inflight: int,
    clap_batch_size: int,
    extractor_parallelism_config: Optional[Dict[str, Dict[str, Any]]] = None,
    strict_extractors: bool = True,
    t_start: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Запускает все extractors и возвращает результаты.
    
    Returns:
        Tuple[extractor_results, per_extractor_report, timings_by_extractor]
    """
    extractor_results = {}
    per_extractor_report = {}
    timings_by_extractor = {}
    total_extractors = len(extractor_keys)
    
    # Store t_start for progress callbacks (use provided t_start or current time as fallback)
    _t_start = t_start if t_start is not None else time.time()
    
    # Извлекаем segments из payload
    families = segments_payload.get("families") or {} if segments_payload else {}
    primary = (families.get("primary") or {}) if isinstance(families, dict) else {}
    clap_f = (families.get("clap") or {}) if isinstance(families, dict) else {}
    tempo_f = (families.get("tempo") or {}) if isinstance(families, dict) else {}
    asr_f = (families.get("asr") or {}) if isinstance(families, dict) else {}
    diar_f = (families.get("diarization") or {}) if isinstance(families, dict) else {}
    emo_f = (families.get("emotion") or {}) if isinstance(families, dict) else {}
    sep_f = (families.get("source_separation") or {}) if isinstance(families, dict) else {}
    
    primary_segments = primary.get("segments") or []
    clap_segments = clap_f.get("segments") or []
    tempo_segments = tempo_f.get("segments") or []
    asr_segments = asr_f.get("segments") or []
    diar_segments = diar_f.get("segments") or []
    emo_segments = emo_f.get("segments") or []
    sep_segments = sep_f.get("segments") or []
    
    # Дополнительные семейства
    pitch_f = families.get("pitch", {})
    pitch_segments = pitch_f.get("segments") or []
    spectral_f = families.get("spectral", {})
    spectral_segments = spectral_f.get("segments") or []
    quality_f = families.get("quality", {})
    quality_segments = quality_f.get("segments") or []
    mfcc_f = families.get("mfcc", {})
    mfcc_segments = mfcc_f.get("segments") or []
    mel_f = families.get("mel", {})
    mel_segments = mel_f.get("segments") or []
    onset_f = families.get("onset", {})
    onset_segments = onset_f.get("segments") or []
    chroma_f = families.get("chroma", {})
    chroma_segments = chroma_f.get("segments") or []
    rhythmic_f = families.get("rhythmic", {})
    rhythmic_segments = rhythmic_f.get("segments") or []
    voice_quality_f = families.get("voice_quality", {})
    voice_quality_segments = voice_quality_f.get("segments") or []
    hpss_f = families.get("hpss", {})
    hpss_segments = hpss_f.get("segments") or []
    key_f = families.get("key", {})
    key_segments = key_f.get("segments") or []
    band_energy_f = families.get("band_energy", {})
    band_energy_segments = band_energy_f.get("segments") or []
    # Fallback для band_energy: если family отсутствует, используем spectral или primary segments
    if not band_energy_segments:
        if spectral_segments:
            band_energy_segments = spectral_segments
        elif primary_segments:
            band_energy_segments = primary_segments
    spectral_entropy_f = families.get("spectral_entropy", {})
    spectral_entropy_segments = spectral_entropy_f.get("segments") or []
    # Fallback для spectral_entropy: если family отсутствует, используем spectral или primary segments
    if not spectral_entropy_segments:
        if spectral_segments:
            spectral_entropy_segments = spectral_segments
        elif primary_segments:
            spectral_entropy_segments = primary_segments
    # Fallback для pitch: если family отсутствует, используем spectral или primary segments
    if not pitch_segments:
        if spectral_segments:
            pitch_segments = spectral_segments
        elif primary_segments:
            pitch_segments = primary_segments
    for idx, key in enumerate(extractor_keys):
        logger.info(f"AudioProcessor | [{idx + 1}/{total_extractors}] Initializing extractor: {key}")
        extractor = processor.extractors.get(key)
        if extractor is None:
            error_msg = "extractor_not_available"
            if strict_extractors:
                raise RuntimeError(f"AudioProcessor | extractor '{key}' is required but not available (fail-fast)")
            safe_log_warning(logger, f"AudioProcessor | extractor '{key}' is not available, skipping (graceful degradation)")
            extractor_results[key] = {"success": False, "payload": None, "error": error_msg, "device_used": "unknown"}
            per_extractor_report[key] = {"status": "error", "error": error_msg}
            continue
        
        # Проверка обязательных зависимостей перед запуском extractor'а
        required_deps = REQUIRED_EXTRACTOR_DEPENDENCIES.get(key, [])
        missing_required = []
        for dep in required_deps:
            dep_result = extractor_results.get(dep)
            if dep_result is None or not dep_result.get("success", False):
                missing_required.append(dep)
        
        if missing_required:
            error_msg = f"missing_required_dependencies: {missing_required}"
            if strict_extractors:
                raise RuntimeError(
                    f"AudioProcessor | extractor '{key}' requires dependencies: {missing_required}. "
                    f"These dependencies must be enabled and executed before '{key}'. "
                    f"Ensure they are in --extractors list and executed successfully."
                )
            safe_log_warning(
                logger,
                f"AudioProcessor | extractor '{key}' requires dependencies: {missing_required}, "
                f"but they are missing or failed. Skipping '{key}' (graceful degradation)."
            )
            extractor_results[key] = {"success": False, "payload": None, "error": error_msg, "device_used": "unknown"}
            per_extractor_report[key] = {"status": "error", "error": error_msg}
            continue
        
        # Проверка опциональных зависимостей (предупреждение, не ошибка)
        optional_deps = OPTIONAL_EXTRACTOR_DEPENDENCIES.get(key, [])
        missing_optional = []
        for dep in optional_deps:
            dep_result = extractor_results.get(dep)
            if dep_result is None or not dep_result.get("success", False):
                missing_optional.append(dep)
        
        if missing_optional:
            safe_log_warning(
                logger,
                f"AudioProcessor | extractor '{key}' has optional dependencies: {missing_optional}, "
                f"but they are missing or failed. Component may have suboptimal performance."
            )
        
        try:
            logger.info(f"AudioProcessor | [{idx + 1}/{total_extractors}] Running extractor: {key}")
            t_e0 = time.time()
            r, effective = run_single_extractor(
                extractor_key=key,
                extractor=extractor,
                audio_path=audio_path,
                tmp_dir=tmp_dir,
                segments_payload=segments_payload,
                run_rs_path=run_rs_path,
                extractor_results=extractor_results,
                extractor_idx=idx,
                total_extractors=total_extractors,
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                segment_parallelism=segment_parallelism,
                max_inflight=max_inflight,
                clap_batch_size=clap_batch_size,
                extractor_parallelism_config=extractor_parallelism_config,
                t_start=_t_start,
                primary_segments=primary_segments,
                clap_segments=clap_segments,
                tempo_segments=tempo_segments,
                asr_segments=asr_segments,
                diar_segments=diar_segments,
                emo_segments=emo_segments,
                sep_segments=sep_segments,
                pitch_segments=pitch_segments,
                spectral_segments=spectral_segments,
                quality_segments=quality_segments,
                mfcc_segments=mfcc_segments,
                mel_segments=mel_segments,
                onset_segments=onset_segments,
                chroma_segments=chroma_segments,
                rhythmic_segments=rhythmic_segments,
                voice_quality_segments=voice_quality_segments,
                hpss_segments=hpss_segments,
                key_segments=key_segments,
                band_energy_segments=band_energy_segments,
                spectral_entropy_segments=spectral_entropy_segments,
            )
            
            t_e1 = time.time()
            wall_time_ms = float((t_e1 - t_e0) * 1000.0)
            
            extractor_results[key] = {
                "success": bool(r.success),
                "payload": r.payload if isinstance(r.payload, dict) else None,
                "error": r.error,
                "processing_time": r.processing_time,
                "device_used": r.device_used,
            }
            if not bool(r.success):
                safe_log_warning(logger, f"AudioProcessor | Extractor '{key}' failed: {r.error}")
            per_extractor_report[key] = {
                "status": "ok" if bool(r.success) else "error",
                "wall_ms": wall_time_ms,  # Будет перезаписано в вызывающем коде
                "reported_ms": float((float(r.processing_time or 0.0)) * 1000.0),
                "segments_count": (
                    int((r.payload or {}).get("segments_count"))
                    if isinstance(r.payload, dict) and (r.payload.get("segments_count") is not None)
                    else None
                ),
                "effective_knobs": effective,
            }
            timings_by_extractor[key] = {
                "wall_ms": wall_time_ms,  # Будет перезаписано в вызывающем коде
                "reported_ms": float((float(r.processing_time or 0.0)) * 1000.0),
            }
            
            # Обновляем прогресс после завершения экстрактора
            # Более точный расчет: используем (idx + 1) для прогресса после завершения
            if total_extractors > 0:
                progress_pct = int(10 + ((idx + 1) / total_extractors) * 70)
            else:
                progress_pct = 10
            total_elapsed = time.time() - t_start if t_start is not None else 0.0
            emit_progress(
                platform_id=platform_id,
                video_id=video_id,
                run_id=run_id,
                component="audio_processor",
                stage_id="run_extractors",
                stage_name="Running extractors",
                progress_pct=progress_pct,
                extractor=key,
                elapsed_sec=wall_time_ms / 1000.0,
                total_elapsed_sec=total_elapsed,
            )
        except Exception as e:
            extractor_results[key] = {
                "success": False,
                "payload": None,
                "error": str(e),
                "device_used": getattr(extractor, "device", "unknown"),
            }
            per_extractor_report[key] = {
                "status": "error",
                "wall_ms": None,
                "reported_ms": None,
                "segments_count": None,
                "effective_knobs": None,
                "error": str(e),
            }
    
    return extractor_results, per_extractor_report, timings_by_extractor

