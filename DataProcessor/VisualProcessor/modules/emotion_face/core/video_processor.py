"""
Основной класс для обработки видео и анализа эмоций.

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (core_face_landmarks)
    2. ✅ Использование face presence из core_face_landmarks (вместо устаревшего face_detection)
    3. ✅ Интеграция с BaseModule через класс EmotionFaceModule
    4. ✅ Единый формат вывода для сохранения в npz
"""
import os
import sys
import time
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# VisualProcessor root (for modules.* and utils.frame_manager)
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

# emotion_face/utils contains _utils.py; append (not insert) so VisualProcessor/utils wins for "utils.*".
_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.video_context import VideoContext

from core.processing_config import (
    ProcessingParams, ProcessingMetrics
)

def safe_array_check(arr):
    """
    Безопасная проверка пустых массивов NumPy.
    Используется для избежания ошибки "The truth value of an empty array is ambiguous".
    """
    if arr is None:
        return False
    if isinstance(arr, np.ndarray):
        return arr.size > 0
    if isinstance(arr, (list, tuple)):
        return len(arr) > 0
    return bool(arr)
from core.memory_manager import memory_context, cleanup_memory
from core.retry_strategy import RetryStrategy, QualityMetrics
from core.validation import ValidationLogic, ValidationCriteria
from core.exceptions import (
    VideoProcessingError, FrameSelectionError,
    EmotionAnalysisError, ValidationError
)
from core.validators import (
    validate_target_length
)

from _utils import (
    segmentation, select_from_segments,
    build_emotion_curve, detect_keyframes, compress_sequence,
    expand_sequence, temporal_smoothing, validate_sequence_quality,
    get_video_type,
    analyze_emotion_profile, sample_for_static_face,
    analyze_emotion_changes,
    get_available_memory_mb,
    compute_steps, process_frames_in_batches
)
from core.advanced_emotion_features import (
    detect_micro_expressions,
    compute_face_asymmetry,
    compute_emotional_individuality
)

EMOTION_CLASSES = {
    0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
    4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
}

from utils.logger import get_logger
logger = get_logger("VideoEmotionProcessor")

def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
    """
    try:
        from pathlib import Path as _Path

        platform_id = event.get("platform_id")
        video_id = event.get("video_id")
        run_id = event.get("run_id")
        if not (platform_id and video_id and run_id):
            return
        runs_root = _Path(rs_path).expanduser().resolve()
        p = runs_root / "state" / str(platform_id) / str(video_id) / str(run_id) / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
    message: Optional[str] = None,
) -> None:
    total_i = int(total) if int(total) > 0 else 1
    done_i = int(done)
    pct = float(min(100.0, max(0.0, (done_i / total_i) * 100.0)))
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "event": "progress",
            "component": "emotion_face",
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "stage": str(stage),
            "done": done_i,
            "total": total_i,
            "progress": pct,
            "message": str(message) if message else None,
            "ts": time.time(),
        },
    )

class VideoEmotionProcessor:
    """
    Основной класс для обработки видео и анализа эмоций.
    Реализует все этапы обработки с четким разделением ответственности.
    """
    
    def __init__(
        self,
        # validate
        min_frames_ratio: float = 0.8,
        min_keyframes: int = 3,
        min_transitions: int = 2,
        min_diversity_threshold: float = 0.2,
        quality_threshold: float = 0.4,
        # perfomance
        memory_threshold_low: int = 2000,
        batch_load_low: int = 20,
        batch_process_low: int = 8,
        memory_threshold_medium: int = 4000,
        batch_load_medium: int = 30,
        batch_process_medium: int = 12,
        memory_threshold_high: int = 8000,
        batch_load_high: int = 50,
        batch_process_high: int = 15,
        batch_load_very_high: int = 80,
        batch_process_very_high: int = 24,
        # logging
        enable_structured_metrics: bool = True,
        # processing
        min_faces_threshold: int = 1,
        target_length: int = 256,
        max_retries: int = 2,
        # keyframes
        transition_threshold: float = 0.3,
        # segmentation
        max_gap_seconds: float = 0.5,
        max_samples_per_segment: int = 10,
        # emonet
        emo_path: str = None,
        # other
        device: str = "cuda",
        # BaseModule integration
        rs_path: Optional[str] = None,
        load_dependency_func: Optional[callable] = None,
    ):
        """
        Инициализация процессора.
        
        Args:
            config_path: Путь к файлу конфигурации. Если None, используется config.yaml.
            rs_path: Путь к хранилищу результатов (для загрузки зависимостей)
            load_dependency_func: Функция для загрузки зависимостей через BaseModule
        """
        self.device = device
        self.rs_path = rs_path
        
        self.metrics = ProcessingMetrics()
        
        self.target_length = validate_target_length(target_length)
        self.max_retries = max_retries
        self.transition_threshold = transition_threshold
        self.quality_threshold = quality_threshold
        self.min_diversity_threshold = min_diversity_threshold
        self.max_gap_seconds = max_gap_seconds
        self.max_samples_per_segment = max_samples_per_segment
        self.min_faces_threshold = min_faces_threshold
        
        self.memory_threshold_low = memory_threshold_low
        self.batch_load_low = batch_load_low
        self.batch_process_low = batch_process_low
        self.memory_threshold_medium = memory_threshold_medium
        self.batch_load_medium = batch_load_medium
        self.batch_process_medium = batch_process_medium
        self.memory_threshold_high = memory_threshold_high
        self.batch_load_high = batch_load_high
        self.batch_process_high = batch_process_high
        self.batch_load_very_high = batch_load_very_high
        self.batch_process_very_high = batch_process_very_high

        # Загружаем frames_with_face с поддержкой BaseModule
        self.frames_with_face, self.core_times_by_index, self.core_landmarks_by_index, self.core_face_confidence = (
            self.frames_with_face_load(
            "auto",
            rs_path=rs_path,
            load_func=load_dependency_func,
            )
        )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EmotionFace requires CUDA device, but cuda is not available")

        if emo_path == "None" or emo_path is None:
            import os
            # dp_models/emonet/pretrained/emonet_8.pth
            p = os.path.dirname(os.path.dirname(__file__))
            data_processor_root = os.path.abspath(os.path.join(p, "..", "..", "..", ".."))
            emo_path = os.path.join(data_processor_root, "dp_models", "emonet", "pretrained", "emonet_8.pth")
        
        self.model = self.load_emonet(path=emo_path)
        
        self.validation_logic = ValidationLogic(
            ValidationCriteria(
                min_frames_ratio=min_frames_ratio,
                min_keyframes=min_keyframes,
                min_transitions=min_transitions,
                min_diversity=self.min_diversity_threshold
            )
        )
        
    def frames_with_face_load(
        self,
        filename,
        rs_path: Optional[str] = None,
        load_func: Optional[callable] = None,
    ):
        """
        Загружает список кадров с лицами из результатов core provider `core_face_landmarks`.
        
        Args:
            filename: Имя файла или "auto" для автоматического поиска
            rs_path: Путь к хранилищу результатов (если None, использует старый метод)
            load_func: Функция для загрузки через BaseModule (load_dependency_results)
            
        Returns:
            Список индексов кадров с лицами
        """
        if not rs_path or load_func is None:
            logger.warning(
                "VideoEmotionProcessor | frames_with_face_load | rs_path/load_func not set; "
                "cannot load core_face_landmarks. Returning empty list."
            )
            return [], {}, {}, {}

        try:
            core = load_func("core_face_landmarks", format="npz")
        except Exception as e:
            logger.warning(f"VideoEmotionProcessor | frames_with_face_load | Failed to load core_face_landmarks: {e}")
            return [], {}, {}, {}

        if not isinstance(core, dict):
            return [], {}, {}, {}

        fi = core.get("frame_indices")
        face_present = core.get("face_present")
        face_landmarks = core.get("face_landmarks")
        times_s = core.get("times_s")
        if fi is None or face_present is None or face_landmarks is None:
            logger.warning("VideoEmotionProcessor | frames_with_face_load | core_face_landmarks missing frame_indices/face_present")
            return [], {}, {}, {}

        try:
            fi_np = np.asarray(fi, dtype=np.int32)
            fp_np = np.asarray(face_present, dtype=bool)
            lm_np = np.asarray(face_landmarks)
            ts_np = np.asarray(times_s) if times_s is not None else None
            if fi_np.shape[0] != fp_np.shape[0]:
                logger.warning(
                    "VideoEmotionProcessor | frames_with_face_load | "
                    f"shape mismatch: frame_indices={fi_np.shape} face_present={fp_np.shape}"
                )
                return [], {}, {}, {}

            frames_with_face: List[int] = []
            times_by_index: Dict[int, float] = {}
            landmarks_by_index: Dict[int, np.ndarray] = {}
            face_confidence: Dict[int, float] = {}

            for i, frame_idx in enumerate(fi_np):
                present_row = fp_np[i]
                if present_row.ndim == 0:
                    has_face = bool(present_row)
                else:
                    has_face = bool(np.any(present_row))
                if not has_face:
                    continue

                frame_idx_int = int(frame_idx)
                frames_with_face.append(frame_idx_int)
                if ts_np is not None and i < len(ts_np):
                    times_by_index[frame_idx_int] = float(ts_np[i])

                # Берем первое присутствующее лицо как primary
                face_idx = 0
                if present_row.ndim > 0:
                    for f in range(len(present_row)):
                        if bool(present_row[f]):
                            face_idx = f
                            break
                try:
                    landmarks_by_index[frame_idx_int] = lm_np[i, face_idx]
                except Exception:
                    pass

                # Базовая уверенность: 1.0 если лицо есть
                face_confidence[frame_idx_int] = 1.0

            logger.info(
                f"VideoEmotionProcessor | frames_with_face_load | loaded {len(frames_with_face)} face frames from core_face_landmarks"
            )
            return sorted(frames_with_face), times_by_index, landmarks_by_index, face_confidence
        except Exception as e:
            logger.warning(f"VideoEmotionProcessor | frames_with_face_load | parse error: {e}")
            return [], {}, {}, {}

    def load_emonet(self, path: str, n_expression: int = 8):
        import importlib.util
        # dp_models/emonet/emonet/models/emonet.py
        # Find DataProcessor root: core/video_processor.py -> emotion_face -> modules -> VisualProcessor -> DataProcessor
        current_file = os.path.abspath(__file__)
        if "DataProcessor" in current_file:
            parts = current_file.split("DataProcessor")
            if len(parts) > 1:
                dp_root = os.path.join(parts[0], "DataProcessor")
                emonet_py = os.path.join(dp_root, "dp_models", "emonet", "emonet", "models", "emonet.py")
            else:
                raise RuntimeError("Could not determine DataProcessor root from file path")
        else:
            raise RuntimeError("Could not find DataProcessor in file path")
        
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
        
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model = EmoNet(n_expression=n_expression).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    
    def process(
        self,
        frame_manager,
        save_path
    ) -> Dict[str, Any]:
        """
        Основной метод обработки видео.
        
        Returns:
            Словарь с результатами обработки.
        
        Raises:
            VideoFileError: Если видео файл некорректен.
            ConfigurationValidationError: Если параметры некорректны.
            VideoProcessingError: При ошибках обработки.
        """
        retry_strategy = RetryStrategy(max_retries=self.max_retries)
        
        # Инициализация параметров
        base_params = ProcessingParams(
            scan_stride_multiplier=1.0,
            keyframe_threshold=self.transition_threshold,
            quality_threshold=self.quality_threshold,
            min_diversity=self.min_diversity_threshold,
            segment_max_gap=self.max_gap_seconds,
            samples_per_segment=self.max_samples_per_segment
        )
        
        current_params = base_params.copy()
        
        try:
            with memory_context():
                    
                total_frames = frame_manager.total_frames
                fps = frame_manager.fps
                meta = frame_manager.meta
                
                # Основной цикл обработки с повторными попытками
                while retry_strategy.attempts <= self.max_retries:
                    try:
                        cleanup_memory()
                        
                        logger.info(f"Попытка {retry_strategy.attempts + 1}/{self.max_retries + 1}")
                        
                        result = self._adaptive_frame_selection(
                            frame_manager, self.frames_with_face, total_frames, fps, current_params
                        )
                        
                        if not result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    result.get("video_type", "UNKNOWN"),
                                    result.get("segments_count", 0),
                                    result.get("faces_found", 0),
                                    log_func=logger.info
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        selected_indices = result["selected_indices"]
                        timeline = result["timeline"]
                        segments = result["segments"]
                        video_type = result["video_type"]
                        
                        emotion_result = self._emotion_analysis_pipeline(
                            frame_manager, selected_indices, self.model, fps
                        )
                        
                        if not emotion_result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    logger.info
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        # Этап 3: Валидация и нормализация
                        validation_result = self._validation_and_retry_logic(
                            emotion_result,
                            selected_indices,
                            self.target_length,
                            video_type,
                            current_params,
                            retry_strategy
                        )
                        
                        # Логируем метрики качества
                        if validation_result.get("quality_metrics"):
                            quality_dict = validation_result["quality_metrics"].to_dict()
                        
                        if validation_result["should_retry"]:
                            if retry_strategy.should_retry(validation_result["quality_metrics"]):
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    validation_result["quality_metrics"],
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    logger.info
                                )
                                retry_strategy.next_attempt()
                                continue
                        
                        # Успешная обработка
                        result = self._save_results(
                            validation_result,
                            emotion_result,
                            selected_indices,
                            save_path,
                            meta,
                            current_params,
                            retry_strategy.attempts + 1,
                            video_type,
                            len(segments),
                            len(timeline)
                        )
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Ошибка в попытке {retry_strategy.attempts}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        if retry_strategy.next_attempt():
                            current_params = retry_strategy.get_safe_params()
                            logger.info("Переход к безопасным параметрам")
                            continue
                        else:
                            return self._build_failure_result(current_params, retry_strategy.attempts, str(e))
                
                # Все попытки неудачны
                return self._build_failure_result(current_params, retry_strategy.attempts)
        
        finally:
            torch.cuda.empty_cache()
    
    def _adaptive_frame_selection(
        self,
        fm,
        timeline,
        total_frames: int,
        fps: float,
        params: ProcessingParams
    ) -> Dict[str, Any]:
        """
        Этап 1: Адаптивный сбор кадров с лицами.
        
        Returns:
            Словарь с результатами: success, selected_indices, timeline, segments, video_type
        """
        try:
            # Динамическое сканирование
            scan_stride, target_scans = compute_steps(total_frames)
            adjusted_scan_stride = int(scan_stride * params.scan_stride_multiplier)
            adjusted_scan_stride = max(1, adjusted_scan_stride)
            
            logger.info(f"Scan stride: {scan_stride} -> {adjusted_scan_stride}")
            
            logger.info(f"Найдено лиц: {len(timeline)} (порог: {params.face_detection_threshold})")
            
            # Сегментация
            segments = segmentation(
                timeline,
                fps=fps,
                max_gap_seconds=params.segment_max_gap
            )
            logger.info(f"Создано {len(segments)} сегментов")
            
            # Определение типа видео
            video_type = get_video_type(timeline, total_frames, segments)
            logger.info(f"Тип видео: {video_type}")
            
            # Выборка кадров
            if video_type == "STATIC_FACE":
                selected_indices = sample_for_static_face(
                    segments,
                    total_frames,
                    fps,
                    target_samples=min(150, total_frames)
                )
            else:
                selected_indices = select_from_segments(
                    segments,
                    total_frames,
                    fps=fps,
                    max_samples_per_segment=params.samples_per_segment
                )
            
            n_frames = len(selected_indices)
            logger.info(f"Выбрано индексов: {n_frames}")
            
            if n_frames < self.min_faces_threshold:
                logger.info(
                    f"Найдено кадров с лицами меньше порога ({n_frames} < {self.min_faces_threshold})"
                )
            

            logger.info(f"Кол-во кадров на выходе 1 Этапа: {len(selected_indices)}")

            return {
                "success": len(selected_indices) > 0,
                "selected_indices": selected_indices,
                "timeline": timeline,
                "segments": segments,
                "video_type": video_type,
                "faces_found": len(timeline),
                "segments_count": len(segments)
            }
        
        except Exception as e:
            logger.error(f"Ошибка в _adaptive_frame_selection: {e}")
            if isinstance(e, VideoProcessingError):
                raise
            raise FrameSelectionError(
                f"Failed to select frames: {e}",
                details={"error": str(e), "total_frames": total_frames}
            ) from e
    
    def _emotion_analysis_pipeline(
        self,
        fm,
        selected_indices: List[int],
        model,
        fps: float,
    ) -> Dict[str, Any]:
        """
        Этап 2: Анализ эмоций в выбранных кадрах.
        
        Returns:
            Словарь с результатами: success, emo_results, emotion_profile, change_analysis
        """
        try:
            # Определение размера батча
            available_memory = get_available_memory_mb()
            
            if available_memory < self.memory_threshold_low:
                batch_load = self.batch_load_low
                batch_process = self.batch_process_low
            elif available_memory < self.memory_threshold_medium:
                batch_load = self.batch_load_medium
                batch_process = self.batch_process_medium
            elif available_memory < self.memory_threshold_high:
                batch_load = self.batch_load_high
                batch_process = self.batch_process_high
            else:
                batch_load = self.batch_load_very_high
                batch_process = self.batch_process_very_high
            
            logger.info(f"Размеры батчей: загрузка={batch_load}, обработка={batch_process}")
            
            # Обработка кадров батчами
            # face_confidence: берём из core_face_landmarks (1.0 для кадров с лицами)
            face_confidence = [self.core_face_confidence.get(int(idx), 1.0) for idx in selected_indices]
            emo_results = process_frames_in_batches(
                fm,
                selected_indices,
                model,
                logger.info,
                batch_size_load=batch_load,
                batch_size_process=batch_process,
                face_confidence=face_confidence
            )
            
            # Анализ эмоционального профиля
            emotion_profile = analyze_emotion_profile(emo_results)
            logger.info(f"Доминирующая эмоция: {emotion_profile['dominant_emotion']}")
            
            # Анализ изменений эмоций
            change_analysis = analyze_emotion_changes(emo_results)
            logger.info(f"Тип изменений: {change_analysis['change_type']}")
            
            # Расширенные фичи: микроэмоции (с улучшенными параметрами)
            logger.info("Вычисление микроэмоций...")
            microexpressions = detect_micro_expressions(
                emo_results, 
                fps=fps,
                min_frames=2,  # Require at least 2 frames (0.03s at 30fps ≈ 1 frame is too short)
                change_threshold=None  # Use adaptive threshold (85th percentile)
            )
            logger.info(f"Найдено микроэмоций: {microexpressions['microexpressions_count']}")
            
            # Расширенные фичи: индивидуальность выражения эмоций
            logger.info("Анализ индивидуальности выражения эмоций...")
            emotional_individuality = compute_emotional_individuality(emo_results, fps=fps)
            logger.info(f"Индекс выразительности: {emotional_individuality['expressivity_index']:.2f}")
            
            # Расширенные фичи: асимметрия лица на основе landmarks
            logger.info("Анализ асимметрии лица (landmarks)...")
            asymmetry_series = []
            for idx in selected_indices:
                lm = self.core_landmarks_by_index.get(int(idx))
                if lm is None:
                    asymmetry_series.append(None)
                    continue
                asym = compute_face_asymmetry(landmarks=lm)
                asymmetry_series.append(asym.get("asymmetry_score"))
            valid_asym = [a for a in asymmetry_series if a is not None]
            face_asymmetry = {
                "asymmetry_series": asymmetry_series,
                "asymmetry_mean": float(np.mean(valid_asym)) if valid_asym else 0.0,
                "asymmetry_std": float(np.std(valid_asym)) if valid_asym else 0.0,
            }
            
            return {
                "success": True,
                "emo_results": emo_results,
                "emotion_profile": emotion_profile,
                "change_analysis": change_analysis,
                "microexpressions": microexpressions,
                "emotional_individuality": emotional_individuality,
                "face_asymmetry": face_asymmetry
            }
        
        except Exception as e:
            logger.error(f"Ошибка в _emotion_analysis_pipeline: {e}")
            if isinstance(e, VideoProcessingError):
                raise
            raise EmotionAnalysisError(
                f"Failed to analyze emotions: {e}",
                details={"error": str(e), "frames_count": len(selected_indices)}
            ) from e
    
    def _validation_and_retry_logic(
        self,
        emotion_result: Dict[str, Any],
        selected_indices: List[int],
        target_length: int,
        video_type: str,
        params: ProcessingParams,
        retry_strategy: RetryStrategy
    ) -> Dict[str, Any]:
        """
        Этап 3: Валидация и нормализация последовательности.
        
        Returns:
            Словарь с результатами: should_retry, quality_metrics, final_indices, smoothed_emotions, keyframes
        """
        try:
            emo_results = emotion_result["emo_results"]
            emotion_profile = emotion_result["emotion_profile"]
            neutral_percentage = emotion_profile["neutral_percentage"]
            
            # Построение кривой эмоций
            emotion_curve = build_emotion_curve(emo_results)
            
            # Детекция ключевых кадров с улучшенными параметрами
            keyframes_indices = detect_keyframes(
                emotion_curve,
                EMOTION_CLASSES,
                threshold=params.keyframe_threshold,
                prominence=0.1,  # Normalized prominence threshold
                min_distance=8  # Minimum distance between peaks (8-12 frames recommended)
            )
            
            # Нормализация до target_length
            n_frames = len(selected_indices)
            if n_frames == target_length:
                final_indices = selected_indices
                final_emotions = emo_results
            elif n_frames > target_length:
                final_indices, final_emotions = compress_sequence(
                    selected_indices,
                    emo_results,
                    keyframes_indices,
                    target_length
                )
            else:
                final_indices, final_emotions = expand_sequence(
                    selected_indices,
                    emo_results,
                    keyframes_indices,
                    target_length
                )
            
            # Сглаживание
            smoothed_emotions = temporal_smoothing(final_emotions, window=3)
            
            # Валидация качества
            quality_metrics_raw = validate_sequence_quality(
                smoothed_emotions,
                min_diversity_threshold=params.min_diversity,
                is_static_face=(video_type == "STATIC_FACE"),
                neutral_percentage=neutral_percentage,
                logger=logger
            )
            
            # Единая логика валидации
            quality_metrics = self.validation_logic.validate_quality(
                smoothed_emotions,
                quality_metrics_raw,
                target_length,
                len(keyframes_indices),
                is_monotonic=quality_metrics_raw.get("is_monotonic", False),
                neutral_percentage=neutral_percentage,
                logger=logger
            )
            
            logger.info(f"Валидация: acceptable={quality_metrics.is_acceptable}")
            
            return {
                "should_retry": not quality_metrics.is_acceptable,
                "quality_metrics": quality_metrics,
                "quality_metrics_raw": quality_metrics_raw,
                "final_indices": final_indices,
                "smoothed_emotions": smoothed_emotions,
                "keyframes_indices": keyframes_indices,
                "emotion_curve": emotion_curve
            }
        
        except Exception as e:
            logger.error(f"Ошибка в _validation_and_retry_logic: {e}")
            if isinstance(e, VideoProcessingError):
                raise
            raise ValidationError(
                f"Failed to validate results: {e}",
                details={"error": str(e), "target_length": target_length}
            ) from e
    
    def _save_results(
        self,
        validation_result: Dict[str, Any],
        emotion_result: Dict[str, Any],
        selected_indices: List[int],
        save_path: str,
        meta: Dict[str, Any],
        params: ProcessingParams,
        attempt_number: int,
        video_type: str,
        segments_count: int,
        faces_found: int
    ) -> Dict[str, Any]:
        """
        Сохранение результатов обработки.
        
        Returns:
            Словарь с результатами обработки.
        """
        quality_metrics = validation_result["quality_metrics"]
        quality_metrics_raw = validation_result["quality_metrics_raw"]
        final_indices = validation_result["final_indices"]
        smoothed_emotions = validation_result["smoothed_emotions"]
        keyframes_indices = validation_result["keyframes_indices"]
        emotion_curve = validation_result["emotion_curve"]
        emo_results = emotion_result["emo_results"]
        emotion_profile = emotion_result["emotion_profile"]
        
        # Расширенные фичи
        microexpressions = emotion_result.get("microexpressions", {})
        emotional_individuality = emotion_result.get("emotional_individuality", {})
        face_asymmetry = emotion_result.get("face_asymmetry", {})
        
        # Подготовка данных для пользователя
        user_data = {
            "original_emotions": emo_results,
            "emotion_profile": emotion_profile,
            "keyframes": [],
            "emotion_curve": emotion_curve,
            "quality_metrics": quality_metrics_raw,
            "processing_params": params.to_dict(),
            "processing_stats": {
                "total_frames": meta["total_frames"],
                "faces_found": faces_found,
                "segments": segments_count,
                "selected_frames": len(selected_indices),
                "final_length": len(smoothed_emotions),
                "keyframes_count": len(keyframes_indices),
                "attempt_number": attempt_number,
                "success": True,
                "video_type": video_type
            },
            # Расширенные фичи (для UI)
            "advanced_features": {
                "microexpressions": microexpressions,
                "emotional_individuality": emotional_individuality,
                "face_asymmetry": face_asymmetry
            }
        }
        
        # Добавляем ключевые кадры
        for idx in keyframes_indices.keys():
            if idx < len(selected_indices):
                user_data["keyframes"].append({
                    "global_index": int(selected_indices[idx]),
                    "local_index": int(idx),
                    "type": keyframes_indices[idx]["type"],
                    "emotion": emo_results[idx] if idx < len(emo_results) else {}
                })
        
        # Подготовка данных для модели (минимальный набор)
        model_data = {
            "indices": [int(idx) for idx in final_indices],
            "emotions": smoothed_emotions,
            "valence": [e["valence"] for e in smoothed_emotions],
            "arousal": [e["arousal"] for e in smoothed_emotions],
            "emotion_confidence": [e.get("emotion_confidence", 1.0) for e in smoothed_emotions],
            "dominant_emotion": [
                max(e.get("emotions", {}).items(), key=lambda x: x[1])[0] if e.get("emotions") else "Neutral"
                for e in smoothed_emotions
            ],
            "intensity": [float(np.sqrt(e.get("valence", 0.0) ** 2 + e.get("arousal", 0.0) ** 2)) for e in smoothed_emotions],
            "sequence_length": len(smoothed_emotions),
            "video_metadata": meta,
            "quality_score": quality_metrics_raw.get("overall_score", 0),
            "processing_attempt": attempt_number
        }
        
        return {
            "success": True,
            "user_data": user_data,
            "model_data": model_data,
            "attempts": attempt_number,
            "final_params": params.to_dict(),
            "metrics": self.metrics.to_dict()
        }
    
    def _build_failure_result(
        self,
        params: ProcessingParams,
        attempts: int,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Создает результат при неудачной обработке.
        
        Returns:
            Словарь с информацией об ошибке.
        """
        return {
            "success": False,
            "error": error or "Failed to process video after retries",
            "attempts": attempts + 1,
            "final_params": params.to_dict(),
            "metrics": self.metrics.to_dict()
        }


def _resolve_bundled_emonet_checkpoint() -> Optional[str]:
    """
    Prefer ModelManager; when it cannot resolve EmoNet (missing dp_models on path, wrong DP_MODELS_ROOT,
    etc.), use the same bundled layout as spec ``emonet_8_inprocess``.
    """
    candidates: List[str] = []
    env_root = str(os.environ.get("DP_MODELS_ROOT") or "").strip()
    if env_root:
        er = os.path.abspath(env_root)
        base = os.path.basename(er)
        if base == "bundled_models":
            candidates.append(os.path.join(er, "visual", "emonet", "emonet_8.pth"))
        candidates.append(os.path.join(er, "bundled_models", "visual", "emonet", "emonet_8.pth"))
    current_file = os.path.abspath(__file__)
    if "DataProcessor" in current_file:
        dp_root = os.path.join(current_file.split("DataProcessor")[0], "DataProcessor")
        candidates.extend(
            [
                os.path.join(dp_root, "dp_models", "bundled_models", "visual", "emonet", "emonet_8.pth"),
                os.path.join(dp_root, "dp_models", "emonet", "pretrained", "emonet_8.pth"),
            ]
        )
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


class EmotionFaceModule(BaseModule):
    """
    Модуль для анализа эмоций на лицах в видео.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует VideoEmotionProcessor для обработки видео.
    
    Зависимости:
    - core_face_landmarks (обязательная) - face_present + landmarks, используется для выбора кадров с лицами
    """

    MODULE_NAME = "emotion_face"
    ARTIFACT_FILENAME = "emotion_face.npz"
    SCHEMA_VERSION = "emotion_face_npz_v3"
    VERSION = "2.0.2"
    
    @property
    def supports_batch(self) -> bool:
        """Указывает, поддерживает ли модуль батчинг."""
        return True
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        min_frames_ratio: float = 0.8,
        min_keyframes: int = 3,
        min_transitions: int = 2,
        min_diversity_threshold: float = 0.2,
        quality_threshold: float = 0.4,
        memory_threshold_low: int = 2000,
        batch_load_low: int = 20,
        batch_process_low: int = 8,
        memory_threshold_medium: int = 4000,
        batch_load_medium: int = 30,
        batch_process_medium: int = 12,
        memory_threshold_high: int = 8000,
        batch_load_high: int = 50,
        batch_process_high: int = 15,
        batch_load_very_high: int = 80,
        batch_process_very_high: int = 24,
        enable_structured_metrics: bool = True,
        min_faces_threshold: int = 1,
        target_length: int = 256,
        max_retries: int = 2,
        transition_threshold: float = 0.3,
        max_gap_seconds: float = 0.5,
        max_samples_per_segment: int = 10,
        # Baseline v1: frames are selected from core_face_landmarks frames (where any face is present),
        # then downsampled by stride (no Segmenter-owned sampling for this module by decision).
        face_frame_stride: int = 4,
        max_frames: int = 200,
        # Multi-face inference
        max_faces_per_frame: int = 2,
        face_bbox_margin: float = 0.20,
        # Feature gating (noisy/expensive) - off by default (module itself is also disabled by default)
        enable_microexpressions: bool = False,
        enable_emotional_individuality: bool = False,
        enable_face_asymmetry: bool = False,
        # Model policy
        emonet_model_spec: Optional[str] = "emonet_8_inprocess",
        emo_path: Optional[str] = None,  # legacy explicit weights path (kept for compatibility)
        device: str = "cuda",
        **kwargs: Any
    ):
        """
        Инициализация EmotionFaceModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            min_frames_ratio: Минимальное соотношение кадров
            min_keyframes: Минимальное количество ключевых кадров
            min_transitions: Минимальное количество переходов
            min_diversity_threshold: Минимальный порог разнообразия
            quality_threshold: Порог качества
            memory_threshold_low: Порог памяти (низкий)
            batch_load_low: Размер батча загрузки (низкий)
            batch_process_low: Размер батча обработки (низкий)
            memory_threshold_medium: Порог памяти (средний)
            batch_load_medium: Размер батча загрузки (средний)
            batch_process_medium: Размер батча обработки (средний)
            memory_threshold_high: Порог памяти (высокий)
            batch_load_high: Размер батча загрузки (высокий)
            batch_process_high: Размер батча обработки (высокий)
            batch_load_very_high: Размер батча загрузки (очень высокий)
            batch_process_very_high: Размер батча обработки (очень высокий)
            enable_structured_metrics: Включить структурированные метрики
            min_faces_threshold: Минимальный порог лиц
            target_length: Целевая длина последовательности
            max_retries: Максимальное количество повторных попыток
            transition_threshold: Порог перехода
            max_gap_seconds: Максимальный разрыв в секундах
            max_samples_per_segment: Максимальное количество сэмплов на сегмент
            emo_path: Путь к модели EmoNet
            device: Устройство для обработки (cuda/cpu)
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        self.emo_path = emo_path
        self.emonet_model_spec = str(emonet_model_spec or "").strip() or None
        self.device = device

        # Validation / quality thresholds (stored for reproducibility + meta config highlights).
        self.min_frames_ratio = float(min_frames_ratio)
        self.min_keyframes = int(min_keyframes)
        self.min_transitions = int(min_transitions)
        self.min_diversity_threshold = float(min_diversity_threshold)
        self.quality_threshold = float(quality_threshold)
        self.transition_threshold = float(transition_threshold)
        self.max_gap_seconds = float(max_gap_seconds)
        self.target_length = int(target_length)

        self.face_frame_stride = int(face_frame_stride)
        self.max_frames = int(max_frames)
        self.max_faces_per_frame = int(max_faces_per_frame)
        self.face_bbox_margin = float(face_bbox_margin)

        self.enable_microexpressions = bool(enable_microexpressions)
        self.enable_emotional_individuality = bool(enable_emotional_individuality)
        self.enable_face_asymmetry = bool(enable_face_asymmetry)

        # ModelManager (baseline policy): resolve and load model locally (no-network).
        self._mm_resolved = None
        self._emonet_model = None
        self._models_used_entry = None
        try:
            from dp_models.manager import get_global_model_manager  # type: ignore
            from dp_models.errors import ModelManagerError  # type: ignore

            if self.emonet_model_spec:
                mm = get_global_model_manager()
                resolved = mm.get(model_name=self.emonet_model_spec)
                self._mm_resolved = resolved
                self._emonet_model = resolved.handle
                self._models_used_entry = resolved.models_used_entry
        except Exception:
            # If ModelManager is not usable, we will fall back to legacy `emo_path` loading in run().
            self._mm_resolved = None
            self._emonet_model = None
            self._models_used_entry = None

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Prefer ModelManager provenance if available.
        if self._models_used_entry is not None:
            return [self._models_used_entry]
        # Legacy fallback (best-effort): keep old digest behavior.
        import hashlib
        import os
        from utils.meta_builder import model_used

        weights_digest = "unknown"
        if self.emo_path and os.path.exists(self.emo_path):
            try:
                with open(self.emo_path, "rb") as f:
                    weights_digest = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass
        return [
            model_used(
                model_name=str(self.emonet_model_spec or "EmoNet"),
                model_version="unknown",
                weights_digest=weights_digest,
                runtime="inprocess",
                engine="torch",
                precision="fp32",
                device=str(self.device or "cuda"),
            )
        ]
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Обязательные:
        - core_face_landmarks: для получения списка кадров с лицами (face_present)
        """
        return ["core_face_landmarks"]

    @staticmethod
    def _load_emonet_from_path(*, emo_path: str, device: str) -> Any:
        """
        Legacy loader for EmoNet weights (discouraged).
        Prefer ModelManager spec `emonet_model_spec`.
        """
        import importlib.util
        # dp_models/emonet/emonet/models/emonet.py
        # Find DataProcessor root: core/video_processor.py -> emotion_face -> modules -> VisualProcessor -> DataProcessor
        current_file = os.path.abspath(__file__)
        if "DataProcessor" in current_file:
            parts = current_file.split("DataProcessor")
            if len(parts) > 1:
                dp_root = os.path.join(parts[0], "DataProcessor")
                emonet_py = os.path.join(dp_root, "dp_models", "emonet", "emonet", "models", "emonet.py")
            else:
                raise RuntimeError("Could not determine DataProcessor root from file path")
        else:
            raise RuntimeError("Could not find DataProcessor in file path")
        
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
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки (не используется напрямую,
                          VideoEmotionProcessor использует frames_with_face из core_face_landmarks)
            config: Конфигурация модуля (не используется, параметры заданы в __init__)
            
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: агрегированные фичи эмоций
            - sequence_features: последовательности эмоций для VisualTransformer
            - summary: метаданные обработки
        """
        # Baseline v1: `frame_indices` is expected to be the already-selected global indices (union-domain).
        # This method runs multi-face inference and produces numeric arrays (no dtype=object for model-facing data).
        import cv2
        from _utils import predict_emonet_batch

        fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
        if fi.size == 0:
            return self._empty_result()

        # Best-effort granular progress
        progress_ctx = config.get("_progress_ctx") if isinstance(config, dict) else None
        rs_path_ev = None
        if isinstance(progress_ctx, dict):
            rs_path_ev = progress_ctx.get("rs_path")
        platform_id_ev = progress_ctx.get("platform_id") if isinstance(progress_ctx, dict) else None
        video_id_ev = progress_ctx.get("video_id") if isinstance(progress_ctx, dict) else None
        run_id_ev = progress_ctx.get("run_id") if isinstance(progress_ctx, dict) else None
        update_every = max(1, int(fi.size) // 10)

        # Required injected inputs from run()
        core = config.get("_core_face_landmarks")
        if not isinstance(core, dict):
            raise RuntimeError("emotion_face | missing injected _core_face_landmarks (no-fallback)")
        face_fi = np.asarray(core["frame_indices"], dtype=np.int32).reshape(-1)
        face_present = np.asarray(core["face_present"], dtype=bool)
        face_landmarks = np.asarray(core["face_landmarks"], dtype=np.float32)
        fi_to_pos = core.get("_fi_to_pos")
        if not isinstance(fi_to_pos, dict):
            raise RuntimeError("emotion_face | missing injected _fi_to_pos mapping (no-fallback)")

        # Load model: ModelManager first; then explicit emo_path; then bundled dp_models checkpoint.
        model = self._emonet_model
        if model is None:
            ckpt = str(self.emo_path).strip() if self.emo_path else ""
            if self.emo_path and ckpt and not os.path.isfile(ckpt):
                raise RuntimeError(f"emotion_face | emo_path does not exist: {ckpt}")
            if not ckpt:
                ckpt = _resolve_bundled_emonet_checkpoint() or ""
            if ckpt and os.path.isfile(ckpt):
                if self.emo_path:
                    self.logger.warning(
                        "emotion_face | loading EmoNet from explicit emo_path (override): %s",
                        ckpt,
                    )
                else:
                    self.logger.info(
                        "emotion_face | loading EmoNet from bundled/local weights (ModelManager unavailable): %s",
                        ckpt,
                    )
                model = self._load_emonet_from_path(emo_path=str(ckpt), device=str(self.device or "cuda"))
                self._emonet_model = model
            else:
                raise RuntimeError(
                    "emotion_face | EmoNet weights not found. "
                    "Ensure `dp_models/bundled_models/visual/emonet/emonet_8.pth` exists and DP_MODELS_ROOT is set, "
                    "or pass --emo-path."
                )

        # Pre-allocate per-frame/per-face outputs
        max_faces = max(1, int(self.max_faces_per_frame))
        N = int(fi.size)
        valence_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
        arousal_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
        conf_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
        probs_faces = np.full((N, max_faces, 8), np.nan, dtype=np.float32)
        face_count = np.zeros((N,), dtype=np.int16)

        # Build list of crops to infer (frame_slot, face_slot, crop_rgb)
        crops: List[np.ndarray] = []
        crop_map: List[tuple[int, int]] = []

        for i in range(N):
            if rs_path_ev and platform_id_ev and video_id_ev and run_id_ev:
                if (i % update_every) == 0:
                    _emit_progress(
                        rs_path=str(rs_path_ev),
                        platform_id=str(platform_id_ev),
                        video_id=str(video_id_ev),
                        run_id=str(run_id_ev),
                        done=int(i),
                        total=int(N),
                        stage="process_frames",
                    )
            gidx = int(fi[i])
            pos = fi_to_pos.get(gidx)
            if pos is None:
                continue
            p = int(pos)
            present_row = face_present[p]  # (FACES,)
            if present_row.ndim != 1:
                present_row = np.asarray(present_row, dtype=bool).reshape(-1)
            lm_row = face_landmarks[p]  # (FACES,468,3)
            if lm_row.ndim != 3:
                continue

            # Pick up to max_faces faces (largest bbox area first)
            candidates: List[tuple[float, int]] = []
            for f in range(int(present_row.shape[0])):
                if not bool(present_row[f]):
                    continue
                lm = lm_row[f]
                if np.isnan(lm).all():
                    continue
                xs = lm[:, 0]
                ys = lm[:, 1]
                # normalized 0..1; bbox area in that space is enough for ranking
                area = float((np.nanmax(xs) - np.nanmin(xs)) * (np.nanmax(ys) - np.nanmin(ys)))
                candidates.append((area, f))
            candidates.sort(reverse=True, key=lambda t: t[0])
            chosen = [f for _area, f in candidates[:max_faces]]
            face_count[i] = int(len(chosen))
            if not chosen:
                continue

            frame = frame_manager.get(gidx)
            if frame is None:
                continue
            # Ensure RGB uint8 (FrameManager contract), but keep a guard.
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            H, W = int(frame.shape[0]), int(frame.shape[1])

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
                # Expand bbox
                mx = (x1 - x0) * float(self.face_bbox_margin)
                my = (y1 - y0) * float(self.face_bbox_margin)
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
                crops.append(crop)
                crop_map.append((i, slot))

        # Batched inference
        if crops:
            results = predict_emonet_batch(crops, model, batch_size=None, use_amp=True, temperature=1.0, face_confidence=None)
            for r, (i, slot) in zip(results, crop_map):
                valence_faces[i, slot] = float(r.get("valence", np.nan))
                arousal_faces[i, slot] = float(r.get("arousal", np.nan))
                conf_faces[i, slot] = float(r.get("emotion_confidence", np.nan))
                probs = r.get("emotions") or {}
                # fixed class order
                cls = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
                probs_faces[i, slot, :] = np.asarray([float(probs.get(c, 0.0)) for c in cls], dtype=np.float32)

        if rs_path_ev and platform_id_ev and video_id_ev and run_id_ev:
            _emit_progress(
                rs_path=str(rs_path_ev),
                platform_id=str(platform_id_ev),
                video_id=str(video_id_ev),
                run_id=str(run_id_ev),
                done=int(N),
                total=int(N),
                stage="process_frames",
            )

        # Aggregate over faces (mean over valid)
        valence = np.nanmean(valence_faces, axis=1).astype(np.float32)
        arousal = np.nanmean(arousal_faces, axis=1).astype(np.float32)
        emotion_confidence = np.nanmean(conf_faces, axis=1).astype(np.float32)
        emotion_probs = np.nanmean(probs_faces, axis=1).astype(np.float32)  # (N,8)
        intensity = np.sqrt(np.square(valence) + np.square(arousal)).astype(np.float32)
        dominant_id = np.nanargmax(emotion_probs, axis=1).astype(np.int8) if emotion_probs.size else np.zeros((N,), dtype=np.int8)

        sequence_features = {
            "frame_indices": fi,
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity,
            "emotion_confidence": emotion_confidence,
            "emotion_probs": emotion_probs,
            "dominant_emotion_id": dominant_id,
            "face_count": face_count.astype(np.int16),
            # per-face
            "valence_faces": valence_faces,
            "arousal_faces": arousal_faces,
            "emotion_confidence_faces": conf_faces,
            "emotion_probs_faces": probs_faces,
        }

        # Minimal aggregates for downstream/UI
        # Используем safe_array_check для проверки пустых массивов перед вызовом .any()
        features = {
            "valence_mean": float(np.nanmean(valence)) if safe_array_check(valence) and np.isfinite(valence).any() else np.nan,
            "arousal_mean": float(np.nanmean(arousal)) if safe_array_check(arousal) and np.isfinite(arousal).any() else np.nan,
            "intensity_mean": float(np.nanmean(intensity)) if safe_array_check(intensity) and np.isfinite(intensity).any() else np.nan,
        }

        summary = {
            "sequence_length": int(N),
            "faces_found_frames": int(np.sum(face_count > 0)),
            "max_faces_per_frame": int(max_faces),
        }

        return {
            "features": features,
            "sequence_features": sequence_features,
            "summary": summary,
            "advanced_features": {},
            "keyframes": [],
        }
    
    def _format_results_for_npz(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует результаты VideoEmotionProcessor в формат для сохранения в npz.
        
        Args:
            result: Результаты из processor.process()
            
        Returns:
            Словарь в формате для npz
        """
        import numpy as np
        
        user_data = result.get("user_data", {})
        model_data = result.get("model_data", {})
        
        # Извлекаем основные данные
        emotions = model_data.get("emotions", [])
        indices = model_data.get("indices", [])
        valence = model_data.get("valence", [])
        arousal = model_data.get("arousal", [])
        emotion_confidence = model_data.get("emotion_confidence", [])
        dominant_emotion = model_data.get("dominant_emotion", [])
        intensity = model_data.get("intensity", [])
        emotion_profile = user_data.get("emotion_profile", {})
        quality_metrics = user_data.get("quality_metrics", {})
        processing_stats = user_data.get("processing_stats", {})
        advanced_features = user_data.get("advanced_features", {})
        
        # Подготавливаем sequence_features для VisualTransformer
        # Используем safe_array_check для проверки пустых массивов вместо булевой проверки
        sequence_features = {
            "emotion_sequence": np.array(emotions, dtype=object) if safe_array_check(emotions) else np.array([], dtype=object),
            "valence_sequence": np.array(valence, dtype=np.float32) if safe_array_check(valence) else np.array([], dtype=np.float32),
            "arousal_sequence": np.array(arousal, dtype=np.float32) if safe_array_check(arousal) else np.array([], dtype=np.float32),
            "emotion_confidence": np.array(emotion_confidence, dtype=np.float32) if safe_array_check(emotion_confidence) else np.array([], dtype=np.float32),
            "dominant_emotion": np.array(dominant_emotion, dtype=object) if safe_array_check(dominant_emotion) else np.array([], dtype=object),
            "intensity": np.array(intensity, dtype=np.float32) if safe_array_check(intensity) else np.array([], dtype=np.float32),
            "frame_indices": np.array(indices, dtype=np.int32) if safe_array_check(indices) else np.array([], dtype=np.int32),
        }
        
        # Подготавливаем агрегированные фичи
        features = {}
        
        # Emotion profile features
        if emotion_profile:
            for key, value in emotion_profile.items():
                if isinstance(value, (int, float, bool)):
                    features[f"emotion_profile_{key}"] = float(value) if isinstance(value, bool) else value
                elif isinstance(value, (list, tuple)):
                    try:
                        features[f"emotion_profile_{key}"] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        features[f"emotion_profile_{key}"] = np.asarray(value, dtype=object)
        
        # Quality metrics
        if quality_metrics:
            for key, value in quality_metrics.items():
                if isinstance(value, (int, float, bool)):
                    features[f"quality_{key}"] = float(value) if isinstance(value, bool) else value
        
        # Processing stats
        if processing_stats:
            for key, value in processing_stats.items():
                if isinstance(value, (int, float, bool)):
                    features[f"processing_{key}"] = float(value) if isinstance(value, bool) else value
        
        # Advanced features
        if advanced_features:
            for category, category_data in advanced_features.items():
                if isinstance(category_data, dict):
                    for key, value in category_data.items():
                        if isinstance(value, (int, float, bool)):
                            features[f"{category}_{key}"] = float(value) if isinstance(value, bool) else value
                        elif isinstance(value, (list, tuple)):
                            try:
                                features[f"{category}_{key}"] = np.asarray(value, dtype=np.float32)
                            except Exception:
                                features[f"{category}_{key}"] = np.asarray(value, dtype=object)
        
        # Model data features (минимальный набор)
        for key in ["processing_attempt", "quality_score"]:
            if key in model_data and isinstance(model_data[key], (int, float, bool)):
                features[key] = float(model_data[key]) if isinstance(model_data[key], bool) else model_data[key]
        
        # Подготавливаем summary
        summary = {
            "success": result.get("success", False),
            "attempts": result.get("attempts", 0),
            "sequence_length": len(emotions),
            "total_frames": processing_stats.get("total_frames", 0),
            "faces_found": processing_stats.get("faces_found", 0),
            "selected_frames": processing_stats.get("selected_frames", 0),
            "keyframes_count": processing_stats.get("keyframes_count", 0),
            "video_type": processing_stats.get("video_type", "UNKNOWN"),
        }
        
        # Формируем итоговый результат
        # keyframes должен быть сохранен как object array (список словарей), а не как обычный массив
        keyframes_list = user_data.get("keyframes", [])
        if keyframes_list:
            # Создаем object array для списка словарей
            keyframes_arr = np.empty(len(keyframes_list), dtype=object)
            for i, kf in enumerate(keyframes_list):
                keyframes_arr[i] = kf
        else:
            keyframes_arr = np.asarray([], dtype=object)
        
        formatted_result = {
            "features": features,
            "sequence_features": sequence_features,
            "summary": summary,
            "advanced_features": advanced_features,
            "keyframes": keyframes_arr,
        }
        
        return formatted_result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат (baseline contract: numeric arrays, no dtype=object for model-facing data)."""
        max_faces = max(1, int(getattr(self, "max_faces_per_frame", 2)))
        return {
            "features": {},
            "sequence_features": {
                "frame_indices": np.asarray([], dtype=np.int32),
                "times_s": np.asarray([], dtype=np.float32),
                "face_present": np.asarray([], dtype=bool),
                "processed_mask": np.asarray([], dtype=bool),
                "valence": np.asarray([], dtype=np.float32),
                "arousal": np.asarray([], dtype=np.float32),
                "intensity": np.asarray([], dtype=np.float32),
                "emotion_confidence": np.asarray([], dtype=np.float32),
                "emotion_probs": np.asarray([], dtype=np.float32).reshape(0, 8),
                "dominant_emotion_id": np.asarray([], dtype=np.int8),
                "face_count": np.asarray([], dtype=np.int16),
                "valence_faces": np.asarray([], dtype=np.float32).reshape(0, max_faces),
                "arousal_faces": np.asarray([], dtype=np.float32).reshape(0, max_faces),
                "emotion_confidence_faces": np.asarray([], dtype=np.float32).reshape(0, max_faces),
                "emotion_probs_faces": np.asarray([], dtype=np.float32).reshape(0, max_faces, 8),
            },
            "frame_indices": np.asarray([], dtype=np.int32),
            "times_s": np.asarray([], dtype=np.float32),
            "face_present": np.asarray([], dtype=bool),
            "processed_mask": np.asarray([], dtype=bool),
            "face_count": np.asarray([], dtype=np.int16),
            "valence": np.asarray([], dtype=np.float32),
            "arousal": np.asarray([], dtype=np.float32),
            "intensity": np.asarray([], dtype=np.float32),
            "emotion_confidence": np.asarray([], dtype=np.float32),
            "emotion_probs": np.asarray([], dtype=np.float32).reshape(0, 8),
            "dominant_emotion_id": np.asarray([], dtype=np.int8),
            "summary": {
                "sequence_length": 0,
                "stage_timings_ms": {},
            },
            "advanced_features": {},
            "keyframes": np.asarray([], dtype=object),
        }

    def _build_ui_payload(self, results: Dict[str, Any]) -> Dict[str, Any]:
        seq = results.get("sequence_features", {}) or {}
        summary = results.get("summary", {}) or {}
        # Используем safe_array_check для проверки keyframes (может быть NumPy массивом)
        keyframes_raw = results.get("keyframes", [])
        if isinstance(keyframes_raw, np.ndarray):
            keyframes = keyframes_raw.tolist() if safe_array_check(keyframes_raw) else []
        else:
            keyframes = keyframes_raw if keyframes_raw else []

        frame_indices = seq.get("frame_indices", np.asarray([], dtype=np.int32))
        times_s = seq.get("times_s", np.asarray([], dtype=np.float32))
        if isinstance(frame_indices, np.ndarray):
            frame_indices_list = frame_indices.astype(int).tolist()
        else:
            frame_indices_list = [int(x) for x in (frame_indices or [])]
        if isinstance(times_s, np.ndarray):
            times_s_list = times_s.astype(float).tolist()
        else:
            times_s_list = [float(x) for x in (times_s or [])]

        emotion_classes = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        payload = {
            "component": self.module_name,
            "schema_version": "emotion_face_ui_v1",
            "frame_indices": frame_indices_list,
            "times_s": times_s_list,
            "emotion_classes": emotion_classes,
            "curves": {
                "valence": {"key": "sequence_features.valence"},
                "arousal": {"key": "sequence_features.arousal"},
                "intensity": {"key": "sequence_features.intensity"},
                "emotion_confidence": {"key": "sequence_features.emotion_confidence"},
                "face_count": {"key": "sequence_features.face_count"},
            },
            "keyframes": keyframes,
            "summary": summary,
        }
        return payload

    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")

        platform_id = str(metadata.get("platform_id"))
        video_id = str(metadata.get("video_id"))
        run_id = str(metadata.get("run_id"))

        uts = metadata.get("union_timestamps_sec") or metadata.get("times_s")
        if uts is None:
            raise RuntimeError("emotion_face | metadata.json missing union_timestamps_sec (no-fallback)")
        uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)

        stage_timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        def _resource_profile_snapshot() -> Dict[str, Any]:
            """
            Best-effort, env-gated resource snapshot for Audit 4.2.
            """
            if str(os.environ.get("VP_RESOURCE_PROFILE", "")).strip().lower() not in ("1", "true", "yes", "on"):
                return {}
            snap: Dict[str, Any] = {}
            try:
                import psutil  # type: ignore

                snap["rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass
            try:
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    snap["cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                    snap["cuda_max_reserved_mb"] = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
                    snap["cuda_device"] = int(torch.cuda.current_device())
            except Exception:
                pass
            return snap

        resource_profile_before = _resource_profile_snapshot()

        frame_manager = None
        try:
            if self.rs_path is not None:
                _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=0, total=1, stage="start")

            # --- load_deps ---
            t_stage = time.perf_counter()
            core = self.load_core_provider("core_face_landmarks", file_name="landmarks.npz")
            if core is None or not isinstance(core, dict):
                raise RuntimeError("emotion_face | missing core_face_landmarks/landmarks.npz (no-fallback)")
            face_fi = np.asarray(core.get("frame_indices"), dtype=np.int32).reshape(-1)
            face_present = np.asarray(core.get("face_present"), dtype=bool)
            face_landmarks = np.asarray(core.get("face_landmarks"), dtype=np.float32)
            if face_fi.size == 0:
                raise RuntimeError("emotion_face | core_face_landmarks has empty frame_indices (unexpected)")
            if face_present.ndim != 2 or face_present.shape[0] != face_fi.shape[0]:
                raise RuntimeError("emotion_face | core_face_landmarks face_present shape mismatch")
            if face_landmarks.ndim != 4 or face_landmarks.shape[0] != face_fi.shape[0]:
                raise RuntimeError("emotion_face | core_face_landmarks face_landmarks shape mismatch")

            any_face = np.any(face_present, axis=1)
            frames_with_face = face_fi[any_face]
            stage_timings["load_deps"] = (time.perf_counter() - t_stage) * 1000.0

            if self.rs_path is not None:
                _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=1, total=1, stage="load_deps")

            # --- axis/select_frames ---
            t_stage = time.perf_counter()
            # Axis for output alignment (Segmenter contract).
            # Preferred: metadata[emotion_face].frame_indices, fallback: core_face_landmarks.frame_indices
            axis_frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
            axis_source = "emotion_face"
            # Используем len() для проверки пустого списка/массива
            if not axis_frame_indices or (isinstance(axis_frame_indices, (list, tuple)) and len(axis_frame_indices) == 0) or (isinstance(axis_frame_indices, np.ndarray) and axis_frame_indices.size == 0):
                axis_frame_indices = face_fi.astype(np.int32).tolist()
                axis_source = "core_face_landmarks"
                self.logger.warning(
                    "emotion_face | missing metadata[emotion_face].frame_indices; "
                    "falling back to core_face_landmarks.frame_indices for axis alignment"
                )
            axis_fi = np.asarray(axis_frame_indices, dtype=np.int32).reshape(-1)
            if axis_fi.size == 0:
                raise RuntimeError("emotion_face | axis frame_indices is empty (no-fallback)")
            if np.any(axis_fi < 0) or np.any(axis_fi >= int(uts_arr.shape[0])):
                raise RuntimeError("emotion_face | axis frame_indices out of range for union_timestamps_sec (no-fallback)")

            # Check alignment: axis should be covered by core_face_landmarks frame_indices.
            # If not fully covered, issue a warning and continue (frames without faces will have face_present=False).
            face_set = set(int(x) for x in face_fi.tolist())
            missing = [int(x) for x in axis_fi.tolist() if int(x) not in face_set]
            if missing:
                self.logger.warning(
                    f"emotion_face | axis frame_indices partially covered by core_face_landmarks.frame_indices. "
                    f"Missing {len(missing)} indices (example: {missing[:5]}). "
                    f"Module will proceed, but these frames will have face_present=False and processed_mask=False."
                )

            # Build face_present axis mask (any face)
            # For frames not in core_face_landmarks, set face_present=False
            fi_to_pos = {int(x): int(i) for i, x in enumerate(face_fi.tolist())}
            axis_face_present = np.asarray(
                [
                    bool(any_face[fi_to_pos[int(fi)]]) if int(fi) in fi_to_pos else False
                    for fi in axis_fi.tolist()
                ],
                dtype=bool,
            ).reshape(-1)

            # Internal sampling among face frames on the axis (stride + cap)
            axis_face_frames = axis_fi[axis_face_present]

            if axis_face_frames.size == 0:
                results = self._empty_result()
                status = "empty"
                empty_reason = "no_faces_in_video"
                selected_fi = np.asarray([], dtype=np.int32)
            else:
                stride = max(1, int(self.face_frame_stride))
                selected_fi = axis_face_frames[::stride].astype(np.int32)
                if int(self.max_frames) > 0 and selected_fi.size > int(self.max_frames):
                    selected_fi = selected_fi[: int(self.max_frames)]
                if selected_fi.size == 0:
                    raise RuntimeError("emotion_face | internal sampling produced empty selection (invalid config)")
                status = "ok"
                empty_reason = None
                results = None
            stage_timings["select_frames"] = (time.perf_counter() - t_stage) * 1000.0

            if self.rs_path is not None:
                _emit_progress(
                    rs_path=str(self.rs_path),
                    platform_id=platform_id,
                    video_id=video_id,
                    run_id=run_id,
                    done=int(selected_fi.size),
                    total=int(max(1, selected_fi.size)),
                    stage="select_frames",
                    message=f"selected={int(selected_fi.size)} stride={max(1, int(self.face_frame_stride))}",
                )

            # --- process_frames ---
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            if results is None:
                times_s_proc = uts_arr[selected_fi].astype(np.float32)
                injected = {
                    "frame_indices": face_fi,
                    "face_present": face_present,
                    "face_landmarks": face_landmarks,
                    "_fi_to_pos": fi_to_pos,
                }
                t_stage = time.perf_counter()
                results = self.process(
                    frame_manager=frame_manager,
                    frame_indices=selected_fi.tolist(),
                    config={
                        "_core_face_landmarks": injected,
                        "_progress_ctx": {
                            "rs_path": str(self.rs_path) if self.rs_path is not None else None,
                            "platform_id": platform_id,
                            "video_id": video_id,
                            "run_id": run_id,
                        },
                    },
                )
                stage_timings["process_frames"] = (time.perf_counter() - t_stage) * 1000.0

                # Build axis-aligned time-series for models (full axis with masks).
                times_s_axis = uts_arr[axis_fi].astype(np.float32)
                selected_set = set(int(x) for x in selected_fi.tolist())
                processed_mask = np.asarray([int(fi) in selected_set for fi in axis_fi.tolist()], dtype=bool)

                # Initialize axis arrays (NaN for floats; -1 for ids).
                N = int(axis_fi.size)
                valence = np.full((N,), np.nan, dtype=np.float32)
                arousal = np.full((N,), np.nan, dtype=np.float32)
                intensity = np.full((N,), np.nan, dtype=np.float32)
                emotion_confidence = np.full((N,), np.nan, dtype=np.float32)
                emotion_probs = np.full((N, 8), np.nan, dtype=np.float32)
                dominant_emotion_id = np.full((N,), -1, dtype=np.int8)

                max_faces = max(1, int(self.max_faces_per_frame))
                valence_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
                arousal_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
                emotion_confidence_faces = np.full((N, max_faces), np.nan, dtype=np.float32)
                emotion_probs_faces = np.full((N, max_faces, 8), np.nan, dtype=np.float32)

                # Face count from core_face_landmarks (any faces), not capped by max_faces_per_frame.
                axis_face_count = np.asarray(
                    [int(np.sum(face_present[fi_to_pos[int(fi)]])) if int(fi) in fi_to_pos else 0 for fi in axis_fi.tolist()],
                    dtype=np.int16,
                ).reshape(-1)

                # Map processed outputs into the axis arrays.
                seq_proc = results.get("sequence_features", {}) or {}
                fi_proc = np.asarray(seq_proc.get("frame_indices"), dtype=np.int32).reshape(-1)
                pos_axis = {int(fi): int(i) for i, fi in enumerate(axis_fi.tolist())}
                for j in range(int(fi_proc.size)):
                    g = int(fi_proc[j])
                    i = pos_axis.get(g)
                    if i is None:
                        continue
                    try:
                        valence[i] = float(np.asarray(seq_proc.get("valence"), dtype=np.float32).reshape(-1)[j])
                        arousal[i] = float(np.asarray(seq_proc.get("arousal"), dtype=np.float32).reshape(-1)[j])
                        intensity[i] = float(np.asarray(seq_proc.get("intensity"), dtype=np.float32).reshape(-1)[j])
                        emotion_confidence[i] = float(np.asarray(seq_proc.get("emotion_confidence"), dtype=np.float32).reshape(-1)[j])
                        emotion_probs[i, :] = np.asarray(seq_proc.get("emotion_probs"), dtype=np.float32).reshape(-1, 8)[j, :]
                        dominant_emotion_id[i] = np.asarray(seq_proc.get("dominant_emotion_id"), dtype=np.int8).reshape(-1)[j]
                        # per-face arrays
                        valence_faces[i, :] = np.asarray(seq_proc.get("valence_faces"), dtype=np.float32).reshape(-1, max_faces)[j, :]
                        arousal_faces[i, :] = np.asarray(seq_proc.get("arousal_faces"), dtype=np.float32).reshape(-1, max_faces)[j, :]
                        emotion_confidence_faces[i, :] = np.asarray(seq_proc.get("emotion_confidence_faces"), dtype=np.float32).reshape(-1, max_faces)[j, :]
                        emotion_probs_faces[i, :, :] = np.asarray(seq_proc.get("emotion_probs_faces"), dtype=np.float32).reshape(-1, max_faces, 8)[j, :, :]
                    except Exception:
                        continue

                # Replace sequence_features with axis-aligned version (keeps UI + render stable).
                seq_axis = {
                    "frame_indices": axis_fi.astype(np.int32),
                    "times_s": times_s_axis.astype(np.float32),
                    "face_present": axis_face_present.astype(bool),
                    "processed_mask": processed_mask.astype(bool),
                    "face_count": axis_face_count.astype(np.int16),
                    "valence": valence,
                    "arousal": arousal,
                    "intensity": intensity,
                    "emotion_confidence": emotion_confidence,
                    "emotion_probs": emotion_probs,
                    "dominant_emotion_id": dominant_emotion_id,
                    "valence_faces": valence_faces,
                    "arousal_faces": arousal_faces,
                    "emotion_confidence_faces": emotion_confidence_faces,
                    "emotion_probs_faces": emotion_probs_faces,
                }
                results["sequence_features"] = seq_axis
                results["frame_indices"] = axis_fi.astype(np.int32)
                results["times_s"] = times_s_axis.astype(np.float32)
                # v3: duplicate model-facing time-series to top-level for easier consumption.
                results["face_present"] = axis_face_present.astype(bool)
                results["processed_mask"] = processed_mask.astype(bool)
                results["face_count"] = axis_face_count.astype(np.int16)
                results["valence"] = valence.astype(np.float32)
                results["arousal"] = arousal.astype(np.float32)
                results["intensity"] = intensity.astype(np.float32)
                results["emotion_confidence"] = emotion_confidence.astype(np.float32)
                results["emotion_probs"] = emotion_probs.astype(np.float32)
                results["dominant_emotion_id"] = dominant_emotion_id.astype(np.int8)
                # axis_source должен быть сохранен как object (scalar), а не как строка
                axis_source_obj = np.empty((), dtype=object)
                axis_source_obj[()] = axis_source
                results["axis_source"] = axis_source_obj

                # Keyframes (baseline: enabled; derived from valence/arousal)
                try:
                    v = np.asarray(seq.get("valence"), dtype=np.float32).reshape(-1)
                    a = np.asarray(seq.get("arousal"), dtype=np.float32).reshape(-1)
                    ts = np.asarray(times_s, dtype=np.float32).reshape(-1)
                    # Используем safe_array_check для проверки пустых массивов
                    if safe_array_check(v) and safe_array_check(a) and safe_array_check(ts) and v.size >= 3 and a.size == v.size and ts.size == v.size:
                        # Convert min-distance in seconds to a frame distance using median dt (no fps).
                        dt = np.diff(ts)
                        dt = dt[np.isfinite(dt) & (dt > 1e-6)]
                        med_dt = float(np.median(dt)) if dt.size else 1.0 / 30.0
                        min_distance_sec = 0.25
                        min_distance = max(1, int(round(min_distance_sec / med_dt)))
                        curve = {"valence": v.tolist(), "arousal": a.tolist()}
                        kf = detect_keyframes(curve, EMOTION_CLASSES, threshold=0.3, smooth_window=5, prominence=0.1, min_distance=min_distance)
                        keyframes: List[Dict[str, Any]] = []
                        for local_idx, info in (kf or {}).items():
                            try:
                                li = int(local_idx)
                            except Exception:
                                continue
                            if li < 0 or li >= int(v.size):
                                continue
                            keyframes.append(
                                {
                                    "global_index": int(selected_fi[li]),
                                    "local_index": li,
                                    "time_s": float(ts[li]),
                                    "type": str(info.get("type") or "event"),
                                    "score": float(info.get("score")) if info.get("score") is not None else None,
                                    "valence": float(info.get("valence", v[li])),
                                    "arousal": float(info.get("arousal", a[li])),
                                }
                            )
                        # keyframes должен быть object array для правильного сохранения
                        if keyframes:
                            keyframes_arr = np.empty(len(keyframes), dtype=object)
                            for i, kf in enumerate(keyframes):
                                keyframes_arr[i] = kf
                        else:
                            keyframes_arr = np.asarray([], dtype=object)
                        results["keyframes"] = keyframes_arr
                except Exception:
                    # Keyframes are best-effort; do not fail the whole run.
                    if "keyframes" not in results or not isinstance(results.get("keyframes"), np.ndarray):
                        results["keyframes"] = np.asarray([], dtype=object)

            # --- build meta.ui_payload ---
            ui_payload = self._build_ui_payload(results)

            # Save metadata (baseline meta contract)
            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": int(results.get("sequence_features", {}).get("frame_indices", np.asarray([], dtype=np.int32)).shape[0]),
                "frames_dir": frames_dir,
                "platform_id": platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "sampling_policy_version": metadata.get("sampling_policy_version"),
                "config_hash": metadata.get("config_hash"),
                "dataprocessor_version": metadata.get("dataprocessor_version") or "unknown",
                "analysis_width": metadata.get("analysis_width") or metadata.get("width"),
                "analysis_height": metadata.get("analysis_height") or metadata.get("height"),
                "status": status,
                "empty_reason": empty_reason,
                "ui_payload": ui_payload,  # will be copied into meta by save_results via boxing
            }
            if resource_profile_before:
                save_metadata["resource_profile_before"] = resource_profile_before
            # Stage timings must live in meta (schema contract).
            save_metadata["stage_timings_ms"] = {k: float(v) for k, v in stage_timings.items()}
            save_metadata["module_sampling_policy_version"] = "segmenter_axis_v1" if axis_source == "emotion_face" else "core_face_landmarks_axis_fallback_v1"
            save_metadata["face_frames_sampling_policy_version"] = f"faces_stride_v1_stride_{max(1, int(self.face_frame_stride))}_cap_{int(self.max_frames)}"
            # Config highlights for reproducibility (thresholds/calibrations/sampling).
            save_metadata["face_frame_stride"] = int(self.face_frame_stride)
            save_metadata["max_frames"] = int(self.max_frames)
            save_metadata["max_faces_per_frame"] = int(self.max_faces_per_frame)
            save_metadata["face_bbox_margin"] = float(self.face_bbox_margin)
            save_metadata["transition_threshold"] = float(self.transition_threshold)
            save_metadata["max_gap_seconds"] = float(self.max_gap_seconds)
            save_metadata["quality_threshold"] = float(self.quality_threshold)
            save_metadata["min_diversity_threshold"] = float(self.min_diversity_threshold)
            save_metadata["target_length"] = int(self.target_length)
            save_metadata["min_frames_ratio"] = float(self.min_frames_ratio)
            save_metadata["min_keyframes"] = int(self.min_keyframes)
            save_metadata["min_transitions"] = int(self.min_transitions)
            save_metadata["emonet_model_spec"] = self.emonet_model_spec
            save_metadata["device"] = str(self.device or "cuda")
            save_metadata["enable_microexpressions"] = bool(self.enable_microexpressions)
            save_metadata["enable_emotional_individuality"] = bool(self.enable_emotional_individuality)
            save_metadata["enable_face_asymmetry"] = bool(self.enable_face_asymmetry)
            try:
                save_metadata["models_used"] = self.get_models_used(config=config or {}, metadata=metadata or {})
            except Exception:
                save_metadata["models_used"] = []

            t_stage = time.perf_counter()
            saved_path = self.save_results(results=results, metadata=save_metadata)
            stage_timings["save"] = (time.perf_counter() - t_stage) * 1000.0
            stage_timings["total"] = (time.perf_counter() - t0) * 1000.0

            if self.rs_path is not None:
                _emit_progress(rs_path=str(self.rs_path), platform_id=platform_id, video_id=video_id, run_id=run_id, done=1, total=1, stage="done")
            return saved_path
        finally:
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    self.logger.exception(f"{self.module_name} | Ошибка при закрытии FrameManager: {e}")
    
    def process_batch(
        self,
        video_contexts: List[VideoContext],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Batch processing для emotion_face.
        
        Args:
            video_contexts: Список VideoContext для каждого видео
            config: Конфигурация модуля
            
        Returns:
            Список результатов для каждого видео
        """
        from utils.emotion_face_batch import process_emotion_face_batch
        
        # Подготавливаем конфиг для batch processing
        batch_config = {
            "face_frame_stride": self.face_frame_stride,
            "max_frames": self.max_frames,
            "max_faces_per_frame": self.max_faces_per_frame,
            "face_bbox_margin": self.face_bbox_margin,
            "emonet_model_spec": self.emonet_model_spec,
            "emo_path": self.emo_path,
            "device": self.device,
        }
        
        # Получаем параметры batch processing из глобального конфига
        max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
        batch_size = config.get("batch_size", 16)
        
        return process_emotion_face_batch(
            video_contexts=video_contexts,
            config=batch_config,
            max_frames_per_batch=max_frames_per_batch,
            batch_size=batch_size,
        )

