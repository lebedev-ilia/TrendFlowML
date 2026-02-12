"""
RhythmicExtractor: извлечение ритмических метрик (beat tracking, регулярность) с использованием librosa/essentia.
Интеграция с общим интерфейсом BaseExtractor и AudioUtils.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (fail-fast, explicit backend selection)
- Per-run storage for .npy files
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Additional parameters for librosa/essentia
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
RHYTHMIC_CONTRACT_VERSION = "rhythmic_contract_v1"

# Threshold for saving large arrays to .npy files
BEAT_TIMES_SAVE_THRESHOLD = 10000


class RhythmicExtractor(BaseExtractor):
    """Экстрактор ритмических метрик: beat tracking, регулярность, плотность ударов."""

    name = "rhythmic"
    version = "2.0.0"
    description = "Ритмические метрики: темп, биты, регулярность"
    category = "rhythm"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.2

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        average_channels: bool = True,
        # Backend selection (no-fallback policy)
        backend: str = "librosa",  # "librosa" | "essentia"
        # Additional librosa parameters
        start_bpm: Optional[float] = None,
        std_bpm: Optional[float] = None,
        ac_size: int = 4,
        max_tempo: Optional[float] = None,
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_metrics: bool = False,
        enable_interval_stats: bool = False,
        enable_regularity_metrics: bool = False,
        enable_beat_times: bool = False,
        enable_tempo_metrics: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация Rhythmic экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            hop_length: Размер hop для анализа
            average_channels: Усреднять каналы для многоканального аудио
            backend: Backend для beat tracking ("librosa" | "essentia")
            start_bpm: Начальный BPM для librosa beat tracking
            std_bpm: Стандартное отклонение BPM для librosa
            ac_size: Размер автокорреляции для librosa
            max_tempo: Максимальный темп для librosa
            enable_basic_metrics: Включить базовые метрики (tempo_bpm, beats_count, beat_density)
            enable_interval_stats: Включить статистики интервалов (avg_period, std_period, min/max/median)
            enable_regularity_metrics: Включить метрики регулярности (regularity, syncopation, etc.)
            enable_beat_times: Включить временные метки ударов (beat_times)
            enable_tempo_metrics: Включить метрики темпа (median_bpm, tempo_variation, etc.)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(sample_rate, hop_length, backend, start_bpm, std_bpm, ac_size, max_tempo)

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.average_channels = bool(average_channels)
        self.backend = str(backend)

        # Librosa parameters
        self.librosa_params = dict(
            start_bpm=start_bpm,
            std_bpm=std_bpm,
            ac_size=ac_size,
            max_tempo=max_tempo,
        )

        # Feature gating flags
        self.enable_basic_metrics = bool(enable_basic_metrics)
        self.enable_interval_stats = bool(enable_interval_stats)
        self.enable_regularity_metrics = bool(enable_regularity_metrics)
        self.enable_beat_times = bool(enable_beat_times)
        self.enable_tempo_metrics = bool(enable_tempo_metrics)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def _validate_parameters(
        self,
        sample_rate: int,
        hop_length: int,
        backend: str,
        start_bpm: Optional[float],
        std_bpm: Optional[float],
        ac_size: int,
        max_tempo: Optional[float],
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            hop_length: Размер hop для анализа
            backend: Backend для beat tracking
            start_bpm: Начальный BPM
            std_bpm: Стандартное отклонение BPM
            ac_size: Размер автокорреляции
            max_tempo: Максимальный темп

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"rhythmic | sample_rate must be positive, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"rhythmic | hop_length must be positive, got {hop_length}")
        if backend not in ["librosa", "essentia"]:
            raise ValueError(f"rhythmic | backend must be 'librosa' or 'essentia', got {backend}")
        if start_bpm is not None and (start_bpm <= 0 or start_bpm > 300):
            raise ValueError(f"rhythmic | start_bpm must be in (0, 300], got {start_bpm}")
        if std_bpm is not None and (std_bpm <= 0 or std_bpm > 100):
            raise ValueError(f"rhythmic | std_bpm must be in (0, 100], got {std_bpm}")
        if ac_size < 1 or ac_size > 16:
            raise ValueError(f"rhythmic | ac_size must be in [1, 16], got {ac_size}")
        if max_tempo is not None and (max_tempo <= 0 or max_tempo > 300):
            raise ValueError(f"rhythmic | max_tempo must be in (0, 300], got {max_tempo}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - rhythmic_audio_load_failed
                - rhythmic_essentia_failed
                - rhythmic_librosa_failed
                - rhythmic_no_beats_detected
                - rhythmic_validation_failed
                - rhythmic_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "rhythmic_audio_load_failed"
        if "essentia" in error_str or context == "essentia_failed":
            return "rhythmic_essentia_failed"
        if "librosa" in error_str or context == "librosa_failed":
            return "rhythmic_librosa_failed"
        if "beat" in error_str or "no beats" in error_str or context == "no_beats_detected":
            return "rhythmic_no_beats_detected"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "rhythmic_validation_failed"

        return "rhythmic_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        try:
            # Check for NaN/inf
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        return False, f"rhythmic | {key} contains NaN or inf: {value}"
                elif isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        return False, f"rhythmic | {key} contains NaN or inf"

            # Validate ranges
            if "rhythm_tempo_bpm" in features:
                tempo = features["rhythm_tempo_bpm"]
                if not (40.0 <= tempo <= 300.0):
                    return False, f"rhythmic | tempo_bpm out of range [40, 300]: {tempo}"

            if "rhythm_regularity" in features:
                regularity = features["rhythm_regularity"]
                if not (0.0 <= regularity <= 1.0):
                    return False, f"rhythmic | regularity out of range [0, 1]: {regularity}"

            if "rhythm_beat_density" in features:
                density = features["rhythm_beat_density"]
                if density < 0.0 or density > 10.0:
                    return False, f"rhythmic | beat_density out of reasonable range [0, 10]: {density}"

            # Validate consistency
            if "rhythm_avg_period_sec" in features and "rhythm_tempo_bpm" in features:
                avg_period = features["rhythm_avg_period_sec"]
                tempo = features["rhythm_tempo_bpm"]
                if avg_period > 0:
                    expected_tempo = 60.0 / avg_period
                    if abs(tempo - expected_tempo) > 10.0:  # Allow 10 BPM tolerance
                        logger.warning(
                            f"rhythmic | tempo_bpm ({tempo}) inconsistent with avg_period ({avg_period}): expected {expected_tempo:.2f}"
                        )

            return True, None
        except Exception as e:
            return False, f"rhythmic | validation error: {e}"

    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Нормализовать аудио сигнал (опционально).

        Args:
            y: Аудио сигнал

        Returns:
            Нормализованный сигнал
        """
        if not self.enable_audio_normalization:
            return y

        max_val = np.max(np.abs(y))
        if max_val > 1e-9:
            return y / max_val
        return y

    def _beat_track_essentia(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """
        Beat tracking с использованием Essentia (fail-fast, no-fallback).

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            (beat_times, tempo)

        Raises:
            RuntimeError: Если Essentia недоступна или произошла ошибка
        """
        try:
            import essentia
            import essentia.standard as es
        except ImportError as e:
            raise RuntimeError(f"rhythmic | Essentia not available: {e}") from e

        try:
            audio = y.astype(np.float32)
            # Onset detection + beat tracking в Essentia
            od = es.OnsetRate()
            onset_rate, onset_times = od(audio)
            bt = es.BeatTrackerMultiFeature()
            beats, ticks = bt(audio)
            beat_times = np.array(beats, dtype=np.float32)

            # Темп оценим как медиану межударных интервалов
            tempo = 0.0
            if beat_times.size > 1:
                intervals = np.diff(beat_times)
                tempo = float(60.0 / (np.median(intervals) + 1e-6))

            return beat_times, tempo
        except Exception as e:
            raise RuntimeError(f"rhythmic | Essentia beat tracking failed: {e}") from e

    def _beat_track_librosa(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """
        Beat tracking с использованием librosa (fail-fast, no-fallback).

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            (beat_times, tempo)

        Raises:
            RuntimeError: Если librosa недоступна или произошла ошибка
        """
        try:
            import librosa
        except ImportError as e:
            raise RuntimeError(f"rhythmic | librosa not available: {e}") from e

        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

            # Build kwargs for beat_track
            # Note: ac_size is not a valid parameter for librosa.beat.beat_track()
            # It was removed in newer versions of librosa
            beat_track_kwargs = {"onset_envelope": onset_env, "sr": sr, "hop_length": self.hop_length}
            if self.librosa_params["start_bpm"] is not None:
                beat_track_kwargs["start_bpm"] = self.librosa_params["start_bpm"]
            if self.librosa_params["std_bpm"] is not None:
                beat_track_kwargs["std_bpm"] = self.librosa_params["std_bpm"]
            # ac_size parameter is not supported in librosa.beat.beat_track() - removed
            if self.librosa_params["max_tempo"] is not None:
                beat_track_kwargs["max_tempo"] = self.librosa_params["max_tempo"]

            tempo, beat_frames = librosa.beat.beat_track(**beat_track_kwargs)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

            return beat_times, float(tempo)
        except Exception as e:
            raise RuntimeError(f"rhythmic | librosa beat tracking failed: {e}") from e

    def _compute_beat_metrics(self, beat_times: np.ndarray, duration: float) -> Dict[str, Any]:
        """
        Вычислить все метрики на основе beat_times.

        Args:
            beat_times: Временные метки ударов
            duration: Длительность аудио (секунды)

        Returns:
            Словарь с метриками
        """
        features: Dict[str, Any] = {}

        if beat_times.size == 0:
            # No beats detected
            if self.enable_basic_metrics:
                features["rhythm_tempo_bpm"] = 0.0
                features["rhythm_beats_count"] = 0
                features["rhythm_beat_density"] = 0.0
            if self.enable_interval_stats:
                features["rhythm_avg_period_sec"] = 0.0
                features["rhythm_period_std_sec"] = 0.0
                features["rhythm_median_period_sec"] = 0.0
                features["rhythm_min_period_sec"] = 0.0
                features["rhythm_max_period_sec"] = 0.0
            if self.enable_regularity_metrics:
                features["rhythm_regularity"] = 0.0
            if self.enable_tempo_metrics:
                features["rhythm_median_bpm"] = 0.0
            return features

        # Интервалы между ударами
        intervals = np.diff(beat_times) if beat_times.size > 1 else np.array([])

        # Basic metrics
        if self.enable_basic_metrics:
            tempo = float(60.0 / (np.median(intervals) + 1e-6)) if intervals.size > 0 else 0.0
            features["rhythm_tempo_bpm"] = tempo
            features["rhythm_beats_count"] = int(beat_times.size)
            features["rhythm_beat_density"] = float(beat_times.size / (duration + 1e-9))

        # Interval stats
        if self.enable_interval_stats:
            features["rhythm_avg_period_sec"] = float(np.mean(intervals)) if intervals.size > 0 else 0.0
            features["rhythm_period_std_sec"] = float(np.std(intervals)) if intervals.size > 0 else 0.0
            features["rhythm_median_period_sec"] = float(np.median(intervals)) if intervals.size > 0 else 0.0
            features["rhythm_min_period_sec"] = float(np.min(intervals)) if intervals.size > 0 else 0.0
            features["rhythm_max_period_sec"] = float(np.max(intervals)) if intervals.size > 0 else 0.0

        # Regularity metrics
        if self.enable_regularity_metrics:
            if intervals.size > 0:
                avg_period = np.mean(intervals)
                std_period = np.std(intervals)
                cv = float(std_period / (avg_period + 1e-9))
                regularity = float(1.0 / (1.0 + cv))
            else:
                regularity = 0.0

            features["rhythm_regularity"] = regularity

            # Additional regularity metrics
            if intervals.size > 1:
                # Syncopation score (based on interval variance)
                syncopation = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
                features["rhythm_syncopation_score"] = syncopation

                # Polyrhythm score (based on interval distribution)
                # Check if intervals cluster around multiple values (polyrhythm indicator)
                interval_hist, _ = np.histogram(intervals, bins=min(10, len(intervals)))
                polyrhythm = float(np.std(interval_hist) / (np.mean(interval_hist) + 1e-9))
                features["rhythm_polyrhythm_score"] = polyrhythm

                # Beat strength (based on consistency)
                beat_strength_mean = float(np.mean(1.0 / (intervals + 1e-9)))
                beat_strength_std = float(np.std(1.0 / (intervals + 1e-9)))
                features["rhythm_beat_strength_mean"] = beat_strength_mean
                features["rhythm_beat_strength_std"] = beat_strength_std

                # Metrical stability (inverse of coefficient of variation)
                metrical_stability = float(1.0 / (cv + 1e-9))
                features["rhythm_metrical_stability"] = metrical_stability

        # Tempo metrics
        if self.enable_tempo_metrics:
            if intervals.size > 0:
                median_period = np.median(intervals)
                median_bpm = float(60.0 / (median_period + 1e-9))
                features["rhythm_median_bpm"] = median_bpm

                # Tempo variation (coefficient of variation of intervals)
                tempo_variation = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
                features["rhythm_tempo_variation"] = tempo_variation

                # Beat consistency (inverse of tempo variation)
                beat_consistency = float(1.0 / (1.0 + tempo_variation))
                features["rhythm_beat_consistency"] = beat_consistency

        return features

    def _save_beat_times_npy(self, beat_times: np.ndarray, component_name: str) -> Optional[str]:
        """
        Сохранить beat_times в .npy файл для больших массивов (per-run storage).

        Args:
            beat_times: Массив временных меток ударов
            component_name: Имя компонента

        Returns:
            Относительный путь к .npy файлу или None
        """
        if beat_times.size < BEAT_TIMES_SAVE_THRESHOLD:
            return None

        if self.artifacts_dir is None:
            logger.warning("rhythmic | artifacts_dir not set, cannot save beat_times to .npy")
            return None

        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            npy_path = os.path.join(self.artifacts_dir, "beat_times.npy")
            np.save(npy_path, beat_times)
            # Return relative path from component directory
            rel_path = "_artifacts/beat_times.npy"
            logger.info(f"rhythmic | Saved beat_times ({beat_times.size} elements) to {npy_path}")
            return rel_path
        except Exception as e:
            logger.warning(f"rhythmic | Failed to save beat_times to .npy: {e}")
            return None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Обработка полного аудио файла (legacy mode, для обратной совместимости).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория

        Returns:
            ExtractorResult с ритмическими метриками
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    False,
                    error=f"rhythmic | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Load audio
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            # Normalize audio if enabled
            y = self._normalize_audio(y)

            duration = float(y.shape[-1] / sr)

            # Beat tracking (explicit backend, fail-fast)
            if self.backend == "essentia":
                beat_times, tempo = self._beat_track_essentia(y, sr)
            elif self.backend == "librosa":
                beat_times, tempo = self._beat_track_librosa(y, sr)
            else:
                raise ValueError(f"rhythmic | Unknown backend: {self.backend}")

            if beat_times.size == 0:
                error_code = self._classify_error(RuntimeError("No beats detected"), "no_beats_detected")
                return self._create_result(
                    False,
                    error=f"rhythmic | No beats detected (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Compute metrics
            features = self._compute_beat_metrics(beat_times, duration)

            # Save beat_times if needed
            beat_times_npy_path = None
            if self.enable_beat_times:
                if beat_times.size >= BEAT_TIMES_SAVE_THRESHOLD:
                    beat_times_npy_path = self._save_beat_times_npy(beat_times, self.name)
                    features["beat_times_npy"] = beat_times_npy_path
                else:
                    features["beat_times"] = beat_times.astype(np.float32).tolist()

            # Add metadata
            features["sample_rate"] = int(sr)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = duration
            features["device_used"] = self.device
            features["backend"] = self.backend
            features["rhythmic_contract_version"] = RHYTHMIC_CONTRACT_VERSION
            
            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_basic_metrics:
                features_enabled.append("basic_metrics")
            if self.enable_interval_stats:
                features_enabled.append("interval_stats")
            if self.enable_regularity_metrics:
                features_enabled.append("regularity_metrics")
            if self.enable_beat_times:
                features_enabled.append("beat_times")
            if self.enable_tempo_metrics:
                features_enabled.append("tempo_metrics")
            features["_features_enabled"] = features_enabled

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"rhythmic | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"rhythmic | {error_code}: {str(e)}", processing_time=dt)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        *,
        segment_parallelism: int = 1,
        max_inflight: Optional[int] = None,
    ) -> ExtractorResult:
        """
        Segmenter-driven rhythmic extraction: compute beat tracking on provided windows (families.rhythmic).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов из Segmenter (families.rhythmic)
            segment_parallelism: Количество параллельных потоков для обработки сегментов
            max_inflight: Максимальное количество одновременно обрабатываемых сегментов

        Returns:
            ExtractorResult с агрегированными метриками по сегментам
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    False,
                    error=f"rhythmic | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("rhythmic | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            all_beat_times: List[np.ndarray] = []
            all_tempos: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            seg_p = max(1, int(segment_parallelism or 1))
            inflight = int(max_inflight) if max_inflight is not None else seg_p
            inflight = max(1, int(inflight))

            def _process_segment(i: int, seg: dict) -> tuple[int, float, np.ndarray, float, float]:
                """Обработать один сегмент."""
                ss = int(seg.get("start_sample", 0))
                es = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))
                start_sec = float(seg.get("start_sec", 0.0))
                end_sec = float(seg.get("end_sec", 0.0))
                duration = end_sec - start_sec

                # Load audio segment
                waveform_t, sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=ss,
                    end_sample=es,
                    target_sr=self.sample_rate,
                    mix_to_mono=self.average_channels,
                )
                waveform_np = self.audio_utils.to_numpy(waveform_t)
                if waveform_np.ndim == 2:
                    waveform_np = np.mean(waveform_np, axis=0) if self.average_channels else waveform_np[0]

                # Normalize audio if enabled
                waveform_np = self._normalize_audio(waveform_np)

                # Beat tracking (explicit backend, fail-fast)
                if self.backend == "essentia":
                    beat_times, tempo = self._beat_track_essentia(waveform_np, int(sr))
                elif self.backend == "librosa":
                    beat_times, tempo = self._beat_track_librosa(waveform_np, int(sr))
                else:
                    raise ValueError(f"rhythmic | Unknown backend: {self.backend}")

                # Adjust beat_times to absolute time (add segment start time)
                if beat_times.size > 0:
                    beat_times = beat_times + start_sec

                return i, center_sec, beat_times, tempo, duration

            # Process segments (sequential or parallel)
            if seg_p <= 1:
                for seg_idx, seg in enumerate(segments):
                    _, center_sec, beat_times, tempo, duration = _process_segment(seg_idx, seg)
                    all_beat_times.append(beat_times)
                    all_tempos.append(tempo)
                    segment_centers.append(center_sec)
                    segment_durations.append(duration)

                    # Progress reporting
                    if self.progress_callback and seg_idx % progress_report_interval == 0:
                        pct = int((seg_idx + 1) * 100 / total_segments)
                        if pct != last_reported_pct:
                            self.progress_callback("rhythmic", seg_idx + 1, total_segments, f"Processed {seg_idx + 1}/{total_segments} segments")
                            last_reported_pct = pct
            else:
                # Parallel processing
                workers = max(1, min(int(seg_p), int(inflight)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_process_segment, i, seg) for i, seg in enumerate(segments)]
                    completed = 0
                    for fut in as_completed(futures):
                        i, center_sec, beat_times, tempo, duration = fut.result()
                        all_beat_times.append(beat_times)
                        all_tempos.append(tempo)
                        segment_centers.append(center_sec)
                        segment_durations.append(duration)
                        completed += 1

                        # Progress reporting
                        if self.progress_callback and completed % progress_report_interval == 0:
                            pct = int(completed * 100 / total_segments)
                            if pct != last_reported_pct:
                                self.progress_callback("rhythmic", completed, total_segments, f"Processed {completed}/{total_segments} segments")
                                last_reported_pct = pct

            # Aggregate metrics across all segments
            if len(all_beat_times) == 0:
                error_code = self._classify_error(RuntimeError("No segments processed"), "no_beats_detected")
                return self._create_result(
                    False,
                    error=f"rhythmic | No segments processed (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Concatenate all beat_times
            all_beats_combined = np.concatenate([bt for bt in all_beat_times if bt.size > 0]) if any(bt.size > 0 for bt in all_beat_times) else np.array([])

            # Compute aggregate metrics
            total_duration = sum(segment_durations)
            features = self._compute_beat_metrics(all_beats_combined, total_duration)

            # Add segment-level aggregates
            if self.enable_tempo_metrics and all_tempos:
                features["rhythm_tempo_mean"] = float(np.mean(all_tempos))
                features["rhythm_tempo_std"] = float(np.std(all_tempos))
                features["rhythm_tempo_min"] = float(np.min(all_tempos))
                features["rhythm_tempo_max"] = float(np.max(all_tempos))

            # Save beat_times if needed
            beat_times_npy_path = None
            if self.enable_beat_times:
                if all_beats_combined.size >= BEAT_TIMES_SAVE_THRESHOLD:
                    beat_times_npy_path = self._save_beat_times_npy(all_beats_combined, self.name)
                    features["beat_times_npy"] = beat_times_npy_path
                else:
                    features["beat_times"] = all_beats_combined.astype(np.float32).tolist()

            # Add metadata
            features["sample_rate"] = int(self.sample_rate)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = total_duration
            features["device_used"] = self.device
            features["backend"] = self.backend
            features["segments_count"] = int(total_segments)
            features["rhythmic_contract_version"] = RHYTHMIC_CONTRACT_VERSION
            
            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_basic_metrics:
                features_enabled.append("basic_metrics")
            if self.enable_interval_stats:
                features_enabled.append("interval_stats")
            if self.enable_regularity_metrics:
                features_enabled.append("regularity_metrics")
            if self.enable_beat_times:
                features_enabled.append("beat_times")
            if self.enable_tempo_metrics:
                features_enabled.append("tempo_metrics")
            features["_features_enabled"] = features_enabled

            # Add segment-level time series if enabled
            if self.enable_beat_times and len(all_beat_times) > 0:
                features["segment_beat_times"] = [bt.astype(np.float32).tolist() for bt in all_beat_times]
                features["segment_centers_sec"] = segment_centers
                features["segment_durations_sec"] = segment_durations

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"rhythmic | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"rhythmic | {error_code}: {str(e)}", processing_time=dt)

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        rhythmic_extractor поддерживает batch processing через extract_batch_segments()
        с CPU parallelism для обработки сегментов из нескольких видео одновременно.
        """
        return True

    def extract_batch_segments(
        self,
        audio_files: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких аудио файлов.
        
        Использует ThreadPoolExecutor для параллельной обработки сегментов из всех файлов.
        Каждый файл обрабатывается изолированно через run_segments().
        
        Args:
            audio_files: Список словарей с ключами:
                - 'input_uri': URI к входному аудио файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'file_id': Идентификатор файла (опционально, для логирования)
                - 'segments': Список сегментов для обработки (обязательно)
            max_workers: Количество параллельных воркеров (None = os.cpu_count())
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        import os as os_module
        
        if max_workers is None:
            max_workers = os_module.cpu_count() or 4
        
        results: List[ExtractorResult] = []
        
        def process_file(file_info: Dict[str, Any]) -> ExtractorResult:
            input_uri = file_info.get("input_uri")
            tmp_path = file_info.get("tmp_path")
            file_id = file_info.get("file_id", input_uri)
            segments = file_info.get("segments", [])
            
            if not input_uri or not tmp_path:
                logger.error(f"rhythmic | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"rhythmic | Missing segments for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing segments",
                )
            
            try:
                return self.run_segments(
                    input_uri=input_uri,
                    tmp_path=tmp_path,
                    segments=segments,
                )
            except Exception as e:
                logger.error(f"rhythmic | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
