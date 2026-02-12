"""
OnsetExtractor: извлечение онсетов (пики атак) с использованием librosa/essentia.
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
- Optional integration with tempo_extractor
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
ONSET_CONTRACT_VERSION = "onset_contract_v1"

# Threshold for saving large arrays to .npy files
ONSET_TIMES_SAVE_THRESHOLD = 10000


class OnsetExtractor(BaseExtractor):
    """Экстрактор онсетов (атаки звука) с поддержкой segment-based обработки."""

    name = "onset"
    version = "2.0.0"
    description = "Определение онсетов (атака звука)"
    category = "rhythm"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 0.8

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        pre_max: int = 3,
        post_max: int = 3,
        pre_avg: int = 3,
        post_avg: int = 5,
        delta: float = 0.2,
        wait: int = 10,
        # Backend selection (no-fallback policy)
        backend: str = "librosa",  # "librosa" | "essentia"
        # Additional librosa parameters
        units: str = "time",  # "time" | "frames"
        backtrack: bool = False,
        energy: bool = False,
        normalize: bool = False,
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_features: bool = False,
        enable_interval_stats: bool = False,
        enable_rhythmic_metrics: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Optional integration with tempo_extractor
        tempo_payload: Optional[Dict[str, Any]] = None,  # Results from tempo_extractor for validation
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация Onset экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            hop_length: Размер hop для анализа онсетов
            pre_max: Количество кадров до максимума для пикового детектора
            post_max: Количество кадров после максимума для пикового детектора
            pre_avg: Количество кадров до для усреднения
            post_avg: Количество кадров после для усреднения
            delta: Минимальная разница для обнаружения онсета
            wait: Минимальное количество кадров между онсетами
            backend: Backend для обнаружения онсетов ("librosa" | "essentia")
            units: Единицы измерения для онсетов ("time" | "frames")
            backtrack: Включить backtrack для обнаружения онсетов
            energy: Использовать энергетический детектор
            normalize: Нормализовать onset envelope
            enable_basic_features: Включить базовые фичи (onset_times, onset_count)
            enable_interval_stats: Включить статистики интервалов (interval_std, interval_min, etc.)
            enable_rhythmic_metrics: Включить ритмические метрики (regularity, clustering, etc.)
            enable_time_series: Включить временные серии (onset_times как time series)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            tempo_payload: Результаты от tempo_extractor для валидации/улучшения результатов
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(
            sample_rate, hop_length, pre_max, post_max, pre_avg, post_avg, delta, wait, backend, units
        )

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.backend = str(backend)
        self.units = str(units)
        self.backtrack = bool(backtrack)
        self.energy = bool(energy)
        self.normalize = bool(normalize)

        self.librosa_params = dict(
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait,
            backtrack=backtrack,
            energy=energy,
            normalize=normalize,
        )

        # Feature gating flags
        self.enable_basic_features = bool(enable_basic_features)
        self.enable_interval_stats = bool(enable_interval_stats)
        self.enable_rhythmic_metrics = bool(enable_rhythmic_metrics)
        self.enable_time_series = bool(enable_time_series)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Optional integration with tempo_extractor
        self.tempo_payload = tempo_payload

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def _validate_parameters(
        self,
        sample_rate: int,
        hop_length: int,
        pre_max: int,
        post_max: int,
        pre_avg: int,
        post_avg: int,
        delta: float,
        wait: int,
        backend: str,
        units: str,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            hop_length: Размер hop для анализа онсетов
            pre_max: Количество кадров до максимума
            post_max: Количество кадров после максимума
            pre_avg: Количество кадров до для усреднения
            post_avg: Количество кадров после для усреднения
            delta: Минимальная разница для обнаружения онсета
            wait: Минимальное количество кадров между онсетами
            backend: Backend для обнаружения онсетов
            units: Единицы измерения для онсетов

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"onset | sample_rate must be positive, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"onset | hop_length must be positive, got {hop_length}")
        if pre_max < 0:
            raise ValueError(f"onset | pre_max must be non-negative, got {pre_max}")
        if post_max < 0:
            raise ValueError(f"onset | post_max must be non-negative, got {post_max}")
        if pre_avg < 0:
            raise ValueError(f"onset | pre_avg must be non-negative, got {pre_avg}")
        if post_avg < 0:
            raise ValueError(f"onset | post_avg must be non-negative, got {post_avg}")
        if delta < 0.0:
            raise ValueError(f"onset | delta must be non-negative, got {delta}")
        if wait < 0:
            raise ValueError(f"onset | wait must be non-negative, got {wait}")
        if backend not in ["librosa", "essentia"]:
            raise ValueError(f"onset | backend must be 'librosa' or 'essentia', got {backend}")
        if units not in ["time", "frames"]:
            raise ValueError(f"onset | units must be 'time' or 'frames', got {units}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - onset_audio_load_failed
                - onset_essentia_failed
                - onset_librosa_failed
                - onset_validation_failed
                - onset_insufficient_data
                - onset_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "onset_audio_load_failed"
        if "essentia" in error_str or context == "essentia_failed":
            return "onset_essentia_failed"
        if "librosa" in error_str or context == "librosa_failed":
            return "onset_librosa_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "onset_validation_failed"
        if "insufficient" in error_str or "empty" in error_str or context == "insufficient_data":
            return "onset_insufficient_data"

        return "onset_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "onset | features must be a dict"

        # Validate onset_times if present
        if "onset_times" in features:
            onset_times = features.get("onset_times")
            if onset_times is not None:
                if isinstance(onset_times, list):
                    onset_times = np.asarray(onset_times, dtype=np.float32)
                if not isinstance(onset_times, np.ndarray):
                    return False, "onset | onset_times must be numpy array or list"
                if np.any(np.isnan(onset_times)) or np.any(np.isinf(onset_times)):
                    return False, "onset | onset_times contains NaN or Inf values"
                if np.any(onset_times < 0):
                    return False, "onset | onset_times contains negative values"
                if len(onset_times) > 0:
                    if not np.all(np.diff(onset_times) >= 0):
                        return False, "onset | onset_times must be monotonically increasing"

        # Validate onset_count if present
        if "onset_count" in features:
            onset_count = features.get("onset_count")
            try:
                onset_count = int(onset_count)
                if onset_count < 0:
                    return False, "onset | onset_count must be non-negative"
                # Consistency check: onset_count == len(onset_times)
                if "onset_times" in features and features.get("onset_times") is not None:
                    onset_times = features.get("onset_times")
                    if isinstance(onset_times, list):
                        onset_times = np.asarray(onset_times, dtype=np.float32)
                    if onset_count != len(onset_times):
                        return False, f"onset | consistency check failed: onset_count ({onset_count}) != len(onset_times) ({len(onset_times)})"
            except (ValueError, TypeError):
                return False, f"onset | onset_count must be int, got {type(onset_count)}"

        # Validate intervals if present
        if "avg_interval_sec" in features:
            avg_interval = features.get("avg_interval_sec")
            if avg_interval is not None:
                try:
                    avg_interval = float(avg_interval)
                    if np.isnan(avg_interval) or np.isinf(avg_interval):
                        return False, "onset | avg_interval_sec is NaN or Inf"
                    if avg_interval < 0:
                        return False, "onset | avg_interval_sec must be non-negative"
                except (ValueError, TypeError):
                    return False, f"onset | avg_interval_sec must be float, got {type(avg_interval)}"

        # Validate interval stats if present
        for key in ["interval_std", "interval_min", "interval_max", "interval_median"]:
            if key in features:
                value = features.get(key)
                if value is not None:
                    try:
                        value = float(value)
                        if np.isnan(value) or np.isinf(value):
                            return False, f"onset | {key} is NaN or Inf"
                        if value < 0 and key != "interval_std":  # interval_std может быть 0
                            return False, f"onset | {key} must be non-negative, got {value}"
                    except (ValueError, TypeError):
                        return False, f"onset | {key} must be float, got {type(value)}"

        # Validate density if present
        if "onset_density_per_sec" in features:
            density = features.get("onset_density_per_sec")
            try:
                density = float(density)
                if np.isnan(density) or np.isinf(density):
                    return False, "onset | onset_density_per_sec is NaN or Inf"
                if density < 0:
                    return False, "onset | onset_density_per_sec must be non-negative"
            except (ValueError, TypeError):
                return False, f"onset | onset_density_per_sec must be float, got {type(density)}"

        return True, None

    def _extract_onsets(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Извлечение онсетов через выбранный backend (no-fallback policy).

        Args:
            y: Аудио сигнал (numpy array)
            sr: Частота дискретизации

        Returns:
            onset_times: Массив времен онсетов в секундах (float32)

        Raises:
            RuntimeError: Если выбранный backend недоступен или произошла ошибка
        """
        if self.backend == "essentia":
            try:
                import essentia.standard as es
                audio = y.astype(np.float32)
                od = es.OnsetRate()
                _, onset_times = od(audio)
                onset_times = np.array(onset_times, dtype=np.float32)
                if onset_times.ndim == 0:
                    onset_times = onset_times.reshape(1)
                return onset_times
            except ImportError:
                raise RuntimeError(
                    f"onset | essentia backend selected but essentia is not available (error_code=onset_essentia_failed)"
                )
            except Exception as e:
                raise RuntimeError(
                    f"onset | essentia backend failed: {e} (error_code=onset_essentia_failed)"
                )
        elif self.backend == "librosa":
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    units=self.units,
                    **self.librosa_params,
                )
                onset_times = np.array(onset_frames, dtype=np.float32)
                if onset_times.ndim == 0:
                    onset_times = onset_times.reshape(1)
                return onset_times
            except Exception as e:
                raise RuntimeError(
                    f"onset | librosa backend failed: {e} (error_code=onset_librosa_failed)"
                )
        else:
            raise RuntimeError(
                f"onset | unknown backend: {self.backend} (error_code=onset_unknown)"
            )

    def _compute_interval_stats(self, onset_times: np.ndarray) -> Dict[str, Optional[float]]:
        """
        Вычисление статистик интервалов между онсетами.

        Args:
            onset_times: Массив времен онсетов

        Returns:
            Словарь со статистиками интервалов
        """
        if onset_times.size <= 1:
            return {
                "interval_std": None,
                "interval_min": None,
                "interval_max": None,
                "interval_median": None,
            }

        intervals = np.diff(onset_times)
        return {
            "interval_std": float(np.std(intervals)),
            "interval_min": float(np.min(intervals)),
            "interval_max": float(np.max(intervals)),
            "interval_median": float(np.median(intervals)),
        }

    def _compute_rhythmic_metrics(
        self, onset_times: np.ndarray, duration: float
    ) -> Dict[str, float]:
        """
        Вычисление дополнительных ритмических метрик для ML/аналитики.

        Args:
            onset_times: Массив времен онсетов
            duration: Длительность аудио в секундах

        Returns:
            Словарь с ритмическими метриками
        """
        metrics = {}

        if onset_times.size <= 1:
            metrics.update(
                {
                    "onset_regularity_score": 0.0,
                    "onset_clustering_score": 0.0,
                    "onset_tempo_estimate": 0.0,
                    "onset_syncopation_score": 0.0,
                    "onset_strength_mean": 0.0,
                    "onset_strength_std": 0.0,
                    "onset_density_variance": 0.0,
                }
            )
            return metrics

        # Regularity score: 1 / (1 + CV), где CV = std/mean интервалов
        intervals = np.diff(onset_times)
        if intervals.size > 0 and np.mean(intervals) > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-9)
            metrics["onset_regularity_score"] = float(1.0 / (1.0 + cv))
        else:
            metrics["onset_regularity_score"] = 0.0

        # Clustering score: мера кластеризации онсетов по времени
        # Используем стандартное отклонение интервалов, нормализованное на средний интервал
        if intervals.size > 0 and np.mean(intervals) > 0:
            normalized_std = np.std(intervals) / (np.mean(intervals) + 1e-9)
            metrics["onset_clustering_score"] = float(1.0 / (1.0 + normalized_std))
        else:
            metrics["onset_clustering_score"] = 0.0

        # Tempo estimate: оценка BPM из интервалов
        if intervals.size > 0 and np.mean(intervals) > 0:
            avg_interval = np.mean(intervals)
            metrics["onset_tempo_estimate"] = float(60.0 / avg_interval)
        else:
            metrics["onset_tempo_estimate"] = 0.0

        # Syncopation score: мера синкопированности (вариация интервалов)
        if intervals.size > 0:
            cv = np.std(intervals) / (np.mean(intervals) + 1e-9)
            metrics["onset_syncopation_score"] = float(cv / (1.0 + cv))
        else:
            metrics["onset_syncopation_score"] = 0.0

        # Onset strength: средняя и стандартное отклонение силы онсетов
        # (используем простую эвристику: обратная величина интервалов)
        if intervals.size > 0:
            strengths = 1.0 / (intervals + 1e-9)
            metrics["onset_strength_mean"] = float(np.mean(strengths))
            metrics["onset_strength_std"] = float(np.std(strengths))
        else:
            metrics["onset_strength_mean"] = 0.0
            metrics["onset_strength_std"] = 0.0

        # Density variance: вариация плотности онсетов по времени
        # Разделяем аудио на окна и считаем плотность в каждом окне
        if duration > 0:
            window_size = min(5.0, duration / 10.0)  # Окно 5 сек или 1/10 длительности
            num_windows = int(duration / window_size) + 1
            densities = []
            for i in range(num_windows):
                window_start = i * window_size
                window_end = min((i + 1) * window_size, duration)
                window_onsets = onset_times[
                    (onset_times >= window_start) & (onset_times < window_end)
                ]
                window_density = len(window_onsets) / (window_end - window_start + 1e-9)
                densities.append(window_density)
            if len(densities) > 1:
                metrics["onset_density_variance"] = float(np.var(densities))
            else:
                metrics["onset_density_variance"] = 0.0
        else:
            metrics["onset_density_variance"] = 0.0

        return metrics

    def _validate_with_tempo(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Опциональная валидация/улучшение результатов с использованием tempo_extractor.

        Args:
            features: Словарь с выходными данными

        Returns:
            Обновленный словарь с дополнительными метриками валидации
        """
        if self.tempo_payload is None:
            return features

        # Добавляем метрики сравнения с tempo
        if "onset_tempo_estimate" in features and "tempo_bpm" in self.tempo_payload:
            tempo_bpm = float(self.tempo_payload.get("tempo_bpm", 0.0))
            onset_tempo = float(features.get("onset_tempo_estimate", 0.0))
            if tempo_bpm > 0 and onset_tempo > 0:
                tempo_diff = abs(onset_tempo - tempo_bpm) / (tempo_bpm + 1e-9)
                features["onset_tempo_consistency"] = float(1.0 / (1.0 + tempo_diff))
            else:
                features["onset_tempo_consistency"] = 0.0

        return features

    def _save_time_series_artifacts(
        self, features: Dict[str, Any], input_uri: str, tmp_path: str
    ) -> Dict[str, Any]:
        """
        Сохранение больших временных серий в .npy файлы (per-run storage).

        Args:
            features: Словарь с выходными данными
            input_uri: URI входного файла
            tmp_path: Временный путь для сохранения

        Returns:
            Обновленный словарь с путями к .npy файлам (если сохранены)
        """
        if self.artifacts_dir is None:
            return features

        # Сохраняем onset_times если размер превышает threshold
        if "onset_times" in features and self.enable_time_series:
            onset_times = features.get("onset_times")
            if onset_times is not None and len(onset_times) > ONSET_TIMES_SAVE_THRESHOLD:
                artifacts_path = Path(self.artifacts_dir)
                artifacts_path.mkdir(parents=True, exist_ok=True)

                npy_path = artifacts_path / "onset_times.npy"
                np.save(str(npy_path), onset_times.astype(np.float32))

                # Заменяем большой массив на путь (relpath внутри _artifacts/)
                features["onset_times_npy"] = "_artifacts/onset_times.npy"
                features["onset_times_shape"] = onset_times.shape
                features["onset_times_elements"] = int(onset_times.size)

        return features

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение онсетов на полном аудио.

        Progress reporting: обновление прогресса для каждого этапа.
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"onset | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("onset", 0, 5, "Loading audio")
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)

            # Опциональная нормализация аудио
            if self.enable_audio_normalization:
                y = self.audio_utils.normalize_audio(y_t)
                y = self.audio_utils.to_numpy(y)

            # Выбираем канал с максимальной RMS энергией для многоканального аудио
            if y.ndim == 2:
                rms = np.mean(y**2, axis=1)
                y = y[np.argmax(rms)]

            # Извлекаем онсеты
            if self.progress_callback:
                self.progress_callback("onset", 1, 5, "Extracting onsets")
            onset_times = self._extract_onsets(y, sr)

            # Вычисляем базовые метрики
            if self.progress_callback:
                self.progress_callback("onset", 2, 5, "Computing metrics")
            duration = float(y.shape[-1] / sr)
            features: Dict[str, Any] = {}

            # Basic features
            if self.enable_basic_features:
                features["onset_times"] = onset_times.astype(np.float32)
                features["onset_count"] = int(onset_times.size)
                features["onset_density_per_sec"] = float(
                    onset_times.size / (duration + 1e-9)
                )
                features["insufficient_onsets"] = onset_times.size <= 1

            # Interval stats
            if self.enable_interval_stats:
                interval_stats = self._compute_interval_stats(onset_times)
                features.update(interval_stats)
                if onset_times.size > 1:
                    intervals = np.diff(onset_times)
                    features["avg_interval_sec"] = float(np.mean(intervals))
                else:
                    features["avg_interval_sec"] = None

            # Rhythmic metrics
            if self.enable_rhythmic_metrics:
                rhythmic_metrics = self._compute_rhythmic_metrics(onset_times, duration)
                features.update(rhythmic_metrics)

            # Time series
            if self.enable_time_series:
                features["onset_times"] = onset_times.astype(np.float32)

            # Метаданные
            features["sample_rate"] = int(sr)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = duration
            features["device_used"] = self.device
            features["backend"] = self.backend

            # Сохраняем большие массивы в .npy
            if self.progress_callback:
                self.progress_callback("onset", 3, 5, "Saving artifacts")
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Валидация выходных данных
            if self.progress_callback:
                self.progress_callback("onset", 4, 5, "Validating output")
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"onset | {error_msg} (error_code={error_code})")

            # Опциональная валидация с tempo_extractor
            features = self._validate_with_tempo(features)

            # Добавляем contract version
            features["onset_contract_version"] = ONSET_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_interval_stats:
                enabled_features.append("interval_stats")
            if self.enable_rhythmic_metrics:
                enabled_features.append("rhythmic_metrics")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_audio_ms": 0.0,  # Audio loading is part of extraction
                "extract_onsets_ms": float(processing_time * 1000.0),
                "compute_metrics_ms": 0.0,  # Metrics computation is part of extraction
                "save_artifacts_ms": 0.0,  # Artifact saving is part of extraction
                "validate_output_ms": 0.0,  # Validation is part of extraction
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            if self.progress_callback:
                self.progress_callback("onset", 5, 5, "Completed")
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"onset | Ошибка извлечения онсетов (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven onset extraction: compute onsets on provided windows (families.onset).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"onset | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("onset | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            onset_times_all: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback(
                            "onset",
                            seg_idx,
                            total_segments,
                            f"Processing segment {seg_idx+1}/{total_segments}",
                        )
                        last_reported_pct = pct

                # Load segment
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))

                wav_t, _sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    target_sr=self.sample_rate,
                )
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)

                # Опциональная нормализация аудио
                if self.enable_audio_normalization:
                    wav_t_normalized = self.audio_utils.normalize_audio(wav_t)
                    wav = self.audio_utils.to_numpy(wav_t_normalized)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)

                # Extract onsets for segment
                seg_onset_times = self._extract_onsets(wav, self.sample_rate)

                # Смещаем времена онсетов относительно начала сегмента
                segment_start_sec = float(start_sample / self.sample_rate)
                seg_onset_times_global = seg_onset_times + segment_start_sec

                # Собираем онсеты
                onset_times_all.extend(seg_onset_times_global.tolist())
                segment_centers.append(center_sec)
                segment_durations.append(float((end_sample - start_sample) / self.sample_rate))

            # Final progress report
            if self.progress_callback:
                self.progress_callback("onset", total_segments, total_segments, "Aggregating results")

            # Aggregate results
            onset_times_all = np.array(onset_times_all, dtype=np.float32)
            onset_times_all = np.unique(onset_times_all)  # Удаляем дубликаты на границах сегментов
            onset_times_all = np.sort(onset_times_all)  # Сортируем по времени

            segment_centers = np.array(segment_centers, dtype=np.float32)
            segment_durations = np.array(segment_durations, dtype=np.float32)

            # Вычисляем метрики на агрегированных данных
            total_duration = float(np.sum(segment_durations))
            features: Dict[str, Any] = {}

            # Basic features
            if self.enable_basic_features:
                features["onset_times"] = onset_times_all
                features["onset_count"] = int(onset_times_all.size)
                features["onset_density_per_sec"] = float(
                    onset_times_all.size / (total_duration + 1e-9)
                )
                features["insufficient_onsets"] = onset_times_all.size <= 1

            # Interval stats
            if self.enable_interval_stats:
                interval_stats = self._compute_interval_stats(onset_times_all)
                features.update(interval_stats)
                if onset_times_all.size > 1:
                    intervals = np.diff(onset_times_all)
                    features["avg_interval_sec"] = float(np.mean(intervals))
                else:
                    features["avg_interval_sec"] = None

            # Rhythmic metrics
            if self.enable_rhythmic_metrics:
                rhythmic_metrics = self._compute_rhythmic_metrics(onset_times_all, total_duration)
                features.update(rhythmic_metrics)

            # Time series
            if self.enable_time_series:
                features["onset_times"] = onset_times_all

            # Per-segment data
            features["segment_centers_sec"] = segment_centers
            features["segment_durations_sec"] = segment_durations
            features["segments_count"] = int(total_segments)

            # Метаданные
            features["sample_rate"] = int(self.sample_rate)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = total_duration
            features["device_used"] = self.device
            features["backend"] = self.backend

            # Сохраняем большие массивы в .npy
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"onset | {error_msg} (error_code={error_code})")

            # Опциональная валидация с tempo_extractor
            features = self._validate_with_tempo(features)

            # Добавляем contract version
            features["onset_contract_version"] = ONSET_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_interval_stats:
                enabled_features.append("interval_stats")
            if self.enable_rhythmic_metrics:
                enabled_features.append("rhythmic_metrics")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_segments_ms": 0.0,  # Segment loading is part of extraction
                "extract_onsets_ms": float(processing_time * 1000.0),
                "aggregate_results_ms": 0.0,  # Aggregation is part of extraction
                "compute_metrics_ms": 0.0,  # Metrics computation is part of extraction
                "save_artifacts_ms": 0.0,  # Artifact saving is part of extraction
                "validate_output_ms": 0.0,  # Validation is part of extraction
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"onset | Ошибка извлечения онсетов на сегментах (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        onset_extractor поддерживает batch processing через extract_batch_segments()
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
        from concurrent.futures import ThreadPoolExecutor
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
                logger.error(f"onset | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"onset | Missing segments for file_id={file_id}")
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
                logger.error(f"onset | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
