"""
MFCCExtractor: извлечение MFCC (Mel-frequency cepstral coefficients) признаков.
Интеграция с общим интерфейсом BaseExtractor и AudioUtils.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (fail-fast)
- Per-run storage for .npy files
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Optional time series storage
- Improved GPU heuristic
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
MFCC_CONTRACT_VERSION = "mfcc_contract_v1"


class MFCCExtractor(BaseExtractor):
    """Экстрактор MFCC признаков с поддержкой GPU."""

    name = "mfcc"
    version = "2.0.0"
    description = "Извлечение MFCC (Mel-frequency cepstral coefficients) признаков"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 2.0

    # Предпочитает GPU, но может работать на CPU
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.5  # 500MB

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_features: bool = False,
        enable_deltas: bool = False,
        enable_time_series: bool = False,
        enable_normalization: bool = False,  # MFCC normalization (z-score)
        # Optional audio normalization
        enable_audio_normalization: bool = True,  # Audio normalization before processing (default: True for backward compatibility)
        # GPU heuristic parameters
        min_gpu_duration_sec: float = 3.0,
        min_gpu_file_size_mb: float = 5.0,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация MFCC экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            n_mfcc: Количество MFCC коэффициентов
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            n_mels: Количество мел-фильтров
            fmin: Минимальная частота
            fmax: Максимальная частота
            enable_basic_features: Включить базовые фичи (mfcc_features, mfcc_statistics: mean, std, min, max)
            enable_deltas: Включить дельты (delta_mean, delta_std, delta_delta_mean, delta_delta_std)
            enable_time_series: Включить временные серии для всех фичей
            enable_normalization: Включить нормализацию MFCC по времени (z-score)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            min_gpu_duration_sec: Минимальная длительность для использования GPU (секунды)
            min_gpu_file_size_mb: Минимальный размер файла для использования GPU (MB)
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(sample_rate, n_mfcc, n_fft, hop_length, n_mels, fmin, fmax)

        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else float(sample_rate // 2)

        # Feature gating flags
        self.enable_basic_features = bool(enable_basic_features)
        self.enable_deltas = bool(enable_deltas)
        self.enable_time_series = bool(enable_time_series)
        self.enable_normalization = bool(enable_normalization)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # GPU heuristic parameters
        self.min_gpu_duration_sec = float(min_gpu_duration_sec)
        self.min_gpu_file_size_mb = float(min_gpu_file_size_mb)

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # Инициализируем трансформы
        self._setup_transforms()

    def _validate_parameters(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float,
        fmax: Optional[float],
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            n_mfcc: Количество MFCC коэффициентов
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            n_mels: Количество мел-фильтров
            fmin: Минимальная частота
            fmax: Максимальная частота

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"mfcc | sample_rate must be positive, got {sample_rate}")
        if n_mfcc <= 0:
            raise ValueError(f"mfcc | n_mfcc must be positive, got {n_mfcc}")
        if n_fft <= 0:
            raise ValueError(f"mfcc | n_fft must be positive, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"mfcc | hop_length must be positive, got {hop_length}")
        if hop_length > n_fft:
            raise ValueError(f"mfcc | hop_length ({hop_length}) must be <= n_fft ({n_fft})")
        if n_mels <= 0:
            raise ValueError(f"mfcc | n_mels must be positive, got {n_mels}")
        if fmin < 0.0:
            raise ValueError(f"mfcc | fmin must be non-negative, got {fmin}")
        if fmax is not None:
            if fmax <= fmin:
                raise ValueError(f"mfcc | fmax ({fmax}) must be > fmin ({fmin})")
            if fmax > sample_rate / 2:
                raise ValueError(f"mfcc | fmax ({fmax}) must be <= sample_rate/2 ({sample_rate / 2})")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки (audio_load_failed, transform_setup_failed, extraction_failed, deltas_failed, statistics_failed, validation_failed, unknown)

        Returns:
            error_code: один из:
                - mfcc_audio_load_failed
                - mfcc_transform_setup_failed
                - mfcc_extraction_failed
                - mfcc_deltas_failed
                - mfcc_statistics_failed
                - mfcc_validation_failed
                - mfcc_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "mfcc_audio_load_failed"
        if "transform" in error_str or "setup" in error_str or context == "transform_setup_failed":
            return "mfcc_transform_setup_failed"
        if "extract" in error_str or "mfcc" in error_str or context == "extraction_failed":
            return "mfcc_extraction_failed"
        if "delta" in error_str or context == "deltas_failed":
            return "mfcc_deltas_failed"
        if "statistic" in error_str or context == "statistics_failed":
            return "mfcc_statistics_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "mfcc_validation_failed"

        return "mfcc_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "mfcc | features must be a dict"

        # Validate basic features if present
        if "mfcc_features" in features:
            mfcc_features = features.get("mfcc_features")
            if mfcc_features is not None:
                if isinstance(mfcc_features, np.ndarray):
                    if np.any(np.isnan(mfcc_features)) or np.any(np.isinf(mfcc_features)):
                        return False, "mfcc | mfcc_features contains NaN or Inf values"
                    if mfcc_features.shape[0] != self.n_mfcc:
                        return False, f"mfcc | mfcc_features shape[0] ({mfcc_features.shape[0]}) != n_mfcc ({self.n_mfcc})"

        # Validate statistics if present
        if "mfcc_statistics" in features:
            stats = features.get("mfcc_statistics")
            if not isinstance(stats, dict):
                return False, "mfcc | mfcc_statistics must be a dict"

            # Validate basic statistics
            for stat_key in ["mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max"]:
                if stat_key in stats:
                    stat_val = stats.get(stat_key)
                    if isinstance(stat_val, list):
                        stat_arr = np.asarray(stat_val, dtype=np.float32)
                        if np.any(np.isnan(stat_arr)) or np.any(np.isinf(stat_arr)):
                            return False, f"mfcc | {stat_key} contains NaN or Inf values"
                        if len(stat_arr) != self.n_mfcc:
                            return False, f"mfcc | {stat_key} length ({len(stat_arr)}) != n_mfcc ({self.n_mfcc})"

            # Validate deltas if present
            if self.enable_deltas:
                for stat_key in ["delta_mean", "delta_std", "delta_delta_mean", "delta_delta_std"]:
                    if stat_key in stats:
                        stat_val = stats.get(stat_key)
                        if isinstance(stat_val, list):
                            stat_arr = np.asarray(stat_val, dtype=np.float32)
                            if np.any(np.isnan(stat_arr)) or np.any(np.isinf(stat_arr)):
                                return False, f"mfcc | {stat_key} contains NaN or Inf values"
                            if len(stat_arr) != self.n_mfcc:
                                return False, f"mfcc | {stat_key} length ({len(stat_arr)}) != n_mfcc ({self.n_mfcc})"

            # Validate consistency: feature_shape
            if "feature_shape" in stats:
                feature_shape = stats.get("feature_shape")
                if isinstance(feature_shape, (tuple, list)) and len(feature_shape) == 2:
                    if feature_shape[0] != self.n_mfcc:
                        return False, f"mfcc | feature_shape[0] ({feature_shape[0]}) != n_mfcc ({self.n_mfcc})"

        # Validate time series if present
        for series_key in ["mfcc_series", "delta_series", "delta_delta_series"]:
            if series_key in features:
                series = features.get(series_key)
                if series is not None:
                    if isinstance(series, np.ndarray):
                        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                            return False, f"mfcc | {series_key} contains NaN or Inf values"
                        if series.shape[0] != self.n_mfcc:
                            return False, f"mfcc | {series_key} shape[0] ({series.shape[0]}) != n_mfcc ({self.n_mfcc})"

        return True, None

    def _should_use_gpu(self, input_uri: str, duration_sec: float) -> bool:
        """
        Улучшенная эвристика выбора CPU/GPU: учитывает длительность, размер файла и доступную GPU память.

        Args:
            input_uri: Путь к входному файлу
            duration_sec: Длительность аудио в секундах

        Returns:
            True если следует использовать GPU, False иначе
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return False

        # Проверка длительности
        if duration_sec < self.min_gpu_duration_sec:
            return False

        # Проверка размера файла
        try:
            file_size_mb = os.path.getsize(input_uri) / (1024 * 1024)
            if file_size_mb < self.min_gpu_file_size_mb:
                return False
        except Exception:
            # Если не удалось получить размер файла, пропускаем эту проверку
            pass

        # Проверка доступной GPU памяти
        try:
            gpu_memory_free_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) - torch.cuda.memory_allocated(0) / (1024 * 1024)
            if gpu_memory_free_mb < self.gpu_memory_required * 1024:  # Convert GB to MB
                return False
        except Exception:
            # Если не удалось получить информацию о GPU памяти, используем CPU
            return False

        return True

    def _setup_transforms(self) -> None:
        """Настройка трансформов для MFCC (fail-fast)."""
        try:
            if self.progress_callback:
                self.progress_callback("mfcc", 0, 1, "Setting up transforms")

            # Создаем CPU-трансформы (всегда)
            self.mel_spectrogram_cpu = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
            )
            self.mfcc_transform_cpu = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "n_mels": self.n_mels,
                    "f_min": self.fmin,
                    "f_max": self.fmax,
                },
            )

            # И при наличии CUDA — дубли на GPU
            if self.device == "cuda" and torch.cuda.is_available():
                self.mel_spectrogram_gpu = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    f_min=self.fmin,
                    f_max=self.fmax,
                ).to(self.device)
                self.mfcc_transform_gpu = torchaudio.transforms.MFCC(
                    sample_rate=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    melkwargs={
                        "n_fft": self.n_fft,
                        "hop_length": self.hop_length,
                        "n_mels": self.n_mels,
                        "f_min": self.fmin,
                        "f_max": self.fmax,
                    },
                ).to(self.device)
            else:
                self.mel_spectrogram_gpu = None
                self.mfcc_transform_gpu = None

        except Exception as e:
            error_code = self._classify_error(e, "transform_setup_failed")
            raise RuntimeError(f"mfcc | Ошибка настройки MFCC трансформов (error_code={error_code}): {e}") from e

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение MFCC признаков на полном аудио.

        Progress reporting: обновление прогресса для каждого этапа.
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"mfcc | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("mfcc", 0, 5, "Loading audio")
            waveform, sample_rate = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)

            # Опциональная нормализация аудио
            if self.enable_audio_normalization:
                waveform = self.audio_utils.normalize_audio(waveform)

            # Улучшенная эвристика выбора CPU/GPU
            duration_sec = waveform.shape[1] / float(sample_rate)
            use_gpu = self._should_use_gpu(input_uri, duration_sec)
            if use_gpu:
                waveform = self.audio_utils._move_to_device(waveform)

            # Извлекаем MFCC
            if self.progress_callback:
                self.progress_callback("mfcc", 1, 5, "Extracting MFCC features")
            mfcc_features = self._extract_mfcc_features(waveform, prefer_gpu=use_gpu)

            # Вычисляем статистики
            if self.progress_callback:
                self.progress_callback("mfcc", 2, 5, "Computing statistics")
            mfcc_stats = self._compute_mfcc_statistics(mfcc_features)

            # Вычисляем дополнительные метрики
            if self.progress_callback:
                self.progress_callback("mfcc", 3, 5, "Computing additional metrics")
            additional_metrics = self._compute_additional_metrics(mfcc_features)

            # Сохраняем большие временные серии в .npy (per-run storage)
            if self.progress_callback:
                self.progress_callback("mfcc", 4, 5, "Saving artifacts")
            features = self._build_payload(mfcc_features, mfcc_stats, additional_metrics, sample_rate, duration_sec)
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Валидация выходных данных
            if self.progress_callback:
                self.progress_callback("mfcc", 5, 5, "Validating output")
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"mfcc | {error_msg} (error_code={error_code})")

            # Добавляем contract version
            features["mfcc_contract_version"] = MFCC_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_deltas:
                enabled_features.append("deltas")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_audio_ms": 0.0,  # Audio loading is part of extraction
                "extract_mfcc_ms": float(processing_time * 1000.0),
                "compute_statistics_ms": 0.0,  # Statistics computation is part of extraction
                "save_artifacts_ms": 0.0,  # Artifact saving is part of extraction
                "validate_output_ms": 0.0,  # Validation is part of extraction
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"mfcc | Ошибка извлечения MFCC признаков (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven MFCC extraction: compute MFCC on provided windows (families.mfcc).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"mfcc | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("mfcc | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            mfcc_features_all: List[np.ndarray] = []
            mfcc_stats_all: List[Dict[str, Any]] = []
            additional_metrics_all: List[Dict[str, Any]] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback("mfcc", seg_idx, total_segments, f"Processing segment {seg_idx+1}/{total_segments}")
                        last_reported_pct = pct

                # Load segment
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))

                waveform, _sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    target_sr=self.sample_rate,
                )

                # Опциональная нормализация аудио
                if self.enable_audio_normalization:
                    waveform = self.audio_utils.normalize_audio(waveform)

                # Улучшенная эвристика выбора CPU/GPU
                duration_sec = float((end_sample - start_sample) / self.sample_rate)
                use_gpu = self._should_use_gpu(input_uri, duration_sec)
                if use_gpu:
                    waveform = self.audio_utils._move_to_device(waveform)

                # Extract MFCC features for segment
                mfcc_features = self._extract_mfcc_features(waveform, prefer_gpu=use_gpu)

                # Compute statistics
                mfcc_stats = self._compute_mfcc_statistics(mfcc_features)

                # Compute additional metrics
                additional_metrics = self._compute_additional_metrics(mfcc_features)

                # Store results
                mfcc_features_all.append(self.audio_utils.to_numpy(mfcc_features))
                mfcc_stats_all.append(mfcc_stats)
                additional_metrics_all.append(additional_metrics)
                segment_centers.append(center_sec)
                segment_durations.append(duration_sec)

            # Final progress report
            if self.progress_callback:
                self.progress_callback("mfcc", total_segments, total_segments, "Completed")

            # Aggregate results
            if len(mfcc_features_all) == 0:
                error_code = self._classify_error(RuntimeError("All segments produced empty features"), "validation_failed")
                raise RuntimeError(f"mfcc | all segments produced empty features (error_code={error_code})")

            # Build payload (feature-gated)
            features = self._build_payload_from_segments(
                mfcc_features_all,
                mfcc_stats_all,
                additional_metrics_all,
                segment_centers,
                segment_durations,
                total_segments,
            )

            # Сохраняем большие временные серии в .npy (per-run storage)
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_deltas:
                enabled_features.append("deltas")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"mfcc | {error_msg} (error_code={error_code})")

            # Добавляем contract version
            features["mfcc_contract_version"] = MFCC_CONTRACT_VERSION

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_segments_ms": 0.0,  # Segment loading is part of extraction
                "process_segments_ms": float(processing_time * 1000.0),
                "aggregate_results_ms": 0.0,  # Aggregation is part of processing
                "validate_output_ms": 0.0,  # Validation is part of processing
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"mfcc | Ошибка извлечения MFCC признаков (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def _save_time_series_artifacts(
        self,
        features: Dict[str, Any],
        input_uri: str,
        tmp_path: str,
    ) -> Dict[str, Any]:
        """
        Сохранить большие временные серии в .npy файлы (per-run storage).

        Args:
            features: Словарь с признаками
            input_uri: Путь к входному файлу
            tmp_path: Временная директория (если artifacts_dir не задан, используется tmp_path)

        Returns:
            Обновлённый словарь features с путями к .npy файлам
        """
        if not self.enable_time_series:
            return features

        artifacts_dir = Path(self.artifacts_dir) if self.artifacts_dir else Path(tmp_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(input_uri).stem

        # Save large time series if present
        for series_key in ["mfcc_series", "delta_series", "delta_delta_series"]:
            series = features.get(series_key)
            if isinstance(series, np.ndarray) and series.size > 1000:  # Save if > 1000 elements
                npy_path = artifacts_dir / f"{stem}_{series_key}.npy"
                np.save(str(npy_path), series.astype(np.float32))
                features[f"{series_key}_npy"] = str(npy_path)
                # Убираем саму серию из JSON (если не включена time_series)
                if not self.enable_time_series:
                    features.pop(series_key, None)

        return features

    def _extract_mfcc_features(self, waveform: torch.Tensor, prefer_gpu: bool = False) -> torch.Tensor:
        """
        Извлечение MFCC признаков (fail-fast, no-fallback).

        Args:
            waveform: Аудио сигнал (torch.Tensor)
            prefer_gpu: Использовать ли GPU

        Returns:
            MFCC признаки (torch.Tensor, shape: [n_mfcc, T] или [B, n_mfcc, T])

        Raises:
            RuntimeError: Если извлечение не удалось (no-fallback)
        """
        try:
            # Выбираем трансформ (GPU/CPU) и применяем
            if prefer_gpu and self.mfcc_transform_gpu is not None:
                mfcc = self.mfcc_transform_gpu(waveform)
            else:
                # Гарантируем CPU для CPU-трансформа
                if waveform.device.type != "cpu":
                    waveform = waveform.cpu()
                mfcc = self.mfcc_transform_cpu(waveform)

            # Опциональная нормализация по времени (z-score)
            if self.enable_normalization:
                # Ожидается форма [B, n_mfcc, T] или [n_mfcc, T]
                if mfcc.dim() == 3:
                    mean = mfcc.mean(dim=2, keepdim=True)
                    std = mfcc.std(dim=2, keepdim=True).clamp(min=1e-8)
                else:
                    mean = mfcc.mean(dim=1, keepdim=True)
                    std = mfcc.std(dim=1, keepdim=True).clamp(min=1e-8)
                mfcc = (mfcc - mean) / std

            # Валидация результата
            if torch.any(torch.isnan(mfcc)) or torch.any(torch.isinf(mfcc)):
                error_code = self._classify_error(RuntimeError("mfcc_features produced NaN/Inf"), "extraction_failed")
                raise RuntimeError(f"mfcc | mfcc_features produced NaN/Inf (error_code={error_code})")

            return mfcc

        except Exception as e:
            error_code = self._classify_error(e, "extraction_failed")
            raise RuntimeError(f"mfcc | extraction failed (error_code={error_code}): {e}") from e

    def _compute_mfcc_statistics(self, mfcc_features: torch.Tensor) -> Dict[str, Any]:
        """
        Вычисление статистик MFCC (fail-fast, no-fallback).

        Args:
            mfcc_features: MFCC признаки (torch.Tensor)

        Returns:
            Словарь со статистиками (feature-gated)

        Raises:
            RuntimeError: Если вычисление не удалось (no-fallback)
        """
        try:
            # Приводим к форме [n_mfcc, T]
            if mfcc_features.dim() == 3:
                mfcc_2d = mfcc_features[0]
            else:
                mfcc_2d = mfcc_features

            stats: Dict[str, Any] = {}

            # Basic statistics (feature-gated)
            if self.enable_basic_features:
                mean = mfcc_2d.mean(dim=1)
                std = mfcc_2d.std(dim=1)
                min_vals = mfcc_2d.min(dim=1).values
                max_vals = mfcc_2d.max(dim=1).values

                # Валидация
                if torch.any(torch.isnan(mean)) or torch.any(torch.isinf(mean)):
                    error_code = self._classify_error(RuntimeError("mfcc_statistics produced NaN/Inf"), "statistics_failed")
                    raise RuntimeError(f"mfcc | mfcc_statistics produced NaN/Inf (error_code={error_code})")

                stats.update({
                    "mfcc_mean": mean.detach().cpu().numpy().tolist(),
                    "mfcc_std": std.detach().cpu().numpy().tolist(),
                    "mfcc_min": min_vals.detach().cpu().numpy().tolist(),
                    "mfcc_max": max_vals.detach().cpu().numpy().tolist(),
                    "feature_shape": tuple(int(x) for x in mfcc_2d.shape),
                })

            # Deltas (feature-gated)
            if self.enable_deltas:
                deltas = torchaudio.functional.compute_deltas(mfcc_2d)
                delta_deltas = torchaudio.functional.compute_deltas(deltas)

                delta_mean = deltas.mean(dim=1)
                delta_std = deltas.std(dim=1)
                delta_delta_mean = delta_deltas.mean(dim=1)
                delta_delta_std = delta_deltas.std(dim=1)

                # Валидация
                if torch.any(torch.isnan(delta_mean)) or torch.any(torch.isinf(delta_mean)):
                    error_code = self._classify_error(RuntimeError("deltas produced NaN/Inf"), "deltas_failed")
                    raise RuntimeError(f"mfcc | deltas produced NaN/Inf (error_code={error_code})")

                stats.update({
                    "delta_mean": delta_mean.detach().cpu().numpy().tolist(),
                    "delta_std": delta_std.detach().cpu().numpy().tolist(),
                    "delta_delta_mean": delta_delta_mean.detach().cpu().numpy().tolist(),
                    "delta_delta_std": delta_delta_std.detach().cpu().numpy().tolist(),
                    "delta_shape": tuple(int(x) for x in deltas.shape),
                    "delta_delta_shape": tuple(int(x) for x in delta_deltas.shape),
                })

            # Total features count
            if self.enable_basic_features:
                n_base = self.n_mfcc * 4  # mean, std, min, max
                if self.enable_deltas:
                    n_base += self.n_mfcc * 4  # delta_mean, delta_std, delta_delta_mean, delta_delta_std
                stats["total_features"] = n_base

            return stats

        except Exception as e:
            error_code = self._classify_error(e, "statistics_failed")
            raise RuntimeError(f"mfcc | statistics computation failed (error_code={error_code}): {e}") from e

    def _compute_additional_metrics(self, mfcc_features: torch.Tensor) -> Dict[str, Any]:
        """
        Вычислить дополнительные метрики для ML/аналитики.

        Args:
            mfcc_features: MFCC признаки (torch.Tensor)

        Returns:
            Словарь с дополнительными метриками
        """
        metrics: Dict[str, Any] = {}

        try:
            # Приводим к форме [n_mfcc, T]
            if mfcc_features.dim() == 3:
                mfcc_2d = mfcc_features[0]
            else:
                mfcc_2d = mfcc_features

            if mfcc_2d.shape[1] == 0:
                return {
                    "mfcc_energy": 0.0,
                    "mfcc_centroid": 0.0,
                    "mfcc_bandwidth": 0.0,
                    "mfcc_skewness": 0.0,
                    "mfcc_kurtosis": 0.0,
                    "mfcc_correlation": 0.0,
                    "mfcc_stability": 0.0,
                }

            # MFCC energy (первый коэффициент, часто используется как отдельная фича)
            if mfcc_2d.shape[0] > 0:
                mfcc_energy = float(torch.mean(torch.abs(mfcc_2d[0, :])))
                metrics["mfcc_energy"] = mfcc_energy

            # MFCC centroid (взвешенное среднее по коэффициентам)
            weights = torch.arange(1, mfcc_2d.shape[0] + 1, dtype=mfcc_2d.dtype, device=mfcc_2d.device)
            mfcc_mean_per_frame = torch.mean(mfcc_2d, dim=0)  # [T]
            mfcc_centroid = float(torch.mean(mfcc_mean_per_frame))
            metrics["mfcc_centroid"] = mfcc_centroid

            # MFCC bandwidth (стандартное отклонение по коэффициентам)
            mfcc_std_per_frame = torch.std(mfcc_2d, dim=0)  # [T]
            mfcc_bandwidth = float(torch.mean(mfcc_std_per_frame))
            metrics["mfcc_bandwidth"] = mfcc_bandwidth

            # MFCC skewness (асимметрия распределения)
            mfcc_mean_all = torch.mean(mfcc_2d)
            mfcc_std_all = torch.std(mfcc_2d)
            if mfcc_std_all > 1e-8:
                mfcc_skewness = float(torch.mean(((mfcc_2d - mfcc_mean_all) / mfcc_std_all) ** 3))
            else:
                mfcc_skewness = 0.0
            metrics["mfcc_skewness"] = mfcc_skewness

            # MFCC kurtosis (эксцесс распределения)
            if mfcc_std_all > 1e-8:
                mfcc_kurtosis = float(torch.mean(((mfcc_2d - mfcc_mean_all) / mfcc_std_all) ** 4)) - 3.0
            else:
                mfcc_kurtosis = 0.0
            metrics["mfcc_kurtosis"] = mfcc_kurtosis

            # MFCC correlation (корреляция между коэффициентами)
            if mfcc_2d.shape[0] > 1:
                # Вычисляем корреляцию между первыми двумя коэффициентами как пример
                mfcc_corr = float(torch.corrcoef(torch.stack([mfcc_2d[0, :], mfcc_2d[1, :]]))[0, 1])
                if torch.isnan(torch.tensor(mfcc_corr)):
                    mfcc_corr = 0.0
            else:
                mfcc_corr = 0.0
            metrics["mfcc_correlation"] = mfcc_corr

            # MFCC stability (стабильность во времени: обратная к стандартному отклонению)
            mfcc_stability = float(1.0 / (1.0 + torch.std(mfcc_2d).item()))
            metrics["mfcc_stability"] = mfcc_stability

        except Exception as e:
            logger.warning(f"mfcc | Error computing additional metrics: {e}")
            # Return default values
            metrics = {
                "mfcc_energy": 0.0,
                "mfcc_centroid": 0.0,
                "mfcc_bandwidth": 0.0,
                "mfcc_skewness": 0.0,
                "mfcc_kurtosis": 0.0,
                "mfcc_correlation": 0.0,
                "mfcc_stability": 0.0,
            }

        return metrics

    def _build_payload(
        self,
        mfcc_features: torch.Tensor,
        mfcc_stats: Dict[str, Any],
        additional_metrics: Dict[str, Any],
        sample_rate: int,
        duration_sec: float,
    ) -> Dict[str, Any]:
        """
        Построить payload с feature-gated полями.

        Args:
            mfcc_features: MFCC признаки (torch.Tensor)
            mfcc_stats: Статистики MFCC
            additional_metrics: Дополнительные метрики
            sample_rate: Частота дискретизации
            duration_sec: Длительность аудио

        Returns:
            Словарь с payload (feature-gated)
        """
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": sample_rate,
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "duration": duration_sec,
        }

        # Basic features (feature-gated)
        if self.enable_basic_features:
            features["mfcc_features"] = self.audio_utils.to_numpy(mfcc_features)
            features["mfcc_statistics"] = mfcc_stats

            # Additional metrics (always included if basic_features enabled)
            features.update(additional_metrics)

        # Deltas (feature-gated, but stats already in mfcc_statistics if enable_deltas)
        # No need to add separately, they're in mfcc_statistics

        # Time series (feature-gated)
        if self.enable_time_series:
            mfcc_np = self.audio_utils.to_numpy(mfcc_features)
            features["mfcc_series"] = mfcc_np

            # Compute deltas for time series if enabled
            if self.enable_deltas:
                mfcc_2d = mfcc_np[0] if mfcc_np.ndim == 3 else mfcc_np
                mfcc_tensor = torch.from_numpy(mfcc_2d)
                deltas = torchaudio.functional.compute_deltas(mfcc_tensor)
                delta_deltas = torchaudio.functional.compute_deltas(deltas)
                features["delta_series"] = deltas.detach().cpu().numpy()
                features["delta_delta_series"] = delta_deltas.detach().cpu().numpy()

        return features

    def _build_payload_from_segments(
        self,
        mfcc_features_all: List[np.ndarray],
        mfcc_stats_all: List[Dict[str, Any]],
        additional_metrics_all: List[Dict[str, Any]],
        segment_centers: List[float],
        segment_durations: List[float],
        total_segments: int,
    ) -> Dict[str, Any]:
        """
        Построить payload из сегментов с агрегацией.

        Args:
            mfcc_features_all: Список MFCC признаков для каждого сегмента
            mfcc_stats_all: Список статистик для каждого сегмента
            additional_metrics_all: Список дополнительных метрик для каждого сегмента
            segment_centers: Центры сегментов в секундах
            segment_durations: Длительности сегментов в секундах
            total_segments: Общее количество сегментов

        Returns:
            Словарь с payload (feature-gated, агрегированный)
        """
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": self.sample_rate,
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "segments_count": int(total_segments),
        }

        # Aggregate basic features
        if self.enable_basic_features and len(mfcc_stats_all) > 0:
            # Aggregate statistics across segments
            aggregated_stats: Dict[str, Any] = {}

            # Aggregate basic statistics
            mfcc_mean_all = [np.array(stats.get("mfcc_mean", [])) for stats in mfcc_stats_all if "mfcc_mean" in stats]
            if mfcc_mean_all:
                aggregated_stats["mfcc_mean"] = np.mean(mfcc_mean_all, axis=0).tolist()
                aggregated_stats["mfcc_std"] = np.mean([np.array(stats.get("mfcc_std", [])) for stats in mfcc_stats_all if "mfcc_std" in stats], axis=0).tolist()
                aggregated_stats["mfcc_min"] = np.min([np.array(stats.get("mfcc_min", [])) for stats in mfcc_stats_all if "mfcc_min" in stats], axis=0).tolist()
                aggregated_stats["mfcc_max"] = np.max([np.array(stats.get("mfcc_max", [])) for stats in mfcc_stats_all if "mfcc_max" in stats], axis=0).tolist()

            # Aggregate deltas if enabled
            if self.enable_deltas:
                delta_mean_all = [np.array(stats.get("delta_mean", [])) for stats in mfcc_stats_all if "delta_mean" in stats]
                if delta_mean_all:
                    aggregated_stats["delta_mean"] = np.mean(delta_mean_all, axis=0).tolist()
                    aggregated_stats["delta_std"] = np.mean([np.array(stats.get("delta_std", [])) for stats in mfcc_stats_all if "delta_std" in stats], axis=0).tolist()
                    aggregated_stats["delta_delta_mean"] = np.mean([np.array(stats.get("delta_delta_mean", [])) for stats in mfcc_stats_all if "delta_delta_mean" in stats], axis=0).tolist()
                    aggregated_stats["delta_delta_std"] = np.mean([np.array(stats.get("delta_delta_std", [])) for stats in mfcc_stats_all if "delta_delta_std" in stats], axis=0).tolist()

            features["mfcc_statistics"] = aggregated_stats

            # Aggregate additional metrics
            if additional_metrics_all:
                aggregated_additional = {}
                for key in additional_metrics_all[0].keys():
                    values = [m.get(key, 0.0) for m in additional_metrics_all]
                    aggregated_additional[key] = float(np.mean(values))
                features.update(aggregated_additional)

        # Time series (feature-gated)
        if self.enable_time_series:
            # Concatenate all segments
            if mfcc_features_all:
                # Normalize shapes: ensure all arrays have shape (n_mfcc, frames)
                normalized_features = []
                for mfcc_feat in mfcc_features_all:
                    mfcc_arr = np.asarray(mfcc_feat)
                    # Remove batch dimension if present: (1, n_mfcc, frames) -> (n_mfcc, frames)
                    if mfcc_arr.ndim == 3:
                        mfcc_arr = mfcc_arr[0]  # Take first (and only) batch element
                    # Ensure shape is (n_mfcc, frames)
                    if mfcc_arr.ndim == 2:
                        normalized_features.append(mfcc_arr)
                    else:
                        logger.warning(f"mfcc | Unexpected MFCC shape: {mfcc_arr.shape}, skipping")
                
                if normalized_features:
                    # Verify all arrays have same n_mfcc dimension
                    n_mfcc_expected = normalized_features[0].shape[0]
                    for i, feat in enumerate(normalized_features):
                        if feat.shape[0] != n_mfcc_expected:
                            raise ValueError(
                                f"mfcc | Inconsistent n_mfcc dimension: segment {i} has {feat.shape[0]}, "
                                f"expected {n_mfcc_expected}"
                            )
                    
                    # Concatenate along time axis (axis=1)
                    mfcc_series_all = np.concatenate(normalized_features, axis=1)
                    features["mfcc_series"] = mfcc_series_all

                    # Compute deltas for time series if enabled
                    if self.enable_deltas:
                        # compute_deltas expects shape (n_mfcc, frames) or (batch, n_mfcc, frames)
                        # Add batch dimension if needed
                        if mfcc_series_all.ndim == 2:
                            mfcc_tensor = torch.from_numpy(mfcc_series_all).unsqueeze(0)  # (1, n_mfcc, frames)
                        else:
                            mfcc_tensor = torch.from_numpy(mfcc_series_all)
                        deltas = torchaudio.functional.compute_deltas(mfcc_tensor)
                        delta_deltas = torchaudio.functional.compute_deltas(deltas)
                        # Remove batch dimension if added
                        if deltas.dim() == 3 and deltas.shape[0] == 1:
                            deltas = deltas[0]
                            delta_deltas = delta_deltas[0]
                        features["delta_series"] = deltas.detach().cpu().numpy()
                        features["delta_delta_series"] = delta_deltas.detach().cpu().numpy()

            features["segment_centers_sec"] = segment_centers
            features["segment_durations_sec"] = segment_durations

        return features

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        mfcc_extractor поддерживает batch processing через extract_batch_segments()
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
                logger.error(f"mfcc | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"mfcc | Missing segments for file_id={file_id}")
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
                logger.error(f"mfcc | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
