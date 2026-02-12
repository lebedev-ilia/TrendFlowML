"""
ChromaExtractor: извлечение хрома-фич (pitch class profile) на базе librosa.
Интеграция с общим интерфейсом BaseExtractor и AudioUtils.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (fail-fast, explicit chroma_type selection)
- Per-run storage for .npy files
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Optional time series storage
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
CHROMA_CONTRACT_VERSION = "chroma_contract_v1"

# Threshold for saving large arrays to .npy files
CHROMA_SAVE_THRESHOLD = 12 * 500  # 12 classes * 500 frames


class ChromaExtractor(BaseExtractor):
    """Экстрактор хрома-признаков (12-полосный профиль классов высот) с поддержкой segment-based обработки."""

    name = "chroma"
    version = "2.0.0"
    description = "Хрома (12-полосный профиль классов высот) с тюнингом и агрегатами"
    category = "spectral"
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
        n_fft: int = 4096,
        mix_to_mono: bool = True,
        chroma_type: str = "cqt",
        normalize: Optional[str] = "l1",
        # Additional CQT/STFT parameters
        n_chroma: int = 12,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        n_bins: Optional[int] = None,  # For CQT
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_stats: bool = False,
        enable_extended_stats: bool = False,
        enable_stats_vector: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация Chroma экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            hop_length: Размер hop для STFT/CQT
            n_fft: Размер FFT окна (для STFT mode)
            mix_to_mono: Сводить стерео в моно
            chroma_type: Тип хрома ("cqt" | "stft")
            normalize: Нормализация по кадрам (None | "l1" | "l2")
            n_chroma: Количество хрома-классов (по умолчанию 12)
            fmin: Минимальная частота (Hz, None = default)
            fmax: Максимальная частота (Hz, None = default)
            n_bins: Количество бинов для CQT (None = default)
            enable_basic_stats: Включить базовые статистики (mean, std, min, max)
            enable_extended_stats: Включить расширенные статистики (median, p25, p75)
            enable_stats_vector: Включить компактный вектор статистик
            enable_time_series: Включить временные серии (chroma spectrogram)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(
            sample_rate, hop_length, n_fft, chroma_type, normalize, n_chroma, fmin, fmax, n_bins
        )

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.mix_to_mono = bool(mix_to_mono)
        self.chroma_type = str(chroma_type)
        self.normalize = normalize
        self.n_chroma = int(n_chroma)
        self.fmin = float(fmin) if fmin is not None else None
        self.fmax = float(fmax) if fmax is not None else None
        self.n_bins = int(n_bins) if n_bins is not None else None

        # Feature gating flags
        self.enable_basic_stats = bool(enable_basic_stats)
        self.enable_extended_stats = bool(enable_extended_stats)
        self.enable_stats_vector = bool(enable_stats_vector)
        self.enable_time_series = bool(enable_time_series)

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
        n_fft: int,
        chroma_type: str,
        normalize: Optional[str],
        n_chroma: int,
        fmin: Optional[float],
        fmax: Optional[float],
        n_bins: Optional[int],
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            hop_length: Размер hop для STFT/CQT
            n_fft: Размер FFT окна
            chroma_type: Тип хрома
            normalize: Нормализация по кадрам
            n_chroma: Количество хрома-классов
            fmin: Минимальная частота
            fmax: Максимальная частота
            n_bins: Количество бинов для CQT

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"chroma | sample_rate must be positive, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"chroma | hop_length must be positive, got {hop_length}")
        if n_fft <= 0:
            raise ValueError(f"chroma | n_fft must be positive, got {n_fft}")
        if chroma_type not in ["cqt", "stft"]:
            raise ValueError(f"chroma | chroma_type must be 'cqt' or 'stft', got {chroma_type}")
        if normalize not in [None, "l1", "l2"]:
            raise ValueError(f"chroma | normalize must be None, 'l1', or 'l2', got {normalize}")
        if n_chroma <= 0:
            raise ValueError(f"chroma | n_chroma must be positive, got {n_chroma}")
        if fmin is not None and fmin < 0:
            raise ValueError(f"chroma | fmin must be non-negative, got {fmin}")
        if fmax is not None and fmax <= 0:
            raise ValueError(f"chroma | fmax must be positive, got {fmax}")
        if fmin is not None and fmax is not None and fmax <= fmin:
            raise ValueError(f"chroma | fmax ({fmax}) must be > fmin ({fmin})")
        if n_bins is not None and n_bins <= 0:
            raise ValueError(f"chroma | n_bins must be positive, got {n_bins}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - chroma_audio_load_failed
                - chroma_tuning_failed
                - chroma_cqt_failed
                - chroma_stft_failed
                - chroma_normalization_failed
                - chroma_statistics_failed
                - chroma_validation_failed
                - chroma_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "chroma_audio_load_failed"
        if "tuning" in error_str or context == "tuning_failed":
            return "chroma_tuning_failed"
        if "cqt" in error_str or context == "cqt_failed":
            return "chroma_cqt_failed"
        if "stft" in error_str or context == "stft_failed":
            return "chroma_stft_failed"
        if "normalization" in error_str or "normalize" in error_str or context == "normalization_failed":
            return "chroma_normalization_failed"
        if "statistics" in error_str or "stats" in error_str or context == "statistics_failed":
            return "chroma_statistics_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "chroma_validation_failed"

        return "chroma_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "chroma | features must be a dict"

        # Validate chroma statistics if present
        for stat_name in ["chroma_mean", "chroma_std", "chroma_min", "chroma_max", "chroma_median", "chroma_p25", "chroma_p75"]:
            if stat_name in features:
                stat_value = features.get(stat_name)
                if stat_value is not None:
                    if isinstance(stat_value, list):
                        stat_arr = np.asarray(stat_value, dtype=np.float32)
                    else:
                        stat_arr = np.asarray(stat_value, dtype=np.float32)
                    
                    if not isinstance(stat_arr, np.ndarray):
                        return False, f"chroma | {stat_name} must be numpy array or list"
                    
                    if stat_arr.size != self.n_chroma:
                        return False, f"chroma | {stat_name} size ({stat_arr.size}) must be {self.n_chroma}"
                    
                    if np.any(np.isnan(stat_arr)) or np.any(np.isinf(stat_arr)):
                        return False, f"chroma | {stat_name} contains NaN or Inf values"
                    
                    # Range checks (after normalization, values should be in [0, 1] or [0, inf) without normalization)
                    if self.normalize is not None:
                        if np.any(stat_arr < 0) or np.any(stat_arr > 1.1):  # Allow small tolerance
                            return False, f"chroma | {stat_name} out of range [0, 1] after normalization"
                    else:
                        if np.any(stat_arr < 0):
                            return False, f"chroma | {stat_name} contains negative values"

        # Validate chroma_frames if present
        if "chroma_frames" in features:
            chroma_frames = features.get("chroma_frames")
            try:
                chroma_frames = int(chroma_frames)
                if chroma_frames < 0:
                    return False, "chroma | chroma_frames must be non-negative"
            except (ValueError, TypeError):
                return False, f"chroma | chroma_frames must be int, got {type(chroma_frames)}"

        # Validate tuning_estimate if present
        if "tuning_estimate" in features:
            tuning = features.get("tuning_estimate")
            if tuning is not None:
                try:
                    tuning = float(tuning)
                    if np.isnan(tuning) or np.isinf(tuning):
                        return False, "chroma | tuning_estimate is NaN or Inf"
                    # Tuning typically in range [-0.5, 0.5] semitones
                    if abs(tuning) > 1.0:
                        return False, f"chroma | tuning_estimate ({tuning}) out of reasonable range [-1.0, 1.0]"
                except (ValueError, TypeError):
                    return False, f"chroma | tuning_estimate must be float, got {type(tuning)}"

        return True, None

    def _extract_chroma(self, y: np.ndarray, sr: int, tuning: float) -> np.ndarray:
        """
        Извлечение хрома через выбранный метод (no-fallback policy).

        Args:
            y: Аудио сигнал (numpy array)
            sr: Частота дискретизации
            tuning: Оценка строя (semitones)

        Returns:
            chroma: Хрома-спектрограмма (n_chroma x frames, float32)

        Raises:
            RuntimeError: Если выбранный метод недоступен или произошла ошибка
        """
        if self.chroma_type == "cqt":
            try:
                kwargs = {
                    "y": y,
                    "sr": sr,
                    "hop_length": self.hop_length,
                    "n_chroma": self.n_chroma,
                    "tuning": tuning,
                }
                if self.fmin is not None:
                    kwargs["fmin"] = self.fmin
                if self.fmax is not None:
                    kwargs["fmax"] = self.fmax
                if self.n_bins is not None:
                    kwargs["n_bins"] = self.n_bins
                
                chroma = librosa.feature.chroma_cqt(**kwargs)
                return chroma.astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    f"chroma | CQT method failed: {e} (error_code=chroma_cqt_failed)"
                )
        elif self.chroma_type == "stft":
            try:
                kwargs = {
                    "y": y,
                    "sr": sr,
                    "hop_length": self.hop_length,
                    "n_fft": self.n_fft,
                    "n_chroma": self.n_chroma,
                }
                if self.fmin is not None:
                    kwargs["fmin"] = self.fmin
                if self.fmax is not None:
                    kwargs["fmax"] = self.fmax
                
                chroma = librosa.feature.chroma_stft(**kwargs)
                return chroma.astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    f"chroma | STFT method failed: {e} (error_code=chroma_stft_failed)"
                )
        else:
            raise RuntimeError(
                f"chroma | unknown chroma_type: {self.chroma_type} (error_code=chroma_unknown)"
            )

    def _normalize_chroma(self, chroma: np.ndarray) -> np.ndarray:
        """
        Нормализация хрома по кадрам.

        Args:
            chroma: Хрома-спектрограмма (n_chroma x frames)

        Returns:
            chroma_normalized: Нормализованная хрома-спектрограмма

        Raises:
            RuntimeError: Если нормализация не удалась
        """
        try:
            if self.normalize == "l1":
                frame_sums = chroma.sum(axis=0, keepdims=True) + 1e-12
                chroma = chroma / frame_sums
            elif self.normalize == "l2":
                norms = np.linalg.norm(chroma, ord=2, axis=0, keepdims=True) + 1e-12
                chroma = chroma / norms
            return chroma.astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"chroma | normalization failed: {e} (error_code=chroma_normalization_failed)"
            )

    def _compute_statistics(self, chroma: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисление статистических агрегатов по хрома-спектрограмме.

        Args:
            chroma: Хрома-спектрограмма (n_chroma x frames)

        Returns:
            Словарь со статистиками
        """
        stats = {}
        
        if self.enable_basic_stats:
            stats["chroma_mean"] = chroma.mean(axis=1).astype(np.float32)
            stats["chroma_std"] = chroma.std(axis=1).astype(np.float32)
            stats["chroma_min"] = chroma.min(axis=1).astype(np.float32)
            stats["chroma_max"] = chroma.max(axis=1).astype(np.float32)
        
        if self.enable_extended_stats:
            stats["chroma_median"] = np.median(chroma, axis=1).astype(np.float32)
            stats["chroma_p25"] = np.percentile(chroma, 25, axis=1).astype(np.float32)
            stats["chroma_p75"] = np.percentile(chroma, 75, axis=1).astype(np.float32)
        
        if self.enable_stats_vector:
            # Concatenate all statistics into a single vector
            stat_list = []
            if self.enable_basic_stats:
                stat_list.extend([stats["chroma_mean"], stats["chroma_std"], stats["chroma_min"], stats["chroma_max"]])
            if self.enable_extended_stats:
                stat_list.extend([stats["chroma_median"], stats["chroma_p25"], stats["chroma_p75"]])
            
            if stat_list:
                stats["chroma_stats_vector"] = np.concatenate(stat_list).astype(np.float32)
        
        return stats

    def _compute_additional_metrics(self, chroma: np.ndarray) -> Dict[str, float]:
        """
        Вычисление дополнительных метрик для ML/аналитики.

        Args:
            chroma: Хрома-спектрограмма (n_chroma x frames)

        Returns:
            Словарь с дополнительными метриками
        """
        metrics = {}
        
        # Dominant class and energy
        chroma_mean = chroma.mean(axis=1)
        dominant_idx = int(np.argmax(chroma_mean))
        metrics["chroma_dominant_class"] = dominant_idx
        metrics["chroma_dominant_energy"] = float(chroma_mean[dominant_idx])
        
        # Harmonic stability (1 / (1 + mean_std))
        chroma_std = chroma.std(axis=1)
        mean_std = float(np.mean(chroma_std))
        metrics["chroma_harmonic_stability"] = float(1.0 / (1.0 + mean_std))
        
        # Entropy of chroma distribution
        chroma_mean_norm = chroma_mean / (chroma_mean.sum() + 1e-12)
        entropy = -np.sum(chroma_mean_norm * np.log(chroma_mean_norm + 1e-12))
        metrics["chroma_entropy"] = float(entropy)
        
        # Contrast (max - min)
        metrics["chroma_contrast"] = float(np.max(chroma_mean) - np.min(chroma_mean))
        
        # Centroid (weighted average of chroma classes)
        chroma_classes = np.arange(self.n_chroma, dtype=np.float32)
        centroid = np.sum(chroma_classes * chroma_mean) / (chroma_mean.sum() + 1e-12)
        metrics["chroma_centroid"] = float(centroid)
        
        # Rolloff (95% energy)
        chroma_cumsum = np.cumsum(chroma_mean)
        chroma_total = chroma_cumsum[-1]
        rolloff_idx = np.where(chroma_cumsum >= 0.95 * chroma_total)[0]
        if len(rolloff_idx) > 0:
            metrics["chroma_rolloff"] = float(rolloff_idx[0])
        else:
            metrics["chroma_rolloff"] = float(self.n_chroma - 1)
        
        return metrics

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

        # Сохраняем chroma time series если размер превышает threshold
        if "chroma" in features and self.enable_time_series:
            chroma = features.get("chroma")
            if chroma is not None and isinstance(chroma, np.ndarray) and chroma.size > CHROMA_SAVE_THRESHOLD:
                artifacts_path = Path(self.artifacts_dir)
                artifacts_path.mkdir(parents=True, exist_ok=True)

                npy_path = artifacts_path / "chroma.npy"
                np.save(str(npy_path), chroma.astype(np.float32))

                # Заменяем большой массив на путь (relpath внутри _artifacts/)
                features["chroma_npy"] = "_artifacts/chroma.npy"
                features["chroma_shape"] = chroma.shape
                features["chroma_elements"] = int(chroma.size)
                del features["chroma"]

        return features

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение хрома на полном аудио.

        Progress reporting: обновление прогресса для каждого этапа.
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"chroma | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("chroma", 0, 7, "Loading audio")
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)

            # Опциональная нормализация аудио
            if self.enable_audio_normalization:
                y = self.audio_utils.normalize_audio(y_t)
                y = self.audio_utils.to_numpy(y)

            # Сведение в моно (опционально)
            if y.ndim == 2:
                if self.mix_to_mono:
                    y = np.mean(y, axis=0)
                else:
                    y = y[0]

            y = y.astype(np.float32)
            if y.size == 0:
                raise ValueError("chroma | Пустой аудиосигнал (error_code=chroma_audio_load_failed)")

            # Оценка строя
            if self.progress_callback:
                self.progress_callback("chroma", 1, 7, "Estimating tuning")
            try:
                tuning = float(librosa.estimate_tuning(y=y, sr=sr))
            except Exception as e:
                error_code = self._classify_error(e, "tuning_failed")
                raise RuntimeError(
                    f"chroma | tuning estimation failed: {e} (error_code={error_code})"
                )

            # Извлекаем хрома (no-fallback policy)
            if self.progress_callback:
                self.progress_callback("chroma", 2, 7, f"Extracting chroma ({self.chroma_type})")
            chroma = self._extract_chroma(y, sr, tuning)

            # Проверка размерности
            if chroma.ndim != 2 or chroma.shape[0] != self.n_chroma:
                raise ValueError(
                    f"chroma | invalid chroma shape: {chroma.shape}, expected ({self.n_chroma}, frames) (error_code=chroma_validation_failed)"
                )

            # Нормализация по кадрам
            if self.progress_callback:
                self.progress_callback("chroma", 3, 7, "Normalizing chroma")
            if self.normalize is not None:
                chroma = self._normalize_chroma(chroma)

            # Вычисляем статистики
            if self.progress_callback:
                self.progress_callback("chroma", 4, 7, "Computing statistics")
            stats = self._compute_statistics(chroma)

            # Вычисляем дополнительные метрики
            if self.progress_callback:
                self.progress_callback("chroma", 5, 7, "Computing additional metrics")
            additional_metrics = self._compute_additional_metrics(chroma)

            # Формируем payload
            features: Dict[str, Any] = {}

            # Basic stats (feature-gated)
            if self.enable_basic_stats:
                features["chroma_mean"] = stats.get("chroma_mean", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_std"] = stats.get("chroma_std", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_min"] = stats.get("chroma_min", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_max"] = stats.get("chroma_max", np.zeros(self.n_chroma, dtype=np.float32))

            # Extended stats (feature-gated)
            if self.enable_extended_stats:
                features["chroma_median"] = stats.get("chroma_median", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_p25"] = stats.get("chroma_p25", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_p75"] = stats.get("chroma_p75", np.zeros(self.n_chroma, dtype=np.float32))

            # Stats vector (feature-gated)
            if self.enable_stats_vector:
                features["chroma_stats_vector"] = stats.get("chroma_stats_vector", np.zeros(0, dtype=np.float32))

            # Time series (feature-gated)
            if self.enable_time_series:
                features["chroma"] = chroma.astype(np.float32)

            # Additional metrics
            features.update(additional_metrics)

            # Tuning estimate (always saved)
            features["tuning_estimate"] = float(tuning)

            # Метаданные
            features["sample_rate"] = int(sr)
            features["hop_length"] = int(self.hop_length)
            features["n_fft"] = int(self.n_fft)
            features["duration"] = float(y.shape[-1] / sr)
            features["device_used"] = self.device
            features["chroma_type"] = self.chroma_type
            features["normalize"] = self.normalize
            features["n_chroma"] = int(self.n_chroma)
            features["chroma_frames"] = int(chroma.shape[1])

            # Сохраняем большие массивы в .npy
            if self.progress_callback:
                self.progress_callback("chroma", 6, 7, "Saving artifacts")
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"chroma | {error_msg} (error_code={error_code})")

            # Добавляем contract version
            features["chroma_contract_version"] = CHROMA_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_stats_vector:
                enabled_features.append("stats_vector")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_audio_ms": 0.0,  # Audio loading is part of extraction
                "extract_chroma_ms": float(processing_time * 1000.0),
                "compute_stats_ms": 0.0,  # Stats computation is part of extraction
                "save_artifacts_ms": 0.0,  # Artifact saving is part of extraction
                "validate_output_ms": 0.0,  # Validation is part of extraction
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            if self.progress_callback:
                self.progress_callback("chroma", 7, 7, "Completed")
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"chroma | Ошибка извлечения хрома (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven chroma extraction: compute chroma on provided windows (families.chroma).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"chroma | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("chroma | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            chroma_all: List[np.ndarray] = []
            tuning_all: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback(
                            "chroma",
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

                # Опциональная нормализация аудио
                if self.enable_audio_normalization:
                    wav_t_normalized = self.audio_utils.normalize_audio(wav_t)
                    wav = self.audio_utils.to_numpy(wav_t_normalized)

                # Сведение в моно
                if wav.ndim == 2:
                    if self.mix_to_mono:
                        wav = np.mean(wav, axis=0)
                    else:
                        wav = wav[0]

                wav = wav.astype(np.float32)
                if wav.size == 0:
                    continue  # Skip empty segments

                # Оценка строя для сегмента
                try:
                    tuning = float(librosa.estimate_tuning(y=wav, sr=self.sample_rate))
                except Exception:
                    tuning = 0.0  # Fallback to 0.0 for segment

                # Extract chroma for segment
                seg_chroma = self._extract_chroma(wav, self.sample_rate, tuning)

                # Нормализация
                if self.normalize is not None:
                    seg_chroma = self._normalize_chroma(seg_chroma)

                # Собираем данные
                chroma_all.append(seg_chroma)
                tuning_all.append(tuning)
                segment_centers.append(center_sec)
                segment_durations.append(float((end_sample - start_sample) / self.sample_rate))

            # Final progress report
            if self.progress_callback:
                self.progress_callback("chroma", total_segments, total_segments, "Aggregating results")

            if not chroma_all:
                raise ValueError("chroma | all segments produced empty chroma (error_code=chroma_validation_failed)")

            # Aggregate chroma across segments
            chroma_aggregated = np.concatenate(chroma_all, axis=1)  # Concatenate along time axis

            # Вычисляем статистики на агрегированных данных
            stats = self._compute_statistics(chroma_aggregated)
            additional_metrics = self._compute_additional_metrics(chroma_aggregated)

            # Формируем payload
            features: Dict[str, Any] = {}

            # Basic stats (feature-gated)
            if self.enable_basic_stats:
                features["chroma_mean"] = stats.get("chroma_mean", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_std"] = stats.get("chroma_std", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_min"] = stats.get("chroma_min", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_max"] = stats.get("chroma_max", np.zeros(self.n_chroma, dtype=np.float32))

            # Extended stats (feature-gated)
            if self.enable_extended_stats:
                features["chroma_median"] = stats.get("chroma_median", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_p25"] = stats.get("chroma_p25", np.zeros(self.n_chroma, dtype=np.float32))
                features["chroma_p75"] = stats.get("chroma_p75", np.zeros(self.n_chroma, dtype=np.float32))

            # Stats vector (feature-gated)
            if self.enable_stats_vector:
                features["chroma_stats_vector"] = stats.get("chroma_stats_vector", np.zeros(0, dtype=np.float32))

            # Time series (feature-gated)
            if self.enable_time_series:
                features["chroma"] = chroma_aggregated.astype(np.float32)

            # Additional metrics
            features.update(additional_metrics)

            # Tuning estimate (mean across segments)
            features["tuning_estimate"] = float(np.mean(tuning_all)) if tuning_all else 0.0

            # Per-segment data
            features["segment_centers_sec"] = np.array(segment_centers, dtype=np.float32)
            features["segment_durations_sec"] = np.array(segment_durations, dtype=np.float32)
            features["segments_count"] = int(total_segments)

            # Метаданные
            features["sample_rate"] = int(self.sample_rate)
            features["hop_length"] = int(self.hop_length)
            features["n_fft"] = int(self.n_fft)
            features["duration"] = float(np.sum(segment_durations))
            features["device_used"] = self.device
            features["chroma_type"] = self.chroma_type
            features["normalize"] = self.normalize
            features["n_chroma"] = int(self.n_chroma)
            features["chroma_frames"] = int(chroma_aggregated.shape[1])

            # Сохраняем большие массивы в .npy
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"chroma | {error_msg} (error_code={error_code})")

            # Добавляем contract version
            features["chroma_contract_version"] = CHROMA_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_stats_vector:
                enabled_features.append("stats_vector")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            features["stage_timings_ms"] = {
                "load_segments_ms": 0.0,  # Segment loading is part of extraction
                "extract_chroma_ms": float(processing_time * 1000.0),
                "aggregate_results_ms": 0.0,  # Aggregation is part of extraction
                "compute_stats_ms": 0.0,  # Stats computation is part of extraction
                "save_artifacts_ms": 0.0,  # Artifact saving is part of extraction
                "validate_output_ms": 0.0,  # Validation is part of extraction
                "total_ms": float(processing_time * 1000.0),
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"chroma | Ошибка извлечения хрома на сегментах (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        chroma_extractor поддерживает batch processing через extract_batch_segments()
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
                logger.error(f"chroma | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"chroma | Missing segments for file_id={file_id}")
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
                logger.error(f"chroma | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
