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
from src.extractors.chroma_extractor.utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
CHROMA_CONTRACT_VERSION = "chroma_contract_v1"

# Threshold for saving large arrays to .npy files
CHROMA_SAVE_THRESHOLD = 12 * 500  # 12 classes * 500 frames


class ChromaExtractor(BaseExtractor):
    """Экстрактор хрома-признаков (12-полосный профиль классов высот) с поддержкой segment-based обработки."""

    name = "chroma"
    version = "2.1.1"
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
        # Audit v3: canonical contract fixes n_chroma=12
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

        # Audit v3: keep contract minimal and stable.
        if self.enable_basic_stats or self.enable_extended_stats or self.enable_stats_vector:
            raise RuntimeError(
                "chroma | Audit v3: basic/extended stats and stats_vector are not supported in audited contract. "
                "Disable enable_basic_stats/enable_extended_stats/enable_stats_vector."
            )
        if self.normalize != "l1":
            raise RuntimeError(f"chroma | Audit v3: normalize must be 'l1', got {self.normalize!r}")

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
        # Audit v3: fix n_chroma to 12 for stable downstream shapes
        if int(n_chroma) != 12:
            raise RuntimeError(f"chroma | Audit v3: n_chroma must be 12, got {n_chroma}")
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

        # Validate canonical outputs (Audit v3)
        chroma_mean = features.get("chroma_mean")
        if chroma_mean is None:
            return False, "chroma | chroma_mean is required"
        chroma_mean_arr = np.asarray(chroma_mean, dtype=np.float32).reshape(-1)
        if chroma_mean_arr.size != self.n_chroma:
            return False, f"chroma | chroma_mean size ({chroma_mean_arr.size}) must be {self.n_chroma}"
        if np.any(~np.isfinite(chroma_mean_arr)) or np.any(chroma_mean_arr < 0):
            return False, "chroma | chroma_mean contains invalid values"

        for k in ["chroma_entropy", "chroma_harmonic_stability", "chroma_contrast", "chroma_dominant_energy", "tuning_estimate"]:
            v = features.get(k)
            if v is None:
                return False, f"chroma | {k} is required"
            try:
                fv = float(v)
            except Exception:
                return False, f"chroma | {k} must be float"
            if not np.isfinite(fv):
                return False, f"chroma | {k} is NaN or Inf"

        # chroma_dominant_class required (int-like, 0..11)
        vdc = features.get("chroma_dominant_class")
        if vdc is None:
            return False, "chroma | chroma_dominant_class is required"
        try:
            idx = int(vdc)
        except Exception:
            return False, "chroma | chroma_dominant_class must be int"
        if idx < 0 or idx >= self.n_chroma:
            return False, f"chroma | chroma_dominant_class out of range: {idx}"

        # Legacy: validate chroma statistics if present (should not be used in audit v3).
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

    def _compute_minimal_metrics(self, chroma: np.ndarray) -> Dict[str, Any]:
        """
        Audit v3: compute minimal stable chroma signals.
        """
        chroma_mean = chroma.mean(axis=1).astype(np.float32)
        dominant_idx = int(np.argmax(chroma_mean))
        dominant_energy = float(chroma_mean[dominant_idx])

        chroma_mean_norm = chroma_mean / (float(np.sum(chroma_mean)) + 1e-12)
        entropy = float(-np.sum(chroma_mean_norm * np.log(chroma_mean_norm + 1e-12)))
        contrast = float(np.max(chroma_mean) - np.min(chroma_mean))

        chroma_std = chroma.std(axis=1).astype(np.float32)
        mean_std = float(np.mean(chroma_std))
        harmonic_stability = float(1.0 / (1.0 + mean_std))

        return {
            "chroma_mean": chroma_mean,
            "chroma_dominant_class": dominant_idx,
            "chroma_dominant_energy": dominant_energy,
            "chroma_entropy": entropy,
            "chroma_contrast": contrast,
            "chroma_harmonic_stability": harmonic_stability,
        }

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
        # Audit v3: NPZ is the only source-of-truth. Do not save external .npy artifacts.
        if "chroma" in features and self.enable_time_series:
            chroma = features.get("chroma")
            if chroma is not None and isinstance(chroma, np.ndarray) and chroma.size > CHROMA_SAVE_THRESHOLD:
                features["chroma_time_series_omitted"] = True
                del features["chroma"]

        return features

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение хрома на полном аудио.

        Progress reporting: обновление прогресса для каждого этапа.
        """
        start_time = time.time()
        t0 = time.perf_counter()
        stage_ms: Dict[str, float] = {}
        res_prof: Optional[Dict[str, Any]] = None
        if resource_profile_enabled():
            res_prof = {"at_start": snapshot_process_resources()}
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
            t_load0 = time.perf_counter()
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            stage_ms["load_audio_ms"] = float((time.perf_counter() - t_load0) * 1000.0)

            # Опциональная нормализация аудио
            if self.enable_audio_normalization:
                t_norm0 = time.perf_counter()
                y = self.audio_utils.normalize_audio(y_t)
                y = self.audio_utils.to_numpy(y)
                stage_ms["normalize_audio_ms"] = float((time.perf_counter() - t_norm0) * 1000.0)

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
            tuning_failed = False
            try:
                t_tune0 = time.perf_counter()
                tuning = float(librosa.estimate_tuning(y=y, sr=sr))
                stage_ms["tuning_ms"] = float((time.perf_counter() - t_tune0) * 1000.0)
            except Exception:
                # Audit v3: deterministic fallback to 0.0 (do not fail the whole extractor)
                tuning = 0.0
                tuning_failed = True
                stage_ms["tuning_ms"] = float((time.perf_counter() - t_tune0) * 1000.0) if "t_tune0" in locals() else 0.0

            # Извлекаем хрома (no-fallback policy)
            if self.progress_callback:
                self.progress_callback("chroma", 2, 7, f"Extracting chroma ({self.chroma_type})")
            t_chr0 = time.perf_counter()
            chroma = self._extract_chroma(y, sr, tuning)
            stage_ms["extract_chroma_ms"] = float((time.perf_counter() - t_chr0) * 1000.0)

            # Проверка размерности
            if chroma.ndim != 2 or chroma.shape[0] != self.n_chroma:
                raise ValueError(
                    f"chroma | invalid chroma shape: {chroma.shape}, expected ({self.n_chroma}, frames) (error_code=chroma_validation_failed)"
                )

            # Нормализация по кадрам
            if self.progress_callback:
                self.progress_callback("chroma", 3, 7, "Normalizing chroma")
            if self.normalize is not None:
                t_cnorm0 = time.perf_counter()
                chroma = self._normalize_chroma(chroma)
                stage_ms["normalize_chroma_ms"] = float((time.perf_counter() - t_cnorm0) * 1000.0)

            # Формируем payload
            features: Dict[str, Any] = {}
            # Minimal stable outputs (Audit v3)
            if self.progress_callback:
                self.progress_callback("chroma", 4, 7, "Computing minimal metrics")
            t_min0 = time.perf_counter()
            features.update(self._compute_minimal_metrics(chroma))
            stage_ms["compute_minimal_ms"] = float((time.perf_counter() - t_min0) * 1000.0)

            # Audit v3: expose in-memory chroma for key_extractor reuse, without persisting it.
            # extractor_runner.py prefers this field over persisted chroma.
            # Avoid extra copies: chroma is already float32 in the audited path.
            features["_shared_chroma"] = chroma

            # Optional debug time series
            if self.enable_time_series:
                # Keep as a view/reference; saver may drop this key if too large.
                features["chroma"] = chroma

            # Tuning estimate (always saved)
            features["tuning_estimate"] = float(tuning)
            features["tuning_failed"] = bool(tuning_failed)

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
            t_save0 = time.perf_counter()
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)
            stage_ms["save_artifacts_ms"] = float((time.perf_counter() - t_save0) * 1000.0)

            # Валидация выходных данных
            t_val0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"chroma | {error_msg} (error_code={error_code})")
            stage_ms["validate_output_ms"] = float((time.perf_counter() - t_val0) * 1000.0)

            # Добавляем contract version
            features["chroma_contract_version"] = CHROMA_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            features["stage_timings_ms"] = {**stage_ms, "total_ms": float((time.perf_counter() - t0) * 1000.0)}
            if res_prof is not None:
                res_prof["at_end"] = snapshot_process_resources()
                features["chroma_resource_profile"] = {
                    **prefix_snapshot("at_start", res_prof.get("at_start", {})),
                    **prefix_snapshot("at_end", res_prof.get("at_end", {})),
                }

            processing_time = time.time() - start_time
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
        t0 = time.perf_counter()
        stage_ms: Dict[str, float] = {}
        res_prof: Optional[Dict[str, Any]] = None
        if resource_profile_enabled():
            res_prof = {"at_start": snapshot_process_resources()}
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

            t_segmeta0 = time.perf_counter()
            # Audit v3: strict alignment by N segments with segment_mask + NaNs.
            segment_centers: List[float] = [float(seg.get("center_sec", 0.0)) for seg in segments]
            segment_durations: List[float] = [
                float(seg.get("end_sec", 0.0) - seg.get("start_sec", 0.0)) for seg in segments
            ]
            segment_mask = np.zeros((total_segments,), dtype=bool)
            chroma_mean_by_segment = np.full((total_segments, self.n_chroma), np.nan, dtype=np.float32)
            stage_ms["load_segments_ms"] = float((time.perf_counter() - t_segmeta0) * 1000.0)

            # Audit v3: tuning is computed once on full audio; if fails -> 0.0.
            tuning_failed = False
            try:
                t_full0 = time.perf_counter()
                full_y_t, full_sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
                full_y = self.audio_utils.to_numpy(full_y_t)
                if self.enable_audio_normalization:
                    full_y_t_norm = self.audio_utils.normalize_audio(full_y_t)
                    full_y = self.audio_utils.to_numpy(full_y_t_norm)
                if full_y.ndim == 2:
                    full_y = np.mean(full_y, axis=0) if self.mix_to_mono else full_y[0]
                full_y = full_y.astype(np.float32)
                stage_ms["load_full_audio_ms"] = float((time.perf_counter() - t_full0) * 1000.0)
                t_tune0 = time.perf_counter()
                tuning = float(librosa.estimate_tuning(y=full_y, sr=int(full_sr)))
                stage_ms["tuning_ms"] = float((time.perf_counter() - t_tune0) * 1000.0)
            except Exception:
                tuning = 0.0
                tuning_failed = True
                if "t_tune0" in locals():
                    stage_ms["tuning_ms"] = float((time.perf_counter() - t_tune0) * 1000.0)

            t_proc0 = time.perf_counter()
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
                    # Keep alignment; mark as invalid
                    continue

                # Extract chroma for segment
                seg_chroma = self._extract_chroma(wav, self.sample_rate, tuning)

                # Нормализация
                if self.normalize is not None:
                    seg_chroma = self._normalize_chroma(seg_chroma)

                # Собираем данные
                # Avoid float64 accumulation (smaller allocs, faster).
                seg_mean = np.mean(seg_chroma, axis=1, dtype=np.float32)  # (12,)
                if seg_mean.size == self.n_chroma and np.all(np.isfinite(seg_mean)):
                    chroma_mean_by_segment[seg_idx, :] = seg_mean
                    segment_mask[seg_idx] = True
            stage_ms["process_segments_ms"] = float((time.perf_counter() - t_proc0) * 1000.0)

            # Final progress report
            if self.progress_callback:
                self.progress_callback("chroma", total_segments, total_segments, "Aggregating results")

            # Require at least one valid segment.
            if not bool(np.any(segment_mask)):
                raise ValueError("chroma | all segments invalid/empty (error_code=chroma_validation_failed)")

            # Aggregate chroma_mean across segments (duration-weighted, ignoring masked rows).
            weights = np.asarray(segment_durations, dtype=np.float32)
            w = np.where(segment_mask, np.maximum(weights, 0.0), 0.0)
            w_sum = float(np.sum(w)) + 1e-12
            t_ag0 = time.perf_counter()
            chroma_mean = np.nansum(chroma_mean_by_segment * w.reshape(-1, 1), axis=0) / w_sum
            stage_ms["aggregate_results_ms"] = float((time.perf_counter() - t_ag0) * 1000.0)

            # Формируем payload
            features: Dict[str, Any] = {}
            features["chroma_mean"] = chroma_mean.astype(np.float32)
            # Compute minimal scalars from chroma_mean and per-segment variation
            dominant_idx = int(np.argmax(chroma_mean))
            features["chroma_dominant_class"] = dominant_idx
            features["chroma_dominant_energy"] = float(chroma_mean[dominant_idx])
            chroma_mean_norm = chroma_mean / (float(np.sum(chroma_mean)) + 1e-12)
            features["chroma_entropy"] = float(-np.sum(chroma_mean_norm * np.log(chroma_mean_norm + 1e-12)))
            features["chroma_contrast"] = float(np.max(chroma_mean) - np.min(chroma_mean))
            # stability: 1/(1+mean std across segments) using masked rows
            std_per_class = np.nanstd(chroma_mean_by_segment, axis=0).astype(np.float32)
            features["chroma_harmonic_stability"] = float(1.0 / (1.0 + float(np.nanmean(std_per_class))))

            # Optional segment-level sequence
            if self.enable_time_series:
                features["segment_centers_sec"] = np.asarray(segment_centers, dtype=np.float32)
                features["segment_durations_sec"] = np.asarray(segment_durations, dtype=np.float32)
                features["segment_mask"] = segment_mask.astype(bool)
                features["chroma_mean_by_segment"] = chroma_mean_by_segment.astype(np.float32)

            # Tuning estimate (single global)
            features["tuning_estimate"] = float(tuning)
            features["tuning_failed"] = bool(tuning_failed)
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
            # Segment mode does not expose full spectrogram frames in audit v3 contract.
            features["chroma_frames"] = 0

            # For key_extractor reuse in the same run: provide a small in-memory proxy chroma.
            # We do not want to persist full spectrograms; this is best-effort.
            features["_shared_chroma"] = np.asarray(chroma_mean, dtype=np.float32).reshape(self.n_chroma, 1)

            # Сохраняем большие массивы в .npy
            t_save0 = time.perf_counter()
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)
            stage_ms["save_artifacts_ms"] = float((time.perf_counter() - t_save0) * 1000.0)

            # Валидация выходных данных
            t_val0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"chroma | {error_msg} (error_code={error_code})")
            stage_ms["validate_output_ms"] = float((time.perf_counter() - t_val0) * 1000.0)

            # Добавляем contract version
            features["chroma_contract_version"] = CHROMA_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            # Add stage timings to payload (for meta/stage_timings_ms)
            features["stage_timings_ms"] = {**stage_ms, "total_ms": float((time.perf_counter() - t0) * 1000.0)}
            if res_prof is not None:
                res_prof["at_end"] = snapshot_process_resources()
                features["chroma_resource_profile"] = {
                    **prefix_snapshot("at_start", res_prof.get("at_start", {})),
                    **prefix_snapshot("at_end", res_prof.get("at_end", {})),
                }

            processing_time = time.time() - start_time
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
