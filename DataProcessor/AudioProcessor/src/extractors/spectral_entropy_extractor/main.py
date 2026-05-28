"""
SpectralEntropyExtractor: извлечение спектральной энтропии и связанных метрик (flatness, spread).

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (explicit method selection: librosa only)
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Integration with spectral_extractor via shared_features
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import capture_spectral_entropy_resource_profile, is_spectral_entropy_resource_profile_enabled

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
SPECTRAL_ENTROPY_CONTRACT_VERSION = "spectral_entropy_contract_v1"


class SpectralEntropyExtractor(BaseExtractor):
    """Экстрактор спектральной энтропии с поддержкой segment-based обработки."""

    name = "spectral_entropy"
    version = "2.0.1"
    description = "Спектральная энтропия, flatness и spread"
    category = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 0.9

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        average_channels: bool = True,
        smoothing_window: int = 0,
        use_mel: bool = False,
        n_mels: int = 128,
        # Feature gating flags (Audit v3 defaults: basic stats enabled)
        enable_basic_stats: bool = True,
        enable_flatness: bool = False,
        enable_spread: bool = False,
        enable_time_series: bool = False,  # legacy/no-op in audited v3 contract
        enable_extended_stats: bool = False,
        enable_dynamics: bool = False,  # legacy/no-op in audited v3 contract
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация SpectralEntropy экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            n_fft: Размер FFT окна
            hop_length: Размер hop для STFT
            average_channels: Усреднять ли каналы для многоканального аудио
            smoothing_window: Размер окна сглаживания (0 = без сглаживания)
            use_mel: Использовать ли mel-шкалу вместо линейной
            n_mels: Количество mel-фильтров (если use_mel=True)
            enable_basic_stats: Включить базовые статистики (mean, std) для entropy
            enable_flatness: Включить метрики flatness
            enable_spread: Включить метрики spread
            enable_time_series: Включить временные серии для всех метрик
            enable_extended_stats: Включить расширенные статистики (min, max, p25, p75)
            enable_dynamics: Включить метрики динамики (для run_segments)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(sample_rate, n_fft, hop_length, n_mels, smoothing_window)

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.average_channels = bool(average_channels)
        self.smoothing_window = max(0, int(smoothing_window))
        self.use_mel = bool(use_mel)
        self.n_mels = int(n_mels)

        # Feature gating flags
        self.enable_basic_stats = bool(enable_basic_stats)
        self.enable_flatness = bool(enable_flatness)
        self.enable_spread = bool(enable_spread)
        self.enable_time_series = bool(enable_time_series)
        self.enable_extended_stats = bool(enable_extended_stats)
        self.enable_dynamics = bool(enable_dynamics)

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
        n_fft: int,
        hop_length: int,
        n_mels: int,
        smoothing_window: int,
    ) -> None:
        """Валидация параметров инициализации."""
        if sample_rate <= 0:
            raise ValueError(f"spectral_entropy | sample_rate must be > 0, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"spectral_entropy | n_fft must be > 0, got {n_fft}")
        if n_fft < 512:
            raise ValueError(f"spectral_entropy | n_fft ({n_fft}) is too small (minimum 512)")
        if hop_length <= 0:
            raise ValueError(f"spectral_entropy | hop_length must be > 0, got {hop_length}")
        if hop_length > n_fft:
            raise ValueError(f"spectral_entropy | hop_length ({hop_length}) must be <= n_fft ({n_fft})")
        if n_mels < 3:
            raise ValueError(f"spectral_entropy | n_mels must be >= 3, got {n_mels}")
        if smoothing_window < 0:
            raise ValueError(f"spectral_entropy | smoothing_window must be >= 0, got {smoothing_window}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: Детальный код ошибки
        """
        error_str = str(error).lower()
        if "file" in error_str or "not found" in error_str or "cannot open" in error_str or context == "audio_load_failed":
            return "audio_load_failed"
        if "too short" in error_str or "empty" in error_str or context == "audio_too_short":
            return "audio_too_short"
        if "stft" in error_str or "fft" in error_str or "spectrum" in error_str or context == "stft_computation_failed":
            return "stft_computation_failed"
        if "entropy" in error_str or context == "entropy_computation_failed":
            return "entropy_computation_failed"
        if "flatness" in error_str or context == "flatness_computation_failed":
            return "flatness_computation_failed"
        if "spread" in error_str or context == "spread_computation_failed":
            return "spread_computation_failed"
        if "parameter" in error_str or "invalid" in error_str or context == "invalid_parameters":
            return "invalid_parameters"
        if "validation" in error_str or context == "validation_failed":
            return "validation_failed"
        return "spectral_entropy_unknown"

    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Нормализация аудио (peak normalization)."""
        if not self.enable_audio_normalization:
            return y
        max_val = np.abs(y).max()
        if max_val > 1e-12:
            return y / max_val
        return y

    def _compute_spectrogram(
        self,
        y: np.ndarray,
        sr: int,
        shared_features: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Вычисление спектрограммы мощности с поддержкой shared_features.

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации
            shared_features: Предвычисленные фичи от других extractors

        Returns:
            S: Спектрограмма мощности [n_freq, n_time]
        """
        # Audit v3: correctness-first. Do NOT reuse shared_features spectrograms.
        # (shared_features may correspond to full-audio STFT and is not safe for segment runs.)
        _ = shared_features

        # Compute spectrogram here
        if not self.use_mel:
            S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2  # power
        else:
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0
            )
        return S.astype(np.float32)

    def _compute_entropy(self, S: np.ndarray) -> np.ndarray:
        """
        Вычисление спектральной энтропии Шеннона.

        Args:
            S: Спектрограмма мощности [n_freq, n_time]

        Returns:
            ent: Спектральная энтропия [n_time]
        """
        eps = 1e-12
        # Нормируем по частотной оси для каждой временной колонки
        P = S / (np.sum(S, axis=0, keepdims=True) + eps)
        # Энтропия Шеннона
        ent = -np.sum(P * np.log2(P + eps), axis=0)
        return ent.astype(np.float32)

    def _compute_flatness(self, S: np.ndarray) -> np.ndarray:
        """
        Вычисление spectral flatness (геометрическое/арифметическое среднее).

        Args:
            S: Спектрограмма мощности [n_freq, n_time]

        Returns:
            flatness: Spectral flatness [n_time]
        """
        eps = 1e-12
        P = S / (np.sum(S, axis=0, keepdims=True) + eps)
        logP = np.log(P + eps)
        # Геометрическое среднее / арифметическое среднее
        flatness = np.exp(np.mean(logP, axis=0)) / (np.mean(P, axis=0) + eps)
        return flatness.astype(np.float32)

    def _compute_spread(self, S: np.ndarray) -> np.ndarray:
        """
        Вычисление spectral spread (стандартное отклонение частотного индекса).

        Args:
            S: Спектрограмма мощности [n_freq, n_time]

        Returns:
            spread: Spectral spread [n_time]
        """
        eps = 1e-12
        P = S / (np.sum(S, axis=0, keepdims=True) + eps)
        # Нормированный индекс частоты (0..1)
        freq_idx = np.linspace(0.0, 1.0, P.shape[0], dtype=np.float32).reshape(-1, 1)
        mu = np.sum(freq_idx * P, axis=0)
        spread = np.sqrt(np.sum(((freq_idx - mu) ** 2) * P, axis=0))
        return spread.astype(np.float32)

    def _apply_smoothing(self, series: np.ndarray) -> np.ndarray:
        """Применение скользящего сглаживания."""
        if self.smoothing_window <= 1:
            return series
        w = self.smoothing_window
        kernel = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(series, kernel, mode="same").astype(np.float32)

    def _calc_stats(self, arr: np.ndarray) -> Dict[str, float]:
        """Вычисление базовых статистик."""
        if len(arr) == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def _calc_extended_stats(self, arr: np.ndarray) -> Dict[str, float]:
        """Вычисление расширенных статистик."""
        if len(arr) == 0:
            return {"min": float("nan"), "max": float("nan"), "p25": float("nan"), "p75": float("nan")}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    def _calc_additional_metrics(
        self,
        entropy_arr: Optional[np.ndarray] = None,
        flatness_arr: Optional[np.ndarray] = None,
        spread_arr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Вычисление дополнительных метрик для ML/аналитики.

        Args:
            entropy_arr: Массив энтропии
            flatness_arr: Массив flatness
            spread_arr: Массив spread

        Returns:
            Словарь с дополнительными метриками
        """
        metrics = {}
        if entropy_arr is not None and len(entropy_arr) > 0:
            metrics["spectral_entropy_variance"] = float(np.var(entropy_arr))
            metrics["spectral_entropy_min"] = float(np.min(entropy_arr))
            metrics["spectral_entropy_max"] = float(np.max(entropy_arr))
        if flatness_arr is not None and len(flatness_arr) > 0:
            metrics["spectral_flatness_variance"] = float(np.var(flatness_arr))
            metrics["spectral_flatness_min"] = float(np.min(flatness_arr))
            metrics["spectral_flatness_max"] = float(np.max(flatness_arr))
        if spread_arr is not None and len(spread_arr) > 0:
            metrics["spectral_spread_variance"] = float(np.var(spread_arr))
            metrics["spectral_spread_min"] = float(np.min(spread_arr))
            metrics["spectral_spread_max"] = float(np.max(spread_arr))
        return metrics

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "spectral_entropy | features must be a dict"

        # Validate entropy stats if present (allow NaN for missing values)
        if "spectral_entropy_stats" in features:
            stats = features.get("spectral_entropy_stats")
            if not isinstance(stats, dict):
                return False, "spectral_entropy | spectral_entropy_stats must be a dict"
            max_entropy = np.log2(self.n_fft // 2 + 1) if not self.use_mel else np.log2(self.n_mels)
            for stat_key in ["mean", "std", "min", "max"]:
                if stat_key in stats:
                    value = float(stats.get(stat_key))
                    if np.isinf(value):
                        return False, f"spectral_entropy | spectral_entropy_stats.{stat_key} is Inf"
                    if np.isnan(value):
                        continue
                    if value < 0 or value > max_entropy:
                        return False, f"spectral_entropy | spectral_entropy_stats.{stat_key} must be in [0, {max_entropy}], got {value}"

        # Validate flatness (should be in [0, 1]) (allow NaN)
        if "spectral_flatness_stats" in features:
            stats = features.get("spectral_flatness_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if np.isnan(value):
                            continue
                        if value < 0.0 or value > 1.0:
                            return False, f"spectral_entropy | spectral_flatness_stats.{stat_key} must be in [0, 1], got {value}"

        # Validate spread (should be >= 0) (allow NaN)
        if "spectral_spread_stats" in features:
            stats = features.get("spectral_spread_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if np.isnan(value):
                            continue
                        if value < 0.0:
                            return False, f"spectral_entropy | spectral_spread_stats.{stat_key} must be >= 0, got {value}"

        # Validate time series if present (legacy; allow empty; forbid NaN/Inf/negative if present)
        for series_key in ["spectral_entropy_series", "spectral_flatness_series", "spectral_spread_series"]:
            if series_key in features:
                series = features.get(series_key)
                if series is not None:
                    if isinstance(series, list):
                        series_arr = np.asarray(series, dtype=np.float32)
                        if np.any(np.isnan(series_arr)) or np.any(np.isinf(series_arr)):
                            return False, f"spectral_entropy | {series_key} contains NaN or Inf values"
                        if np.any(series_arr < 0):
                            return False, f"spectral_entropy | {series_key} contains negative values"

        return True, None

    def run(self, input_uri: str, tmp_path: str, shared_features: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        """
        Извлечение спектральной энтропии на полном аудио.

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            shared_features: Предвычисленные фичи от других extractors

        Returns:
            ExtractorResult с payload
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        spectral_entropy_resource_profile: Optional[Dict[str, Any]] = None
        if is_spectral_entropy_resource_profile_enabled():
            spectral_entropy_resource_profile = {
                "at_start": capture_spectral_entropy_resource_profile(stage="at_start"),
            }
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            t0 = time.perf_counter()
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0

            # Проверка минимальной длительности (Audit v3: short audio -> empty)
            duration = len(y) / sr
            if duration < 1.0:
                payload = {
                    "status": "empty",
                    "empty_reason": "audio_too_short",
                    "device_used": self.device,
                    "sample_rate": sr,
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "use_mel": self.use_mel,
                    "n_mels": self.n_mels,
                    "average_channels": self.average_channels,
                    "smoothing_window": self.smoothing_window,
                    "duration": float(duration),
                    "segments_count": 0,
                    # model-facing scalars (NaN = missing)
                    "spectral_entropy_mean": float("nan"),
                    "spectral_entropy_std": float("nan"),
                    # canonical axis (synthetic single window)
                    "segment_start_sec": [0.0],
                    "segment_end_sec": [float(duration)],
                    "segment_center_sec": [0.5 * float(duration)],
                    "segment_mask": [False],
                    # per-segment arrays (N=1)
                    "entropy_mean_by_segment": [float("nan")],
                    "entropy_std_by_segment": [float("nan")],
                    "spectral_entropy_contract_version": SPECTRAL_ENTROPY_CONTRACT_VERSION,
                    "_features_enabled": ["basic_stats"],
                }
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload["stage_timings_ms"] = stage_timings_ms
                if spectral_entropy_resource_profile is not None:
                    spectral_entropy_resource_profile["at_end"] = capture_spectral_entropy_resource_profile(stage="at_end")
                    payload["spectral_entropy_resource_profile"] = spectral_entropy_resource_profile
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            # Обработка многоканального аудио
            if y.ndim == 2:
                if self.average_channels:
                    y = np.mean(y, axis=0)  # mix to mono
                else:
                    y = y[0]  # use first channel

            # Нормализация аудио (опционально)
            t0 = time.perf_counter()
            y = self._normalize_audio(y)
            stage_timings_ms["normalize_audio_ms"] = (time.perf_counter() - t0) * 1000.0

            # Вычисление спектрограммы
            try:
                t0 = time.perf_counter()
                S = self._compute_spectrogram(y, sr, shared_features)
                stage_timings_ms["spectrogram_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                error_code = self._classify_error(e, "stft_computation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Ошибка вычисления спектрограммы: {str(e)} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Вычисление метрик
            try:
                t0 = time.perf_counter()
                ent = self._compute_entropy(S)
                ent = self._apply_smoothing(ent)
                stage_timings_ms["entropy_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                error_code = self._classify_error(e, "entropy_computation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Ошибка вычисления энтропии: {str(e)} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            flatness = None
            spread = None
            if self.enable_flatness:
                try:
                    flatness = self._compute_flatness(S)
                    flatness = self._apply_smoothing(flatness)
                except Exception as e:
                    error_code = self._classify_error(e, "flatness_computation_failed")
                    logger.warning(f"spectral_entropy | Ошибка вычисления flatness: {str(e)} (error_code={error_code})")
                    flatness = None

            if self.enable_spread:
                try:
                    spread = self._compute_spread(S)
                    spread = self._apply_smoothing(spread)
                except Exception as e:
                    error_code = self._classify_error(e, "spread_computation_failed")
                    logger.warning(f"spectral_entropy | Ошибка вычисления spread: {str(e)} (error_code={error_code})")
                    spread = None

            # Формирование payload
            t0 = time.perf_counter()
            payload: Dict[str, Any] = {
                "status": "ok",
                "empty_reason": "none",
                "device_used": self.device,
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "use_mel": self.use_mel,
                "n_mels": self.n_mels,
                "average_channels": self.average_channels,
                "smoothing_window": self.smoothing_window,
                "duration": float(duration),
                "spectral_entropy_contract_version": SPECTRAL_ENTROPY_CONTRACT_VERSION,
            }

            # Feature gating: basic stats
            if self.enable_basic_stats:
                payload["spectral_entropy_stats"] = self._calc_stats(ent)
                if self.enable_extended_stats:
                    payload["spectral_entropy_stats"].update(self._calc_extended_stats(ent))
                payload["spectral_entropy_mean"] = float(payload["spectral_entropy_stats"].get("mean", float("nan")))
                payload["spectral_entropy_std"] = float(payload["spectral_entropy_stats"].get("std", float("nan")))
            else:
                payload["spectral_entropy_mean"] = float("nan")
                payload["spectral_entropy_std"] = float("nan")

            # Feature gating: flatness
            if self.enable_flatness and flatness is not None:
                payload["spectral_flatness_stats"] = self._calc_stats(flatness)
                if self.enable_extended_stats:
                    payload["spectral_flatness_stats"].update(self._calc_extended_stats(flatness))

            # Feature gating: spread
            if self.enable_spread and spread is not None:
                payload["spectral_spread_stats"] = self._calc_stats(spread)
                if self.enable_extended_stats:
                    payload["spectral_spread_stats"].update(self._calc_extended_stats(spread))

            # Audit v3: per-segment arrays are the primary time-axis contract.
            payload["segment_start_sec"] = [0.0]
            payload["segment_end_sec"] = [float(duration)]
            payload["segment_center_sec"] = [0.5 * float(duration)]
            payload["segment_mask"] = [True]
            payload["entropy_mean_by_segment"] = [float(payload.get("spectral_entropy_mean", float("nan")))]
            payload["entropy_std_by_segment"] = [float(payload.get("spectral_entropy_std", float("nan")))]

            # Additional ML/analytics metrics
            additional_metrics = self._calc_additional_metrics(ent, flatness, spread)
            if additional_metrics:
                payload.update(additional_metrics)

            # Features enabled list
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_flatness:
                enabled_features.append("flatness")
            if self.enable_spread:
                enabled_features.append("spread")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            payload["_features_enabled"] = enabled_features
            stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0

            # Валидация выходных данных
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(payload)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Валидация выходных данных не прошла: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            processing_time = time.time() - start_time
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload["stage_timings_ms"] = stage_timings_ms
            if spectral_entropy_resource_profile is not None:
                spectral_entropy_resource_profile["at_end"] = capture_spectral_entropy_resource_profile(stage="at_end")
                payload["spectral_entropy_resource_profile"] = spectral_entropy_resource_profile
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "spectral_entropy_unknown")
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(
                success=False,
                error=f"spectral_entropy | Неожиданная ошибка: {str(e)} (error_code={error_code})",
                processing_time=processing_time,
            )

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        shared_features: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """
        Извлечение спектральной энтропии по сегментам.

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов из segments.json
            shared_features: Предвычисленные фичи от других extractors

        Returns:
            ExtractorResult с payload
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        spectral_entropy_resource_profile: Optional[Dict[str, Any]] = None
        if is_spectral_entropy_resource_profile_enabled():
            spectral_entropy_resource_profile = {
                "at_start": capture_spectral_entropy_resource_profile(stage="at_start"),
            }
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            if not segments:
                payload = {
                    "status": "error",
                    "empty_reason": "none",
                    "error_code": "invalid_parameters",
                }
                return self._create_result(
                    success=False,
                    error="spectral_entropy | Пустой список сегментов (error_code=invalid_parameters)",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            total_segments = len(segments)
            t0_process = time.perf_counter()

            segment_start_sec = [0.0 for _ in range(total_segments)]
            segment_end_sec = [0.0 for _ in range(total_segments)]
            segment_center_sec = [0.0 for _ in range(total_segments)]
            segment_mask = [False for _ in range(total_segments)]

            entropy_mean_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)
            entropy_std_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)
            entropy_min_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)
            entropy_max_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)

            flatness_mean_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)
            flatness_std_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)

            spread_mean_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)
            spread_std_by_segment = np.full((total_segments,), np.nan, dtype=np.float32)

            seg_duration_sec = np.zeros((total_segments,), dtype=np.float32)

            # Обработка сегментов
            for i, seg in enumerate(segments):
                if self.progress_callback:
                    self.progress_callback(
                        "spectral_entropy",
                        i + 1,
                        total_segments,
                        f"Processing segment {i+1}/{total_segments}",
                    )

                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                start_sec = float(seg.get("start_sec", 0.0))
                end_sec = float(seg.get("end_sec", 0.0))
                segment_start_sec[i] = start_sec
                segment_end_sec[i] = end_sec
                segment_center_sec[i] = 0.5 * (start_sec + end_sec) if end_sec > start_sec else start_sec
                
                # Вычисление длительности: предпочтительно из start_sec/end_sec, fallback на samples
                if end_sec > start_sec:
                    duration = end_sec - start_sec
                elif end_sample > start_sample:
                    # Fallback: вычисляем из samples (используем sample_rate из сегмента или self.sample_rate)
                    segment_sr = int(seg.get("sample_rate", self.sample_rate))
                    duration = (end_sample - start_sample) / float(segment_sr)
                else:
                    duration = 0.0
                seg_duration_sec[i] = float(max(0.0, duration))

                # Проверка минимальной длительности (до загрузки аудио)
                if duration < 0.1:  # Минимум 100ms для сегмента
                    logger.warning(f"spectral_entropy | Сегмент {i} слишком короткий: {duration:.3f}s < 0.1s, пропускаем")
                    continue

                # Загрузка сегмента
                try:
                    y_seg_t, sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        target_sr=self.sample_rate,
                    )
                    y_seg = self.audio_utils.to_numpy(y_seg_t)
                except Exception as e:
                    error_code = self._classify_error(e, "audio_load_failed")
                    logger.warning(f"spectral_entropy | Ошибка загрузки сегмента {i}: {str(e)} (error_code={error_code})")
                    continue

                # Дополнительная проверка после загрузки
                if y_seg.size == 0:
                    logger.warning(f"spectral_entropy | Сегмент {i} пустой после загрузки, пропускаем")
                    continue

                # Обработка многоканального аудио
                if y_seg.ndim == 2:
                    if self.average_channels:
                        y_seg = np.mean(y_seg, axis=0)  # mix to mono
                    else:
                        y_seg = y_seg[0]  # use first channel

                # Нормализация аудио (опционально)
                y_seg = self._normalize_audio(y_seg)

                # Вычисление спектрограммы
                try:
                    S = self._compute_spectrogram(y_seg, sr, shared_features)
                except Exception as e:
                    error_code = self._classify_error(e, "stft_computation_failed")
                    logger.warning(f"spectral_entropy | Ошибка вычисления спектрограммы для сегмента {i}: {str(e)} (error_code={error_code})")
                    continue

                # Вычисление метрик
                try:
                    ent = self._compute_entropy(S)
                    ent = self._apply_smoothing(ent)
                except Exception as e:
                    error_code = self._classify_error(e, "entropy_computation_failed")
                    logger.warning(f"spectral_entropy | Ошибка вычисления энтропии для сегмента {i}: {str(e)} (error_code={error_code})")
                    continue

                # Per-segment entropy stats
                entropy_mean_by_segment[i] = float(np.mean(ent)) if ent.size else np.nan
                entropy_std_by_segment[i] = float(np.std(ent)) if ent.size else np.nan
                if self.enable_extended_stats and ent.size:
                    entropy_min_by_segment[i] = float(np.min(ent))
                    entropy_max_by_segment[i] = float(np.max(ent))

                if self.enable_flatness:
                    try:
                        flatness = self._compute_flatness(S)
                        flatness = self._apply_smoothing(flatness)
                        if flatness.size:
                            flatness_mean_by_segment[i] = float(np.mean(flatness))
                            flatness_std_by_segment[i] = float(np.std(flatness))
                    except Exception as e:
                        error_code = self._classify_error(e, "flatness_computation_failed")
                        logger.warning(f"spectral_entropy | Ошибка вычисления flatness для сегмента {i}: {str(e)} (error_code={error_code})")

                if self.enable_spread:
                    try:
                        spread = self._compute_spread(S)
                        spread = self._apply_smoothing(spread)
                        if spread.size:
                            spread_mean_by_segment[i] = float(np.mean(spread))
                            spread_std_by_segment[i] = float(np.std(spread))
                    except Exception as e:
                        error_code = self._classify_error(e, "spread_computation_failed")
                        logger.warning(f"spectral_entropy | Ошибка вычисления spread для сегмента {i}: {str(e)} (error_code={error_code})")

                segment_mask[i] = True

            # Проверка на пустые результаты
            valid_idx = np.asarray(segment_mask, dtype=bool)
            if not np.any(valid_idx):
                # Audit v3: empty when all segments failed
                payload = {
                    "status": "empty",
                    "empty_reason": "spectral_entropy_all_segments_failed",
                    "device_used": self.device,
                    "sample_rate": self.sample_rate,
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "use_mel": self.use_mel,
                    "n_mels": self.n_mels,
                    "average_channels": self.average_channels,
                    "smoothing_window": self.smoothing_window,
                    "duration": float(max(segment_end_sec) if segment_end_sec else 0.0),
                    "segments_count": int(total_segments),
                    "spectral_entropy_mean": float("nan"),
                    "spectral_entropy_std": float("nan"),
                    "segment_start_sec": segment_start_sec,
                    "segment_end_sec": segment_end_sec,
                    "segment_center_sec": segment_center_sec,
                    "segment_mask": [bool(x) for x in segment_mask],
                    "entropy_mean_by_segment": entropy_mean_by_segment.tolist(),
                    "entropy_std_by_segment": entropy_std_by_segment.tolist(),
                    "spectral_entropy_contract_version": SPECTRAL_ENTROPY_CONTRACT_VERSION,
                    "_features_enabled": ["basic_stats"],
                }
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_process) * 1000.0
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload["stage_timings_ms"] = stage_timings_ms
                if spectral_entropy_resource_profile is not None:
                    spectral_entropy_resource_profile["at_end"] = capture_spectral_entropy_resource_profile(stage="at_end")
                    payload["spectral_entropy_resource_profile"] = spectral_entropy_resource_profile
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            # Global mean/std pooled over segments (duration-weighted)
            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_process) * 1000.0
            t0 = time.perf_counter()
            w = seg_duration_sec[valid_idx]
            m = entropy_mean_by_segment[valid_idx]
            # per-segment variance from std
            v = (entropy_std_by_segment[valid_idx] ** 2).astype(np.float64)
            wsum = float(np.sum(w)) if w.size else 0.0
            if wsum <= 0:
                global_mean = float(np.nanmean(m)) if m.size else float("nan")
                global_var = float(np.nanmean(v)) if v.size else float("nan")
            else:
                global_mean = float(np.sum(w * m) / wsum)
                global_var = float(np.sum(w * (v + (m - global_mean) ** 2)) / wsum)
            global_std = float(np.sqrt(global_var)) if np.isfinite(global_var) else float("nan")
            stage_timings_ms["aggregate_metrics_ms"] = (time.perf_counter() - t0) * 1000.0
            
            payload: Dict[str, Any] = {
                "status": "ok",
                "empty_reason": "none",
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "use_mel": self.use_mel,
                "n_mels": self.n_mels,
                "average_channels": self.average_channels,
                "smoothing_window": self.smoothing_window,
                "duration": float(max(segment_end_sec) if segment_end_sec else 0.0),
                "segments_count": int(total_segments),
                "spectral_entropy_contract_version": SPECTRAL_ENTROPY_CONTRACT_VERSION,
                "spectral_entropy_mean": float(global_mean),
                "spectral_entropy_std": float(global_std),
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": segment_center_sec,
                "segment_mask": [bool(x) for x in segment_mask],
                "entropy_mean_by_segment": entropy_mean_by_segment.tolist(),
                "entropy_std_by_segment": entropy_std_by_segment.tolist(),
            }

            if self.enable_extended_stats:
                payload["entropy_min_by_segment"] = entropy_min_by_segment.tolist()
                payload["entropy_max_by_segment"] = entropy_max_by_segment.tolist()

            if self.enable_flatness:
                payload["flatness_mean_by_segment"] = flatness_mean_by_segment.tolist()
                payload["flatness_std_by_segment"] = flatness_std_by_segment.tolist()

            if self.enable_spread:
                payload["spread_mean_by_segment"] = spread_mean_by_segment.tolist()
                payload["spread_std_by_segment"] = spread_std_by_segment.tolist()

            # Features enabled list
            enabled_features = []
            enabled_features.append("basic_stats")
            if self.enable_flatness:
                enabled_features.append("flatness")
            if self.enable_spread:
                enabled_features.append("spread")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            payload["_features_enabled"] = enabled_features

            # Валидация выходных данных
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(payload)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Валидация выходных данных не прошла: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            processing_time = time.time() - start_time
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload["stage_timings_ms"] = stage_timings_ms
            if spectral_entropy_resource_profile is not None:
                spectral_entropy_resource_profile["at_end"] = capture_spectral_entropy_resource_profile(stage="at_end")
                payload["spectral_entropy_resource_profile"] = spectral_entropy_resource_profile
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "spectral_entropy_unknown")
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(
                success=False,
                error=f"spectral_entropy | Неожиданная ошибка: {str(e)} (error_code={error_code})",
                processing_time=processing_time,
            )
