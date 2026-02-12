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

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
SPECTRAL_ENTROPY_CONTRACT_VERSION = "spectral_entropy_contract_v1"


class SpectralEntropyExtractor(BaseExtractor):
    """Экстрактор спектральной энтропии с поддержкой segment-based обработки."""

    name = "spectral_entropy"
    version = "2.0.0"
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
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_stats: bool = False,
        enable_flatness: bool = False,
        enable_spread: bool = False,
        enable_time_series: bool = False,
        enable_extended_stats: bool = False,
        enable_dynamics: bool = False,
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
        # Try to reuse provided spectrogram if present in shared_features
        if shared_features:
            # Try to get STFT magnitude from spectral_extractor
            stft_magnitude = shared_features.get("stft_magnitude")
            if stft_magnitude is not None:
                if not isinstance(stft_magnitude, np.ndarray):
                    stft_magnitude = np.array(stft_magnitude)
                if stft_magnitude.ndim == 2:
                    # Convert magnitude to power spectrum
                    S = stft_magnitude.astype(np.float32) ** 2
                    logger.debug(f"spectral_entropy | Reusing STFT from shared_features: {S.shape}")
                    return S
                else:
                    logger.warning(f"spectral_entropy | Invalid stft_magnitude shape in shared_features: {stft_magnitude.shape}, recomputing")

            # Try to get mel spectrogram
            mel_spectrogram = shared_features.get("mel_spectrogram")
            if mel_spectrogram is not None and self.use_mel:
                if not isinstance(mel_spectrogram, np.ndarray):
                    mel_spectrogram = np.array(mel_spectrogram)
                if mel_spectrogram.ndim == 2:
                    S = mel_spectrogram.astype(np.float32) ** 2
                    logger.debug(f"spectral_entropy | Reusing mel spectrogram from shared_features: {S.shape}")
                    return S

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
            return {"mean": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def _calc_extended_stats(self, arr: np.ndarray) -> Dict[str, float]:
        """Вычисление расширенных статистик."""
        if len(arr) == 0:
            return {"min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0}
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

        # Validate entropy stats if present
        if "spectral_entropy_stats" in features:
            stats = features.get("spectral_entropy_stats")
            if not isinstance(stats, dict):
                return False, "spectral_entropy | spectral_entropy_stats must be a dict"
            max_entropy = np.log2(self.n_fft // 2 + 1) if not self.use_mel else np.log2(self.n_mels)
            for stat_key in ["mean", "std", "min", "max"]:
                if stat_key in stats:
                    value = float(stats.get(stat_key))
                    if np.isnan(value) or np.isinf(value):
                        return False, f"spectral_entropy | spectral_entropy_stats.{stat_key} is NaN or Inf"
                    if value < 0 or value > max_entropy:
                        return False, f"spectral_entropy | spectral_entropy_stats.{stat_key} must be in [0, {max_entropy}], got {value}"

        # Validate flatness (should be in [0, 1])
        if "spectral_flatness_stats" in features:
            stats = features.get("spectral_flatness_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if value < 0.0 or value > 1.0:
                            return False, f"spectral_entropy | spectral_flatness_stats.{stat_key} must be in [0, 1], got {value}"

        # Validate spread (should be >= 0)
        if "spectral_spread_stats" in features:
            stats = features.get("spectral_spread_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if value < 0.0:
                            return False, f"spectral_entropy | spectral_spread_stats.{stat_key} must be >= 0, got {value}"

        # Validate time series if present
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
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)

            # Проверка минимальной длительности
            duration = len(y) / sr
            if duration < 1.0:
                error_code = self._classify_error(ValueError("Audio too short"), "audio_too_short")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Аудио слишком короткое: {duration:.2f}s < 1.0s (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Обработка многоканального аудио
            if y.ndim == 2:
                if self.average_channels:
                    y = np.mean(y, axis=0)  # mix to mono
                else:
                    y = y[0]  # use first channel

            # Нормализация аудио (опционально)
            y = self._normalize_audio(y)

            # Вычисление спектрограммы
            try:
                S = self._compute_spectrogram(y, sr, shared_features)
            except Exception as e:
                error_code = self._classify_error(e, "stft_computation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Ошибка вычисления спектрограммы: {str(e)} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Вычисление метрик
            try:
                ent = self._compute_entropy(S)
                ent = self._apply_smoothing(ent)
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
            payload: Dict[str, Any] = {
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

            # Feature gating: time series
            if self.enable_time_series:
                payload["spectral_entropy_series"] = ent.tolist()
                if flatness is not None:
                    payload["spectral_flatness_series"] = flatness.tolist()
                if spread is not None:
                    payload["spectral_spread_series"] = spread.tolist()

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
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_dynamics:
                enabled_features.append("dynamics")
            payload["_features_enabled"] = enabled_features

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(payload)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Валидация выходных данных не прошла: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            processing_time = time.time() - start_time
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
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            if not segments:
                error_code = self._classify_error(ValueError("Empty segments"), "invalid_parameters")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Пустой список сегментов (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            total_segments = len(segments)
            entropy_series_all = []
            flatness_series_all = []
            spread_series_all = []

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
                
                # Вычисление длительности: предпочтительно из start_sec/end_sec, fallback на samples
                if end_sec > start_sec:
                    duration = end_sec - start_sec
                elif end_sample > start_sample:
                    # Fallback: вычисляем из samples (используем sample_rate из сегмента или self.sample_rate)
                    segment_sr = int(seg.get("sample_rate", self.sample_rate))
                    duration = (end_sample - start_sample) / float(segment_sr)
                else:
                    duration = 0.0

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
                    entropy_series_all.extend(ent.tolist())
                except Exception as e:
                    error_code = self._classify_error(e, "entropy_computation_failed")
                    logger.warning(f"spectral_entropy | Ошибка вычисления энтропии для сегмента {i}: {str(e)} (error_code={error_code})")
                    continue

                if self.enable_flatness:
                    try:
                        flatness = self._compute_flatness(S)
                        flatness = self._apply_smoothing(flatness)
                        flatness_series_all.extend(flatness.tolist())
                    except Exception as e:
                        error_code = self._classify_error(e, "flatness_computation_failed")
                        logger.warning(f"spectral_entropy | Ошибка вычисления flatness для сегмента {i}: {str(e)} (error_code={error_code})")

                if self.enable_spread:
                    try:
                        spread = self._compute_spread(S)
                        spread = self._apply_smoothing(spread)
                        spread_series_all.extend(spread.tolist())
                    except Exception as e:
                        error_code = self._classify_error(e, "spread_computation_failed")
                        logger.warning(f"spectral_entropy | Ошибка вычисления spread для сегмента {i}: {str(e)} (error_code={error_code})")

            # Проверка на пустые результаты
            if len(entropy_series_all) == 0:
                error_code = self._classify_error(ValueError("All segments failed"), "entropy_computation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Все сегменты не удалось обработать (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Агрегация результатов
            entropy_arr = np.asarray(entropy_series_all, dtype=np.float32)
            flatness_arr = np.asarray(flatness_series_all, dtype=np.float32) if flatness_series_all else None
            spread_arr = np.asarray(spread_series_all, dtype=np.float32) if spread_series_all else None

            # Формирование payload
            # Вычисляем общую длительность из сегментов
            total_duration = 0.0
            for seg in segments:
                start_sec = float(seg.get("start_sec", 0.0))
                end_sec = float(seg.get("end_sec", 0.0))
                if end_sec > start_sec:
                    total_duration += (end_sec - start_sec)
                else:
                    start_sample = int(seg.get("start_sample", 0))
                    end_sample = int(seg.get("end_sample", 0))
                    if end_sample > start_sample:
                        segment_sr = int(seg.get("sample_rate", self.sample_rate))
                        total_duration += (end_sample - start_sample) / float(segment_sr)
            
            payload: Dict[str, Any] = {
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "use_mel": self.use_mel,
                "n_mels": self.n_mels,
                "average_channels": self.average_channels,
                "smoothing_window": self.smoothing_window,
                "duration": total_duration,
                "segments_count": int(total_segments),
                "spectral_entropy_contract_version": SPECTRAL_ENTROPY_CONTRACT_VERSION,
            }

            # Feature gating: basic stats
            if self.enable_basic_stats:
                payload["spectral_entropy_stats"] = self._calc_stats(entropy_arr)
                if self.enable_extended_stats:
                    payload["spectral_entropy_stats"].update(self._calc_extended_stats(entropy_arr))

            # Feature gating: flatness
            if self.enable_flatness and flatness_arr is not None and len(flatness_arr) > 0:
                payload["spectral_flatness_stats"] = self._calc_stats(flatness_arr)
                if self.enable_extended_stats:
                    payload["spectral_flatness_stats"].update(self._calc_extended_stats(flatness_arr))

            # Feature gating: spread
            if self.enable_spread and spread_arr is not None and len(spread_arr) > 0:
                payload["spectral_spread_stats"] = self._calc_stats(spread_arr)
                if self.enable_extended_stats:
                    payload["spectral_spread_stats"].update(self._calc_extended_stats(spread_arr))

            # Feature gating: time series
            if self.enable_time_series:
                payload["spectral_entropy_series"] = entropy_arr.tolist()
                if flatness_arr is not None:
                    payload["spectral_flatness_series"] = flatness_arr.tolist()
                if spread_arr is not None:
                    payload["spectral_spread_series"] = spread_arr.tolist()

            # Feature gating: dynamics metrics
            if self.enable_dynamics:
                # Стабильность энтропии
                payload["spectral_entropy_stability"] = float(np.var(entropy_arr))
                # Количество переходов (простой эвристический метод)
                transitions = np.sum(np.abs(np.diff(entropy_arr)) > np.std(entropy_arr))
                payload["spectral_entropy_transitions_count"] = int(transitions)
                payload["spectral_entropy_transitions_rate"] = float(transitions / len(entropy_arr)) if len(entropy_arr) > 0 else 0.0
                # Распределение и разнообразие
                payload["spectral_entropy_distribution"] = {
                    "low": float(np.sum(entropy_arr < np.percentile(entropy_arr, 33)) / len(entropy_arr)),
                    "medium": float(np.sum((entropy_arr >= np.percentile(entropy_arr, 33)) & (entropy_arr < np.percentile(entropy_arr, 67))) / len(entropy_arr)),
                    "high": float(np.sum(entropy_arr >= np.percentile(entropy_arr, 67)) / len(entropy_arr)),
                }
                payload["spectral_entropy_diversity"] = float(len(np.unique(np.round(entropy_arr, decimals=2))) / len(entropy_arr))

            # Additional ML/analytics metrics
            additional_metrics = self._calc_additional_metrics(entropy_arr, flatness_arr, spread_arr)
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
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_dynamics:
                enabled_features.append("dynamics")
            payload["_features_enabled"] = enabled_features

            # Валидация выходных данных
            is_valid, error_msg = self._validate_output(payload)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral_entropy | Валидация выходных данных не прошла: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            processing_time = time.time() - start_time
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
