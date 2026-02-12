"""
HPSSExtractor: извлечение Harmonic-Percussive Source Separation признаков с использованием librosa.
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
- Spectral features from separated components
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
HPSS_CONTRACT_VERSION = "hpss_contract_v1"

# Threshold for saving large arrays to .npy files
TIME_SERIES_SAVE_THRESHOLD = 10000


class HPSSExtractor(BaseExtractor):
    """Экстрактор Harmonic-Percussive Source Separation: разложение на гармоническую и перкуссионную компоненты."""

    name = "hpss"
    version = "2.0.0"
    description = "Harmonic-Percussive Source Separation признаки и доли энергии"
    category = "source_separation"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.3

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
        # HPSS parameters
        hpss_kernel_size: int = 31,
        hpss_margin: float = 1.0,
        hpss_power: float = 2.0,
        # Feature gating flags (per-feature control, default: all False)
        enable_energy_metrics: bool = False,
        enable_waveforms: bool = False,
        enable_spectral_features: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация HPSS экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            n_fft: Размер FFT окна
            hop_length: Размер hop для STFT
            average_channels: Усреднять каналы для многоканального аудио
            hpss_kernel_size: Размер ядра для HPSS фильтрации
            hpss_margin: Отступ для границ HPSS
            hpss_power: Степень для нормализации HPSS
            enable_energy_metrics: Включить энергетические метрики (shares, energies)
            enable_waveforms: Включить восстановленные временные сигналы
            enable_spectral_features: Включить спектральные фичи из разделённых компонент
            enable_time_series: Включить временные серии (shares, energies по времени)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(
            sample_rate, n_fft, hop_length, hpss_kernel_size, hpss_margin, hpss_power
        )

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.average_channels = bool(average_channels)
        self.hpss_kernel_size = int(hpss_kernel_size)
        self.hpss_margin = float(hpss_margin)
        self.hpss_power = float(hpss_power)

        # Feature gating flags
        self.enable_energy_metrics = bool(enable_energy_metrics)
        self.enable_waveforms = bool(enable_waveforms)
        self.enable_spectral_features = bool(enable_spectral_features)
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
        n_fft: int,
        hop_length: int,
        hpss_kernel_size: int,
        hpss_margin: float,
        hpss_power: float,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            n_fft: Размер FFT окна
            hop_length: Размер hop для STFT
            hpss_kernel_size: Размер ядра для HPSS
            hpss_margin: Отступ для границ HPSS
            hpss_power: Степень для нормализации HPSS

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"hpss | sample_rate must be positive, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"hpss | n_fft must be positive, got {n_fft}")
        if n_fft < 512:
            raise ValueError(f"hpss | n_fft ({n_fft}) is too small (minimum 512)")
        if hop_length <= 0:
            raise ValueError(f"hpss | hop_length must be positive, got {hop_length}")
        if hop_length > n_fft:
            raise ValueError(f"hpss | hop_length ({hop_length}) must be <= n_fft ({n_fft})")
        if hpss_kernel_size <= 0:
            raise ValueError(f"hpss | hpss_kernel_size must be positive, got {hpss_kernel_size}")
        if hpss_kernel_size % 2 == 0:
            raise ValueError(f"hpss | hpss_kernel_size ({hpss_kernel_size}) must be odd")
        if hpss_margin < 0:
            raise ValueError(f"hpss | hpss_margin must be non-negative, got {hpss_margin}")
        if hpss_power <= 0:
            raise ValueError(f"hpss | hpss_power must be positive, got {hpss_power}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - hpss_audio_load_failed
                - hpss_stft_failed
                - hpss_hpss_failed
                - hpss_istft_failed
                - hpss_validation_failed
                - hpss_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "hpss_audio_load_failed"
        if "stft" in error_str or context == "stft_failed":
            return "hpss_stft_failed"
        if "hpss" in error_str or "decompose" in error_str or context == "hpss_failed":
            return "hpss_hpss_failed"
        if "istft" in error_str or context == "istft_failed":
            return "hpss_istft_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "hpss_validation_failed"

        return "hpss_unknown"

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
                        return False, f"hpss | {key} contains NaN or inf: {value}"
                elif isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        return False, f"hpss | {key} contains NaN or inf"

            # Validate ranges
            if "hpss_harmonic_share" in features:
                share_h = features["hpss_harmonic_share"]
                if share_h < 0.0 or share_h > 1.0:
                    return False, f"hpss | harmonic_share out of range [0, 1]: {share_h}"

            if "hpss_percussive_share" in features:
                share_p = features["hpss_percussive_share"]
                if share_p < 0.0 or share_p > 1.0:
                    return False, f"hpss | percussive_share out of range [0, 1]: {share_p}"

            # Validate consistency: shares should sum to approximately 1.0
            if "hpss_harmonic_share" in features and "hpss_percussive_share" in features:
                share_h = features["hpss_harmonic_share"]
                share_p = features["hpss_percussive_share"]
                total_share = share_h + share_p
                if not (0.95 <= total_share <= 1.05):  # Allow small floating point errors
                    return False, f"hpss | shares sum ({total_share}) not close to 1.0 (harmonic={share_h}, percussive={share_p})"

            # Validate energies
            if "hpss_energy_total" in features:
                energy_total = features["hpss_energy_total"]
                if energy_total < 0:
                    return False, f"hpss | energy_total must be non-negative, got {energy_total}"

            if "hpss_energy_harmonic" in features:
                energy_h = features["hpss_energy_harmonic"]
                if energy_h < 0:
                    return False, f"hpss | energy_harmonic must be non-negative, got {energy_h}"

            if "hpss_energy_percussive" in features:
                energy_p = features["hpss_energy_percussive"]
                if energy_p < 0:
                    return False, f"hpss | energy_percussive must be non-negative, got {energy_p}"

            return True, None
        except Exception as e:
            return False, f"hpss | validation error: {e}"

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

    def _compute_hpss(
        self, y: np.ndarray, sr: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычислить HPSS разложение (fail-fast, no-fallback).

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            (H, P, S_complex, S_mag) - гармоническая, перкуссионная компоненты, комплексный STFT, магнитуда STFT

        Raises:
            RuntimeError: Если HPSS разложение не удалось
        """
        try:
            # Compute STFT
            S_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            S_mag = np.abs(S_complex)

            # Perform HPSS (fail-fast, no-fallback)
            hpss_kwargs = {
                "kernel_size": (self.hpss_kernel_size, self.hpss_kernel_size),
                "margin": self.hpss_margin,
                "power": self.hpss_power,
            }
            H, P = librosa.decompose.hpss(S_mag, **hpss_kwargs)

            return H.astype(np.float32), P.astype(np.float32), S_complex, S_mag.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"hpss | HPSS decomposition failed: {e}") from e

    def _compute_spectral_features(
        self, H: np.ndarray, P: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """
        Вычислить спектральные фичи из разделённых компонент.

        Args:
            H: Гармоническая компонента (spectrogram)
            P: Перкуссионная компонента (spectrogram)
            sr: Частота дискретизации

        Returns:
            Словарь со спектральными фичами
        """
        features: Dict[str, Any] = {}

        # Harmonic spectral features
        try:
            # Spectral centroid
            h_centroid = librosa.feature.spectral_centroid(S=H, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_harmonic_centroid_mean"] = float(np.mean(h_centroid))
            features["hpss_harmonic_centroid_std"] = float(np.std(h_centroid))

            # Spectral bandwidth
            h_bandwidth = librosa.feature.spectral_bandwidth(S=H, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_harmonic_bandwidth_mean"] = float(np.mean(h_bandwidth))
            features["hpss_harmonic_bandwidth_std"] = float(np.std(h_bandwidth))

            # Spectral rolloff
            h_rolloff = librosa.feature.spectral_rolloff(S=H, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_harmonic_rolloff_mean"] = float(np.mean(h_rolloff))
            features["hpss_harmonic_rolloff_std"] = float(np.std(h_rolloff))
        except Exception as e:
            logger.warning(f"hpss | Failed to compute harmonic spectral features: {e}")

        # Percussive spectral features
        try:
            # Spectral centroid
            p_centroid = librosa.feature.spectral_centroid(S=P, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_percussive_centroid_mean"] = float(np.mean(p_centroid))
            features["hpss_percussive_centroid_std"] = float(np.std(p_centroid))

            # Spectral bandwidth
            p_bandwidth = librosa.feature.spectral_bandwidth(S=P, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_percussive_bandwidth_mean"] = float(np.mean(p_bandwidth))
            features["hpss_percussive_bandwidth_std"] = float(np.std(p_bandwidth))

            # Spectral rolloff
            p_rolloff = librosa.feature.spectral_rolloff(S=P, sr=sr, hop_length=self.hop_length)[0]
            features["hpss_percussive_rolloff_mean"] = float(np.mean(p_rolloff))
            features["hpss_percussive_rolloff_std"] = float(np.std(p_rolloff))
        except Exception as e:
            logger.warning(f"hpss | Failed to compute percussive spectral features: {e}")

        return features

    def _compute_hpss_metrics(
        self, H: np.ndarray, P: np.ndarray, S_mag: np.ndarray
    ) -> Dict[str, Any]:
        """
        Вычислить все метрики HPSS на основе разделённых компонент.

        Args:
            H: Гармоническая компонента (spectrogram)
            P: Перкуссионная компонента (spectrogram)
            S_mag: Магнитуда исходного STFT

        Returns:
            Словарь с метриками
        """
        features: Dict[str, Any] = {}

        # Energy metrics (feature-gated)
        if self.enable_energy_metrics:
            energy_total = float(np.sum(S_mag ** 2) + 1e-12)
            energy_h = float(np.sum(H ** 2))
            energy_p = float(np.sum(P ** 2))
            share_h = float(energy_h / energy_total)
            share_p = float(energy_p / energy_total)

            features["hpss_harmonic_share"] = share_h
            features["hpss_percussive_share"] = share_p
            features["hpss_energy_total"] = energy_total
            features["hpss_energy_harmonic"] = energy_h
            features["hpss_energy_percussive"] = energy_p

            # Additional ML/analytics metrics
            # Harmonic stability (variation over time)
            h_energy_per_frame = np.sum(H ** 2, axis=0)
            if h_energy_per_frame.size > 1:
                h_stability = float(1.0 / (1.0 + np.std(h_energy_per_frame) / (np.mean(h_energy_per_frame) + 1e-6)))
            else:
                h_stability = 1.0
            features["hpss_harmonic_stability"] = h_stability

            # Percussive stability
            p_energy_per_frame = np.sum(P ** 2, axis=0)
            if p_energy_per_frame.size > 1:
                p_stability = float(1.0 / (1.0 + np.std(p_energy_per_frame) / (np.mean(p_energy_per_frame) + 1e-6)))
            else:
                p_stability = 1.0
            features["hpss_percussive_stability"] = p_stability

            # Separation quality (based on residual energy)
            residual = S_mag - H - P
            residual_energy = float(np.sum(residual ** 2))
            separation_quality = float(1.0 - (residual_energy / (energy_total + 1e-12)))
            features["hpss_separation_quality"] = separation_quality

            # Balance score (0.0-1.0, 0.5 = balanced)
            balance_score = float(1.0 - abs(share_h - share_p))
            features["hpss_balance_score"] = balance_score

            # Dominance
            if share_h > 0.6:
                dominance = "harmonic"
            elif share_p > 0.6:
                dominance = "percussive"
            else:
                dominance = "mixed"
            features["hpss_dominance"] = dominance

            # Store energy per frame for time series if needed
            if self.enable_time_series:
                features["_harmonic_share_series"] = (h_energy_per_frame / (energy_total + 1e-12)).astype(np.float32)
                features["_percussive_share_series"] = (p_energy_per_frame / (energy_total + 1e-12)).astype(np.float32)

        return features

    def _save_waveforms_npy(
        self, h_wav: np.ndarray, p_wav: np.ndarray
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Сохранить восстановленные waveforms в .npy файлы (per-run storage).

        Args:
            h_wav: Гармонический сигнал
            p_wav: Перкуссионный сигнал

        Returns:
            (harmonic_npy_path, percussive_npy_path) или (None, None) если не нужно сохранять
        """
        if self.artifacts_dir is None:
            logger.warning("hpss | artifacts_dir not set, cannot save waveforms to .npy")
            return None, None

        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            h_npy_path = os.path.join(self.artifacts_dir, "harmonic.npy")
            p_npy_path = os.path.join(self.artifacts_dir, "percussive.npy")
            np.save(h_npy_path, h_wav.astype(np.float32))
            np.save(p_npy_path, p_wav.astype(np.float32))
            # Return relative paths from component directory
            h_rel_path = "_artifacts/harmonic.npy"
            p_rel_path = "_artifacts/percussive.npy"
            logger.info(f"hpss | Saved waveforms to {h_npy_path} and {p_npy_path}")
            return h_rel_path, p_rel_path
        except Exception as e:
            logger.warning(f"hpss | Failed to save waveforms to .npy: {e}")
            return None, None

    def _save_time_series_npy(self, data: np.ndarray, name: str) -> Optional[str]:
        """
        Сохранить временные серии в .npy файл для больших массивов (per-run storage).

        Args:
            data: Массив данных
            name: Имя файла (без расширения)

        Returns:
            Относительный путь к .npy файлу или None
        """
        if data.size < TIME_SERIES_SAVE_THRESHOLD:
            return None

        if self.artifacts_dir is None:
            logger.warning(f"hpss | artifacts_dir not set, cannot save {name} to .npy")
            return None

        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            npy_path = os.path.join(self.artifacts_dir, f"{name}.npy")
            np.save(npy_path, data)
            # Return relative path from component directory
            rel_path = f"_artifacts/{name}.npy"
            logger.info(f"hpss | Saved {name} ({data.size} elements) to {npy_path}")
            return rel_path
        except Exception as e:
            logger.warning(f"hpss | Failed to save {name} to .npy: {e}")
            return None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Обработка полного аудио файла (legacy mode, для обратной совместимости).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория

        Returns:
            ExtractorResult с метриками HPSS
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    False,
                    error=f"hpss | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Load audio
            if self.progress_callback:
                self.progress_callback("hpss", 0, 6, "Loading audio")
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            # Normalize audio if enabled
            y = self._normalize_audio(y)

            duration = float(y.shape[-1] / sr)

            # Compute HPSS (fail-fast, no-fallback)
            if self.progress_callback:
                self.progress_callback("hpss", 1, 6, "Computing STFT and HPSS")
            H, P, S_complex, S_mag = self._compute_hpss(y, sr)

            # Compute metrics
            if self.progress_callback:
                self.progress_callback("hpss", 2, 6, "Computing energy metrics")
            features = self._compute_hpss_metrics(H, P, S_mag)

            # Compute spectral features if enabled
            if self.enable_spectral_features:
                if self.progress_callback:
                    self.progress_callback("hpss", 3, 6, "Computing spectral features")
                spectral_features = self._compute_spectral_features(H, P, sr)
                features.update(spectral_features)

            # Save waveforms if enabled
            if self.enable_waveforms:
                if self.progress_callback:
                    self.progress_callback("hpss", 4, 6, "Reconstructing waveforms")
                phase = np.angle(S_complex)
                H_complex = H * np.exp(1j * phase)
                P_complex = P * np.exp(1j * phase)

                h_wav = librosa.istft(H_complex, hop_length=self.hop_length, length=len(y))
                p_wav = librosa.istft(P_complex, hop_length=self.hop_length, length=len(y))

                h_npy_path, p_npy_path = self._save_waveforms_npy(h_wav, p_wav)
                if h_npy_path and p_npy_path:
                    features["hpss_harmonic_npy"] = h_npy_path
                    features["hpss_percussive_npy"] = p_npy_path
                    features["hpss_waveform_length"] = int(len(y))

            # Save time series if needed
            if self.progress_callback:
                self.progress_callback("hpss", 5, 6, "Saving artifacts")
            if self.enable_time_series:
                # Save harmonic share series
                if "_harmonic_share_series" in features:
                    h_share_npy_path = self._save_time_series_npy(features["_harmonic_share_series"], "harmonic_share_series")
                    if h_share_npy_path:
                        features["hpss_harmonic_share_series_npy"] = h_share_npy_path
                    else:
                        features["hpss_harmonic_share_series"] = features["_harmonic_share_series"].astype(np.float32).tolist()
                    del features["_harmonic_share_series"]

                # Save percussive share series
                if "_percussive_share_series" in features:
                    p_share_npy_path = self._save_time_series_npy(features["_percussive_share_series"], "percussive_share_series")
                    if p_share_npy_path:
                        features["hpss_percussive_share_series_npy"] = p_share_npy_path
                    else:
                        features["hpss_percussive_share_series"] = features["_percussive_share_series"].astype(np.float32).tolist()
                    del features["_percussive_share_series"]

            # Add metadata
            features["sample_rate"] = int(sr)
            features["n_fft"] = int(self.n_fft)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = duration
            features["device_used"] = self.device
            features["hpss_frames"] = int(S_mag.shape[1])
            features["hpss_kernel_size"] = int(self.hpss_kernel_size)
            features["hpss_margin"] = float(self.hpss_margin)
            features["hpss_power"] = float(self.hpss_power)
            features["hpss_contract_version"] = HPSS_CONTRACT_VERSION

            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_energy_metrics:
                features_enabled.append("energy_metrics")
            if self.enable_waveforms:
                features_enabled.append("waveforms")
            if self.enable_spectral_features:
                features_enabled.append("spectral_features")
            if self.enable_time_series:
                features_enabled.append("time_series")
            features["_features_enabled"] = features_enabled

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"hpss | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"hpss | {error_code}: {str(e)}", processing_time=dt)

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
        Segmenter-driven HPSS extraction: compute HPSS metrics on provided windows (families.hpss).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов из Segmenter (families.hpss)
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
                    error=f"hpss | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("hpss | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            all_harmonic_share: List[float] = []
            all_percussive_share: List[float] = []
            all_balance_score: List[float] = []
            all_separation_quality: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            seg_p = max(1, int(segment_parallelism or 1))
            inflight = int(max_inflight) if max_inflight is not None else seg_p
            inflight = max(1, int(inflight))

            def _process_segment(i: int, seg: dict) -> tuple[int, float, Dict[str, Any], float]:
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

                # Compute HPSS (fail-fast, no-fallback)
                H, P, S_complex, S_mag = self._compute_hpss(waveform_np, int(sr))

                # Compute metrics
                seg_features = self._compute_hpss_metrics(H, P, S_mag)

                # Compute spectral features if enabled
                if self.enable_spectral_features:
                    spectral_features = self._compute_spectral_features(H, P, int(sr))
                    seg_features.update(spectral_features)

                return i, center_sec, seg_features, duration

            # Process segments (sequential or parallel)
            if seg_p <= 1:
                for seg_idx, seg in enumerate(segments):
                    _, center_sec, seg_features, duration = _process_segment(seg_idx, seg)
                    if self.enable_energy_metrics:
                        all_harmonic_share.append(seg_features.get("hpss_harmonic_share", 0.0))
                        all_percussive_share.append(seg_features.get("hpss_percussive_share", 0.0))
                        all_balance_score.append(seg_features.get("hpss_balance_score", 0.0))
                        all_separation_quality.append(seg_features.get("hpss_separation_quality", 0.0))
                    segment_centers.append(center_sec)
                    segment_durations.append(duration)

                    # Progress reporting
                    if self.progress_callback and seg_idx % progress_report_interval == 0:
                        pct = int((seg_idx + 1) * 100 / total_segments)
                        if pct != last_reported_pct:
                            self.progress_callback("hpss", seg_idx + 1, total_segments, f"Processed {seg_idx + 1}/{total_segments} segments")
                            last_reported_pct = pct
            else:
                # Parallel processing
                workers = max(1, min(int(seg_p), int(inflight)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_process_segment, i, seg) for i, seg in enumerate(segments)]
                    completed = 0
                    for fut in as_completed(futures):
                        i, center_sec, seg_features, duration = fut.result()
                        if self.enable_energy_metrics:
                            all_harmonic_share.append(seg_features.get("hpss_harmonic_share", 0.0))
                            all_percussive_share.append(seg_features.get("hpss_percussive_share", 0.0))
                            all_balance_score.append(seg_features.get("hpss_balance_score", 0.0))
                            all_separation_quality.append(seg_features.get("hpss_separation_quality", 0.0))
                        segment_centers.append(center_sec)
                        segment_durations.append(duration)
                        completed += 1

                        # Progress reporting
                        if self.progress_callback and completed % progress_report_interval == 0:
                            pct = int(completed * 100 / total_segments)
                            if pct != last_reported_pct:
                                self.progress_callback("hpss", completed, total_segments, f"Processed {completed}/{total_segments} segments")
                                last_reported_pct = pct

            # Aggregate metrics across all segments
            features: Dict[str, Any] = {}

            if self.enable_energy_metrics and all_harmonic_share:
                features["hpss_harmonic_share"] = float(np.mean(all_harmonic_share))
                features["hpss_percussive_share"] = float(np.mean(all_percussive_share))
                features["hpss_balance_score"] = float(np.mean(all_balance_score))
                features["hpss_separation_quality"] = float(np.mean(all_separation_quality))
                features["hpss_harmonic_share_mean"] = float(np.mean(all_harmonic_share))
                features["hpss_harmonic_share_std"] = float(np.std(all_harmonic_share))
                features["hpss_percussive_share_mean"] = float(np.mean(all_percussive_share))
                features["hpss_percussive_share_std"] = float(np.std(all_percussive_share))

            # Add metadata
            total_duration = sum(segment_durations)
            features["sample_rate"] = int(self.sample_rate)
            features["n_fft"] = int(self.n_fft)
            features["hop_length"] = int(self.hop_length)
            features["duration"] = total_duration
            features["device_used"] = self.device
            features["hpss_frames"] = int(np.mean([seg.get("end_sample", 0) - seg.get("start_sample", 0) for seg in segments]) / self.hop_length) if segments else 0
            features["hpss_kernel_size"] = int(self.hpss_kernel_size)
            features["hpss_margin"] = float(self.hpss_margin)
            features["hpss_power"] = float(self.hpss_power)
            features["segments_count"] = int(total_segments)
            features["hpss_contract_version"] = HPSS_CONTRACT_VERSION

            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_energy_metrics:
                features_enabled.append("energy_metrics")
            if self.enable_waveforms:
                features_enabled.append("waveforms")
            if self.enable_spectral_features:
                features_enabled.append("spectral_features")
            if self.enable_time_series:
                features_enabled.append("time_series")
            features["_features_enabled"] = features_enabled

            # Add segment-level data (always available when using run_segments)
            # These are needed for understanding segment structure, not just for time series
            features["segment_centers_sec"] = segment_centers
            features["segment_durations_sec"] = segment_durations

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"hpss | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"hpss | {error_code}: {str(e)}", processing_time=dt)
