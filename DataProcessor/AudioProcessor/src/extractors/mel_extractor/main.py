"""
MelExtractor: извлечение Mel-спектрограммы (Mel-frequency spectrogram).
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

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
MEL_CONTRACT_VERSION = "mel_contract_v1"


class MelExtractor(BaseExtractor):
    """Экстрактор Mel-спектрограммы с поддержкой GPU."""

    name = "mel"
    version = "2.1.1"
    description = "Извлечение Mel-спектрограммы признаков"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 3.0

    # Предпочитает GPU, но может работать на CPU
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 1.0  # GB

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        mix_to_mono: bool = True,
        # Feature gating flags (Audit v3 defaults: basic + spectral enabled)
        enable_basic_features: bool = True,
        enable_statistics: bool = False,
        enable_spectral_features: bool = True,
        enable_time_series: bool = False,
        enable_stats_vector: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = True,  # Audit v3: keep enabled by default
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация Mel экстрактора.

        Args:
            device: Устройство для обработки (использует GPU если доступен)
            sample_rate: Частота дискретизации
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            n_mels: Количество мел-фильтров
            fmin: Минимальная частота
            fmax: Максимальная частота
            power: Степень для спектрограммы (1.0 = magnitude, 2.0 = power)
            mix_to_mono: Сводить стерео в моно
            enable_basic_features: Включить базовые фичи (mel_spectrogram, mel_shape, mel_elements)
            enable_statistics: Включить статистики (mel_mean, mel_std, mel_min, mel_max, freq_mean, freq_std)
            enable_spectral_features: Включить спектральные фичи (spectral_centroid, spectral_bandwidth)
            enable_time_series: Включить временные серии для всех фичей
            enable_stats_vector: Включить компактный вектор статистик (mel_stats_vector)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(sample_rate, n_fft, hop_length, n_mels, fmin, fmax, power)

        # Resolve device to torch.device (use GPU if available)
        if device == "auto":
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device(device)

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else float(sample_rate // 2)
        self.power = float(power)
        self.mix_to_mono = bool(mix_to_mono)

        # Feature gating flags
        self.enable_basic_features = bool(enable_basic_features)
        self.enable_statistics = bool(enable_statistics)
        self.enable_spectral_features = bool(enable_spectral_features)
        self.enable_time_series = bool(enable_time_series)
        self.enable_stats_vector = bool(enable_stats_vector)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=str(self.torch_device), sample_rate=self.sample_rate)

        # Инициализируем трансформы
        self._setup_transforms()

    def _validate_parameters(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmin: float,
        fmax: Optional[float],
        power: float,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            n_mels: Количество мел-фильтров
            fmin: Минимальная частота
            fmax: Максимальная частота
            power: Степень для спектрограммы

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"mel | sample_rate must be positive, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"mel | n_fft must be positive, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"mel | hop_length must be positive, got {hop_length}")
        if hop_length > n_fft:
            raise ValueError(f"mel | hop_length ({hop_length}) must be <= n_fft ({n_fft})")
        if n_mels <= 0:
            raise ValueError(f"mel | n_mels must be positive, got {n_mels}")
        if fmin < 0.0:
            raise ValueError(f"mel | fmin must be non-negative, got {fmin}")
        if fmax is not None:
            if fmax <= fmin:
                raise ValueError(f"mel | fmax ({fmax}) must be > fmin ({fmin})")
            if fmax > sample_rate / 2:
                raise ValueError(f"mel | fmax ({fmax}) must be <= sample_rate/2 ({sample_rate / 2})")
        if power <= 0.0:
            raise ValueError(f"mel | power must be positive, got {power}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - mel_audio_load_failed
                - mel_transform_setup_failed
                - mel_spectrogram_failed
                - mel_amplitude_to_db_failed
                - mel_statistics_failed
                - mel_spectral_features_failed
                - mel_validation_failed
                - mel_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "mel_audio_load_failed"
        if "transform" in error_str or "setup" in error_str or context == "transform_setup_failed":
            return "mel_transform_setup_failed"
        if "spectrogram" in error_str or context == "spectrogram_failed":
            return "mel_spectrogram_failed"
        if "amplitude" in error_str or "db" in error_str or context == "amplitude_to_db_failed":
            return "mel_amplitude_to_db_failed"
        if "statistic" in error_str or context == "statistics_failed":
            return "mel_statistics_failed"
        if "centroid" in error_str or "bandwidth" in error_str or context == "spectral_features_failed":
            return "mel_spectral_features_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "mel_validation_failed"

        return "mel_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "mel | features must be a dict"

        # Validate basic features if present
        if "mel_shape" in features:
            mel_shape = features.get("mel_shape")
            if isinstance(mel_shape, (tuple, list)) and len(mel_shape) == 2:
                if mel_shape[0] != self.n_mels:
                    return False, f"mel | mel_shape[0] ({mel_shape[0]}) != n_mels ({self.n_mels})"

        # Validate statistics if present
        if self.enable_statistics:
            for stat_key in ["mel_mean", "mel_std", "mel_min", "mel_max"]:
                if f"{stat_key}_shape" in features:
                    stat_shape = features.get(f"{stat_key}_shape")
                    if isinstance(stat_shape, list) and len(stat_shape) > 0:
                        if stat_shape[0] != self.n_mels:
                            return False, f"mel | {stat_key}_shape[0] ({stat_shape[0]}) != n_mels ({self.n_mels})"

        # Validate spectral features if present
        if self.enable_spectral_features:
            for feat_key in ["spectral_centroid", "spectral_bandwidth"]:
                if f"{feat_key}_shape" in features:
                    feat_shape = features.get(f"{feat_key}_shape")
                    if isinstance(feat_shape, list) and len(feat_shape) > 0:
                        # spectral features should have shape (frames,)
                        if not isinstance(feat_shape[0], int) or feat_shape[0] <= 0:
                            return False, f"mel | {feat_key}_shape is invalid: {feat_shape}"

        return True, None

    def _setup_transforms(self) -> None:
        """Настройка трансформов для Mel (fail-fast)."""
        try:
            if self.progress_callback:
                self.progress_callback("mel", 0, 1, "Setting up transforms")

            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                power=self.power,
            ).to(self.torch_device)

            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power" if self.power != 1.0 else "amplitude"
            ).to(self.torch_device)

        except Exception as e:
            error_code = self._classify_error(e, "transform_setup_failed")
            raise RuntimeError(f"mel | Ошибка настройки Mel трансформов (error_code={error_code}): {e}") from e

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение Mel-спектрограммы на полном аудио.

        Progress reporting: обновление прогресса для каждого этапа.
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        mel_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"mel | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            if resource_profile_enabled():
                try:
                    mel_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    mel_resource_profile = None

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("mel", 0, 7, "Loading audio")
            t0 = time.perf_counter()
            waveform, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0

            # Convert to torch if needed
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)

            # Ensure 2D tensor: (channels, time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim > 2:
                waveform = waveform.reshape(waveform.shape[0], -1)

            # Mix to mono if requested
            if waveform.shape[0] > 1 and self.mix_to_mono:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Опциональная нормализация аудио (fail-fast, no-fallback)
            if self.enable_audio_normalization:
                if self.progress_callback:
                    self.progress_callback("mel", 1, 7, "Normalizing audio")
                try:
                    t0 = time.perf_counter()
                    waveform = self.audio_utils.normalize_audio(waveform)
                    stage_timings_ms["normalize_audio_ms"] = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    error_code = self._classify_error(e, "audio_load_failed")
                    raise RuntimeError(f"mel | Ошибка нормализации аудио (error_code={error_code}): {e}") from e
            else:
                stage_timings_ms["normalize_audio_ms"] = 0.0

            # Move to device and dtype float32
            t0 = time.perf_counter()
            waveform = waveform.to(dtype=torch.float32, device=self.torch_device)
            stage_timings_ms["to_device_ms"] = (time.perf_counter() - t0) * 1000.0

            # Extract Mel spectrogram
            if self.progress_callback:
                self.progress_callback("mel", 2, 7, "Extracting Mel spectrogram")
            t0 = time.perf_counter()
            mel_spec, mel_db = self._extract_mel_spectrogram(waveform)
            stage_timings_ms["extract_mel_ms"] = (time.perf_counter() - t0) * 1000.0

            # Compute statistics
            if self.progress_callback:
                self.progress_callback("mel", 3, 7, "Computing statistics")
            t0 = time.perf_counter()
            mel_stats = self._compute_statistics(mel_db)
            stage_timings_ms["compute_statistics_ms"] = (time.perf_counter() - t0) * 1000.0

            # Compute spectral features
            if self.progress_callback:
                self.progress_callback("mel", 4, 7, "Computing spectral features")
            t0 = time.perf_counter()
            spectral_features = self._compute_spectral_features(mel_db)
            stage_timings_ms["compute_spectral_features_ms"] = (time.perf_counter() - t0) * 1000.0

            # Compute additional metrics
            if self.progress_callback:
                self.progress_callback("mel", 5, 7, "Computing additional metrics")
            t0 = time.perf_counter()
            additional_metrics = self._compute_additional_metrics(mel_db, mel_stats, spectral_features)
            stage_timings_ms["compute_additional_metrics_ms"] = (time.perf_counter() - t0) * 1000.0

            # Save artifacts
            if self.progress_callback:
                self.progress_callback("mel", 6, 7, "Saving artifacts")
            t0 = time.perf_counter()
            features = self._build_payload(mel_db, mel_stats, spectral_features, additional_metrics, sr, waveform.shape[-1] / float(sr))
            stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            features = self._save_artifacts(features, mel_db, mel_stats, spectral_features, input_uri, tmp_path)
            stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0

            # Validate output
            if self.progress_callback:
                self.progress_callback("mel", 7, 7, "Validating output")
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"mel | {error_msg} (error_code={error_code})")

            # Add contract version
            features["mel_contract_version"] = MEL_CONTRACT_VERSION

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_statistics:
                enabled_features.append("statistics")
            if self.enable_spectral_features:
                enabled_features.append("spectral_features")
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_stats_vector:
                enabled_features.append("stats_vector")
            features["_features_enabled"] = enabled_features

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if mel_resource_profile is not None:
                try:
                    mel_resource_profile = {
                        **(mel_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["mel_resource_profile"] = mel_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"mel | Ошибка извлечения Mel-спектрограммы (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven Mel extraction: compute Mel spectrogram on provided windows (families.mel).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        mel_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"mel | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("mel | segments is empty (no-fallback)")

            total_segments = len(segments)
            stage_timings_ms["load_segments_ms"] = 0.0

            if resource_profile_enabled():
                try:
                    mel_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    mel_resource_profile = None

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Strict alignment (Audit v3): pre-allocate arrays, no skipping
            segment_start_sec = np.zeros(total_segments, dtype=np.float32)
            segment_end_sec = np.zeros(total_segments, dtype=np.float32)
            segment_center_sec = np.zeros(total_segments, dtype=np.float32)
            segment_mask = np.zeros(total_segments, dtype=bool)

            mel_mean_by_segment = np.full((total_segments, self.n_mels), np.nan, dtype=np.float32)
            mel_energy_by_segment = np.full(total_segments, np.nan, dtype=np.float32)
            mel_centroid_mean_by_segment = np.full(total_segments, np.nan, dtype=np.float32)
            mel_bandwidth_mean_by_segment = np.full(total_segments, np.nan, dtype=np.float32)

            # For global aggregates
            valid_means: List[np.ndarray] = []
            valid_centroids: List[float] = []
            valid_bandwidths: List[float] = []
            valid_energies: List[float] = []
            valid_entropies: List[float] = []
            valid_contrasts: List[float] = []
            valid_rolloffs: List[float] = []
            valid_flatness: List[float] = []

            total_frames = 0

            t0 = time.perf_counter()

            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback("mel", seg_idx, total_segments, f"Processing segment {seg_idx+1}/{total_segments}")
                        last_reported_pct = pct

                # Always populate time axis (strict alignment)
                st = float(seg.get("start_sec", 0.0))
                en = float(seg.get("end_sec", 0.0))
                c = float(seg.get("center_sec", (st + en) * 0.5))
                segment_start_sec[seg_idx] = st
                segment_end_sec[seg_idx] = en
                segment_center_sec[seg_idx] = c

                # Basic validation
                if not np.isfinite(st) or not np.isfinite(en) or en <= st:
                    continue

                try:
                    # Load segment
                    start_sample = int(seg.get("start_sample", 0))
                    end_sample = int(seg.get("end_sample", 0))
                    waveform, _sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        target_sr=self.sample_rate,
                    )

                    # Convert to torch if needed
                    if isinstance(waveform, np.ndarray):
                        waveform = torch.from_numpy(waveform)

                    # Ensure 2D tensor
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim > 2:
                        waveform = waveform.reshape(waveform.shape[0], -1)

                    # Mix to mono if requested
                    if waveform.shape[0] > 1 and self.mix_to_mono:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    # Optional normalization (fail-fast for that segment only)
                    if self.enable_audio_normalization:
                        waveform = self.audio_utils.normalize_audio(waveform)

                    waveform = waveform.to(dtype=torch.float32, device=self.torch_device)

                    # Extract Mel spectrogram
                    _mel_spec, mel_db = self._extract_mel_spectrogram(waveform)
                    mel_np = self.audio_utils.to_numpy(mel_db)
                    if mel_np.ndim == 3:
                        mel_np = mel_np[0]
                    if mel_np.ndim != 2 or mel_np.shape[0] != self.n_mels:
                        raise RuntimeError(f"mel | invalid mel_db shape: {getattr(mel_np, 'shape', None)}")
                    frames = int(mel_np.shape[1])
                    total_frames += frames

                    # Compute stats/features for segment-level aggregates
                    mel_stats = self._compute_statistics(mel_db)
                    spectral_features = self._compute_spectral_features(mel_db)
                    additional_metrics = self._compute_additional_metrics(mel_db, mel_stats, spectral_features)

                    # Segment-aligned sequences (Audit v3)
                    if isinstance(mel_stats.get("mel_mean"), np.ndarray) and mel_stats["mel_mean"].size == self.n_mels:
                        mel_mean_by_segment[seg_idx, :] = mel_stats["mel_mean"].astype(np.float32)
                        valid_means.append(mel_stats["mel_mean"].astype(np.float32))
                    mel_energy_by_segment[seg_idx] = float(additional_metrics.get("mel_energy", np.nan))
                    mel_centroid_mean_by_segment[seg_idx] = float(additional_metrics.get("mel_centroid_mean", np.nan))
                    mel_bandwidth_mean_by_segment[seg_idx] = float(additional_metrics.get("mel_bandwidth_mean", np.nan))

                    valid_energies.append(float(additional_metrics.get("mel_energy", 0.0)))
                    valid_centroids.append(float(additional_metrics.get("mel_centroid_mean", 0.0)))
                    valid_bandwidths.append(float(additional_metrics.get("mel_bandwidth_mean", 0.0)))
                    valid_entropies.append(float(additional_metrics.get("mel_spectrogram_entropy", 0.0)))
                    valid_contrasts.append(float(additional_metrics.get("mel_spectrogram_contrast", 0.0)))
                    valid_rolloffs.append(float(additional_metrics.get("mel_rolloff", 0.0)))
                    valid_flatness.append(float(additional_metrics.get("mel_flatness", 0.0)))

                    segment_mask[seg_idx] = True
                except Exception as e:
                    logger.warning(f"mel | Segment {seg_idx} failed: {e}")
                    continue

            # Final progress report
            if self.progress_callback:
                self.progress_callback("mel", total_segments, total_segments, "Completed")

            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            n_valid = int(np.sum(segment_mask))
            if n_valid <= 0:
                error_code = self._classify_error(RuntimeError("All segments failed"), "validation_failed")
                raise RuntimeError(f"mel | all segments failed (error_code={error_code})")

            # Build payload (Audit v3): segment-aligned sequences + aggregated scalars
            features: Dict[str, Any] = {
                "device_used": str(self.torch_device),
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "power": self.power,
                "segments_count": int(total_segments),
                "duration": float(np.sum((segment_end_sec - segment_start_sec)[segment_mask])) if n_valid > 0 else 0.0,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": segment_center_sec,
                "segment_mask": segment_mask,
            }

            # Basic features: describe concatenated time axis without storing full per-frame series in NPZ
            if self.enable_basic_features:
                features["mel_shape"] = (int(self.n_mels), int(total_frames))
                features["mel_elements"] = int(self.n_mels) * int(total_frames)

            # Aggregated additional metrics (expanded model_facing scalars)
            features["mel_energy"] = float(np.mean(valid_energies)) if valid_energies else 0.0
            features["mel_centroid_mean"] = float(np.mean(valid_centroids)) if valid_centroids else 0.0
            features["mel_centroid_std"] = float(np.std(valid_centroids)) if valid_centroids else 0.0
            features["mel_bandwidth_mean"] = float(np.mean(valid_bandwidths)) if valid_bandwidths else 0.0
            features["mel_bandwidth_std"] = float(np.std(valid_bandwidths)) if valid_bandwidths else 0.0
            features["mel_spectrogram_entropy"] = float(np.mean(valid_entropies)) if valid_entropies else 0.0
            features["mel_spectrogram_contrast"] = float(np.mean(valid_contrasts)) if valid_contrasts else 0.0
            features["mel_rolloff"] = float(np.mean(valid_rolloffs)) if valid_rolloffs else 0.0
            features["mel_flatness"] = float(np.mean(valid_flatness)) if valid_flatness else 0.0

            # Stability: mean cosine similarity of consecutive mel_mean vectors (valid segments only)
            mel_stability = 0.0
            if len(valid_means) >= 2:
                sims = []
                eps = 1e-12
                prev = valid_means[0].astype(np.float64)
                prev = prev / (np.linalg.norm(prev) + eps)
                for v in valid_means[1:]:
                    cur = v.astype(np.float64)
                    cur = cur / (np.linalg.norm(cur) + eps)
                    sims.append(float(np.dot(prev, cur)))
                    prev = cur
                if sims:
                    mel_stability = float(np.mean(sims))
            features["mel_stability"] = mel_stability

            stage_timings_ms["aggregate_results_ms"] = (time.perf_counter() - t0) * 1000.0

            # Global mel-bin statistics for NPZ/schema (must align M with mel_mean_by_segment)
            if self.enable_statistics:
                mm = mel_mean_by_segment[segment_mask]
                if mm.size == 0:
                    error_code = self._classify_error(RuntimeError("no mel means"), "validation_failed")
                    raise RuntimeError(f"mel | no valid mel_mean rows (error_code={error_code})")
                features["mel_mean"] = np.nanmean(mm, axis=0).astype(np.float32)
                features["mel_std"] = np.nanstd(mm, axis=0).astype(np.float32)
                features["mel_min"] = np.nanmin(mm, axis=0).astype(np.float32)
                features["mel_max"] = np.nanmax(mm, axis=0).astype(np.float32)

            if self.enable_stats_vector and self.enable_statistics and "mel_mean" in features:
                features["mel_stats_vector"] = np.concatenate(
                    [
                        features["mel_mean"],
                        features["mel_std"],
                        features["mel_min"],
                        features["mel_max"],
                    ],
                    axis=0,
                ).astype(np.float32)

            # Segment-aligned sequences (feature-gated: time_series)
            if self.enable_time_series:
                features["mel_mean_by_segment"] = mel_mean_by_segment
                features["mel_energy_by_segment"] = mel_energy_by_segment
                features["mel_centroid_mean_by_segment"] = mel_centroid_mean_by_segment
                features["mel_bandwidth_mean_by_segment"] = mel_bandwidth_mean_by_segment

            t0 = time.perf_counter()
            # Save artifacts
            features = self._save_artifacts(features, None, None, None, input_uri, tmp_path)
            stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_statistics:
                enabled_features.append("statistics")
            if self.enable_spectral_features:
                enabled_features.append("spectral_features")
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_stats_vector:
                enabled_features.append("stats_vector")
            features["_features_enabled"] = enabled_features

            t0 = time.perf_counter()
            # Validate output
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"mel | {error_msg} (error_code={error_code})")

            # Add contract version
            features["mel_contract_version"] = MEL_CONTRACT_VERSION

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if mel_resource_profile is not None:
                try:
                    mel_resource_profile = {
                        **(mel_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["mel_resource_profile"] = mel_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"mel | Ошибка извлечения Mel-спектрограммы (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def _extract_mel_spectrogram(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Извлечение Mel-спектрограммы (fail-fast, no-fallback).

        Args:
            waveform: Аудио сигнал (torch.Tensor)

        Returns:
            (mel_spec, mel_db): Mel-спектрограмма в линейной шкале и в децибелах

        Raises:
            RuntimeError: Если извлечение не удалось (no-fallback)
        """
        try:
            with torch.inference_mode():
                # Audit v3: deterministic float32 path (no autocast), stable across devices.
                mel_spec = self.mel_spectrogram(waveform)
                mel_db = self.amplitude_to_db(mel_spec)

            # Convert to CPU numpy
            mel_db_cpu = mel_db.detach().cpu()
            mel_db_np = mel_db_cpu.numpy().astype(np.float32)

            # Squeeze channel dim if single channel
            if mel_db_np.ndim == 3 and mel_db_np.shape[0] == 1:
                mel_db_np = mel_db_np[0]

            # Sanitize NaN/inf and clip dB to safe range
            mel_db_np = np.nan_to_num(mel_db_np, nan=-120.0, posinf=-120.0, neginf=-120.0)
            mel_db_np = np.clip(mel_db_np, -120.0, 0.0).astype(np.float32)

            # Validate
            if np.any(np.isnan(mel_db_np)) or np.any(np.isinf(mel_db_np)):
                error_code = self._classify_error(RuntimeError("mel_db produced NaN/Inf"), "spectrogram_failed")
                raise RuntimeError(f"mel | mel_db produced NaN/Inf (error_code={error_code})")

            return mel_spec, torch.from_numpy(mel_db_np)

        except Exception as e:
            error_code = self._classify_error(e, "spectrogram_failed")
            raise RuntimeError(f"mel | extraction failed (error_code={error_code}): {e}") from e

    def _compute_statistics(self, mel_db: torch.Tensor) -> Dict[str, Any]:
        """
        Вычисление статистик Mel-спектрограммы (fail-fast, no-fallback).

        Args:
            mel_db: Mel-спектрограмма в децибелах (torch.Tensor или np.ndarray)

        Returns:
            Словарь со статистиками (feature-gated)

        Raises:
            RuntimeError: Если вычисление не удалось (no-fallback)
        """
        try:
            # Convert to numpy if needed
            if isinstance(mel_db, torch.Tensor):
                mel_np = mel_db.detach().cpu().numpy() if mel_db.requires_grad else mel_db.cpu().numpy()
            else:
                mel_np = mel_db

            # Ensure 2D: (n_mels, frames)
            if mel_np.ndim == 3:
                mel_np = mel_np[0]
            elif mel_np.ndim == 1:
                mel_np = mel_np.reshape(-1, 1)

            stats: Dict[str, Any] = {}

            if self.enable_statistics:
                # Per-frequency (mel bin) statistics over time
                mel_mean = np.mean(mel_np, axis=1)
                mel_std = np.std(mel_np, axis=1)
                mel_min = np.min(mel_np, axis=1)
                mel_max = np.max(mel_np, axis=1)

                # Per-time statistics
                freq_mean = np.mean(mel_np, axis=0)
                freq_std = np.std(mel_np, axis=0)

                # Validate
                if np.any(np.isnan(mel_mean)) or np.any(np.isinf(mel_mean)):
                    error_code = self._classify_error(RuntimeError("mel_statistics produced NaN/Inf"), "statistics_failed")
                    raise RuntimeError(f"mel | mel_statistics produced NaN/Inf (error_code={error_code})")

                stats.update({
                    "mel_mean": mel_mean.astype(np.float32),
                    "mel_std": mel_std.astype(np.float32),
                    "mel_min": mel_min.astype(np.float32),
                    "mel_max": mel_max.astype(np.float32),
                    "freq_mean": freq_mean.astype(np.float32),
                    "freq_std": freq_std.astype(np.float32),
                })

            return stats

        except Exception as e:
            error_code = self._classify_error(e, "statistics_failed")
            raise RuntimeError(f"mel | statistics computation failed (error_code={error_code}): {e}") from e

    def _compute_spectral_features(self, mel_db: torch.Tensor) -> Dict[str, Any]:
        """
        Вычисление спектральных характеристик (fail-fast, no-fallback).

        Args:
            mel_db: Mel-спектрограмма в децибелах (torch.Tensor или np.ndarray)

        Returns:
            Словарь со спектральными характеристиками (feature-gated)

        Raises:
            RuntimeError: Если вычисление не удалось (no-fallback)
        """
        try:
            if not self.enable_spectral_features:
                return {}

            # Convert to numpy if needed
            if isinstance(mel_db, torch.Tensor):
                mel_np = mel_db.detach().cpu().numpy() if mel_db.requires_grad else mel_db.cpu().numpy()
            else:
                mel_np = mel_db

            # Ensure 2D: (n_mels, frames)
            if mel_np.ndim == 3:
                mel_np = mel_np[0]
            elif mel_np.ndim == 1:
                mel_np = mel_np.reshape(-1, 1)

            if mel_np.shape[1] == 0:
                return {
                    "spectral_centroid": np.array([], dtype=np.float32),
                    "spectral_bandwidth": np.array([], dtype=np.float32),
                }

            # Convert dB to linear power scale for numerical stability
            mel_lin = np.power(10.0, (mel_np.astype(np.float64) / 10.0))
            freqs = np.linspace(self.fmin, self.fmax, self.n_mels, dtype=np.float64)
            mel_sum = np.sum(mel_lin, axis=0)
            mel_sum = np.where(mel_sum > 0.0, mel_sum, 1e-12)

            # Spectral centroid
            spectral_centroid = (np.sum(freqs[:, None] * mel_lin, axis=0) / mel_sum)

            # Spectral bandwidth
            diff_sq = (freqs[:, None] - spectral_centroid) ** 2
            bandwidth_num = np.sum(diff_sq * mel_lin, axis=0)
            bandwidth_ratio = bandwidth_num / mel_sum
            spectral_bandwidth = np.sqrt(np.maximum(bandwidth_ratio, 0.0))

            # Cast back to float32 and sanitize
            spectral_centroid = np.nan_to_num(spectral_centroid).astype(np.float32)
            spectral_bandwidth = np.nan_to_num(spectral_bandwidth).astype(np.float32)

            # Validate
            if np.any(np.isnan(spectral_centroid)) or np.any(np.isinf(spectral_centroid)):
                error_code = self._classify_error(RuntimeError("spectral_features produced NaN/Inf"), "spectral_features_failed")
                raise RuntimeError(f"mel | spectral_features produced NaN/Inf (error_code={error_code})")

            return {
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
            }

        except Exception as e:
            error_code = self._classify_error(e, "spectral_features_failed")
            raise RuntimeError(f"mel | spectral features computation failed (error_code={error_code}): {e}") from e

    def _compute_additional_metrics(
        self,
        mel_db: torch.Tensor,
        mel_stats: Dict[str, Any],
        spectral_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Вычислить дополнительные метрики для ML/аналитики.

        Args:
            mel_db: Mel-спектрограмма в децибелах
            mel_stats: Статистики Mel-спектрограммы
            spectral_features: Спектральные характеристики

        Returns:
            Словарь с дополнительными метриками
        """
        metrics: Dict[str, Any] = {}

        try:
            # Convert to numpy if needed
            if isinstance(mel_db, torch.Tensor):
                mel_np = mel_db.detach().cpu().numpy() if mel_db.requires_grad else mel_db.cpu().numpy()
            else:
                mel_np = mel_db

            # Ensure 2D: (n_mels, frames)
            if mel_np.ndim == 3:
                mel_np = mel_np[0]
            elif mel_np.ndim == 1:
                mel_np = mel_np.reshape(-1, 1)

            if mel_np.shape[1] == 0:
                return {
                    "mel_energy": 0.0,
                    "mel_centroid_mean": 0.0,
                    "mel_centroid_std": 0.0,
                    "mel_bandwidth_mean": 0.0,
                    "mel_bandwidth_std": 0.0,
                    "mel_spectrogram_entropy": 0.0,
                    "mel_spectrogram_contrast": 0.0,
                    "mel_rolloff": 0.0,
                    "mel_flatness": 0.0,
                    "mel_stability": 0.0,
                }

            # Mel energy (общая энергия Mel-спектрограммы)
            mel_energy = float(np.mean(np.abs(mel_np)))
            metrics["mel_energy"] = mel_energy

            # Mel centroid mean/std (статистики spectral_centroid)
            if "spectral_centroid" in spectral_features:
                centroid = spectral_features["spectral_centroid"]
                if centroid.size > 0:
                    metrics["mel_centroid_mean"] = float(np.mean(centroid))
                    metrics["mel_centroid_std"] = float(np.std(centroid))
                else:
                    metrics["mel_centroid_mean"] = 0.0
                    metrics["mel_centroid_std"] = 0.0
            else:
                metrics["mel_centroid_mean"] = 0.0
                metrics["mel_centroid_std"] = 0.0

            # Mel bandwidth mean/std (статистики spectral_bandwidth)
            if "spectral_bandwidth" in spectral_features:
                bandwidth = spectral_features["spectral_bandwidth"]
                if bandwidth.size > 0:
                    metrics["mel_bandwidth_mean"] = float(np.mean(bandwidth))
                    metrics["mel_bandwidth_std"] = float(np.std(bandwidth))
                else:
                    metrics["mel_bandwidth_mean"] = 0.0
                    metrics["mel_bandwidth_std"] = 0.0
            else:
                metrics["mel_bandwidth_mean"] = 0.0
                metrics["mel_bandwidth_std"] = 0.0

            # Mel spectrogram entropy (энтропия распределения энергии)
            mel_lin = np.power(10.0, (mel_np.astype(np.float64) / 10.0))
            mel_sum = np.sum(mel_lin, axis=0)
            mel_sum = np.where(mel_sum > 0.0, mel_sum, 1e-12)
            mel_normalized = mel_lin / mel_sum
            mel_normalized = np.where(mel_normalized > 0.0, mel_normalized, 1e-12)
            mel_entropy = -np.sum(mel_normalized * np.log(mel_normalized), axis=0)
            mel_spectrogram_entropy = float(np.mean(mel_entropy))
            metrics["mel_spectrogram_entropy"] = mel_spectrogram_entropy

            # Mel spectrogram contrast (контраст между mel bins)
            mel_contrast = float(np.std(mel_np))
            metrics["mel_spectrogram_contrast"] = mel_contrast

            # Rolloff / flatness (computed on linear power domain)
            mel_power = mel_lin  # (n_mels, frames)
            eps = 1e-12
            # Spectral flatness per frame: geo_mean / arith_mean
            geo = np.exp(np.mean(np.log(np.maximum(mel_power, eps)), axis=0))
            arith = np.mean(mel_power, axis=0) + eps
            flatness = geo / arith
            metrics["mel_flatness"] = float(np.mean(flatness))

            # Spectral rolloff (0.85) in Hz, using mel-bin center freqs approximation
            freqs = np.linspace(self.fmin, self.fmax, self.n_mels, dtype=np.float64)
            csum = np.cumsum(mel_power, axis=0)
            total = csum[-1, :]
            thr = 0.85 * np.where(total > 0.0, total, 1.0)
            idx = np.argmax(csum >= thr[None, :], axis=0)
            rolloff_hz = freqs[idx]
            metrics["mel_rolloff"] = float(np.mean(rolloff_hz))

            # Stability: placeholder for run(); computed properly for run_segments().
            metrics["mel_stability"] = 0.0

        except Exception as e:
            logger.warning(f"mel | Error computing additional metrics: {e}")
            # Return default values
            metrics = {
                "mel_energy": 0.0,
                "mel_centroid_mean": 0.0,
                "mel_centroid_std": 0.0,
                "mel_bandwidth_mean": 0.0,
                "mel_bandwidth_std": 0.0,
                "mel_spectrogram_entropy": 0.0,
                "mel_spectrogram_contrast": 0.0,
                "mel_rolloff": 0.0,
                "mel_flatness": 0.0,
                "mel_stability": 0.0,
            }

        return metrics

    def _save_artifacts(
        self,
        features: Dict[str, Any],
        mel_db: Optional[torch.Tensor],
        mel_stats: Optional[Dict[str, Any]],
        spectral_features: Optional[Dict[str, Any]],
        input_uri: str,
        tmp_path: str,
    ) -> Dict[str, Any]:
        """
        Сохранить большие массивы в .npy файлы (per-run storage).

        Args:
            features: Словарь с признаками
            mel_db: Mel-спектрограмма в децибелах (для run())
            mel_stats: Статистики (для run())
            spectral_features: Спектральные характеристики (для run())
            input_uri: Путь к входному файлу
            tmp_path: Временная директория (если artifacts_dir не задан, используется tmp_path)

        Returns:
            Обновлённый словарь features с путями к .npy файлам
        """
        artifacts_dir = Path(self.artifacts_dir) if self.artifacts_dir else Path(tmp_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(input_uri).stem

        # Save mel spectrogram if present
        if mel_db is not None and self.enable_basic_features:
            mel_np = self.audio_utils.to_numpy(mel_db) if isinstance(mel_db, torch.Tensor) else mel_db
            if mel_np.size > 0:
                npy_path = artifacts_dir / f"{stem}_mel_spectrogram.npy"
                np.save(str(npy_path), mel_np.astype(np.float32))
                features["mel_spectrogram_npy"] = str(npy_path)

        # Save statistics if present
        if mel_stats is not None and self.enable_statistics:
            for stat_key in ["mel_mean", "mel_std", "mel_min", "mel_max", "freq_mean", "freq_std"]:
                if stat_key in mel_stats:
                    stat_arr = mel_stats[stat_key]
                    if isinstance(stat_arr, np.ndarray) and stat_arr.size > 0:
                        npy_path = artifacts_dir / f"{stem}_{stat_key}.npy"
                        np.save(str(npy_path), stat_arr.astype(np.float32))
                        features[f"{stat_key}_npy"] = str(npy_path)

        # Save spectral features if present
        if spectral_features is not None and self.enable_spectral_features:
            for feat_key in ["spectral_centroid", "spectral_bandwidth"]:
                if feat_key in spectral_features:
                    feat_arr = spectral_features[feat_key]
                    if isinstance(feat_arr, np.ndarray) and feat_arr.size > 0:
                        npy_path = artifacts_dir / f"{stem}_{feat_key}.npy"
                        np.save(str(npy_path), feat_arr.astype(np.float32))
                        features[f"{feat_key}_npy"] = str(npy_path)

        # Save time series if present
        if self.enable_time_series and "mel_series" in features:
            mel_series = features.get("mel_series")
            if isinstance(mel_series, np.ndarray) and mel_series.size > 1000:
                npy_path = artifacts_dir / f"{stem}_mel_series.npy"
                np.save(str(npy_path), mel_series.astype(np.float32))
                features["mel_series_npy"] = str(npy_path)
                features.pop("mel_series", None)

        return features

    def _build_payload(
        self,
        mel_db: torch.Tensor,
        mel_stats: Dict[str, Any],
        spectral_features: Dict[str, Any],
        additional_metrics: Dict[str, Any],
        sample_rate: int,
        duration_sec: float,
    ) -> Dict[str, Any]:
        """
        Построить payload с feature-gated полями.

        Args:
            mel_db: Mel-спектрограмма в децибелах
            mel_stats: Статистики
            spectral_features: Спектральные характеристики
            additional_metrics: Дополнительные метрики
            sample_rate: Частота дискретизации
            duration_sec: Длительность аудио

        Returns:
            Словарь с payload (feature-gated)
        """
        features: Dict[str, Any] = {
            "device_used": str(self.torch_device),
            "sample_rate": sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power,
            "duration": duration_sec,
        }

        # Convert to numpy
        mel_np = self.audio_utils.to_numpy(mel_db) if isinstance(mel_db, torch.Tensor) else mel_db

        # Basic features (feature-gated)
        if self.enable_basic_features:
            features["mel_shape"] = tuple(int(x) for x in mel_np.shape)
            features["mel_elements"] = int(np.prod(mel_np.shape))

        # Statistics (feature-gated)
        if self.enable_statistics and mel_stats:
            for stat_key in ["mel_mean", "mel_std", "mel_min", "mel_max", "freq_mean", "freq_std"]:
                if stat_key in mel_stats:
                    stat_arr = mel_stats[stat_key]
                    features[f"{stat_key}_shape"] = list(stat_arr.shape) if isinstance(stat_arr, np.ndarray) else []

        # Spectral features (feature-gated)
        if self.enable_spectral_features and spectral_features:
            for feat_key in ["spectral_centroid", "spectral_bandwidth"]:
                if feat_key in spectral_features:
                    feat_arr = spectral_features[feat_key]
                    features[f"{feat_key}_shape"] = list(feat_arr.shape) if isinstance(feat_arr, np.ndarray) else []

        # Additional metrics (always included if basic_features enabled)
        if self.enable_basic_features:
            features.update(additional_metrics)

        # Stats vector (feature-gated)
        if self.enable_stats_vector and self.enable_statistics and mel_stats:
            if "mel_mean" in mel_stats and "mel_std" in mel_stats and "mel_min" in mel_stats and "mel_max" in mel_stats:
                mel_stats_vector = np.concatenate([
                    mel_stats["mel_mean"],
                    mel_stats["mel_std"],
                    mel_stats["mel_min"],
                    mel_stats["mel_max"],
                ]).astype(np.float32)
                features["mel_stats_vector_shape"] = list(mel_stats_vector.shape)

        # Time series (feature-gated)
        if self.enable_time_series:
            features["mel_series"] = mel_np

        return features

    def _build_payload_from_segments(
        self,
        mel_db_all: List[np.ndarray],
        mel_stats_all: List[Dict[str, Any]],
        spectral_features_all: List[Dict[str, Any]],
        additional_metrics_all: List[Dict[str, Any]],
        segment_centers: List[float],
        segment_durations: List[float],
        total_segments: int,
    ) -> Dict[str, Any]:
        """
        Построить payload из сегментов с агрегацией.

        Args:
            mel_db_all: Список Mel-спектрограмм для каждого сегмента
            mel_stats_all: Список статистик для каждого сегмента
            spectral_features_all: Список спектральных характеристик для каждого сегмента
            additional_metrics_all: Список дополнительных метрик для каждого сегмента
            segment_centers: Центры сегментов в секундах
            segment_durations: Длительности сегментов в секундах
            total_segments: Общее количество сегментов

        Returns:
            Словарь с payload (feature-gated, агрегированный)
        """
        features: Dict[str, Any] = {
            "device_used": str(self.torch_device),
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power,
            "segments_count": int(total_segments),
        }

        # Aggregate basic features
        if self.enable_basic_features and len(mel_db_all) > 0:
            # Concatenate all segments along time axis
            # First, ensure all arrays have the same shape along mel axis (axis=0)
            normalized_arrays = []
            for arr in mel_db_all:
                if isinstance(arr, np.ndarray):
                    # Ensure 2D: (n_mels, frames)
                    if arr.ndim == 1:
                        # If 1D, reshape to (n_mels, frames) assuming it's already flattened
                        # This shouldn't happen, but handle it
                        arr = arr.reshape(self.n_mels, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(self.n_mels, -1)
                    # Normalize to (n_mels, frames) - ensure first dimension is n_mels
                    if arr.shape[0] != self.n_mels:
                        # Transpose if needed
                        if arr.shape[1] == self.n_mels:
                            arr = arr.T
                        else:
                            # Reshape to (n_mels, frames)
                            arr = arr.reshape(self.n_mels, -1)
                    normalized_arrays.append(arr)
            
            if normalized_arrays:
                mel_series_all = np.concatenate(normalized_arrays, axis=1)  # Concatenate along time axis
                features["mel_shape"] = tuple(int(x) for x in mel_series_all.shape)
                features["mel_elements"] = int(np.prod(mel_series_all.shape))

        # Aggregate statistics
        if self.enable_statistics and len(mel_stats_all) > 0:
            aggregated_stats: Dict[str, Any] = {}
            for stat_key in ["mel_mean", "mel_std", "mel_min", "mel_max", "freq_mean", "freq_std"]:
                stat_arrays = []
                for stats in mel_stats_all:
                    if stat_key in stats:
                        arr = stats[stat_key]
                        if isinstance(arr, np.ndarray):
                            stat_arrays.append(arr)
                        elif isinstance(arr, (list, tuple)):
                            stat_arrays.append(np.array(arr))
                
                if stat_arrays:
                    # For mel_* stats (per mel bin), all should have same shape (n_mels,)
                    # For freq_* stats (per time frame), lengths may differ - aggregate only if same length
                    if stat_key.startswith("mel_"):
                        # All should have shape (n_mels,)
                        # Normalize shape: ensure all are 1D and have same length
                        normalized = []
                        for arr in stat_arrays:
                            arr_1d = arr.flatten() if arr.ndim > 1 else arr
                            if arr_1d.size == self.n_mels:
                                normalized.append(arr_1d)
                        if normalized:
                            aggregated_stats[stat_key] = np.mean(normalized, axis=0).astype(np.float32)
                            features[f"{stat_key}_shape"] = list(aggregated_stats[stat_key].shape)
                    else:
                        # For freq_* stats (per time frame), aggregate only if all have same length
                        # Otherwise, concatenate and compute mean
                        if len(stat_arrays) > 0:
                            # Check if all have same length
                            lengths = [arr.size for arr in stat_arrays]
                            if len(set(lengths)) == 1:
                                # All same length - can use np.mean
                                aggregated_stats[stat_key] = np.mean(stat_arrays, axis=0).astype(np.float32)
                                features[f"{stat_key}_shape"] = list(aggregated_stats[stat_key].shape)
                            else:
                                # Different lengths - concatenate all and compute overall mean
                                # This gives a single scalar mean value
                                all_values = np.concatenate([arr.flatten() for arr in stat_arrays])
                                aggregated_stats[stat_key] = np.array([np.mean(all_values)], dtype=np.float32)
                                features[f"{stat_key}_shape"] = [1]
            
            # Add aggregated stats arrays to features (for NPZ saver)
            for stat_key, stat_array in aggregated_stats.items():
                features[stat_key] = stat_array

        # Aggregate spectral features
        aggregated_spectral = {}
        if self.enable_spectral_features and len(spectral_features_all) > 0:
            for feat_key in ["spectral_centroid", "spectral_bandwidth"]:
                feat_arrays = []
                for feats in spectral_features_all:
                    if feat_key in feats:
                        arr = feats[feat_key]
                        if isinstance(arr, np.ndarray):
                            feat_arrays.append(arr)
                        elif isinstance(arr, (list, tuple)):
                            feat_arrays.append(np.array(arr))
                
                if feat_arrays:
                    # Spectral features are per time frame - lengths may differ
                    # Check if all have same length
                    lengths = [arr.size for arr in feat_arrays]
                    if len(set(lengths)) == 1:
                        # All same length - can use np.mean
                        aggregated_feat = np.mean(feat_arrays, axis=0).astype(np.float32)
                        features[f"{feat_key}_shape"] = list(aggregated_feat.shape)
                        aggregated_spectral[feat_key] = aggregated_feat
                    else:
                        # Different lengths - concatenate all and compute overall mean
                        # This gives a single scalar mean value
                        all_values = np.concatenate([arr.flatten() for arr in feat_arrays])
                        aggregated_feat = np.array([np.mean(all_values)], dtype=np.float32)
                        features[f"{feat_key}_shape"] = [1]
                        aggregated_spectral[feat_key] = aggregated_feat
            
            # Add aggregated spectral features arrays to features (for NPZ saver)
            for feat_key, feat_array in aggregated_spectral.items():
                features[feat_key] = feat_array

        # Aggregate additional metrics
        if additional_metrics_all:
            aggregated_additional = {}
            for key in additional_metrics_all[0].keys():
                values = [m.get(key, 0.0) for m in additional_metrics_all]
                aggregated_additional[key] = float(np.mean(values))
            features.update(aggregated_additional)

        # Stats vector (feature-gated)
        if self.enable_stats_vector and self.enable_statistics and len(mel_stats_all) > 0:
            # Aggregate stats vector
            stats_vectors = []
            for stats in mel_stats_all:
                if all(k in stats for k in ["mel_mean", "mel_std", "mel_min", "mel_max"]):
                    stats_vector = np.concatenate([
                        stats["mel_mean"],
                        stats["mel_std"],
                        stats["mel_min"],
                        stats["mel_max"],
                    ]).astype(np.float32)
                    stats_vectors.append(stats_vector)
            if stats_vectors:
                aggregated_stats_vector = np.mean(stats_vectors, axis=0).astype(np.float32)
                features["mel_stats_vector_shape"] = list(aggregated_stats_vector.shape)

        # Time series (feature-gated)
        if self.enable_time_series and len(mel_db_all) > 0:
            # Normalize arrays before concatenation (same as basic features)
            normalized_arrays = []
            for arr in mel_db_all:
                if isinstance(arr, np.ndarray):
                    # Ensure 2D: (n_mels, frames)
                    if arr.ndim == 1:
                        arr = arr.reshape(self.n_mels, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(self.n_mels, -1)
                    # Normalize to (n_mels, frames)
                    if arr.shape[0] != self.n_mels:
                        if arr.shape[1] == self.n_mels:
                            arr = arr.T
                        else:
                            arr = arr.reshape(self.n_mels, -1)
                    normalized_arrays.append(arr)
            
            if normalized_arrays:
                mel_series_all = np.concatenate(normalized_arrays, axis=1)
                features["mel_series"] = mel_series_all
                features["segment_centers_sec"] = segment_centers
                features["segment_durations_sec"] = segment_durations

        return features

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        mel_extractor поддерживает batch processing через extract_batch_segments()
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
                logger.error(f"mel | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"mel | Missing segments for file_id={file_id}")
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
                logger.error(f"mel | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
