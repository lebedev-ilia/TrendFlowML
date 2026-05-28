"""
SpectralExtractor: извлечение базовых спектральных признаков (centroid, bandwidth, flatness, rolloff, ZCR, contrast, slope).
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
- Batch processing support (CPU parallelism)
"""
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import capture_spectral_resource_profile, is_spectral_resource_profile_enabled

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
SPECTRAL_CONTRACT_VERSION = "spectral_contract_v1"


class SpectralExtractor(BaseExtractor):
    """Извлекает базовые спектральные признаки и их статистики."""

    name = "spectral"
    version = "2.0.1"
    description = "Спектральные признаки: centroid, bandwidth, flatness, rolloff, ZCR, contrast, slope"
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
        n_fft: int = 2048,
        average_channels: bool = True,
        keep_contrast_bands: bool = True,
        # Feature gating flags (Audit v3: basic_features enabled by default)
        enable_basic_features: bool = True,
        enable_contrast: bool = False,
        enable_advanced_features: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация spectral extractor.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            hop_length: Размер шага между кадрами (samples)
            n_fft: Размер окна FFT (samples)
            average_channels: Усреднять ли каналы для многоканального аудио
            keep_contrast_bands: Сохранять ли полные данные контраста по полосам
            enable_basic_features: Включить базовые признаки (centroid, bandwidth, flatness, rolloff, ZCR)
            enable_contrast: Включить контраст (contrast stats + contrast_bands)
            enable_advanced_features: Включить продвинутые признаки (slope, flatness_db)
            enable_time_series: Включить временные серии для каждого признака
            enable_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (feature_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)
        
        # Validate parameters
        self._validate_parameters(sample_rate, hop_length, n_fft)
        
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)
        self.average_channels = bool(average_channels)
        self.keep_contrast_bands = bool(keep_contrast_bands)
        
        # Feature gating flags
        self.enable_basic_features = bool(enable_basic_features)
        self.enable_contrast = bool(enable_contrast)
        self.enable_advanced_features = bool(enable_advanced_features)
        self.enable_time_series = bool(enable_time_series)
        
        # Optional normalization
        self.enable_normalization = bool(enable_normalization)
        
        # Progress callback
        self.progress_callback = progress_callback
        
        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

    def _validate_parameters(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).
        
        Args:
            sample_rate: Частота дискретизации
            hop_length: Размер шага между кадрами
            n_fft: Размер окна FFT
        
        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"spectral | sample_rate must be positive, got {sample_rate}")
        if hop_length <= 0:
            raise ValueError(f"spectral | hop_length must be positive, got {hop_length}")
        if n_fft <= 0:
            raise ValueError(f"spectral | n_fft must be positive, got {n_fft}")
        if n_fft < 512:
            raise ValueError(f"spectral | n_fft ({n_fft}) is too small (minimum 512)")
        if hop_length > n_fft:
            raise ValueError(f"spectral | hop_length ({hop_length}) must be <= n_fft ({n_fft})")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.
        
        Args:
            error: Исключение
            context: Контекст ошибки (audio_load_failed, centroid_failed, bandwidth_failed, flatness_failed, rolloff_failed, zcr_failed, contrast_failed, slope_failed, validation_failed, unknown)
        
        Returns:
            error_code: один из:
                - spectral_audio_load_failed
                - spectral_centroid_failed
                - spectral_bandwidth_failed
                - spectral_flatness_failed
                - spectral_rolloff_failed
                - spectral_zcr_failed
                - spectral_contrast_failed
                - spectral_slope_failed
                - spectral_validation_failed
                - spectral_unknown
        """
        # Сначала проверяем context (более надежно)
        if context == "audio_load_failed":
            return "spectral_audio_load_failed"
        if context == "centroid_failed":
            return "spectral_centroid_failed"
        if context == "bandwidth_failed":
            return "spectral_bandwidth_failed"
        if context == "flatness_failed":
            return "spectral_flatness_failed"
        if context == "rolloff_failed":
            return "spectral_rolloff_failed"
        if context == "zcr_failed":
            return "spectral_zcr_failed"
        if context == "contrast_failed":
            return "spectral_contrast_failed"
        if context == "slope_failed":
            return "spectral_slope_failed"
        if context == "validation_failed":
            return "spectral_validation_failed"
        
        # Затем проверяем строку ошибки (fallback)
        error_str = str(error).lower()
        
        if "audio" in error_str or "load" in error_str:
            return "spectral_audio_load_failed"
        if "centroid" in error_str:
            return "spectral_centroid_failed"
        if "bandwidth" in error_str:
            return "spectral_bandwidth_failed"
        if "flatness" in error_str:
            return "spectral_flatness_failed"
        if "rolloff" in error_str:
            return "spectral_rolloff_failed"
        if "zcr" in error_str or "zero_crossing" in error_str:
            return "spectral_zcr_failed"
        if "slope" in error_str:
            return "spectral_slope_failed"
        if "validation" in error_str or "invalid" in error_str or "no features enabled" in error_str or "empty features" in error_str:
            return "spectral_validation_failed"
        # contrast проверяем последним, чтобы не ловить случайные упоминания
        if "contrast" in error_str:
            return "spectral_contrast_failed"
        
        return "spectral_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.
        
        Args:
            features: Словарь с выходными данными
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "spectral | features must be a dict"
        
        # Validate basic features stats if present
        for feature_name in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
            stats_key = f"{feature_name}_stats"
            if stats_key in features:
                stats = features.get(stats_key)
                if not isinstance(stats, dict):
                    return False, f"spectral | {stats_key} must be a dict"
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = stats.get(stat_key)
                        try:
                            value = float(value)
                            if np.isnan(value) or np.isinf(value):
                                return False, f"spectral | {stats_key}.{stat_key} is NaN or Inf"
                            if value < 0:
                                return False, f"spectral | {stats_key}.{stat_key} must be non-negative, got {value}"
                        except (ValueError, TypeError):
                            return False, f"spectral | {stats_key}.{stat_key} must be float, got {type(value)}"
                
                # Validate consistency: min <= mean <= max
                if all(k in stats for k in ["min", "mean", "max"]):
                    min_val = float(stats.get("min", 0.0))
                    mean_val = float(stats.get("mean", 0.0))
                    max_val = float(stats.get("max", 0.0))
                    if not (min_val <= mean_val <= max_val):
                        return False, f"spectral | {stats_key} consistency check failed: min ({min_val}) <= mean ({mean_val}) <= max ({max_val})"
        
        # Validate flatness (should be in [0, 1])
        if "spectral_flatness_stats" in features:
            stats = features.get("spectral_flatness_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if value < 0.0 or value > 1.0:
                            return False, f"spectral | spectral_flatness_stats.{stat_key} must be in [0, 1], got {value}"
        
        # Validate ZCR (should be in [0, 1])
        if "zcr_stats" in features:
            stats = features.get("zcr_stats")
            if isinstance(stats, dict):
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stats:
                        value = float(stats.get(stat_key))
                        if value < 0.0 or value > 1.0:
                            return False, f"spectral | zcr_stats.{stat_key} must be in [0, 1], got {value}"
        
        # Validate time series if present
        for series_key in ["centroid_series", "bandwidth_series", "flatness_series", "rolloff_series", "zcr_series", "contrast_series", "slope_series"]:
            if series_key in features:
                series = features.get(series_key)
                if series is not None:
                    if isinstance(series, list):
                        series_arr = np.asarray(series, dtype=np.float32)
                        if np.any(np.isnan(series_arr)) or np.any(np.isinf(series_arr)):
                            return False, f"spectral | {series_key} contains NaN or Inf values"
                        if np.any(series_arr < 0) and series_key not in ["slope_series"]:  # slope can be negative
                            return False, f"spectral | {series_key} contains negative values"
        
        return True, None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение спектральных признаков на полном аудио.
        
        Progress reporting: обновление прогресса для каждого признака.
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        spectral_resource_profile: Optional[Dict[str, Any]] = None
        if is_spectral_resource_profile_enabled():
            spectral_resource_profile = {
                "at_start": capture_spectral_resource_profile(stage="at_start"),
            }
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("spectral", 0, 8, "Loading audio")
            t0 = time.perf_counter()
            y_t, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Опциональная нормализация
            if self.enable_normalization:
                t0 = time.perf_counter()
                y_t = self.audio_utils.normalize_audio(y_t)
                stage_timings_ms["normalize_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            duration = float(y.shape[-1] / sr)
            if duration < 1.0:
                nan_f = float("nan")
                payload_empty = {
                    "status": "empty",
                    "empty_reason": "audio_too_short",
                    "device_used": self.device,
                    "sample_rate": sr,
                    "hop_length": self.hop_length,
                    "n_fft": self.n_fft,
                    "duration": duration,
                    "segments_count": 0,
                    "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
                    "segment_start_sec": [0.0],
                    "segment_end_sec": [duration],
                    "segment_center_sec": [0.5 * duration],
                    "segment_mask": [False],
                    "_features_enabled": ["basic_features"] if self.enable_basic_features else [],
                }
                if self.enable_basic_features:
                    payload_empty["centroid_mean_by_segment"] = [nan_f]
                    payload_empty["bandwidth_mean_by_segment"] = [nan_f]
                    payload_empty["flatness_mean_by_segment"] = [nan_f]
                    payload_empty["rolloff_mean_by_segment"] = [nan_f]
                    payload_empty["zcr_mean_by_segment"] = [nan_f]
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload_empty["stage_timings_ms"] = stage_timings_ms
                if spectral_resource_profile is not None:
                    spectral_resource_profile["at_end"] = capture_spectral_resource_profile(stage="at_end")
                    payload_empty["spectral_resource_profile"] = spectral_resource_profile
                return self._create_result(success=True, payload=payload_empty, processing_time=time.time() - start_time)

            # Извлекаем признаки
            t0 = time.perf_counter()
            features = self._extract_spectral_features(y, sr)
            stage_timings_ms["extract_features_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Сохраняем большие временные серии в .npy (per-run storage)
            if self.progress_callback:
                self.progress_callback("spectral", 7, 8, "Saving artifacts")
            t0 = time.perf_counter()
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)
            stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Валидация выходных данных
            if self.progress_callback:
                self.progress_callback("spectral", 8, 8, "Validating output")
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"spectral | {error_msg} (error_code={error_code})")
            
            # Добавляем contract version
            features["spectral_contract_version"] = SPECTRAL_CONTRACT_VERSION
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_contrast:
                enabled_features.append("contrast")
            if self.enable_advanced_features:
                enabled_features.append("advanced_features")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features
            
            # Add stage timings to payload (for meta/stage_timings_ms)
            processing_time = time.time() - start_time
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms
            if spectral_resource_profile is not None:
                spectral_resource_profile["at_end"] = capture_spectral_resource_profile(stage="at_end")
                features["spectral_resource_profile"] = spectral_resource_profile

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"spectral | Ошибка извлечения spectral features (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven spectral extraction: compute spectral features on provided windows (families.spectral).
        Audit v3: strict alignment, canonical axis (segment_start_sec/end/center/mask), per-segment arrays.
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        spectral_resource_profile: Optional[Dict[str, Any]] = None
        if is_spectral_resource_profile_enabled():
            spectral_resource_profile = {
                "at_start": capture_spectral_resource_profile(stage="at_start"),
            }
        nan_f = float("nan")
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"spectral | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("spectral | segments is empty (no-fallback)")

            has_any_features_enabled = (
                self.enable_basic_features or self.enable_contrast or self.enable_advanced_features
            )
            if not has_any_features_enabled:
                error_code = self._classify_error(RuntimeError("No features enabled"), "validation_failed")
                raise RuntimeError(
                    f"spectral | no features enabled (error_code={error_code}). "
                    f"Enable at least one: enable_basic_features, enable_contrast, enable_advanced_features"
                )

            total_segments = len(segments)
            t0_process = time.perf_counter()
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Canonical axis (strict alignment)
            segment_start_sec: List[float] = [0.0] * total_segments
            segment_end_sec: List[float] = [0.0] * total_segments
            segment_center_sec: List[float] = [0.0] * total_segments
            segment_mask: List[bool] = [False] * total_segments

            # Per-segment arrays (NaN for failed)
            centroid_by_seg: List[float] = [nan_f] * total_segments
            bandwidth_by_seg: List[float] = [nan_f] * total_segments
            flatness_by_seg: List[float] = [nan_f] * total_segments
            rolloff_by_seg: List[float] = [nan_f] * total_segments
            zcr_by_seg: List[float] = [nan_f] * total_segments
            contrast_by_seg: List[float] = [nan_f] * total_segments
            slope_by_seg: List[float] = [nan_f] * total_segments

            for seg_idx, seg in enumerate(segments):
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback("spectral", seg_idx, total_segments, f"Processing segment {seg_idx+1}/{total_segments}")
                        last_reported_pct = pct

                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                start_sec = float(seg.get("start_sec", start_sample / self.sample_rate))
                end_sec = float(seg.get("end_sec", end_sample / self.sample_rate))
                center_sec = float(seg.get("center_sec", 0.5 * (start_sec + end_sec) if end_sec > start_sec else start_sec))

                segment_start_sec[seg_idx] = start_sec
                segment_end_sec[seg_idx] = end_sec
                segment_center_sec[seg_idx] = center_sec

                try:
                    wav_t, _sr = self.audio_utils.load_audio_segment(
                        input_uri, start_sample=start_sample, end_sample=end_sample, target_sr=self.sample_rate
                    )
                    if self.enable_normalization:
                        wav_t = self.audio_utils.normalize_audio(wav_t)
                    wav = self.audio_utils.to_numpy(wav_t)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)

                    seg_features = self._extract_spectral_features(wav, self.sample_rate)
                    segment_mask[seg_idx] = True

                    if self.enable_basic_features:
                        s = seg_features.get("spectral_centroid_stats") or {}
                        centroid_by_seg[seg_idx] = float(s.get("mean", nan_f))
                        s = seg_features.get("spectral_bandwidth_stats") or {}
                        bandwidth_by_seg[seg_idx] = float(s.get("mean", nan_f))
                        s = seg_features.get("spectral_flatness_stats") or {}
                        flatness_by_seg[seg_idx] = float(s.get("mean", nan_f))
                        s = seg_features.get("spectral_rolloff_stats") or {}
                        rolloff_by_seg[seg_idx] = float(s.get("mean", nan_f))
                        s = seg_features.get("zcr_stats") or {}
                        zcr_by_seg[seg_idx] = float(s.get("mean", nan_f))
                    if self.enable_contrast:
                        s = seg_features.get("spectral_contrast_stats") or {}
                        contrast_by_seg[seg_idx] = float(s.get("mean", nan_f))
                    if self.enable_advanced_features:
                        s = seg_features.get("spectral_slope_stats") or {}
                        slope_by_seg[seg_idx] = float(s.get("mean", nan_f))
                except Exception:
                    segment_mask[seg_idx] = False

            if self.progress_callback:
                self.progress_callback("spectral", total_segments, total_segments, "Completed")

            n_valid = sum(1 for m in segment_mask if m)
            if n_valid == 0:
                _span = (
                    float(max(segment_end_sec) - min(segment_start_sec))
                    if segment_end_sec and segment_start_sec
                    else 0.0
                )
                payload_empty: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "spectral_all_segments_failed",
                    "device_used": self.device,
                    "sample_rate": self.sample_rate,
                    "hop_length": self.hop_length,
                    "n_fft": self.n_fft,
                    "duration": _span,
                    "segments_count": total_segments,
                    "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
                    "segment_start_sec": segment_start_sec,
                    "segment_end_sec": segment_end_sec,
                    "segment_center_sec": segment_center_sec,
                    "segment_mask": segment_mask,
                    "_features_enabled": ["basic_features"] if self.enable_basic_features else [],
                }
                if self.enable_basic_features:
                    payload_empty["centroid_mean_by_segment"] = centroid_by_seg
                    payload_empty["bandwidth_mean_by_segment"] = bandwidth_by_seg
                    payload_empty["flatness_mean_by_segment"] = flatness_by_seg
                    payload_empty["rolloff_mean_by_segment"] = rolloff_by_seg
                    payload_empty["zcr_mean_by_segment"] = zcr_by_seg
                if self.enable_contrast:
                    payload_empty["contrast_mean_by_segment"] = contrast_by_seg
                if self.enable_advanced_features:
                    payload_empty["slope_mean_by_segment"] = slope_by_seg
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_process) * 1000.0
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload_empty["stage_timings_ms"] = stage_timings_ms
                if spectral_resource_profile is not None:
                    spectral_resource_profile["at_end"] = capture_spectral_resource_profile(stage="at_end")
                    payload_empty["spectral_resource_profile"] = spectral_resource_profile
                return self._create_result(success=True, payload=payload_empty, processing_time=time.time() - start_time)

            # Aggregate stats from valid segments only
            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_process) * 1000.0
            t0_agg = time.perf_counter()
            valid_centroid = np.asarray([v for i, v in enumerate(centroid_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_bandwidth = np.asarray([v for i, v in enumerate(bandwidth_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_flatness = np.asarray([v for i, v in enumerate(flatness_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_rolloff = np.asarray([v for i, v in enumerate(rolloff_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_zcr = np.asarray([v for i, v in enumerate(zcr_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_contrast = np.asarray([v for i, v in enumerate(contrast_by_seg) if segment_mask[i]], dtype=np.float32)
            valid_slope = np.asarray([v for i, v in enumerate(slope_by_seg) if segment_mask[i]], dtype=np.float32)

            _span_ok = (
                float(max(segment_end_sec) - min(segment_start_sec))
                if segment_end_sec and segment_start_sec
                else 0.0
            )
            features: Dict[str, Any] = {
                "status": "ok",
                "empty_reason": None,
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
                "n_fft": self.n_fft,
                "duration": _span_ok,
                "segments_count": int(total_segments),
                "spectral_contract_version": SPECTRAL_CONTRACT_VERSION,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": segment_center_sec,
                "segment_mask": segment_mask,
            }

            if self.enable_basic_features:
                features["spectral_centroid_stats"] = self._calc_stats(valid_centroid)
                features["spectral_bandwidth_stats"] = self._calc_stats(valid_bandwidth)
                features["spectral_flatness_stats"] = self._calc_stats(valid_flatness)
                features["spectral_rolloff_stats"] = self._calc_stats(valid_rolloff)
                features["zcr_stats"] = self._calc_stats(valid_zcr)
                features["centroid_mean_by_segment"] = centroid_by_seg
                features["bandwidth_mean_by_segment"] = bandwidth_by_seg
                features["flatness_mean_by_segment"] = flatness_by_seg
                features["rolloff_mean_by_segment"] = rolloff_by_seg
                features["zcr_mean_by_segment"] = zcr_by_seg
                features.update(self._calc_additional_metrics(valid_centroid, valid_bandwidth, valid_flatness, valid_rolloff, valid_zcr))

            if self.enable_contrast:
                features["contrast_mean_by_segment"] = contrast_by_seg
                if valid_contrast.size > 0:
                    features["spectral_contrast_stats"] = self._calc_stats(valid_contrast)
                    features["spectral_contrast_variance"] = float(np.var(valid_contrast))

            if self.enable_advanced_features:
                features["slope_mean_by_segment"] = slope_by_seg
                if valid_slope.size > 0:
                    features["spectral_slope_stats"] = self._calc_stats(valid_slope)
                    features["spectral_slope_stability"] = float(1.0 / (1.0 + np.std(valid_slope)))

            enabled_features = []
            if self.enable_basic_features:
                enabled_features.append("basic_features")
            if self.enable_contrast:
                enabled_features.append("contrast")
            if self.enable_advanced_features:
                enabled_features.append("advanced_features")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            processing_time = time.time() - start_time
            stage_timings_ms["aggregate_results_ms"] = (time.perf_counter() - t0_agg) * 1000.0
            t0_val = time.perf_counter()

            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0_val) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"spectral | {error_msg} (error_code={error_code})")

            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms
            if spectral_resource_profile is not None:
                spectral_resource_profile["at_end"] = capture_spectral_resource_profile(stage="at_end")
                features["spectral_resource_profile"] = spectral_resource_profile
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"spectral | Ошибка извлечения spectral features (error_code={error_code}): {e}"
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
        for series_key in ["centroid_series", "bandwidth_series", "flatness_series", "rolloff_series", "zcr_series", "contrast_series", "slope_series"]:
            series = features.get(series_key)
            if isinstance(series, list) and len(series) > 1000:  # Save if > 1000 elements
                npy_path = artifacts_dir / f"{stem}_{series_key}.npy"
                np.save(str(npy_path), np.asarray(series, dtype=np.float32))
                features[f"{series_key}_npy"] = str(npy_path)
                # Убираем саму серию из JSON (если не включена time_series)
                if not self.enable_time_series:
                    features.pop(series_key, None)
        
        return features

    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Извлечение спектральных признаков с использованием librosa (no-fallback policy).
        
        Args:
            audio: Аудио сигнал (моно, numpy array)
            sr: Частота дискретизации
        
        Returns:
            Словарь с признаками (feature-gated)
        
        Raises:
            RuntimeError: Если признак не может быть вычислен (no-fallback)
        """
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": sr,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "average_channels": self.average_channels,
            "keep_contrast_bands": self.keep_contrast_bands,
            "duration": float(audio.shape[-1] / sr),
        }

        # Basic features (fail-fast, no-fallback)
        if self.enable_basic_features:
            if self.progress_callback:
                self.progress_callback("spectral", 1, 8, "Computing centroid")
            try:
                centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
                if centroid.size == 0 or np.all(np.isnan(centroid)) or np.all(centroid <= 0):
                    error_code = self._classify_error(RuntimeError("centroid produced empty/invalid output"), "centroid_failed")
                    raise RuntimeError(f"spectral | centroid produced empty/invalid output (error_code={error_code})")
                features["spectral_centroid_stats"] = self._calc_stats(centroid.astype(np.float32))
                if self.enable_time_series:
                    features["centroid_series"] = centroid.astype(np.float32).tolist()
            except Exception as e:
                error_code = self._classify_error(e, "centroid_failed")
                raise RuntimeError(f"spectral | centroid failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("spectral", 2, 8, "Computing bandwidth")
            try:
                bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
                if bandwidth.size == 0 or np.all(np.isnan(bandwidth)) or np.all(bandwidth <= 0):
                    error_code = self._classify_error(RuntimeError("bandwidth produced empty/invalid output"), "bandwidth_failed")
                    raise RuntimeError(f"spectral | bandwidth produced empty/invalid output (error_code={error_code})")
                features["spectral_bandwidth_stats"] = self._calc_stats(bandwidth.astype(np.float32))
                if self.enable_time_series:
                    features["bandwidth_series"] = bandwidth.astype(np.float32).tolist()
            except Exception as e:
                error_code = self._classify_error(e, "bandwidth_failed")
                raise RuntimeError(f"spectral | bandwidth failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("spectral", 3, 8, "Computing flatness")
            try:
                flatness = librosa.feature.spectral_flatness(y=audio, n_fft=self.n_fft, hop_length=self.hop_length)[0]
                if flatness.size == 0 or np.all(np.isnan(flatness)) or np.any(flatness < 0) or np.any(flatness > 1):
                    error_code = self._classify_error(RuntimeError("flatness produced empty/invalid output"), "flatness_failed")
                    raise RuntimeError(f"spectral | flatness produced empty/invalid output (error_code={error_code})")
                features["spectral_flatness_stats"] = self._calc_stats(flatness.astype(np.float32))
                if self.enable_time_series:
                    features["flatness_series"] = flatness.astype(np.float32).tolist()
            except Exception as e:
                error_code = self._classify_error(e, "flatness_failed")
                raise RuntimeError(f"spectral | flatness failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("spectral", 4, 8, "Computing rolloff")
            try:
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
                if rolloff.size == 0 or np.all(np.isnan(rolloff)) or np.all(rolloff <= 0):
                    error_code = self._classify_error(RuntimeError("rolloff produced empty/invalid output"), "rolloff_failed")
                    raise RuntimeError(f"spectral | rolloff produced empty/invalid output (error_code={error_code})")
                features["spectral_rolloff_stats"] = self._calc_stats(rolloff.astype(np.float32))
                if self.enable_time_series:
                    features["rolloff_series"] = rolloff.astype(np.float32).tolist()
            except Exception as e:
                error_code = self._classify_error(e, "rolloff_failed")
                raise RuntimeError(f"spectral | rolloff failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("spectral", 5, 8, "Computing ZCR")
            try:
                zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)[0]
                if zcr.size == 0 or np.all(np.isnan(zcr)) or np.any(zcr < 0) or np.any(zcr > 1):
                    error_code = self._classify_error(RuntimeError("zcr produced empty/invalid output"), "zcr_failed")
                    raise RuntimeError(f"spectral | zcr produced empty/invalid output (error_code={error_code})")
                features["zcr_stats"] = self._calc_stats(zcr.astype(np.float32))
                if self.enable_time_series:
                    features["zcr_series"] = zcr.astype(np.float32).tolist()
            except Exception as e:
                error_code = self._classify_error(e, "zcr_failed")
                raise RuntimeError(f"spectral | zcr failed (error_code={error_code}): {e}") from e

            # Additional ML/analytics metrics
            features.update(self._calc_additional_metrics(
                centroid if self.enable_basic_features else np.array([]),
                bandwidth if self.enable_basic_features else np.array([]),
                flatness if self.enable_basic_features else np.array([]),
                rolloff if self.enable_basic_features else np.array([]),
                zcr if self.enable_basic_features else np.array([]),
            ))

        # Contrast (fail-fast, no-fallback)
        if self.enable_contrast:
            if self.progress_callback:
                self.progress_callback("spectral", 6, 8, "Computing contrast")
            try:
                contrast_full = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
                if contrast_full.size == 0 or np.all(np.isnan(contrast_full)):
                    error_code = self._classify_error(RuntimeError("contrast produced empty/invalid output"), "contrast_failed")
                    raise RuntimeError(f"spectral | contrast produced empty/invalid output (error_code={error_code})")
                contrast = contrast_full.mean(axis=0)
                features["spectral_contrast_stats"] = self._calc_stats(contrast.astype(np.float32))
                if self.keep_contrast_bands:
                    features["spectral_contrast_bands"] = contrast_full.astype(np.float32).tolist()
                if self.enable_time_series:
                    features["contrast_series"] = contrast.astype(np.float32).tolist()
                # Additional metrics
                features["spectral_contrast_variance"] = float(np.var(contrast))
            except Exception as e:
                error_code = self._classify_error(e, "contrast_failed")
                raise RuntimeError(f"spectral | contrast failed (error_code={error_code}): {e}") from e

        # Advanced features (fail-fast, no-fallback)
        if self.enable_advanced_features:
            if self.progress_callback:
                self.progress_callback("spectral", 7, 8, "Computing slope and flatness_db")
            try:
                S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)) + 1e-12
                S_db = 20.0 * np.log10(S)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
                x_f = (freqs - freqs.mean()) / (freqs.std() + 1e-12)
                x_centered = x_f[:, None]
                y_centered = S_db - S_db.mean(axis=0, keepdims=True)
                num = np.sum(x_centered * y_centered, axis=0)
                den = np.sum(x_centered * x_centered, axis=0) + 1e-12
                spectral_slope = (num / den).astype(np.float32)
                if spectral_slope.size == 0 or np.all(np.isnan(spectral_slope)):
                    error_code = self._classify_error(RuntimeError("slope produced empty/invalid output"), "slope_failed")
                    raise RuntimeError(f"spectral | slope produced empty/invalid output (error_code={error_code})")
                features["spectral_slope_stats"] = self._calc_stats(spectral_slope)
                if self.enable_time_series:
                    features["slope_series"] = spectral_slope.tolist()
                # Additional metrics
                features["spectral_slope_stability"] = float(1.0 / (1.0 + np.std(spectral_slope)))
            except Exception as e:
                error_code = self._classify_error(e, "slope_failed")
                raise RuntimeError(f"spectral | slope failed (error_code={error_code}): {e}") from e

            try:
                if self.enable_basic_features and "spectral_flatness_stats" in features:
                    flatness_mean = features["spectral_flatness_stats"].get("mean", 0.0)
                    spectral_flatness_db = (10.0 * np.log10(flatness_mean + 1e-12))
                    features["spectral_flatness_db_stats"] = self._calc_stats(np.array([spectral_flatness_db], dtype=np.float32))
            except Exception as e:
                error_code = self._classify_error(e, "flatness_failed")
                raise RuntimeError(f"spectral | flatness_db failed (error_code={error_code}): {e}") from e

        return features

    def _calc_stats(self, x: np.ndarray) -> Dict[str, float]:
        """
        Вычислить статистики для массива.
        Audit v3: NaN для пустого массива (no zero placeholders).
        
        Args:
            x: Массив значений
        
        Returns:
            Словарь со статистиками (mean, std, min, max, median)
        """
        nan_f = float("nan")
        if x.size == 0:
            return {
                "mean": nan_f,
                "std": nan_f,
                "min": nan_f,
                "max": nan_f,
                "median": nan_f,
            }
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": float(np.median(x)),
        }

    def _calc_additional_metrics(
        self,
        centroid: np.ndarray,
        bandwidth: np.ndarray,
        flatness: np.ndarray,
        rolloff: np.ndarray,
        zcr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Вычислить дополнительные метрики для ML/аналитики.
        
        Args:
            centroid: Массив значений центроида
            bandwidth: Массив значений bandwidth
            flatness: Массив значений flatness
            rolloff: Массив значений rolloff
            zcr: Массив значений ZCR
        
        Returns:
            Словарь с дополнительными метриками
        """
        metrics: Dict[str, Any] = {}
        nan_f = float("nan")
        if centroid.size == 0:
            return {
                "spectral_centroid_median": nan_f,
                "spectral_bandwidth_ratio": nan_f,
                "spectral_rolloff_ratio": nan_f,
                "spectral_flatness_entropy": nan_f,
                "spectral_features_correlation": {},
            }
        
        # Spectral centroid median
        metrics["spectral_centroid_median"] = float(np.median(centroid))
        
        # Spectral bandwidth ratio (bandwidth / centroid, relative width)
        if np.mean(centroid) > 0:
            metrics["spectral_bandwidth_ratio"] = float(np.mean(bandwidth) / np.mean(centroid))
        else:
            metrics["spectral_bandwidth_ratio"] = 0.0
        
        # Spectral rolloff ratio (rolloff / sample_rate, relative rolloff)
        if self.sample_rate > 0:
            metrics["spectral_rolloff_ratio"] = float(np.mean(rolloff) / self.sample_rate)
        else:
            metrics["spectral_rolloff_ratio"] = 0.0
        
        # Spectral flatness entropy (entropy of flatness distribution)
        if flatness.size > 0:
            # Normalize flatness to probabilities
            flatness_norm = flatness / (np.sum(flatness) + 1e-12)
            flatness_norm = flatness_norm[flatness_norm > 0]
            if flatness_norm.size > 0:
                metrics["spectral_flatness_entropy"] = float(-np.sum(flatness_norm * np.log2(flatness_norm + 1e-12)))
            else:
                metrics["spectral_flatness_entropy"] = 0.0
        else:
            metrics["spectral_flatness_entropy"] = 0.0
        
        # Spectral features correlation
        if all(arr.size > 1 for arr in [centroid, bandwidth, flatness, rolloff, zcr]):
            try:
                # Stack features
                feature_matrix = np.stack([
                    centroid[:min(len(centroid), len(bandwidth), len(flatness), len(rolloff), len(zcr))],
                    bandwidth[:min(len(centroid), len(bandwidth), len(flatness), len(rolloff), len(zcr))],
                    flatness[:min(len(centroid), len(bandwidth), len(flatness), len(rolloff), len(zcr))],
                    rolloff[:min(len(centroid), len(bandwidth), len(flatness), len(rolloff), len(zcr))],
                    zcr[:min(len(centroid), len(bandwidth), len(flatness), len(rolloff), len(zcr))],
                ], axis=0)
                # Compute correlation matrix
                corr_matrix = np.corrcoef(feature_matrix)
                metrics["spectral_features_correlation"] = {
                    "centroid_bandwidth": float(corr_matrix[0, 1]),
                    "centroid_flatness": float(corr_matrix[0, 2]),
                    "centroid_rolloff": float(corr_matrix[0, 3]),
                    "centroid_zcr": float(corr_matrix[0, 4]),
                    "bandwidth_flatness": float(corr_matrix[1, 2]),
                    "bandwidth_rolloff": float(corr_matrix[1, 3]),
                    "bandwidth_zcr": float(corr_matrix[1, 4]),
                    "flatness_rolloff": float(corr_matrix[2, 3]),
                    "flatness_zcr": float(corr_matrix[2, 4]),
                    "rolloff_zcr": float(corr_matrix[3, 4]),
                }
            except Exception:
                metrics["spectral_features_correlation"] = {}
        else:
            metrics["spectral_features_correlation"] = {}
        
        return metrics
    
    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        spectral_extractor поддерживает CPU parallelism для обработки сегментов из нескольких видео.
        """
        return True
    
    def extract_batch_segments(
        self,
        audio_files_with_segments: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
        max_segments_per_batch: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких видео с CPU parallelism.
        
        Для CPU extractors используется ThreadPoolExecutor для параллельной обработки
        сегментов из разных видео. Это позволяет ускорить обработку на многоядерных CPU.
        
        Args:
            audio_files_with_segments: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'segments': Список сегментов для обработки
                - 'file_id': Идентификатор файла (для логирования)
            max_workers: Количество параллельных воркеров для CPU extractors (None = auto)
            max_segments_per_batch: Не используется для CPU extractors (оставлено для совместимости)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        start_time = time.time()
        
        if not audio_files_with_segments:
            return []
        
        # Проверяем, поддерживает ли экстрактор run_segments
        if not hasattr(self, 'run_segments'):
            self.logger.error(f"{self.name} does not support run_segments()")
            return [
                self._create_result(
                    success=False,
                    error=f"{self.name} does not support run_segments()",
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
        
        # Определяем количество воркеров
        if max_workers is None or max_workers <= 0:
            import os
            max_workers = min(len(audio_files_with_segments), os.cpu_count() or 1)
        
        results: List[ExtractorResult] = [None] * len(audio_files_with_segments)  # type: ignore
        
        def process_single_file(file_info: Dict[str, Any], file_idx: int) -> tuple[int, ExtractorResult]:
            """Обработка одного файла с сегментами."""
            input_uri = file_info.get("input_uri")
            tmp_path = file_info.get("tmp_path")
            segments = file_info.get("segments", [])
            file_id = file_info.get("file_id", input_uri)
            
            if not input_uri or not tmp_path:
                self.logger.error(f"Missing input_uri or tmp_path for file_id={file_id}")
                return file_idx, self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                    processing_time=0.0,
                )
            
            if not segments:
                self.logger.warning(f"No segments provided for file_id={file_id}")
                return file_idx, self._create_result(
                    success=False,
                    error="No segments provided",
                    processing_time=0.0,
                )
            
            try:
                result = self.run_segments(input_uri, tmp_path, segments)
                return file_idx, result
            except Exception as e:
                self.logger.error(f"Error processing segments for file_id={file_id}: {e}")
                return file_idx, self._create_result(
                    success=False,
                    error=str(e),
                    processing_time=0.0,
                )
        
        # Параллельная обработка файлов через ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, file_info, idx): idx
                for idx, file_info in enumerate(audio_files_with_segments)
            }
            
            for future in as_completed(futures):
                try:
                    file_idx, result = future.result()
                    results[file_idx] = result
                except Exception as e:
                    file_idx = futures[future]
                    self.logger.error(f"Error in extract_batch_segments for file_idx={file_idx}: {e}")
                    results[file_idx] = self._create_result(
                        success=False,
                        error=str(e),
                        processing_time=0.0,
                    )
        
        return results
