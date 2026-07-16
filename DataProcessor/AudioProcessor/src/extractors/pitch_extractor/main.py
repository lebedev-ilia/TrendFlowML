"""
PitchExtractor: извлечение основной частоты (f0) с использованием PYIN/YIN (и CREPE при наличии).
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
- Additional ML/analytics metrics
"""
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
PITCH_CONTRACT_VERSION = "pitch_contract_v1"


class PitchExtractor(BaseExtractor):
    name: str = "pitch"
    version: str = "2.0.1"
    description: str = "Оценка основной частоты (f0) с помощью PYIN/YIN/CREPE (опционально)"
    category: str = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 2.0

    gpu_required = False
    gpu_preferred = False  # torchcrepe может использовать GPU, но не требуется
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 22050,
        fmin: float = 50.0,
        fmax: float = 2000.0,
        hop_length: int = 512,
        frame_length: int = 2048,
        backend: str = "classic",  # classic | torchcrepe
        channel_mode: str = "first",  # first | mean | max
        torchcrepe_batch_size: int = 1,
        # Feature gating flags (Audit v3 default: basic_stats=True)
        enable_basic_stats: bool = True,
        enable_stability_metrics: bool = False,
        enable_delta_features: bool = False,
        enable_method_stats: bool = False,
        enable_time_series: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ) -> None:
        """
        Инициализация pitch extractor.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            fmin: Минимальная частота f0 (Hz)
            fmax: Максимальная частота f0 (Hz)
            hop_length: Размер шага между кадрами (samples)
            frame_length: Размер окна анализа (samples)
            backend: Backend для pitch ("classic" | "torchcrepe")
            channel_mode: Режим обработки многоканального аудио ("first" | "mean" | "max")
            torchcrepe_batch_size: Размер батча для torchcrepe
            enable_basic_stats: Включить базовые статистики (f0_mean, f0_std, f0_min, f0_max, f0_median)
            enable_stability_metrics: Включить метрики стабильности (pitch_variation, pitch_stability, pitch_range)
            enable_delta_features: Включить delta-признаки (f0_delta_mean, f0_delta_std, f0_delta_abs_mean)
            enable_method_stats: Включить статистики по каждому методу (PYIN, YIN, torchcrepe)
            enable_time_series: Включить временные серии (f0_series_pyin, f0_series_yin, f0_series_torchcrepe)
            progress_callback: Callback для прогресса (method_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device)
        
        # Validate parameters
        self._validate_parameters(fmin, fmax, hop_length, frame_length, sample_rate)
        
        self.sample_rate = int(sample_rate)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.hop_length = int(hop_length)
        self.frame_length = int(frame_length)
        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)
        self.backend = str(backend)
        self.channel_mode = str(channel_mode)
        self.torchcrepe_batch_size = int(torchcrepe_batch_size)
        
        # Feature gating flags
        self.enable_basic_stats = bool(enable_basic_stats)
        self.enable_stability_metrics = bool(enable_stability_metrics)
        self.enable_delta_features = bool(enable_delta_features)
        self.enable_method_stats = bool(enable_method_stats)
        self.enable_time_series = bool(enable_time_series)
        
        # Progress callback
        self.progress_callback = progress_callback
        
        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

    def _validate_parameters(
        self,
        fmin: float,
        fmax: float,
        hop_length: int,
        frame_length: int,
        sample_rate: int,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).
        
        Args:
            fmin: Минимальная частота f0
            fmax: Максимальная частота f0
            hop_length: Размер шага между кадрами
            frame_length: Размер окна анализа
            sample_rate: Частота дискретизации
        
        Raises:
            ValueError: Если параметры невалидны
        """
        if fmin <= 0:
            raise ValueError(f"pitch | fmin must be positive, got {fmin}")
        if fmax <= 0:
            raise ValueError(f"pitch | fmax must be positive, got {fmax}")
        if fmin >= fmax:
            raise ValueError(f"pitch | fmin ({fmin}) must be < fmax ({fmax})")
        if fmin < 20.0:
            raise ValueError(f"pitch | fmin ({fmin}) is too low (minimum 20 Hz)")
        if fmax > 8000.0:
            raise ValueError(f"pitch | fmax ({fmax}) is too high (maximum 8000 Hz)")
        if hop_length <= 0:
            raise ValueError(f"pitch | hop_length must be positive, got {hop_length}")
        if frame_length <= 0:
            raise ValueError(f"pitch | frame_length must be positive, got {frame_length}")
        if sample_rate <= 0:
            raise ValueError(f"pitch | sample_rate must be positive, got {sample_rate}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.
        
        Args:
            error: Исключение
            context: Контекст ошибки (audio_load_failed, torchcrepe_failed, pyin_failed, yin_failed, all_methods_failed, validation_failed, unknown)
        
        Returns:
            error_code: один из:
                - pitch_audio_load_failed
                - pitch_torchcrepe_failed
                - pitch_pyin_failed
                - pitch_yin_failed
                - pitch_all_methods_failed
                - pitch_validation_failed
                - pitch_unknown
        """
        error_str = str(error).lower()
        
        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "pitch_audio_load_failed"
        if "torchcrepe" in error_str or context == "torchcrepe_failed":
            return "pitch_torchcrepe_failed"
        if "pyin" in error_str or context == "pyin_failed":
            return "pitch_pyin_failed"
        if "yin" in error_str or context == "yin_failed":
            return "pitch_yin_failed"
        if "all methods" in error_str or "empty" in error_str or context == "all_methods_failed":
            return "pitch_all_methods_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "pitch_validation_failed"
        
        return "pitch_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.
        
        Args:
            features: Словарь с выходными данными
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "pitch | features must be a dict"
        
        # Validate f0_mean if present
        if "f0_mean" in features:
            f0_mean = features.get("f0_mean")
            try:
                f0_mean = float(f0_mean)
                if not (self.fmin <= f0_mean <= self.fmax):
                    return False, f"pitch | f0_mean ({f0_mean}) out of range [{self.fmin}, {self.fmax}]"
                if np.isnan(f0_mean) or np.isinf(f0_mean):
                    return False, f"pitch | f0_mean is NaN or Inf"
            except (ValueError, TypeError):
                return False, f"pitch | f0_mean must be float, got {type(f0_mean)}"
        
        # Validate f0_min, f0_max, f0_std if present
        for key in ["f0_min", "f0_max", "f0_std"]:
            if key in features:
                value = features.get(key)
                try:
                    value = float(value)
                    if np.isnan(value) or np.isinf(value):
                        return False, f"pitch | {key} is NaN or Inf"
                    if value < 0 and key != "f0_std":  # f0_std может быть 0, но не отрицательным
                        return False, f"pitch | {key} must be non-negative, got {value}"
                except (ValueError, TypeError):
                    return False, f"pitch | {key} must be float, got {type(value)}"
        
        # Validate consistency: f0_min <= f0_mean <= f0_max
        # Допуск 1e-3 Hz для float32 rounding при агрегации одинаковых значений
        if all(k in features for k in ["f0_min", "f0_mean", "f0_max"]):
            f0_min = float(features.get("f0_min", 0.0))
            f0_mean = float(features.get("f0_mean", 0.0))
            f0_max = float(features.get("f0_max", 0.0))
            _eps = 1e-3  # float32 accumulation tolerance
            if not (f0_min - _eps <= f0_mean <= f0_max + _eps):
                return False, f"pitch | consistency check failed: f0_min ({f0_min}) <= f0_mean ({f0_mean}) <= f0_max ({f0_max})"
        
        # Validate time series if present
        for key in ["f0_series_pyin", "f0_series_yin", "f0_series_torchcrepe"]:
            if key in features:
                series = features.get(key)
                if series is not None:
                    if isinstance(series, list):
                        series_arr = np.asarray(series, dtype=np.float32)
                        if np.any(np.isnan(series_arr)) or np.any(np.isinf(series_arr)):
                            return False, f"pitch | {key} contains NaN or Inf values"
                        if np.any(series_arr < 0):
                            return False, f"pitch | {key} contains negative values"
        
        return True, None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение pitch на полном аудио.
        
        Progress reporting: обновление прогресса для каждого метода (PYIN, YIN, torchcrepe).
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        pitch_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"pitch | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            if resource_profile_enabled():
                try:
                    pitch_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    pitch_resource_profile = None

            # Загружаем аудио через общую утилиту (с ресемплингом до sample_rate)
            if self.progress_callback:
                self.progress_callback("pitch", 0, 4, "Loading audio")
            t0 = time.perf_counter()
            waveform_t, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            waveform_t = self.audio_utils.normalize_audio(waveform_t)
            stage_timings_ms["normalize_audio_ms"] = (time.perf_counter() - t0) * 1000.0

            # Преобразуем к моно с учетом многоканального входа
            if waveform_t.dim() == 2 and waveform_t.shape[0] > 1:
                if self.channel_mode == "mean":
                    waveform_t = waveform_t.mean(dim=0, keepdim=True)
                elif self.channel_mode == "max":
                    waveform_t, _ = waveform_t.max(dim=0, keepdim=True)
                else:
                    waveform_t = waveform_t[:1, :]

            # В librosa ожидается ndarray (моно)
            audio_np = waveform_t.squeeze(0).cpu().numpy()

            # Извлекаем признаки
            if self.progress_callback:
                self.progress_callback("pitch", 1, 4, "Extracting pitch features")
            t0 = time.perf_counter()
            features = self._extract_pitch_features(audio_np, sr)
            stage_timings_ms["extract_pitch_ms"] = (time.perf_counter() - t0) * 1000.0

            # Audit v3: duration, fmin, fmax, frame_length, hop_length, backend
            duration = float(audio_np.shape[-1] / sr)
            features["duration"] = duration
            features["fmin"] = self.fmin
            features["fmax"] = self.fmax
            features["frame_length"] = self.frame_length
            features["hop_length"] = self.hop_length
            features["backend"] = self.backend

            # Audit v3: canonical segment axis (empty for run())
            features["segment_start_sec"] = np.array([], dtype=np.float32)
            features["segment_end_sec"] = np.array([], dtype=np.float32)
            features["segment_center_sec"] = np.array([], dtype=np.float32)
            features["segment_mask"] = np.array([], dtype=bool)
            
            # Сохраняем f0_series в .npy (debug-only), путь в f0_series_npy
            if self.progress_callback:
                self.progress_callback("pitch", 2, 4, "Saving artifacts")
            t0 = time.perf_counter()
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)
            stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Валидация выходных данных
            if self.progress_callback:
                self.progress_callback("pitch", 3, 4, "Validating output")
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"pitch | {error_msg} (error_code={error_code})")
            
            # Добавляем contract version
            features["pitch_contract_version"] = PITCH_CONTRACT_VERSION
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_stability_metrics:
                enabled_features.append("stability_metrics")
            if self.enable_delta_features:
                enabled_features.append("delta_features")
            if self.enable_method_stats:
                enabled_features.append("method_stats")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if pitch_resource_profile is not None:
                try:
                    pitch_resource_profile = {
                        **(pitch_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["pitch_resource_profile"] = pitch_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            if self.progress_callback:
                self.progress_callback("pitch", 4, 4, "Completed")
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"pitch | Ошибка извлечения pitch (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven pitch extraction: compute f0 on provided windows (families.pitch).
        
        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {"load_segments_ms": 0.0}
        pitch_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"pitch | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("pitch | segments is empty (no-fallback)")

            total_segments = len(segments)

            if resource_profile_enabled():
                try:
                    pitch_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    pitch_resource_profile = None
            
            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments (Audit v3: canonical segment axis, segment_mask)
            f0_series_all: List[float] = []
            segment_starts: List[float] = []
            segment_ends: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []
            segment_mask: List[bool] = []
            
            t0 = time.perf_counter()
            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback("pitch", seg_idx, total_segments, f"Processing segment {seg_idx+1}/{total_segments}")
                        last_reported_pct = pct
                
                # Load segment
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))
                seg_start_sec = float(start_sample / self.sample_rate)
                seg_end_sec = float(end_sample / self.sample_rate)
                seg_dur = float((end_sample - start_sample) / self.sample_rate)
                
                segment_starts.append(seg_start_sec)
                segment_ends.append(seg_end_sec)
                segment_centers.append(center_sec)
                segment_durations.append(seg_dur)
                
                wav_t, _sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    target_sr=self.sample_rate,
                )
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                
                # Extract pitch for segment (catch empty/silent segments)
                try:
                    seg_features = self._extract_pitch_features(wav, self.sample_rate)
                    has_pitch = bool(seg_features.get("f0_mean")) and float(seg_features.get("f0_mean", 0)) > 0
                except Exception:
                    has_pitch = False
                segment_mask.append(has_pitch)
                if has_pitch:
                    f0_series_all.append(float(seg_features.get("f0_mean")))
            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Final progress report
            if self.progress_callback:
                self.progress_callback("pitch", total_segments, total_segments, "Completed")
            
            # Audit v3: all segments empty -> status=empty, not error
            if len(f0_series_all) == 0:
                total_duration = float(sum(segment_durations))
                empty_features: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "pitch_all_segments_empty",
                    "device_used": self.device,
                    "sample_rate": self.sample_rate,
                    "segments_count": int(total_segments),
                    "duration": total_duration,
                    "fmin": self.fmin,
                    "fmax": self.fmax,
                    "frame_length": self.frame_length,
                    "hop_length": self.hop_length,
                    "backend": self.backend,
                    "pitch_contract_version": PITCH_CONTRACT_VERSION,
                    "segment_start_sec": np.array(segment_starts, dtype=np.float32),
                    "segment_end_sec": np.array(segment_ends, dtype=np.float32),
                    "segment_center_sec": np.array(segment_centers, dtype=np.float32),
                    "segment_mask": np.array(segment_mask, dtype=bool),
                    "_features_enabled": [],
                }
                stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
                empty_features["stage_timings_ms"] = stage_timings_ms

                if pitch_resource_profile is not None:
                    try:
                        pitch_resource_profile = {
                            **(pitch_resource_profile or {}),
                            **prefix_snapshot("at_end", snapshot_process_resources()),
                        }
                    except Exception:
                        pass
                empty_features["pitch_resource_profile"] = pitch_resource_profile

                processing_time = time.time() - start_time
                self._log_extraction_success(input_uri, processing_time)
                return self._create_result(success=True, payload=empty_features, processing_time=processing_time)
            
            t0 = time.perf_counter()
            f0_arr = np.asarray(f0_series_all, dtype=np.float32)
            total_duration = float(sum(segment_durations))
            
            # Build payload (feature-gated)
            features: Dict[str, Any] = {
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "segments_count": int(total_segments),
                "duration": total_duration,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "frame_length": self.frame_length,
                "hop_length": self.hop_length,
                "backend": self.backend,
                "f0_method": "aggregated",
                "pitch_contract_version": PITCH_CONTRACT_VERSION,
                "segment_start_sec": np.array(segment_starts, dtype=np.float32),
                "segment_end_sec": np.array(segment_ends, dtype=np.float32),
                "segment_center_sec": np.array(segment_centers, dtype=np.float32),
                "segment_mask": np.array(segment_mask, dtype=bool),
            }
            
            # Basic stats (feature-gated)
            if self.enable_basic_stats:
                features.update({
                    "f0_mean": float(np.mean(f0_arr)),
                    "f0_std": float(np.std(f0_arr)),
                    "f0_min": float(np.min(f0_arr)),
                    "f0_max": float(np.max(f0_arr)),
                    "f0_median": float(np.median(f0_arr)),
                })
            
            # Stability metrics (feature-gated)
            if self.enable_stability_metrics:
                if f0_arr.size > 1:
                    diff = np.diff(f0_arr)
                    features.update({
                        "pitch_variation": float(np.std(diff)),
                        "pitch_stability": float(1.0 / (1.0 + np.std(diff))),
                        "pitch_range": float(np.max(f0_arr) - np.min(f0_arr)),
                    })
                else:
                    features.update({
                        "pitch_variation": 0.0,
                        "pitch_stability": 0.0,
                        "pitch_range": 0.0,
                    })
            
            # Delta features (feature-gated)
            if self.enable_delta_features:
                if f0_arr.size > 1:
                    diff = np.diff(f0_arr)
                    features.update({
                        "f0_delta_mean": float(np.mean(diff)),
                        "f0_delta_std": float(np.std(diff)),
                        "f0_delta_abs_mean": float(np.mean(np.abs(diff))),
                    })
                else:
                    features.update({
                        "f0_delta_mean": 0.0,
                        "f0_delta_std": 0.0,
                        "f0_delta_abs_mean": 0.0,
                    })
            
            # Additional ML/analytics metrics
            if self.enable_basic_stats:
                # Pitch contour smoothness (inverse of second derivative variance)
                if f0_arr.size > 2:
                    second_diff = np.diff(np.diff(f0_arr))
                    features["pitch_contour_smoothness"] = float(1.0 / (1.0 + np.std(second_diff)))
                else:
                    features["pitch_contour_smoothness"] = 0.0
                
                # Pitch jump count (large jumps > 2 semitones)
                if f0_arr.size > 1:
                    jumps = np.abs(np.diff(f0_arr))
                    semitone_threshold = f0_arr.mean() * 0.12  # ~2 semitones
                    features["pitch_jump_count"] = int(np.sum(jumps > semitone_threshold))
                else:
                    features["pitch_jump_count"] = 0
                
                # Pitch octave distribution
                if f0_arr.size > 0:
                    octave_bins = [50, 100, 200, 400, 800, 1600]
                    hist, _ = np.histogram(f0_arr, bins=octave_bins)
                    total = float(np.sum(hist))
                    if total > 0:
                        features["pitch_octave_distribution"] = {f"octave_{i}": float(count / total) for i, count in enumerate(hist)}
                    else:
                        features["pitch_octave_distribution"] = {}
                
                # Pitch skewness, kurtosis (Q6: pitch_centroid removed, duplicate of f0_mean)
                # Guard на std=0 (монотонный pitch): scipy возвращает NaN → заменяем на 0.0
                if f0_arr.size > 2:
                    from scipy import stats
                    sk = stats.skew(f0_arr)
                    ku = stats.kurtosis(f0_arr)
                    features["pitch_skewness"] = float(sk) if np.isfinite(sk) else 0.0
                    features["pitch_kurtosis"] = float(ku) if np.isfinite(ku) else 0.0
                else:
                    features["pitch_skewness"] = 0.0
                    features["pitch_kurtosis"] = 0.0
            
            # Time series (feature-gated)
            if self.enable_time_series:
                features["f0_series"] = f0_series_all
                features["segment_centers_sec"] = segment_centers
                features["segment_durations_sec"] = segment_durations
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_stability_metrics:
                enabled_features.append("stability_metrics")
            if self.enable_delta_features:
                enabled_features.append("delta_features")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            stage_timings_ms["aggregate_results_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Валидация выходных данных
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"pitch | {error_msg} (error_code={error_code})")

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if pitch_resource_profile is not None:
                try:
                    pitch_resource_profile = {
                        **(pitch_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["pitch_resource_profile"] = pitch_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"pitch | Ошибка извлечения pitch (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def _save_time_series_artifacts(
        self,
        features: Dict[str, Any],
        input_uri: str,
        tmp_path: str,
    ) -> Dict[str, Any]:
        """
        Сохранить f0_series в .npy (debug-only). Q7: f0_series не в NPZ, путь в meta.extra.f0_series_npy.
        """
        if not self.enable_time_series:
            return features
        
        artifacts_dir = Path(self.artifacts_dir) if self.artifacts_dir else Path(tmp_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_uri).stem
        
        # Save torchcrepe series if present
        series = features.get("f0_series_torchcrepe")
        if isinstance(series, list) and len(series) > 0:
            npy_path = artifacts_dir / f"{stem}_f0_torchcrepe.npy"
            np.save(str(npy_path), np.asarray(series, dtype=np.float32))
            features["f0_series_torchcrepe_npy"] = str(npy_path)
            features["f0_series_npy"] = str(npy_path)  # canonical path for meta.extra
            features["f0_count_torchcrepe"] = int(len(series))
            features.pop("f0_series_torchcrepe", None)
            return features
        
        # Save pyin/yin series (classic backend)
        for key in ["f0_series_pyin", "f0_series_yin"]:
            series = features.get(key)
            if series is not None:
                arr = np.asarray(series, dtype=np.float32) if isinstance(series, list) else series
                if arr.size > 0:
                    suffix = key.replace("f0_series_", "")
                    npy_path = artifacts_dir / f"{stem}_f0_{suffix}.npy"
                    np.save(str(npy_path), arr)
                    features["f0_series_npy"] = str(npy_path)
                features.pop(key, None)
                break
        
        return features

    def _extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Извлечение pitch признаков с использованием выбранного backend (no-fallback policy).
        
        Args:
            audio: Аудио сигнал (моно, numpy array)
            sr: Частота дискретизации
        
        Returns:
            Словарь с признаками (feature-gated)
        
        Raises:
            RuntimeError: Если выбранный backend падает (no-fallback)
        """
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": sr,
        }

        # Если выбран backend torchcrepe — используем его (fail-fast, no-fallback)
        if self.backend == "torchcrepe":
            if self.progress_callback:
                self.progress_callback("pitch", 1, 3, "Running torchcrepe")
            try:
                f0_tc = self._extract_torchcrepe(audio, sr)
                if f0_tc is None or f0_tc.size == 0:
                    error_code = self._classify_error(RuntimeError("torchcrepe produced empty output"), "torchcrepe_failed")
                    raise RuntimeError(f"pitch | torchcrepe produced empty output (error_code={error_code})")
                
                feats = self._calc_stats(f0_tc, prefix="torchcrepe", feature_gated=True)
                features.update(feats)
                
                # Итог на базе torchcrepe (feature-gated)
                if self.enable_basic_stats:
                    features.update({
                        "f0_mean": feats.get("f0_mean_torchcrepe", 0.0),
                        "f0_std": feats.get("f0_std_torchcrepe", 0.0),
                        "f0_min": feats.get("f0_min_torchcrepe", 0.0),
                        "f0_max": feats.get("f0_max_torchcrepe", 0.0),
                        "f0_median": feats.get("f0_median_torchcrepe", 0.0),
                    })
                features["f0_method"] = "torchcrepe"
                
                # Stability metrics (feature-gated)
                if self.enable_stability_metrics and f0_tc.size > 1:
                    diff = np.diff(f0_tc)
                    features.update({
                        "pitch_variation": float(np.std(diff)),
                        "pitch_stability": float(1.0 / (1.0 + np.std(diff))),
                        "pitch_range": float(feats.get("f0_max_torchcrepe", 0.0) - feats.get("f0_min_torchcrepe", 0.0)),
                    })
                
                # Delta features (feature-gated)
                if self.enable_delta_features and f0_tc.size > 1:
                    diff = np.diff(f0_tc)
                    features.update({
                        "f0_delta_mean": float(np.mean(diff)),
                        "f0_delta_std": float(np.std(diff)),
                        "f0_delta_abs_mean": float(np.mean(np.abs(diff))),
                    })
                
                # Additional ML/analytics metrics (feature-gated)
                if self.enable_basic_stats:
                    features.update(self._calc_additional_metrics(f0_tc))
                
                return features
            except Exception as e:
                error_code = self._classify_error(e, "torchcrepe_failed")
                raise RuntimeError(f"pitch | torchcrepe failed (error_code={error_code}): {e}") from e

        # Classic backend: PYIN и YIN (оба метода запускаются, выбирается лучший)
        if self.progress_callback:
            self.progress_callback("pitch", 1, 3, "Running PYIN")
        
        # PYIN (наиболее устойчивый из классики, требует моно)
        f0_pyin_clean = np.array([])
        voiced_flag_clean = np.array([])
        voiced_probs_mean = 0.0
        try:
            f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            
            # Фильтруем NaN и значения вне [fmin, fmax] — PYIN/YIN могут вернуть out-of-range на тихом сигнале
            _pyin_valid = (f0_pyin is not None) and (~np.isnan(f0_pyin)) & (f0_pyin >= self.fmin) & (f0_pyin <= self.fmax)
            f0_pyin_clean = f0_pyin[_pyin_valid] if f0_pyin is not None else np.array([])
            voiced_flag_clean = voiced_flag[~np.isnan(voiced_flag)] if voiced_flag is not None else np.array([])
            voiced_probs_mean = float(np.nanmean(voiced_probs)) if voiced_probs is not None else 0.0
            
            if f0_pyin_clean.size > 0:
                if self.enable_method_stats:
                    features.update(self._calc_stats(f0_pyin_clean, prefix="pyin", feature_gated=True))
                if self.enable_method_stats:
                    features["voiced_fraction_pyin"] = float(np.mean(voiced_flag_clean)) if voiced_flag_clean.size > 0 else 0.0
                    features["voiced_probability_mean_pyin"] = voiced_probs_mean
        except Exception as e:
            error_code = self._classify_error(e, "pyin_failed")
            raise RuntimeError(f"pitch | PYIN failed (error_code={error_code}): {e}") from e

        # YIN
        if self.progress_callback:
            self.progress_callback("pitch", 2, 3, "Running YIN")
        
        f0_yin_clean = np.array([])
        try:
            f0_yin = librosa.yin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            
            # Фильтруем NaN и значения вне [fmin, fmax] — YIN может вернуть out-of-range на тихом/нулевом сигнале
            _yin_valid = (f0_yin is not None) and (~np.isnan(f0_yin)) & (f0_yin >= self.fmin) & (f0_yin <= self.fmax)
            f0_yin_clean = f0_yin[_yin_valid] if f0_yin is not None else np.array([])

            if f0_yin_clean.size > 0:
                if self.enable_method_stats:
                    features.update(self._calc_stats(f0_yin_clean, prefix="yin", feature_gated=True))
        except Exception as e:
            error_code = self._classify_error(e, "yin_failed")
            raise RuntimeError(f"pitch | YIN failed (error_code={error_code}): {e}") from e

        # Выбор лучшего метода (PYIN vs YIN)
        if self.progress_callback:
            self.progress_callback("pitch", 3, 3, "Selecting best method")
        
        if f0_pyin_clean.size == 0 and f0_yin_clean.size == 0:
            error_code = self._classify_error(RuntimeError("All methods returned empty output"), "all_methods_failed")
            raise RuntimeError(f"pitch | all methods returned empty/invalid output (error_code={error_code})")
        
        # Выбор лучшего метода на основе взвешенной оценки
        score_pyin = (
            (0.6 * float(np.mean(f0_pyin_clean)) if f0_pyin_clean.size > 0 else 0.0) +
            (0.3 * float(np.mean(voiced_flag_clean)) * 100.0 if voiced_flag_clean.size > 0 else 0.0) +
            (0.1 * float(f0_pyin_clean.size))
        )
        score_yin = (
            (0.6 * float(np.mean(f0_yin_clean)) if f0_yin_clean.size > 0 else 0.0) +
            (0.3 * 0.0 * 100.0) +  # YIN не имеет voiced_fraction
            (0.1 * float(f0_yin_clean.size))
        )
        best_method = "pyin" if score_pyin >= score_yin else "yin"
        best_series = f0_pyin_clean if best_method == "pyin" else f0_yin_clean
        
        if best_series.size > 0:
            # Basic stats (feature-gated)
            if self.enable_basic_stats:
                features.update({
                    "f0_mean": float(np.mean(best_series)),
                    "f0_std": float(np.std(best_series)),
                    "f0_min": float(np.min(best_series)),
                    "f0_max": float(np.max(best_series)),
                    "f0_median": float(np.median(best_series)),
                })
            features["f0_method"] = best_method
            
            # Stability metrics (feature-gated)
            if self.enable_stability_metrics and best_series.size > 1:
                diff = np.diff(best_series)
                features.update({
                    "pitch_variation": float(np.std(diff)),
                    "pitch_stability": float(1.0 / (1.0 + np.std(diff))),
                    "pitch_range": float(np.max(best_series) - np.min(best_series)),
                })
            
            # Delta features (feature-gated)
            if self.enable_delta_features and best_series.size > 1:
                diff = np.diff(best_series)
                features.update({
                    "f0_delta_mean": float(np.mean(diff)),
                    "f0_delta_std": float(np.std(diff)),
                    "f0_delta_abs_mean": float(np.mean(np.abs(diff))),
                })
            
            # Additional ML/analytics metrics (feature-gated)
            if self.enable_basic_stats:
                features.update(self._calc_additional_metrics(best_series))
            
            # Time series (feature-gated)
            if self.enable_time_series:
                features[f"f0_series_{best_method}"] = best_series.tolist()
        else:
            error_code = self._classify_error(RuntimeError("Best method produced empty output"), "all_methods_failed")
            raise RuntimeError(f"pitch | best method produced empty output (error_code={error_code})")

        return features

    def _calc_additional_metrics(self, f0: np.ndarray) -> Dict[str, Any]:
        """
        Вычислить дополнительные метрики для ML/аналитики.
        
        Args:
            f0: Массив значений f0
        
        Returns:
            Словарь с дополнительными метриками
        """
        metrics: Dict[str, Any] = {}
        
        if f0.size == 0:
            return {
                "pitch_contour_smoothness": 0.0,
                "pitch_jump_count": 0,
                "pitch_octave_distribution": {},
                "pitch_skewness": 0.0,
                "pitch_kurtosis": 0.0,
            }
        
        # Pitch contour smoothness (inverse of second derivative variance)
        if f0.size > 2:
            second_diff = np.diff(np.diff(f0))
            metrics["pitch_contour_smoothness"] = float(1.0 / (1.0 + np.std(second_diff)))
        else:
            metrics["pitch_contour_smoothness"] = 0.0
        
        # Pitch jump count (large jumps > 2 semitones)
        if f0.size > 1:
            jumps = np.abs(np.diff(f0))
            semitone_threshold = np.mean(f0) * 0.12  # ~2 semitones
            metrics["pitch_jump_count"] = int(np.sum(jumps > semitone_threshold))
        else:
            metrics["pitch_jump_count"] = 0
        
        # Pitch octave distribution
        octave_bins = [50, 100, 200, 400, 800, 1600]
        hist, _ = np.histogram(f0, bins=octave_bins)
        total = float(np.sum(hist))
        if total > 0:
            metrics["pitch_octave_distribution"] = {f"octave_{i}": float(count / total) for i, count in enumerate(hist)}
        else:
            metrics["pitch_octave_distribution"] = {}
        
        # Pitch skewness, kurtosis (Q6: pitch_centroid removed, duplicate of f0_mean)
        if f0.size > 2:
            try:
                from scipy import stats
                metrics["pitch_skewness"] = float(stats.skew(f0))
                metrics["pitch_kurtosis"] = float(stats.kurtosis(f0))
            except ImportError:
                # Fallback: manual calculation if scipy not available
                mean = np.mean(f0)
                std = np.std(f0)
                if std > 0:
                    metrics["pitch_skewness"] = float(np.mean(((f0 - mean) / std) ** 3))
                    metrics["pitch_kurtosis"] = float(np.mean(((f0 - mean) / std) ** 4) - 3.0)
                else:
                    metrics["pitch_skewness"] = 0.0
                    metrics["pitch_kurtosis"] = 0.0
        else:
            metrics["pitch_skewness"] = 0.0
            metrics["pitch_kurtosis"] = 0.0
        
        return metrics

    def _extract_torchcrepe(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Извлечение f0 через torchcrepe (PyTorch, GPU/CPU)."""
        try:
            import torch
            import torchcrepe
        except ImportError as e:
            raise RuntimeError(f"pitch | torchcrepe not installed: {e}") from e

        try:
            # Приводим к 16 kHz — torchcrepe обучен на 16 kHz
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32, copy=False)
                sr_tc = 16000
            else:
                audio_16k = audio.astype(np.float32, copy=False)
                sr_tc = sr

            # Вход torchcrepe: torch.Tensor [batch=1, time]
            wav = torch.from_numpy(audio_16k).unsqueeze(0)
            # Перемещаем на устройство экстрактора, если это cuda и доступно
            if self.device == "cuda" and torch.cuda.is_available():
                wav = wav.cuda(non_blocking=True)

            # frame hop/size: torchcrepe использует hop размером по частоте кадров (default 80 samples @16kHz ~ 200Hz)
            fmin = float(self.fmin)
            fmax = float(self.fmax)
            model = "tiny"  # точнее, можно "tiny" для скорости
            with torch.inference_mode():
                f0, pd = torchcrepe.predict(
                    wav, sr_tc,
                    fmin=fmin, fmax=fmax,
                    model=model,
                    batch_size=self.torchcrepe_batch_size,
                    device=wav.device,
                    return_periodicity=True,
                )
                # Очистка и перенос на CPU
                f0 = f0.squeeze(0).float().cpu().numpy()
                pd = pd.squeeze(0).float().cpu().numpy()

            # Фильтр по периодичности (доверие), убираем нули
            mask = pd > 0.1
            f0 = f0[mask]
            f0 = f0[f0 > 0]
            return f0 if f0.size > 0 else None
        except Exception as e:
            raise RuntimeError(f"pitch | torchcrepe extraction failed: {e}") from e

    def _calc_stats(self, f0: np.ndarray, prefix: str, feature_gated: bool = False) -> Dict[str, Any]:
        """
        Вычислить статистики для f0 массива.
        
        Args:
            f0: Массив значений f0
            prefix: Префикс для ключей (pyin, yin, torchcrepe)
            feature_gated: Если True, возвращает только если enable_method_stats или enable_time_series
        
        Returns:
            Словарь со статистиками
        """
        if feature_gated and not (self.enable_method_stats or self.enable_time_series):
            return {}
        
        f0 = f0.astype(np.float32, copy=False)
        stats: Dict[str, Any] = {}
        
        if self.enable_method_stats:
            stats.update({
                f"f0_mean_{prefix}": float(np.mean(f0)) if f0.size else 0.0,
                f"f0_std_{prefix}": float(np.std(f0)) if f0.size else 0.0,
                f"f0_min_{prefix}": float(np.min(f0)) if f0.size else 0.0,
                f"f0_max_{prefix}": float(np.max(f0)) if f0.size else 0.0,
                f"f0_median_{prefix}": float(np.median(f0)) if f0.size else 0.0,
                f"f0_count_{prefix}": int(f0.size),
            })
        
        if self.enable_time_series:
            stats[f"f0_series_{prefix}"] = f0.tolist()
        
        return stats
