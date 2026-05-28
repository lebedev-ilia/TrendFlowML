"""
QualityExtractor: извлечение метрик качества аудио (DC offset, clipping, crest factor, dynamic range).
Audit v3: snr_db removed (duplicate of dynamic_range_db), dc_offset_abs removed, enable_basic_metrics=True default.
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
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
QUALITY_CONTRACT_VERSION = "quality_contract_v1"


class QualityExtractor(BaseExtractor):
    """Извлекает базовые метрики качества аудио."""

    name = "quality"
    version = "2.0.1"
    description = "Метрики качества аудио: DC offset, clipping, crest factor, dynamic range"
    category = "quality"
    dependencies = ["numpy"]
    estimated_duration = 0.5

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        average_channels: bool = True,
        frame_len_ms: float = 50.0,
        hop_ms: float = 25.0,
        clip_threshold: float = 0.999,
        # Feature gating flags (Audit v3: basic_metrics=True by default)
        enable_basic_metrics: bool = True,
        enable_dynamic_metrics: bool = False,
        enable_frame_analysis: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация quality extractor.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            average_channels: Усреднять ли каналы для многоканального аудио
            frame_len_ms: Длина кадра для анализа уровней (мс)
            hop_ms: Шаг между кадрами (мс)
            clip_threshold: Порог для определения клиппинга (0.0-1.0)
            enable_basic_metrics: Включить базовые метрики (dc_offset, clipping_ratio, crest_factor_db)
            enable_dynamic_metrics: Включить динамические метрики (dynamic_range_db)
            enable_frame_analysis: Включить анализ кадров (frame-level метрики)
            enable_time_series: Включить временные серии для всех метрик
            enable_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)
        
        # Validate parameters
        self._validate_parameters(sample_rate, frame_len_ms, hop_ms, clip_threshold)
        
        self.sample_rate = int(sample_rate)
        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)
        self.average_channels = bool(average_channels)
        self.frame_len_ms = float(frame_len_ms)
        self.hop_ms = float(hop_ms)
        self.clip_threshold = float(clip_threshold)
        
        # Feature gating flags
        self.enable_basic_metrics = bool(enable_basic_metrics)
        self.enable_dynamic_metrics = bool(enable_dynamic_metrics)
        self.enable_frame_analysis = bool(enable_frame_analysis)
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
        frame_len_ms: float,
        hop_ms: float,
        clip_threshold: float,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).
        
        Args:
            sample_rate: Частота дискретизации
            frame_len_ms: Длина кадра (мс)
            hop_ms: Шаг между кадрами (мс)
            clip_threshold: Порог клиппинга
        
        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"quality | sample_rate must be positive, got {sample_rate}")
        if frame_len_ms <= 0:
            raise ValueError(f"quality | frame_len_ms must be positive, got {frame_len_ms}")
        if hop_ms <= 0:
            raise ValueError(f"quality | hop_ms must be positive, got {hop_ms}")
        if hop_ms > frame_len_ms:
            raise ValueError(f"quality | hop_ms ({hop_ms}) must be <= frame_len_ms ({frame_len_ms})")
        if clip_threshold < 0.0 or clip_threshold > 1.0:
            raise ValueError(f"quality | clip_threshold must be in [0, 1], got {clip_threshold}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.
        
        Args:
            error: Исключение
            context: Контекст ошибки (audio_load_failed, dc_offset_failed, clipping_failed, crest_factor_failed, dynamic_range_failed, frame_analysis_failed, validation_failed, unknown)
        
        Returns:
            error_code: один из:
                - quality_audio_load_failed
                - quality_dc_offset_failed
                - quality_clipping_failed
                - quality_crest_factor_failed
                - quality_dynamic_range_failed
                - quality_frame_analysis_failed
                - quality_validation_failed
                - quality_unknown
        """
        error_str = str(error).lower()
        
        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "quality_audio_load_failed"
        if "dc" in error_str or "offset" in error_str or context == "dc_offset_failed":
            return "quality_dc_offset_failed"
        if "clip" in error_str or context == "clipping_failed":
            return "quality_clipping_failed"
        if "crest" in error_str or context == "crest_factor_failed":
            return "quality_crest_factor_failed"
        if "dynamic" in error_str or "range" in error_str or context == "dynamic_range_failed":
            return "quality_dynamic_range_failed"
        if "frame" in error_str or context == "frame_analysis_failed":
            return "quality_frame_analysis_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "quality_validation_failed"
        
        return "quality_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.
        
        Args:
            features: Словарь с выходными данными
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(features, dict):
            return False, "quality | features must be a dict"
        
        # Validate basic metrics if present
        if "dc_offset" in features:
            dc_offset = features.get("dc_offset")
            try:
                dc_offset = float(dc_offset)
                if np.isnan(dc_offset) or np.isinf(dc_offset):
                    return False, "quality | dc_offset is NaN or Inf"
            except (ValueError, TypeError):
                return False, f"quality | dc_offset must be float, got {type(dc_offset)}"
        
        if "clipping_ratio" in features:
            clipping_ratio = features.get("clipping_ratio")
            try:
                clipping_ratio = float(clipping_ratio)
                if np.isnan(clipping_ratio) or np.isinf(clipping_ratio):
                    return False, "quality | clipping_ratio is NaN or Inf"
                if clipping_ratio < 0.0 or clipping_ratio > 1.0:
                    return False, f"quality | clipping_ratio must be in [0, 1], got {clipping_ratio}"
            except (ValueError, TypeError):
                return False, f"quality | clipping_ratio must be float, got {type(clipping_ratio)}"
        
        if "crest_factor_db" in features:
            crest_factor_db = features.get("crest_factor_db")
            try:
                crest_factor_db = float(crest_factor_db)
                if np.isnan(crest_factor_db) or np.isinf(crest_factor_db):
                    return False, "quality | crest_factor_db is NaN or Inf"
                if crest_factor_db < 0.0:
                    return False, f"quality | crest_factor_db must be non-negative, got {crest_factor_db}"
            except (ValueError, TypeError):
                return False, f"quality | crest_factor_db must be float, got {type(crest_factor_db)}"
        
        # Validate dynamic metrics if present
        if "dynamic_range_db" in features:
            dynamic_range_db = features.get("dynamic_range_db")
            try:
                dynamic_range_db = float(dynamic_range_db)
                if np.isnan(dynamic_range_db) or np.isinf(dynamic_range_db):
                    return False, "quality | dynamic_range_db is NaN or Inf"
                if dynamic_range_db < 0.0:
                    return False, f"quality | dynamic_range_db must be non-negative, got {dynamic_range_db}"
            except (ValueError, TypeError):
                return False, f"quality | dynamic_range_db must be float, got {type(dynamic_range_db)}"
        
        # Validate time series if present
        for series_key in ["frame_levels_db_series", "frame_rms_series", "clipping_segments_series"]:
            if series_key in features:
                series = features.get(series_key)
                if series is not None:
                    if isinstance(series, list):
                        series_arr = np.asarray(series, dtype=np.float32)
                        if np.any(np.isnan(series_arr)) or np.any(np.isinf(series_arr)):
                            return False, f"quality | {series_key} contains NaN or Inf values"
                        if series_key == "clipping_segments_series":
                            if np.any((series_arr < 0) | (series_arr > 1)):
                                return False, f"quality | {series_key} contains values outside [0, 1]"
        
        return True, None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение метрик качества на полном аудио.
        
        Progress reporting: обновление прогресса для каждой метрики.
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        quality_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"quality | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            if resource_profile_enabled():
                try:
                    quality_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    quality_resource_profile = None

            # Загружаем аудио
            if self.progress_callback:
                self.progress_callback("quality", 0, 6, "Loading audio")
            t0 = time.perf_counter()
            wav_t, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Опциональная нормализация
            if self.enable_normalization:
                t0 = time.perf_counter()
                wav_t = self.audio_utils.normalize_audio(wav_t)
                stage_timings_ms["normalize_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            else:
                stage_timings_ms["normalize_audio_ms"] = 0.0
            
            x = self.audio_utils.to_numpy(wav_t)
            if x.ndim == 2:
                x = np.mean(x, axis=0) if self.average_channels else x[0]
            
            x = x.astype(np.float32)
            
            # Извлекаем метрики
            t0 = time.perf_counter()
            features = self._extract_quality_metrics(x, sr)
            stage_timings_ms["extract_metrics_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Audit v3: params, canonical segment axis (empty for run())
            features["frame_len_ms"] = self.frame_len_ms
            features["hop_ms"] = self.hop_ms
            features["clip_threshold"] = self.clip_threshold
            features["average_channels"] = self.average_channels
            features["segment_start_sec"] = np.array([], dtype=np.float32)
            features["segment_end_sec"] = np.array([], dtype=np.float32)
            features["segment_center_sec"] = np.array([], dtype=np.float32)
            features["segment_mask"] = np.array([], dtype=bool)
            
            # Сохраняем временные серии в .npy (Audit v3: only in .npy, path in meta.extra)
            if self.progress_callback:
                self.progress_callback("quality", 5, 6, "Saving artifacts")
            t0 = time.perf_counter()
            features = self._save_time_series_artifacts(features, input_uri, tmp_path)
            stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Валидация выходных данных
            if self.progress_callback:
                self.progress_callback("quality", 6, 6, "Validating output")
            t0 = time.perf_counter()
            is_valid, error_msg = self._validate_output(features)
            stage_timings_ms["validate_output_ms"] = (time.perf_counter() - t0) * 1000.0
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                raise ValueError(f"quality | {error_msg} (error_code={error_code})")
            
            # Добавляем contract version
            features["quality_contract_version"] = QUALITY_CONTRACT_VERSION
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_metrics:
                enabled_features.append("basic_metrics")
            if self.enable_dynamic_metrics:
                enabled_features.append("dynamic_metrics")
            if self.enable_frame_analysis:
                enabled_features.append("frame_analysis")
            if self.enable_time_series:
                enabled_features.append("time_series")
            features["_features_enabled"] = enabled_features

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if quality_resource_profile is not None:
                try:
                    quality_resource_profile = {
                        **(quality_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["quality_resource_profile"] = quality_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"quality | Ошибка извлечения quality metrics (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
    ) -> ExtractorResult:
        """
        Segmenter-driven quality extraction: compute quality metrics on provided windows (families.quality).
        
        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {"load_segments_ms": 0.0}
        quality_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"quality | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("quality | segments is empty (no-fallback)")

            total_segments = len(segments)

            if resource_profile_enabled():
                try:
                    quality_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    quality_resource_profile = None
            
            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments (Audit v3: canonical axis, empty semantics)
            dc_offset_all: List[float] = []
            clipping_ratio_all: List[float] = []
            crest_factor_db_all: List[float] = []
            dynamic_range_db_all: List[float] = []
            clipping_segments_all: List[int] = []
            segment_starts: List[float] = []
            segment_ends: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []
            segment_mask: List[bool] = []
            
            t0 = time.perf_counter()
            for seg_idx, seg in enumerate(segments):
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx / total_segments) * 100)
                    if pct != last_reported_pct:
                        self.progress_callback("quality", seg_idx, total_segments, f"Processing segment {seg_idx+1}/{total_segments}")
                        last_reported_pct = pct
                
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
                
                try:
                    wav_t, _sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        target_sr=self.sample_rate,
                    )
                    if self.enable_normalization:
                        wav_t = self.audio_utils.normalize_audio(wav_t)
                    wav = self.audio_utils.to_numpy(wav_t)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                    wav = wav.astype(np.float32)
                    seg_features = self._extract_quality_metrics(wav, self.sample_rate)
                except Exception:
                    segment_mask.append(False)
                    continue
                
                if seg_features.get("dc_offset") is not None:
                    segment_mask.append(True)
                    dc_offset_all.append(float(seg_features.get("dc_offset", 0.0)))
                    clipping_ratio_all.append(float(seg_features.get("clipping_ratio", 0.0)))
                    crest_factor_db_all.append(float(seg_features.get("crest_factor_db", 0.0)))
                    if seg_features.get("dynamic_range_db") is not None:
                        dynamic_range_db_all.append(float(seg_features.get("dynamic_range_db", 0.0)))
                    if seg_features.get("clipping_ratio", 0.0) > 0.0:
                        clipping_segments_all.append(seg_idx)
                else:
                    segment_mask.append(False)
            
            if self.progress_callback:
                self.progress_callback("quality", total_segments, total_segments, "Completed")

            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0) * 1000.0
            
            # Audit v3: all segments empty -> status=empty, not error
            if len(dc_offset_all) == 0:
                total_duration = float(sum(segment_durations))
                empty_features: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "quality_all_segments_empty",
                    "device_used": self.device,
                    "sample_rate": self.sample_rate,
                    "segments_count": int(total_segments),
                    "duration": total_duration,
                    "frame_len_ms": self.frame_len_ms,
                    "hop_ms": self.hop_ms,
                    "clip_threshold": self.clip_threshold,
                    "average_channels": self.average_channels,
                    "quality_contract_version": QUALITY_CONTRACT_VERSION,
                    "segment_start_sec": np.array(segment_starts, dtype=np.float32),
                    "segment_end_sec": np.array(segment_ends, dtype=np.float32),
                    "segment_center_sec": np.array(segment_centers, dtype=np.float32),
                    "segment_mask": np.array(segment_mask, dtype=bool),
                    "_features_enabled": [],
                }
                stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
                empty_features["stage_timings_ms"] = stage_timings_ms

                if quality_resource_profile is not None:
                    try:
                        quality_resource_profile = {
                            **(quality_resource_profile or {}),
                            **prefix_snapshot("at_end", snapshot_process_resources()),
                        }
                    except Exception:
                        pass
                empty_features["quality_resource_profile"] = quality_resource_profile

                processing_time = time.time() - start_time
                self._log_extraction_success(input_uri, processing_time)
                return self._create_result(success=True, payload=empty_features, processing_time=processing_time)
            
            t0 = time.perf_counter()
            total_duration = float(sum(segment_durations))
            features: Dict[str, Any] = {
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "segments_count": int(total_segments),
                "duration": total_duration,
                "frame_len_ms": self.frame_len_ms,
                "hop_ms": self.hop_ms,
                "clip_threshold": self.clip_threshold,
                "average_channels": self.average_channels,
                "quality_contract_version": QUALITY_CONTRACT_VERSION,
                "segment_start_sec": np.array(segment_starts, dtype=np.float32),
                "segment_end_sec": np.array(segment_ends, dtype=np.float32),
                "segment_center_sec": np.array(segment_centers, dtype=np.float32),
                "segment_mask": np.array(segment_mask, dtype=bool),
            }
            
            if self.enable_basic_metrics:
                dc_offset_arr = np.asarray(dc_offset_all, dtype=np.float32)
                clipping_ratio_arr = np.asarray(clipping_ratio_all, dtype=np.float32)
                crest_factor_db_arr = np.asarray(crest_factor_db_all, dtype=np.float32)
                features.update({
                    "dc_offset": float(np.mean(dc_offset_arr)),
                    "clipping_ratio": float(np.mean(clipping_ratio_arr)),
                    "crest_factor_db": float(np.mean(crest_factor_db_arr)),
                })
                features.update(self._calc_additional_metrics(
                    dc_offset_arr,
                    clipping_ratio_arr,
                    crest_factor_db_arr,
                    np.asarray(dynamic_range_db_all, dtype=np.float32) if len(dynamic_range_db_all) > 0 else np.array([]),
                ))
                features["clipping_segments_count"] = int(len(clipping_segments_all))
            
            if self.enable_dynamic_metrics and len(dynamic_range_db_all) > 0:
                dynamic_range_db_arr = np.asarray(dynamic_range_db_all, dtype=np.float32)
                features["dynamic_range_db"] = float(np.mean(dynamic_range_db_arr))
                features["dynamic_range_stability"] = float(1.0 / (1.0 + np.std(dynamic_range_db_arr)))
            
            if self.enable_time_series:
                features["dc_offset_series"] = dc_offset_all
                features["clipping_ratio_series"] = clipping_ratio_all
                features["crest_factor_db_series"] = crest_factor_db_all
                if len(dynamic_range_db_all) > 0:
                    features["dynamic_range_db_series"] = dynamic_range_db_all
                features["segment_centers_sec"] = segment_centers
                features["segment_durations_sec"] = segment_durations
                t_save0 = time.perf_counter()
                features = self._save_time_series_artifacts(features, input_uri, tmp_path)
                stage_timings_ms["save_artifacts_ms"] = (time.perf_counter() - t_save0) * 1000.0
            else:
                stage_timings_ms["save_artifacts_ms"] = 0.0
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_metrics:
                enabled_features.append("basic_metrics")
            if self.enable_dynamic_metrics:
                enabled_features.append("dynamic_metrics")
            if self.enable_frame_analysis:
                enabled_features.append("frame_analysis")
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
                raise ValueError(f"quality | {error_msg} (error_code={error_code})")

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            features["stage_timings_ms"] = stage_timings_ms

            if quality_resource_profile is not None:
                try:
                    quality_resource_profile = {
                        **(quality_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            features["quality_resource_profile"] = quality_resource_profile

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            error_msg = f"quality | Ошибка извлечения quality metrics (error_code={error_code}): {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def _save_time_series_artifacts(
        self,
        features: Dict[str, Any],
        input_uri: str,
        tmp_path: str,
    ) -> Dict[str, Any]:
        """
        Сохранить временные серии в .npy (Audit v3: only in .npy, path in meta.extra).
        """
        if not self.enable_time_series:
            return features
        artifacts_dir = Path(self.artifacts_dir) if self.artifacts_dir else Path(tmp_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_uri).stem
        for series_key in ["frame_levels_db_series", "frame_rms_series", "clipping_segments_series", "dc_offset_series", "clipping_ratio_series", "crest_factor_db_series", "dynamic_range_db_series"]:
            series = features.get(series_key)
            if series is not None:
                arr = np.asarray(series, dtype=np.float32) if isinstance(series, list) else series
                if arr.size > 0:
                    npy_path = artifacts_dir / f"{stem}_{series_key}.npy"
                    np.save(str(npy_path), arr)
                    features[f"{series_key}_npy"] = str(npy_path)
                features.pop(series_key, None)
        return features

    def _extract_quality_metrics(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Извлечение метрик качества с использованием numpy (no-fallback policy).
        
        Args:
            audio: Аудио сигнал (моно, numpy array)
            sr: Частота дискретизации
        
        Returns:
            Словарь с метриками (feature-gated)
        
        Raises:
            RuntimeError: Если метрика не может быть вычислена (no-fallback)
        """
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": sr,
            "duration": float(audio.shape[-1] / sr),
        }

        eps = 1e-12

        # Basic metrics (fail-fast, no-fallback)
        if self.enable_basic_metrics:
            if self.progress_callback:
                self.progress_callback("quality", 1, 6, "Computing DC offset")
            try:
                dc_offset = float(np.mean(audio))
                if np.isnan(dc_offset) or np.isinf(dc_offset):
                    error_code = self._classify_error(RuntimeError("dc_offset produced NaN/Inf"), "dc_offset_failed")
                    raise RuntimeError(f"quality | dc_offset produced NaN/Inf (error_code={error_code})")
                features["dc_offset"] = dc_offset
            except Exception as e:
                error_code = self._classify_error(e, "dc_offset_failed")
                raise RuntimeError(f"quality | dc_offset failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("quality", 2, 6, "Computing clipping ratio")
            try:
                clipping_ratio = float(np.mean(np.abs(audio) >= self.clip_threshold))
                if np.isnan(clipping_ratio) or np.isinf(clipping_ratio) or clipping_ratio < 0.0 or clipping_ratio > 1.0:
                    error_code = self._classify_error(RuntimeError("clipping_ratio produced invalid output"), "clipping_failed")
                    raise RuntimeError(f"quality | clipping_ratio produced invalid output (error_code={error_code})")
                features["clipping_ratio"] = clipping_ratio
            except Exception as e:
                error_code = self._classify_error(e, "clipping_failed")
                raise RuntimeError(f"quality | clipping failed (error_code={error_code}): {e}") from e

            if self.progress_callback:
                self.progress_callback("quality", 3, 6, "Computing crest factor")
            try:
                rms = float(np.sqrt(np.mean(audio**2) + eps))
                peak = float(np.max(np.abs(audio) + eps))
                crest_factor_db = float(20.0 * np.log10((peak + eps) / (rms + eps)))
                if np.isnan(crest_factor_db) or np.isinf(crest_factor_db) or crest_factor_db < 0.0:
                    error_code = self._classify_error(RuntimeError("crest_factor_db produced invalid output"), "crest_factor_failed")
                    raise RuntimeError(f"quality | crest_factor_db produced invalid output (error_code={error_code})")
                features["crest_factor_db"] = crest_factor_db
            except Exception as e:
                error_code = self._classify_error(e, "crest_factor_failed")
                raise RuntimeError(f"quality | crest_factor failed (error_code={error_code}): {e}") from e

        # Dynamic metrics (fail-fast, no-fallback)
        if self.enable_dynamic_metrics:
            if self.progress_callback:
                self.progress_callback("quality", 4, 6, "Computing dynamic range and SNR")
            levels = self._compute_frame_levels(audio, sr)
            
            if len(levels) == 0:
                error_code = self._classify_error(RuntimeError("frame levels computation produced empty output"), "dynamic_range_failed")
                raise RuntimeError(f"quality | frame levels computation produced empty output (error_code={error_code})")
            
            try:
                p5 = float(np.percentile(levels, 5))
                p95 = float(np.percentile(levels, 95))
                dynamic_range_db = float(p95 - p5)
                if np.isnan(dynamic_range_db) or np.isinf(dynamic_range_db) or dynamic_range_db < 0.0:
                    error_code = self._classify_error(RuntimeError("dynamic_range_db produced invalid output"), "dynamic_range_failed")
                    raise RuntimeError(f"quality | dynamic_range_db produced invalid output (error_code={error_code})")
                features["dynamic_range_db"] = dynamic_range_db
            except Exception as e:
                error_code = self._classify_error(e, "dynamic_range_failed")
                raise RuntimeError(f"quality | dynamic_range failed (error_code={error_code}): {e}") from e

            # snr_db removed (Audit v3): was duplicate of dynamic_range_db (p95-p5)

            # Additional ML/analytics metrics
            if self.enable_basic_metrics:
                features.update(self._calc_additional_metrics(
                    np.array([features.get("dc_offset", 0.0)]),
                    np.array([features.get("clipping_ratio", 0.0)]),
                    np.array([features.get("crest_factor_db", 0.0)]),
                    np.array([dynamic_range_db]),
                ))

        # Frame analysis (feature-gated)
        if self.enable_frame_analysis:
            if self.progress_callback:
                self.progress_callback("quality", 5, 6, "Computing frame analysis")
            levels = self._compute_frame_levels(audio, sr)
            if len(levels) > 0:
                features["frame_levels_distribution"] = {
                    "mean": float(np.mean(levels)),
                    "std": float(np.std(levels)),
                    "min": float(np.min(levels)),
                    "max": float(np.max(levels)),
                    "median": float(np.median(levels)),
                }
                
                # Time series for frame analysis
                if self.enable_time_series:
                    features["frame_levels_db_series"] = levels.tolist() if isinstance(levels, np.ndarray) else levels
                    
                    # Compute RMS series
                    frame_len = max(1, int((self.frame_len_ms / 1000.0) * sr))
                    hop = max(1, int((self.hop_ms / 1000.0) * sr))
                    rms_series = []
                    if len(audio) >= frame_len:
                        n_frames = 1 + (len(audio) - frame_len) // hop
                        for i in range(0, len(audio) - frame_len + 1, hop):
                            frm = audio[i : i + frame_len]
                            rms = np.sqrt(np.mean(frm * frm) + eps)
                            rms_series.append(float(rms))
                    features["frame_rms_series"] = rms_series
                    
                    # Clipping segments series
                    if self.enable_basic_metrics:
                        clipping_segments = []
                        for i in range(0, len(audio) - frame_len + 1, hop):
                            frm = audio[i : i + frame_len]
                            clipping = float(np.mean(np.abs(frm) >= self.clip_threshold))
                            clipping_segments.append(clipping)
                        features["clipping_segments_series"] = clipping_segments

        return features

    def _compute_frame_levels(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Вычислить уровни кадров в dB.
        
        Args:
            audio: Аудио сигнал
            sr: Частота дискретизации
        
        Returns:
            Массив уровней кадров в dB
        """
        frame_len = max(1, int((self.frame_len_ms / 1000.0) * sr))
        hop = max(1, int((self.hop_ms / 1000.0) * sr))
        eps = 1e-12
        
        levels: list = []
        if len(audio) >= frame_len:
            n_frames = 1 + (len(audio) - frame_len) // hop
            if n_frames > 0:
                try:
                    shape = (n_frames, frame_len)
                    strides = (audio.strides[-1] * hop, audio.strides[-1])
                    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
                    rms_frames = np.sqrt(np.mean(frames * frames, axis=1) + eps)
                    levels = (20.0 * np.log10(rms_frames + eps)).tolist()
                except Exception:
                    # Fallback к циклу при отсутствии поддержки
                    levels = []
                    for i in range(0, len(audio) - frame_len + 1, hop):
                        frm = audio[i : i + frame_len]
                        lvl = 20.0 * np.log10(np.sqrt(np.mean(frm * frm) + eps))
                        levels.append(lvl)
        
        return np.asarray(levels, dtype=np.float32) if levels else np.array([], dtype=np.float32)

    def _calc_additional_metrics(
        self,
        dc_offset: np.ndarray,
        clipping_ratio: np.ndarray,
        crest_factor_db: np.ndarray,
        dynamic_range_db: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Вычислить дополнительные метрики для ML/аналитики (analytics tier).
        Audit v3: dc_offset_abs removed (duplicate), snr_db/snr_stability removed.
        """
        metrics: Dict[str, Any] = {}
        if dc_offset.size == 0:
            return {
                "clipping_segments_count": 0,
                "crest_factor_median": 0.0,
                "dynamic_range_stability": 0.0,
                "quality_score": 0.0,
            }
        if clipping_ratio.size > 0:
            metrics["clipping_segments_count"] = int(np.sum(clipping_ratio > 0.0))
        if crest_factor_db.size > 0:
            metrics["crest_factor_median"] = float(np.median(crest_factor_db))
        if dynamic_range_db.size > 0:
            metrics["dynamic_range_stability"] = float(1.0 / (1.0 + np.std(dynamic_range_db)))
        quality_components = []
        if dc_offset.size > 0:
            quality_components.append(max(0.0, 1.0 - min(1.0, np.abs(np.mean(dc_offset)))))
        if clipping_ratio.size > 0:
            quality_components.append(max(0.0, 1.0 - np.mean(clipping_ratio)))
        if crest_factor_db.size > 0:
            quality_components.append(float(np.clip(np.mean(crest_factor_db) / 40.0, 0.0, 1.0)))
        if dynamic_range_db.size > 0:
            quality_components.append(min(1.0, np.mean(dynamic_range_db) / 100.0))
        metrics["quality_score"] = float(np.mean(quality_components)) if quality_components else 0.0
        return metrics

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        quality_extractor поддерживает batch processing через extract_batch_segments()
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
                logger.error(f"quality | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )
            
            if not segments:
                logger.error(f"quality | Missing segments for file_id={file_id}")
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
                logger.error(f"quality | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
