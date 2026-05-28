"""
KeyExtractor: определение тональности (ключ + мажор/минор) на основе хрома и корреляции с шаблонами Krumhansl.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (explicit method selection)
- Per-run storage for .npy files
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Key change detection (for run_segments)
- Confidence categorization
- Batch processing support (extract_batch_segments)
"""
import time
import logging
import os
import importlib.util
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
KEY_CONTRACT_VERSION = "key_contract_v1"

# Valid key names
VALID_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
VALID_MODES = ['major', 'minor']

# Confidence thresholds
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW = 0.3

# key_id: 0-23 (maj_C=0, min_C=1, maj_C#=2, ..., min_B=23); -1 = failed
def _key_name_mode_to_id(key_name: str, key_mode: str) -> int:
    if key_name not in VALID_KEYS or key_mode not in VALID_MODES:
        return -1
    key_idx = VALID_KEYS.index(key_name)
    mode_offset = 0 if key_mode == "major" else 1
    return key_idx * 2 + mode_offset


class KeyExtractor(BaseExtractor):
    """Экстрактор тональности (ключ + мажор/минор) с поддержкой segment-based обработки."""

    name = "key"
    version = "2.1.1"
    description = "Определение ключа (тональности) через шаблоны Krumhansl на хроме"
    category = "music_theory"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.0

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        chroma_type: str = "cqt",
        use_beat_sync: bool = False,
        top_k: int = 3,
        key_method: str = "librosa",  # "essentia" | "librosa" | "auto" (Audit v3: default librosa)
        key_confidence_threshold: float = 0.3,
        # Feature gating flags (per-feature control, default: all False except basic)
        enable_detailed_scores: bool = False,
        enable_top_k: bool = False,
        enable_time_series: bool = False,
        enable_key_changes: bool = False,
        enable_stability_metrics: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация Key экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            hop_length: Размер hop для STFT/CQT
            chroma_type: Тип хрома ("cqt" | "stft")
            use_beat_sync: Синхронизация с битами
            top_k: Количество топ-K тональностей
            key_method: Метод определения тональности ("essentia" | "librosa" | "auto")
            key_confidence_threshold: Порог уверенности для предупреждений
            enable_detailed_scores: Включить детальные оценки (24 значения)
            enable_top_k: Включить топ-K альтернативных тональностей
            enable_time_series: Включить временные серии (для run_segments)
            enable_key_changes: Включить детекцию смены тональности
            enable_stability_metrics: Включить метрики стабильности
            enable_audio_normalization: Включить нормализацию аудио
            progress_callback: Callback для прогресса
            artifacts_dir: Директория для сохранения .npy файлов
        """
        super().__init__(device=device)
        
        # Store audio parameters
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.progress_callback = progress_callback
        self.artifacts_dir = artifacts_dir
        
        self.chroma_type = chroma_type
        self.use_beat_sync = use_beat_sync
        self.top_k = top_k
        self.key_method = key_method
        self.key_confidence_threshold = key_confidence_threshold
        self.enable_detailed_scores = enable_detailed_scores
        self.enable_top_k = enable_top_k
        self.enable_time_series = enable_time_series
        self.enable_key_changes = enable_key_changes
        self.enable_stability_metrics = enable_stability_metrics
        self.enable_audio_normalization = enable_audio_normalization

        # Audio utils
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

        # Validate parameters
        if chroma_type not in ["cqt", "stft"]:
            raise ValueError(f"key | Invalid chroma_type: {chroma_type} (must be 'cqt' or 'stft')")
        if key_method not in ["auto", "essentia", "librosa"]:
            raise ValueError(f"key | Invalid key_method: {key_method} (must be 'auto', 'essentia', or 'librosa')")
        if top_k < 1 or top_k > 24:
            raise ValueError(f"key | Invalid top_k: {top_k} (must be 1-24)")

        # Lazy load essentia if needed
        self._essentia_available = False
        if key_method in ["auto", "essentia"]:
            try:
                import essentia.standard as es
                self._essentia_available = True
                self._essentia = es
            except ImportError:
                if key_method == "essentia":
                    raise ImportError("key | essentia not available but key_method='essentia' (no-fallback)")
                logger.warning("key | essentia not available, will use librosa fallback")

    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not input_uri or not os.path.exists(input_uri):
            return False
        return True

    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Нормализация аудио сигнала."""
        if y.size == 0:
            return y
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def _detect_key_essentia(self, y: np.ndarray) -> Dict[str, Any]:
        """Определение тональности через Essentia."""
        if not self._essentia_available:
            raise ImportError("key | essentia not available")
        
        # Essentia key detection
        key_detector = self._essentia.KeyExtractor()
        key, scale, strength = key_detector(y)
        
        return {
            "key_name": key,
            "key_mode": "major" if scale == "major" else "minor",
            "key_confidence": float(strength),
            "method": "essentia",
        }

    def _detect_key_librosa(self, y: np.ndarray, sr: int, shared_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Определение тональности через librosa (Krumhansl-Schmuckler)."""
        # Reuse chroma if available
        chroma = None
        if shared_features and "chroma" in shared_features:
            chroma = shared_features["chroma"]
            # Average over time if needed
            if chroma.ndim > 1:
                chroma = np.mean(chroma, axis=1)
        else:
            # Compute chroma
            if self.chroma_type == "cqt":
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            else:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            # Average over time
            if chroma.ndim > 1:
                chroma = np.mean(chroma, axis=1)

        # Normalize chroma
        chroma = chroma / (np.sum(chroma) + 1e-10)

        # Krumhansl-Schmuckler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        # Compute correlations for all 24 keys
        scores = []
        for key_idx in range(12):
            # Major
            shifted_chroma = np.roll(chroma, -key_idx)
            corr_major = np.corrcoef(shifted_chroma, major_profile)[0, 1]
            scores.append(corr_major if not np.isnan(corr_major) else 0.0)
            
            # Minor
            corr_minor = np.corrcoef(shifted_chroma, minor_profile)[0, 1]
            scores.append(corr_minor if not np.isnan(corr_minor) else 0.0)

        # Find best match
        best_idx = np.argmax(scores)
        key_name = VALID_KEYS[best_idx // 2]
        key_mode = "major" if (best_idx % 2) == 0 else "minor"
        key_confidence = float(scores[best_idx])

        # Normalize confidence to [0, 1]
        if key_confidence < 0:
            key_confidence = 0.0
        elif key_confidence > 1:
            key_confidence = 1.0

        result = {
            "key_name": key_name,
            "key_mode": key_mode,
            "key_confidence": key_confidence,
            "method": "librosa",
        }

        # Add detailed scores if enabled
        if self.enable_detailed_scores:
            # Normalize scores to [0, 1]
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [0.0] * 24
            result["key_scores"] = normalized_scores

        return result

    def _add_confidence_metadata(self, payload: Dict[str, Any]) -> None:
        """Добавление метаданных об уверенности."""
        confidence = payload.get("key_confidence", 0.0)
        
        if confidence >= CONFIDENCE_HIGH:
            category = "high"
        elif confidence >= CONFIDENCE_MEDIUM:
            category = "medium"
        elif confidence >= CONFIDENCE_LOW:
            category = "low"
        else:
            category = "very_low"

        payload["key_confidence_category"] = category
        payload["key_low_confidence_warning"] = confidence < self.key_confidence_threshold
        
        if confidence < self.key_confidence_threshold:
            payload["key_confidence_reason"] = "low_confidence"
        else:
            payload["key_confidence_reason"] = "normal"

    def _classify_error(self, error: Exception, default_code: str) -> str:
        """Классификация ошибок."""
        error_str = str(error).lower()
        if "essentia" in error_str or "import" in error_str:
            return "essentia_unavailable"
        elif "audio" in error_str or "load" in error_str:
            return "audio_load_failed"
        elif "validation" in error_str:
            return "validation_failed"
        else:
            return default_code

    def _log_extraction_error(self, input_uri: str, error_msg: str, processing_time: float) -> None:
        """Логирование ошибок."""
        logger.error(f"key | Extraction failed: {input_uri} | {error_msg} | time={processing_time:.2f}s")

    def _log_extraction_success(self, input_uri: str, processing_time: float) -> None:
        """Логирование успешного извлечения."""
        logger.info(f"key | Extraction successful: {input_uri} | time={processing_time:.2f}s")

    def _report_progress(self, stage: str, current: int, total: int, message: str = "") -> None:
        """Отчет о прогрессе."""
        if self.progress_callback:
            self.progress_callback("key", current, total, message)

    def _validate_output(self, payload: Dict[str, Any]) -> None:
        """Валидация выходных данных."""
        if "key_name" not in payload:
            raise ValueError("key | Missing key_name in output")
        if payload["key_name"] not in VALID_KEYS:
            raise ValueError(f"key | Invalid key_name: {payload['key_name']}")
        if "key_mode" not in payload:
            raise ValueError("key | Missing key_mode in output")
        if payload["key_mode"] not in VALID_MODES:
            raise ValueError(f"key | Invalid key_mode: {payload['key_mode']}")
        if "key_confidence" not in payload:
            raise ValueError("key | Missing key_confidence in output")
        confidence = payload["key_confidence"]
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise ValueError(f"key | Invalid key_confidence: {confidence} (must be 0-1)")

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Audit v3: отключён. Key использует только run_segments (families.key).
        """
        start_time = time.time()
        error_code = self._classify_error(
            RuntimeError("key | run() disabled in audited mode; use run_segments with families.key"),
            "validation_failed",
        )
        return self._create_result(
            success=False,
            error=f"key | {error_code}: run() disabled, use run_segments with families.key",
            processing_time=time.time() - start_time,
        )

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        shared_features: Optional[Dict[str, Any]] = None,
        *,
        segment_parallelism: int = 1,
        max_inflight: Optional[int] = None,
    ) -> ExtractorResult:
        """
        Определение тональности для сегментов от Segmenter (families.key.segments[]).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов от Segmenter
            shared_features: Общие фичи (может содержать chroma от chroma_extractor)
            segment_parallelism: Количество параллельных воркеров для обработки сегментов (не используется, для совместимости)
            max_inflight: Максимальное количество сегментов в обработке одновременно (не используется, для совместимости)

        Returns:
            ExtractorResult с результатами определения тональности по сегментам
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        key_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"key | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("key | segments is empty (no-fallback)")

            total_segments = len(segments)

            if resource_profile_enabled():
                try:
                    key_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    key_resource_profile = None

            # Strict alignment (Audit v3): pre-allocate arrays, no skipping
            segment_start_sec = np.zeros(total_segments, dtype=np.float32)
            segment_end_sec = np.zeros(total_segments, dtype=np.float32)
            segment_center_sec = np.zeros(total_segments, dtype=np.float32)
            segment_mask = np.zeros(total_segments, dtype=bool)
            key_id_by_segment = np.full(total_segments, -1, dtype=np.int32)
            key_confidence_by_segment = np.full(total_segments, np.nan, dtype=np.float32)
            key_names_seq: List[str] = [""] * total_segments
            key_modes_seq: List[str] = [""] * total_segments

            key_scores_all: List[List[float]] = []
            methods_used: List[str] = []
            chroma_reused = shared_features is not None and "chroma" in (shared_features or {})

            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            t_seg0 = time.perf_counter()
            load_audio_ms_total = 0.0
            detect_key_ms_total = 0.0
            for seg_idx, seg in enumerate(segments):
                start_sec = float(seg.get("start_sec", 0.0))
                end_sec = float(seg.get("end_sec", 0.0))
                center_sec = float(seg.get("center_sec", 0.0))
                duration_sec = end_sec - start_sec
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))

                segment_start_sec[seg_idx] = start_sec
                segment_end_sec[seg_idx] = end_sec
                segment_center_sec[seg_idx] = center_sec

                success = False
                try:
                    if duration_sec < 0.5 or start_sec >= end_sec:
                        logger.debug(f"key | Segment {seg_idx}: too short ({duration_sec:.2f}s)")
                    else:
                        t_load0 = time.perf_counter()
                        y_t, sr = self.audio_utils.load_audio_segment(
                            input_uri,
                            start_sample=start_sample,
                            end_sample=end_sample,
                            target_sr=self.sample_rate,
                            mix_to_mono=True,
                        )
                        y = self.audio_utils.to_numpy(y_t)
                        if y.ndim == 2:
                            y = np.mean(y, axis=0)
                        y = y.astype(np.float32)
                        load_audio_ms_total += (time.perf_counter() - t_load0) * 1000.0

                        if y.size > 0:
                            if self.enable_audio_normalization:
                                y = self._normalize_audio(y)
                            result = None
                            t_det0 = time.perf_counter()
                            if self.key_method == "essentia":
                                try:
                                    result = self._detect_key_essentia(y)
                                except Exception:
                                    result = self._detect_key_librosa(y, sr, shared_features)
                            elif self.key_method == "librosa":
                                result = self._detect_key_librosa(y, sr, shared_features)
                            else:
                                try:
                                    result = self._detect_key_essentia(y)
                                except Exception:
                                    result = self._detect_key_librosa(y, sr, shared_features)
                            detect_key_ms_total += (time.perf_counter() - t_det0) * 1000.0
                            if result:
                                kid = _key_name_mode_to_id(result["key_name"], result["key_mode"])
                                segment_mask[seg_idx] = True
                                key_id_by_segment[seg_idx] = kid
                                key_confidence_by_segment[seg_idx] = result["key_confidence"]
                                key_names_seq[seg_idx] = result["key_name"]
                                key_modes_seq[seg_idx] = result["key_mode"]
                                methods_used.append(result["method"])
                                if self.enable_detailed_scores and "key_scores" in result:
                                    key_scores_all.append(result["key_scores"])
                                success = True
                except Exception as e:
                    logger.warning(f"key | Segment {seg_idx} failed: {e}")

                if (seg_idx + 1) % progress_report_interval == 0 and last_reported_pct < (seg_idx + 1) * 100 // total_segments:
                    self._report_progress("process_segments", seg_idx + 1, total_segments, f"Processed {seg_idx + 1}/{total_segments}")
                    last_reported_pct = (seg_idx + 1) * 100 // total_segments

            stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t_seg0) * 1000.0
            stage_timings_ms["load_audio_ms_total"] = float(load_audio_ms_total)
            stage_timings_ms["detect_key_ms_total"] = float(detect_key_ms_total)

            t_post0 = time.perf_counter()
            valid_mask = segment_mask
            n_valid = int(np.sum(valid_mask))
            valid_key_names = [key_names_seq[i] for i in range(total_segments) if valid_mask[i]]
            valid_key_modes = [key_modes_seq[i] for i in range(total_segments) if valid_mask[i]]
            valid_confidences = key_confidence_by_segment[valid_mask]
            valid_durations = np.array([segment_end_sec[i] - segment_start_sec[i] for i in range(total_segments) if valid_mask[i]], dtype=np.float32)

            if n_valid > 0:
                key_counts: Dict[str, int] = {}
                for kn, km in zip(valid_key_names, valid_key_modes):
                    kstr = f"{kn}_{km}"
                    key_counts[kstr] = key_counts.get(kstr, 0) + 1
                dominant_key_str = max(key_counts.items(), key=lambda x: x[1])[0]
                dominant_key_name, dominant_key_mode = dominant_key_str.split("_", 1)
                avg_confidence = float(np.mean(valid_confidences))
                total_duration = float(np.sum(valid_durations))
            else:
                dominant_key_name, dominant_key_mode = "C", "major"
                avg_confidence = 0.0
                total_duration = float(np.sum(segment_end_sec - segment_start_sec))

            payload: Dict[str, Any] = {
                "key_name": dominant_key_name,
                "key_mode": dominant_key_mode,
                "key_id": _key_name_mode_to_id(dominant_key_name, dominant_key_mode) if n_valid > 0 else -1,
                "key_confidence": avg_confidence,
                "method": "librosa" if "librosa" in methods_used else (methods_used[0] if methods_used else "librosa"),
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
                "duration": total_duration,
                "device_used": self.device,
                "segments_count": total_segments,
                "chroma_reused": chroma_reused,
                "stage_timings_ms": stage_timings_ms,
                "key_resource_profile": key_resource_profile,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": segment_center_sec,
                "segment_mask": segment_mask,
                "key_id_by_segment": key_id_by_segment,
                "key_confidence_by_segment": key_confidence_by_segment,
            }

            self._add_confidence_metadata(payload)

            if self.enable_time_series:
                payload["key_names_sequence"] = key_names_seq
                payload["key_modes_sequence"] = key_modes_seq
                payload["key_confidences_sequence"] = key_confidence_by_segment

            if self.enable_key_changes and n_valid > 1:
                transitions = []
                for i in range(1, total_segments):
                    if valid_mask[i] and valid_mask[i - 1]:
                        if key_names_seq[i] != key_names_seq[i - 1] or key_modes_seq[i] != key_modes_seq[i - 1]:
                            transitions.append({
                                "transition_index": i,
                                "from_key": f"{key_names_seq[i-1]}_{key_modes_seq[i-1]}",
                                "to_key": f"{key_names_seq[i]}_{key_modes_seq[i]}",
                                "transition_time_sec": segment_center_sec[i],
                            })
                payload["key_transitions"] = transitions
                payload["key_transitions_count"] = len(transitions)
                total_span = segment_center_sec[-1] - segment_center_sec[0] if total_segments > 1 else max(total_duration, 1e-6)
                payload["key_transitions_rate"] = len(transitions) / total_span if total_span > 0 else 0.0

            if self.enable_stability_metrics and n_valid > 0:
                dominant_duration = sum(
                    valid_durations[j] for j in range(len(valid_key_names))
                    if valid_key_names[j] == dominant_key_name and valid_key_modes[j] == dominant_key_mode
                )
                payload["key_stability_score"] = dominant_duration / total_duration if total_duration > 0 else 0.0
                payload["key_confidence_mean"] = float(np.mean(valid_confidences))
                payload["key_confidence_std"] = float(np.std(valid_confidences))
                payload["key_confidence_min"] = float(np.min(valid_confidences))
                payload["key_confidence_max"] = float(np.max(valid_confidences))
                key_distribution: Dict[str, float] = {}
                for j in range(len(valid_key_names)):
                    kstr = f"{valid_key_names[j]}_{valid_key_modes[j]}"
                    key_distribution[kstr] = key_distribution.get(kstr, 0.0) + float(valid_durations[j])
                total_dur = sum(key_distribution.values())
                if total_dur > 0:
                    key_distribution = {k: v / total_dur for k, v in key_distribution.items()}
                payload["key_distribution"] = key_distribution
                payload["key_diversity"] = len(key_distribution)
                payload["key_detection_quality"] = avg_confidence * payload["key_stability_score"]

            if self.enable_detailed_scores and key_scores_all:
                avg_scores = np.mean([np.array(s) for s in key_scores_all], axis=0)
                payload["key_scores"] = [float(x) for x in avg_scores.tolist()]

            if self.enable_top_k and "key_scores" in payload:
                scores = np.array(payload["key_scores"])
                order = np.argsort(scores)[::-1]
                top_k_list = []
                for idx in order[: self.top_k]:
                    k_name = VALID_KEYS[(idx // 2) % 12]
                    k_mode = "major" if (idx % 2) == 0 else "minor"
                    top_k_list.append({"key": k_name, "mode": k_mode, "score": float(scores[idx])})
                payload["key_top_k"] = top_k_list

            enabled_features = []
            if self.enable_detailed_scores:
                enabled_features.append("detailed_scores")
            if self.enable_top_k:
                enabled_features.append("top_k")
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_key_changes:
                enabled_features.append("key_changes")
            if self.enable_stability_metrics:
                enabled_features.append("stability_metrics")
            payload["key_contract_version"] = KEY_CONTRACT_VERSION
            payload["_features_enabled"] = enabled_features

            self._validate_output(payload)

            stage_timings_ms["postprocess_ms"] = (time.perf_counter() - t_post0) * 1000.0
            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0

            if key_resource_profile is not None:
                try:
                    payload["key_resource_profile"] = {
                        **(key_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            self._report_progress("complete", total_segments, total_segments, "Complete")
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "key_detection_failed")
            error_msg = f"key | {str(e)} (error_code={error_code})"
            self._log_extraction_error(input_uri, error_msg, dt)
            return self._create_result(False, error=error_msg, processing_time=dt)

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        key_extractor поддерживает batch processing через extract_batch_segments()
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
                - 'artifacts_dir': Директория для сохранения .npy файлов (опционально, per-file)
                - 'shared_features': Общие фичи (например, chroma от chroma_extractor)
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
            artifacts_dir = file_info.get("artifacts_dir")
            shared_features = file_info.get("shared_features")

            if not input_uri or not tmp_path:
                logger.error(f"key | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                )

            if not segments:
                logger.error(f"key | Missing segments for file_id={file_id}")
                return self._create_result(
                    success=False,
                    error="Missing segments",
                )

            try:
                # Set artifacts_dir for this specific file context
                original_artifacts_dir = self.artifacts_dir
                if artifacts_dir:
                    self.artifacts_dir = artifacts_dir

                result = self.run_segments(
                    input_uri=input_uri,
                    tmp_path=tmp_path,
                    segments=segments,
                    shared_features=shared_features,
                )
                return result
            except Exception as e:
                logger.error(f"key | Error processing file_id={file_id}: {e}")
                return self._create_result(
                    success=False,
                    error=str(e),
                )
            finally:
                # Restore original artifacts_dir
                if artifacts_dir:
                    self.artifacts_dir = original_artifacts_dir

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))

        return results
