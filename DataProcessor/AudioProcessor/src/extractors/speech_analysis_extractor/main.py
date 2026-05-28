"""
SpeechAnalysisExtractor (non-baseline, production-safe).

Goal:
- Provide a compact "speech overview" by combining:
  - ASR token IDs (inprocess Whisper model via ModelManager) on Segmenter families.asr windows
  - Speaker diarization (inprocess pyannote.audio + whisperx via ModelManager) on Segmenter families.diarization windows
  - Optional pitch (signal processing) on full audio

Important:
- No raw transcript text is stored.
- No "alignment" between ASR tokens and diarization speakers is attempted (Whisper token timing is not available).
- No runtime downloads (ModelManager enforced by sub-extractors).
- NO FALLBACK: компонент использует результаты от зависимых компонентов (asr, speaker_diarization, pitch) из extractor_results.
  Если зависимости не предоставлены и feature flags включены, компонент падает с ошибкой (fail-fast).
- Размеры моделей определяются самими зависимыми компонентами, а не этим компонентом.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import capture_speech_analysis_resource_profile, is_speech_analysis_resource_profile_enabled

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
SPEECH_ANALYSIS_CONTRACT_VERSION = "speech_analysis_contract_v1"


class SpeechAnalysisExtractor(BaseExtractor):
    name = "speech_analysis_extractor"
    version = "2.1.1"
    description = "Speech analysis bundle: ASR token stats + diarization + optional pitch (no raw text)"
    category = "speech"
    dependencies = ["dp_models", "numpy", "librosa"]
    estimated_duration = 10.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # Inprocess ASR/diarization models; pitch is CPU

    def __init__(
        self,
        device: str = "auto",
        *,
        sample_rate: int = 16000,
        pitch_enabled: bool = False,
        pitch_backend: str = "classic",
        # Feature gating flags (per-feature control, default: all False)
        enable_asr_metrics: bool = False,
        enable_diarization_metrics: bool = False,
        enable_pitch_metrics: bool = False,
        # Silence detection
        silence_peak_threshold: float = 1e-3,
        silence_rms_threshold: float = 1e-4,
        enable_silence_detection: bool = True,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ):
        """
        Инициализация bundle экстрактора анализа речи.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            pitch_enabled: Включить анализ pitch
            pitch_backend: Backend для pitch (classic/torchcrepe)
            enable_asr_metrics: Включить ASR метрики (требует включения extractor "asr" - обязательная зависимость, падение при отсутствии)
            enable_diarization_metrics: Включить diarization метрики (требует включения extractor "speaker_diarization" - обязательная зависимость, падение при отсутствии)
            enable_pitch_metrics: Включить pitch метрики (требует pitch_enabled=true и включения extractor "pitch" - обязательная зависимость, падение при отсутствии)
            silence_peak_threshold: Порог peak для детекции тишины
            silence_rms_threshold: Порог RMS для детекции тишины
            enable_silence_detection: Включить проверку на тишину
            progress_callback: Callback для прогресса (extractor_name, current, total, message)
        
        Примечание: Компонент использует результаты от зависимых компонентов (asr, speaker_diarization, pitch) из extractor_results.
        Размеры моделей определяются самими зависимыми компонентами, а не этим компонентом.
        NO FALLBACK: если зависимости не предоставлены, компонент падает с ошибкой (fail-fast).
        """
        super().__init__(device=device)
        self.sample_rate = int(sample_rate)
        self.pitch_enabled = bool(pitch_enabled)
        
        # Feature gating flags
        self.enable_asr_metrics = bool(enable_asr_metrics)
        self.enable_diarization_metrics = bool(enable_diarization_metrics)
        self.enable_pitch_metrics = bool(enable_pitch_metrics)
        
        # Silence detection
        self.silence_peak_threshold = float(silence_peak_threshold)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.enable_silence_detection = bool(enable_silence_detection)
        
        # Progress callback
        self.progress_callback = progress_callback

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # Примечание: ASR, diarization и pitch результаты должны быть получены из extractor_results
        # (передаются через run_bundle из оркестратора)
        # Компонент не создает под-экстракторы, а использует существующие результаты
        # NO FALLBACK: если зависимости не предоставлены, компонент падает с ошибкой

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.
        
        Args:
            error: Исключение
            context: Контекст ошибки (asr_failed, diarization_failed, pitch_failed, silence_detection_failed, validation_failed, segments_invalid)
        
        Returns:
            error_code: один из:
                - asr_failed
                - diarization_failed
                - pitch_failed
                - silence_detection_failed
                - validation_failed
                - segments_invalid
                - audio_too_short
                - unknown
        """
        error_str = str(error).lower()
        
        if "asr" in error_str or "whisper" in error_str or context == "asr_failed":
            return "asr_failed"
        if "diarization" in error_str or "speaker" in error_str or context == "diarization_failed":
            return "diarization_failed"
        if "pitch" in error_str or "f0" in error_str or context == "pitch_failed":
            return "pitch_failed"
        if "silence" in error_str or context == "silence_detection_failed":
            return "silence_detection_failed"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "validation_failed"
        if "segments" in error_str or "empty" in error_str or context == "segments_invalid":
            return "segments_invalid"
        if "too short" in error_str or "<5" in error_str:
            return "audio_too_short"
        
        return "unknown"

    def _validate_segments(self, segments: List[Dict[str, Any]], segment_type: str) -> tuple[bool, Optional[str]]:
        """
        Полная валидация структуры сегментов: проверка обязательных полей, типов, диапазонов.
        
        Args:
            segments: Список сегментов
            segment_type: Тип сегментов (asr/diarization)
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(segments, list):
            return False, f"speech_analysis | {segment_type}_segments must be a list, got {type(segments)}"
        
        if len(segments) == 0:
            return False, f"speech_analysis | {segment_type}_segments is empty (no-fallback)"
        
        required_fields = ["start_sample", "end_sample", "start_sec", "end_sec", "center_sec"]
        
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                return False, f"speech_analysis | {segment_type}_segments[{i}] must be a dict, got {type(seg)}"
            
            for field in required_fields:
                if field not in seg:
                    return False, f"speech_analysis | {segment_type}_segments[{i}] missing required field: {field}"
            
            # Validate types and ranges
            try:
                start_sample = int(seg["start_sample"])
                end_sample = int(seg["end_sample"])
                start_sec = float(seg["start_sec"])
                end_sec = float(seg["end_sec"])
                center_sec = float(seg["center_sec"])
                
                if start_sample < 0 or end_sample < 0:
                    return False, f"speech_analysis | {segment_type}_segments[{i}] has negative sample indices"
                if start_sample >= end_sample:
                    return False, f"speech_analysis | {segment_type}_segments[{i}] has start_sample >= end_sample"
                if start_sec < 0 or end_sec < 0:
                    return False, f"speech_analysis | {segment_type}_segments[{i}] has negative time indices"
                if start_sec >= end_sec:
                    return False, f"speech_analysis | {segment_type}_segments[{i}] has start_sec >= end_sec"
                if not (start_sec <= center_sec <= end_sec):
                    return False, f"speech_analysis | {segment_type}_segments[{i}] has center_sec outside [start_sec, end_sec]"
            except (ValueError, TypeError) as e:
                return False, f"speech_analysis | {segment_type}_segments[{i}] has invalid field types: {e}"
        
        return True, None

    def _validate_asr_payload(self, payload: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация payload от ASR extractor: проверка типов, диапазонов, наличия обязательных полей.
        
        Args:
            payload: Payload от ASR extractor
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(payload, dict):
            return False, "speech_analysis | asr_payload must be a dict"
        
        # Check required fields
        required_fields = ["segments_count"]
        for field in required_fields:
            if field not in payload:
                return False, f"speech_analysis | asr_payload missing required field: {field}"
        
        # Validate segments_count
        segments_count = payload.get("segments_count")
        if not isinstance(segments_count, (int, np.integer)) or segments_count < 0:
            return False, f"speech_analysis | asr_payload.segments_count must be non-negative int, got {type(segments_count)}"
        
        # Validate token_ids_by_segment if present
        token_ids_by_segment = payload.get("token_ids_by_segment")
        if token_ids_by_segment is not None:
            if not isinstance(token_ids_by_segment, list):
                return False, f"speech_analysis | asr_payload.token_ids_by_segment must be a list, got {type(token_ids_by_segment)}"
            if len(token_ids_by_segment) != segments_count:
                return False, f"speech_analysis | asr_payload.token_ids_by_segment length ({len(token_ids_by_segment)}) != segments_count ({segments_count})"
        
        # Validate lang_id_by_segment if present
        lang_id_by_segment = payload.get("lang_id_by_segment")
        if lang_id_by_segment is not None:
            if not isinstance(lang_id_by_segment, (list, np.ndarray)):
                return False, f"speech_analysis | asr_payload.lang_id_by_segment must be a list or array, got {type(lang_id_by_segment)}"
            if len(lang_id_by_segment) != segments_count:
                return False, f"speech_analysis | asr_payload.lang_id_by_segment length ({len(lang_id_by_segment)}) != segments_count ({segments_count})"
        
        return True, None

    def _validate_diarization_payload(self, payload: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация payload от diarization extractor: проверка типов, диапазонов, наличия обязательных полей.
        
        Args:
            payload: Payload от diarization extractor
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(payload, dict):
            return False, "speech_analysis | diarization_payload must be a dict"
        
        # Check required fields
        required_fields = ["segments_count", "speaker_count"]
        for field in required_fields:
            if field not in payload:
                return False, f"speech_analysis | diarization_payload missing required field: {field}"
        
        # Validate segments_count
        segments_count = payload.get("segments_count")
        if not isinstance(segments_count, (int, np.integer)) or segments_count < 0:
            return False, f"speech_analysis | diarization_payload.segments_count must be non-negative int, got {type(segments_count)}"
        
        # Validate speaker_count
        speaker_count = payload.get("speaker_count")
        if not isinstance(speaker_count, (int, np.integer)) or speaker_count < 0:
            return False, f"speech_analysis | diarization_payload.speaker_count must be non-negative int, got {type(speaker_count)}"
        
        # Validate speaker_segments if present
        speaker_segments = payload.get("speaker_segments")
        if speaker_segments is not None:
            if not isinstance(speaker_segments, list):
                return False, f"speech_analysis | diarization_payload.speaker_segments must be a list, got {type(speaker_segments)}"
            for i, seg in enumerate(speaker_segments):
                if not isinstance(seg, dict):
                    return False, f"speech_analysis | diarization_payload.speaker_segments[{i}] must be a dict, got {type(seg)}"
                if "speaker_id" not in seg:
                    return False, f"speech_analysis | diarization_payload.speaker_segments[{i}] missing speaker_id"
        
        # Validate speaker_ids if present
        speaker_ids = payload.get("speaker_ids")
        if speaker_ids is not None:
            if not isinstance(speaker_ids, (list, np.ndarray)):
                return False, f"speech_analysis | diarization_payload.speaker_ids must be a list or array, got {type(speaker_ids)}"
        
        return True, None

    def _validate_pitch_payload(self, payload: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация payload от pitch extractor: проверка типов, диапазонов, наличия обязательных полей.
        
        Args:
            payload: Payload от pitch extractor
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(payload, dict):
            return False, "speech_analysis | pitch_payload must be a dict"
        
        # Validate f0_mean if present
        f0_mean = payload.get("f0_mean")
        if f0_mean is not None:
            try:
                f0_mean = float(f0_mean)
                if f0_mean < 0 or f0_mean > 2000:
                    return False, f"speech_analysis | pitch_payload.f0_mean out of range [0, 2000]: {f0_mean}"
            except (ValueError, TypeError):
                return False, f"speech_analysis | pitch_payload.f0_mean must be float, got {type(f0_mean)}"
        
        # Validate f0_std if present
        f0_std = payload.get("f0_std")
        if f0_std is not None:
            try:
                f0_std = float(f0_std)
                if f0_std < 0:
                    return False, f"speech_analysis | pitch_payload.f0_std must be non-negative, got {f0_std}"
            except (ValueError, TypeError):
                return False, f"speech_analysis | pitch_payload.f0_std must be float, got {type(f0_std)}"
        
        return True, None

    def run_bundle(
        self,
        input_uri: str,
        tmp_path: str,
        *,
        asr_segments: List[Dict[str, Any]],
        diar_segments: List[Dict[str, Any]],
        asr_result: Optional[Dict[str, Any]] = None,  # Результат от asr_extractor (обязателен если enable_asr_metrics=True) - словарь из extractor_results
        diarization_result: Optional[Dict[str, Any]] = None,  # Результат от speaker_diarization_extractor (обязателен если enable_diarization_metrics=True) - словарь из extractor_results
        pitch_result: Optional[Dict[str, Any]] = None,  # Результат от pitch_extractor (обязателен если pitch_enabled=True и enable_pitch_metrics=True) - словарь из extractor_results
    ) -> ExtractorResult:
        """
        Segmenter-driven speech analysis: combine ASR + diarization + optional pitch.
        
        Новая архитектура: компонент использует существующие результаты зависимых компонентов
        (asr_result, diarization_result, pitch_result) вместо запуска под-экстракторов.
        
        Args:
            input_uri: URI к входному аудио файлу
            tmp_path: Путь к временной директории
            asr_segments: Список сегментов для ASR (от Segmenter families.asr)
            diar_segments: Список сегментов для diarization (от Segmenter families.diarization)
            asr_result: Результат от asr_extractor (обязателен если enable_asr_metrics=True, падение при отсутствии)
            diarization_result: Результат от speaker_diarization_extractor (обязателен если enable_diarization_metrics=True, падение при отсутствии)
            pitch_result: Результат от pitch_extractor (обязателен если pitch_enabled=True и enable_pitch_metrics=True, падение при отсутствии)
        
        Progress reporting: для этапов обработки (silence detection, aggregation).
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        timings = {}  # Детальное профилирование этапов (seconds)
        stage_timings_ms: Dict[str, float] = {}

        speech_analysis_resource_profile: Optional[Dict[str, Any]] = None
        if is_speech_analysis_resource_profile_enabled():
            speech_analysis_resource_profile = {
                "at_start": capture_speech_analysis_resource_profile(stage="at_start"),
            }
        
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            
            # Validate segments structure
            is_valid, error_msg = self._validate_segments(asr_segments, "asr")
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "segments_invalid")
                raise ValueError(f"speech_analysis | {error_msg} (error_code={error_code})")
            
            is_valid, error_msg = self._validate_segments(diar_segments, "diarization")
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "segments_invalid")
                raise ValueError(f"speech_analysis | {error_msg} (error_code={error_code})")

            dur_sec = float(
                max(
                    max((float(s.get("end_sec", 0.0)) for s in asr_segments), default=0.0),
                    max((float(s.get("end_sec", 0.0)) for s in diar_segments), default=0.0),
                )
            )
            if dur_sec < 5.0:
                # Audit v3: return valid empty instead of error
                payload: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "audio_too_short",
                    "duration_sec": float(dur_sec),
                    "sample_rate": int(self.sample_rate),
                    "device_used": self.device,
                    "speech_analysis_contract_version": SPEECH_ANALYSIS_CONTRACT_VERSION,
                }
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload["stage_timings_ms"] = stage_timings_ms
                if speech_analysis_resource_profile is not None:
                    speech_analysis_resource_profile["at_end"] = capture_speech_analysis_resource_profile(stage="at_end")
                    payload["speech_analysis_resource_profile"] = speech_analysis_resource_profile
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            # Этап 1: Silence detection (if enabled)
            t_silence_start = time.perf_counter()
            if self.enable_silence_detection:
                try:
                    probe = diar_segments[0]
                    wav_t, _sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=int(probe.get("start_sample")),
                        end_sample=int(probe.get("end_sample")),
                        target_sr=self.sample_rate,
                    )
                    wav = self.audio_utils.to_numpy(wav_t)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                    rms, peak = self._rms_and_peak(wav)
                except Exception as e:
                    error_code = self._classify_error(e, "silence_detection_failed")
                    raise RuntimeError(f"speech_analysis | failed to probe audio for silence detection: {e} (error_code={error_code})") from e

                if peak < self.silence_peak_threshold and rms < self.silence_rms_threshold:
                    # Используем стандартное empty_reason из каноничного словаря
                    # (audio_silent - специфичная причина, но предпочтительно использовать стандартное значение)
                    payload: Dict[str, Any] = {
                        "status": "empty",
                        "empty_reason": "audio_missing_or_extract_failed",  # Стандартное значение из каноничного словаря
                        "duration_sec": float(dur_sec),
                        "sample_rate": int(self.sample_rate),
                        "device_used": self.device,
                        "speech_analysis_contract_version": SPEECH_ANALYSIS_CONTRACT_VERSION,
                    }
                    timings["silence_detection_sec"] = time.perf_counter() - t_silence_start
                    stage_timings_ms["silence_detection_ms"] = float(timings["silence_detection_sec"] * 1000.0)
                    stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                    payload["stage_timings_ms"] = stage_timings_ms
                    if speech_analysis_resource_profile is not None:
                        speech_analysis_resource_profile["at_end"] = capture_speech_analysis_resource_profile(stage="at_end")
                        payload["speech_analysis_resource_profile"] = speech_analysis_resource_profile
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
            
            t_silence_end = time.perf_counter()
            timings["silence_detection_sec"] = t_silence_end - t_silence_start
            stage_timings_ms["silence_detection_ms"] = float(timings["silence_detection_sec"] * 1000.0)
            logger.info(f"speech_analysis | silence detection completed: {timings['silence_detection_sec']:.3f}s")

            # Этап 2: Получение результатов от зависимых компонентов
            # ASR
            t_asr_start = time.perf_counter()
            logger.info(f"speech_analysis | enable_asr_metrics={self.enable_asr_metrics}, asr_result provided={asr_result is not None}")
            if self.enable_asr_metrics:
                if asr_result is None:
                    error_code = self._classify_error(RuntimeError("ASR result not provided"), "asr_failed")
                    raise RuntimeError(f"speech_analysis | asr_result is required when enable_asr_metrics=True, but was not provided. Ensure 'asr' extractor is enabled in config. (error_code={error_code})")
                
                if not asr_result.get("success", False):
                    error_code = self._classify_error(RuntimeError(asr_result.get("error") or "ASR failed"), "asr_failed")
                    raise RuntimeError(f"speech_analysis | asr_result indicates failure: {asr_result.get('error')} (error_code={error_code})")
                
                asr_payload = asr_result.get("payload") or {}
                is_valid, error_msg = self._validate_asr_payload(asr_payload)
                if not is_valid:
                    error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                    raise ValueError(f"speech_analysis | {error_msg} (error_code={error_code})")
                
                if self.progress_callback:
                    self.progress_callback("asr", 1, 1, "ASR result loaded from dependency")
            else:
                asr_payload = {}
            
            t_asr_end = time.perf_counter()
            timings["asr_sec"] = t_asr_end - t_asr_start
            stage_timings_ms["asr_ms"] = float(timings["asr_sec"] * 1000.0)
            logger.info(f"speech_analysis | ASR result processing completed: {timings['asr_sec']:.3f}s")
            
            # Diarization
            t_diar_start = time.perf_counter()
            if self.enable_diarization_metrics:
                if diarization_result is None:
                    error_code = self._classify_error(RuntimeError("Diarization result not provided"), "diarization_failed")
                    raise RuntimeError(f"speech_analysis | diarization_result is required when enable_diarization_metrics=True, but was not provided. Ensure 'speaker_diarization' extractor is enabled in config. (error_code={error_code})")
                
                if not diarization_result.get("success", False):
                    error_code = self._classify_error(RuntimeError(diarization_result.get("error") or "Diarization failed"), "diarization_failed")
                    raise RuntimeError(f"speech_analysis | diarization_result indicates failure: {diarization_result.get('error')} (error_code={error_code})")
                
                diar_payload = diarization_result.get("payload") or {}
                is_valid, error_msg = self._validate_diarization_payload(diar_payload)
                if not is_valid:
                    error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                    raise ValueError(f"speech_analysis | {error_msg} (error_code={error_code})")
                
                if self.progress_callback:
                    self.progress_callback("diarization", 1, 1, "Diarization result loaded from dependency")
            else:
                diar_payload = {}
            
            t_diar_end = time.perf_counter()
            timings["diarization_sec"] = t_diar_end - t_diar_start
            stage_timings_ms["diarization_ms"] = float(timings["diarization_sec"] * 1000.0)
            logger.info(f"speech_analysis | diarization result processing completed: {timings['diarization_sec']:.3f}s")

            # Pitch (требует pitch_result если enable_pitch_metrics=True - обязательная зависимость, падение при отсутствии)
            pitch_payload = None
            if self.pitch_enabled and self.enable_pitch_metrics:
                t_pitch_start = time.perf_counter()
                if pitch_result is None:
                    error_code = self._classify_error(RuntimeError("Pitch result not provided"), "pitch_failed")
                    raise RuntimeError(f"speech_analysis | pitch_result is required when pitch_enabled=True and enable_pitch_metrics=True, but was not provided. Ensure 'pitch' extractor is enabled in config. (error_code={error_code})")
                
                if not pitch_result.get("success", False):
                    error_code = self._classify_error(RuntimeError(pitch_result.get("error") or "Pitch failed"), "pitch_failed")
                    raise RuntimeError(f"speech_analysis | pitch_result indicates failure: {pitch_result.get('error')} (error_code={error_code})")
                
                pitch_payload = pitch_result.get("payload") or {}
                
                if self.progress_callback:
                    self.progress_callback("pitch", 1, 1, "Pitch result loaded from dependency")
                
                # Validate pitch payload
                is_valid, error_msg = self._validate_pitch_payload(pitch_payload)
                if not is_valid:
                    error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                    raise ValueError(f"speech_analysis | {error_msg} (error_code={error_code})")
                
                t_pitch_end = time.perf_counter()
                timings["pitch_sec"] = t_pitch_end - t_pitch_start
                stage_timings_ms["pitch_ms"] = float(timings["pitch_sec"] * 1000.0)
                logger.info(f"speech_analysis | pitch processing completed: {timings['pitch_sec']:.3f}s")

            # Этап 3: Aggregate results (feature-gated)
            t_aggregates_start = time.perf_counter()
            payload_out: Dict[str, Any] = {
                "duration_sec": float(dur_sec),
                "sample_rate": int(self.sample_rate),
                "device_used": self.device,
                "speech_analysis_contract_version": SPEECH_ANALYSIS_CONTRACT_VERSION,
            }
            
            # ASR metrics (feature-gated)
            if self.enable_asr_metrics:
                token_ids_by_segment = asr_payload.get("token_ids_by_segment") or []
                if isinstance(token_ids_by_segment, list):
                    token_counts = np.asarray([len(x or []) for x in token_ids_by_segment], dtype=np.float32)
                else:
                    token_counts = np.zeros((0,), dtype=np.float32)

                lang_ids = np.asarray(asr_payload.get("lang_id_by_segment") or [], dtype=np.int32).reshape(-1)

                token_total = float(np.sum(token_counts)) if token_counts.size else 0.0
                token_mean = float(np.mean(token_counts)) if token_counts.size else 0.0
                token_std = float(np.std(token_counts)) if token_counts.size else 0.0
                token_density = float(token_total / max(1e-6, dur_sec))
                
                # Additional ASR metrics
                speech_rate_wpm = float(asr_payload.get("speech_rate_wpm", 0.0) or 0.0)
                lang_distribution = asr_payload.get("lang_distribution")
                if lang_distribution is None:
                    lang_distribution = {}
                if not isinstance(lang_distribution, dict):
                    lang_distribution = {}
                
                payload_out.update({
                    "asr_segments_count": int(asr_payload.get("segments_count") or len(token_counts)),
                    "asr_token_total": float(token_total),
                    "asr_token_mean": float(token_mean),
                    "asr_token_std": float(token_std),
                    "asr_token_density_per_sec": float(token_density),
                    "asr_speech_rate_wpm": float(speech_rate_wpm),
                    "asr_lang_distribution": {str(k): float(v) for k, v in lang_distribution.items()},
                    "asr_lang_id_by_segment": lang_ids.tolist(),
                })
            
            # Diarization metrics (feature-gated)
            if self.enable_diarization_metrics:
                # Backward-compatible: prefer strict v2 structured arrays; fall back to legacy speaker_segments list[dict].
                speaker_ids = np.asarray(diar_payload.get("speaker_ids") or [], dtype=np.int32).reshape(-1)
                speaker_count = int(diar_payload.get("speaker_count") or speaker_ids.size or 0)

                speaker_duration_sec = diar_payload.get("speaker_duration_sec")
                if isinstance(speaker_duration_sec, (list, np.ndarray)) and len(speaker_duration_sec) == speaker_count:
                    dur_arr = np.asarray(speaker_duration_sec, dtype=np.float32).reshape(-1)
                    total_speech_dur = float(np.sum(dur_arr)) if dur_arr.size else 0.0
                    dominant_share = float(np.max(dur_arr) / max(1e-6, total_speech_dur)) if dur_arr.size else 0.0
                    diar_segments_count = int(diar_payload.get("speaker_turns_count") or len(diar_payload.get("turn_start_sec") or []))
                else:
                    speaker_segments = diar_payload.get("speaker_segments") or []
                    if not isinstance(speaker_segments, list):
                        speaker_segments = []
                    # Dominant speaker share by total duration over diar windows
                    dur_by_spk: Dict[int, float] = {}
                    for s in speaker_segments:
                        try:
                            sid = int(s.get("speaker_id", 0))
                            d = float(s.get("duration", float(s.get("end", 0.0)) - float(s.get("start", 0.0))) or 0.0)
                            dur_by_spk[sid] = dur_by_spk.get(sid, 0.0) + max(0.0, d)
                        except Exception:
                            continue
                    total_speech_dur = float(sum(dur_by_spk.values())) if dur_by_spk else 0.0
                    dominant_share = float(max(dur_by_spk.values()) / max(1e-6, total_speech_dur)) if dur_by_spk else 0.0
                    diar_segments_count = int(diar_payload.get("segments_count") or len(speaker_segments))
                
                # Additional diarization metrics
                speaker_balance_score = float(diar_payload.get("speaker_balance_score", 0.0) or 0.0)
                speaker_transitions_count = int(diar_payload.get("speaker_transitions_count", 0) or 0)
                
                payload_out.update({
                    "diar_segments_count": int(diar_segments_count),
                    "speaker_count": int(speaker_count),
                    "dominant_speaker_share": float(dominant_share),
                    "speaker_balance_score": float(speaker_balance_score),
                    "speaker_transitions_count": int(speaker_transitions_count),
                    "speaker_ids": speaker_ids.tolist(),
                })
            
            # Pitch metrics (feature-gated)
            if self.enable_pitch_metrics and pitch_payload is not None:
                pitch_f0_mean = float(pitch_payload.get("f0_mean", 0.0) or 0.0)
                pitch_f0_std = float(pitch_payload.get("f0_std", 0.0) or 0.0)
                
                # Additional pitch metrics
                f0_min = float(pitch_payload.get("f0_min", 0.0) or 0.0)
                f0_max = float(pitch_payload.get("f0_max", 0.0) or 0.0)
                f0_range = float(f0_max - f0_min) if f0_max > f0_min else 0.0
                
                # Pitch stability (inverse of std/mean ratio, normalized)
                pitch_stability = 0.0
                if pitch_f0_mean > 0:
                    cv = pitch_f0_std / pitch_f0_mean  # Coefficient of variation
                    pitch_stability = float(1.0 / (1.0 + cv))  # 0 = unstable, 1 = stable
                
                # Pitch distribution (if available)
                f0_series = pitch_payload.get("f0_series") or []
                pitch_distribution = {}
                if isinstance(f0_series, (list, np.ndarray)) and len(f0_series) > 0:
                    f0_arr = np.asarray(f0_series, dtype=np.float32)
                    f0_arr = f0_arr[f0_arr > 0]  # Filter out zeros/invalid
                    if f0_arr.size > 0:
                        # Distribution by octaves (simplified)
                        octave_bins = [50, 100, 200, 400, 800, 1600]
                        hist, _ = np.histogram(f0_arr, bins=octave_bins)
                        total = float(np.sum(hist))
                        if total > 0:
                            for i, count in enumerate(hist):
                                pitch_distribution[f"octave_{i}"] = float(count / total)
                
                payload_out.update({
                    "pitch_enabled": True,
                    "pitch_f0_mean": float(pitch_f0_mean),
                    "pitch_f0_std": float(pitch_f0_std),
                    "pitch_f0_min": float(f0_min),
                    "pitch_f0_max": float(f0_max),
                    "pitch_f0_range": float(f0_range),
                    "pitch_stability": float(pitch_stability),
                    "pitch_distribution": {str(k): float(v) for k, v in pitch_distribution.items()},
                })
            else:
                payload_out["pitch_enabled"] = False
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_asr_metrics:
                enabled_features.append("asr_metrics")
            if self.enable_diarization_metrics:
                enabled_features.append("diarization_metrics")
            # Only flag pitch_metrics when pitch dependency was merged (avoids tabular NaN with pitch_enabled=false).
            if self.enable_pitch_metrics and pitch_payload is not None:
                enabled_features.append("pitch_metrics")
            
            payload_out["_features_enabled"] = enabled_features
            logger.info(f"speech_analysis | _features_enabled={enabled_features}, payload keys with 'asr': {[k for k in payload_out.keys() if 'asr' in k.lower()]}")
            
            t_aggregates_end = time.perf_counter()
            timings["aggregates_sec"] = t_aggregates_end - t_aggregates_start
            stage_timings_ms["aggregates_ms"] = float(timings.get("aggregates_sec", 0.0) * 1000.0)
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload_out["stage_timings_ms"] = stage_timings_ms
            if speech_analysis_resource_profile is not None:
                speech_analysis_resource_profile["at_end"] = capture_speech_analysis_resource_profile(stage="at_end")
                payload_out["speech_analysis_resource_profile"] = speech_analysis_resource_profile
            
            # Log detailed profiling
            total_time = time.time() - start_time
            logger.info(f"speech_analysis | run_bundle completed: duration={dur_sec:.2f}s, enabled_features={enabled_features}")
            logger.info(f"speech_analysis | profiling: silence={timings.get('silence_detection_sec', 0):.3f}s, asr={timings.get('asr_sec', 0):.3f}s, diarization={timings.get('diarization_sec', 0):.3f}s, pitch={timings.get('pitch_sec', 0):.3f}s, aggregates={timings.get('aggregates_sec', 0):.3f}s, total={total_time:.3f}s")

            return self._create_result(True, payload=payload_out, processing_time=total_time)
        except Exception as e:
            error_code = self._classify_error(e, "unknown")
            return self._create_result(
                False,
                error=f"speech_analysis | error ({error_code}): {str(e)}",
                processing_time=time.time() - start_time
            )

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Стандартный метод run() не поддерживается для speech_analysis_extractor.
        
        Компонент требует вызова через run_bundle() с сегментами и результатами зависимых компонентов.
        """
        return self._create_result(
            success=False,
            error="speech_analysis_extractor | requires Segmenter window families; use run_bundle(asr_segments, diar_segments).",
            processing_time=0.0,
        )
    
    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        speech_analysis_extractor является bundle extractor и не требует GPU batching,
        но должен быть batch-safe (изоляция данных между файлами).
        """
        return False  # Не требует GPU batching, но batch-safe через run_bundle() для каждого файла
    
    def extract_batch(
        self,
        audio_files: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка нескольких аудио файлов.
        
        По умолчанию: последовательный вызов run_bundle() для каждого файла.
        Компонент batch-safe: каждый файл обрабатывается изолированно через run_bundle().
        
        Args:
            audio_files: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'file_id': Идентификатор файла (опционально, для логирования)
                - 'asr_segments': Список сегментов для ASR (обязательно)
                - 'diar_segments': Список сегментов для diarization (обязательно)
                - 'asr_result': Результат от asr_extractor (опционально, если enable_asr_metrics=True)
                - 'diarization_result': Результат от speaker_diarization_extractor (опционально, если enable_diarization_metrics=True)
                - 'pitch_result': Результат от pitch_extractor (опционально, если pitch_enabled=True и enable_pitch_metrics=True)
            max_workers: Не используется (компонент не поддерживает параллельную обработку)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        results: List[ExtractorResult] = []
        for file_info in audio_files:
            input_uri = file_info.get("input_uri")
            tmp_path = file_info.get("tmp_path")
            file_id = file_info.get("file_id", input_uri)
            
            if not input_uri or not tmp_path:
                self.logger.error(f"speech_analysis | Missing input_uri or tmp_path for file_id={file_id}")
                results.append(self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                ))
                continue
            
            asr_segments = file_info.get("asr_segments", [])
            diar_segments = file_info.get("diar_segments", [])
            
            if not asr_segments or not diar_segments:
                self.logger.error(f"speech_analysis | Missing segments for file_id={file_id}")
                results.append(self._create_result(
                    success=False,
                    error="Missing asr_segments or diar_segments",
                ))
                continue
            
            try:
                result = self.run_bundle(
                    input_uri=input_uri,
                    tmp_path=tmp_path,
                    asr_segments=asr_segments,
                    diar_segments=diar_segments,
                    asr_result=file_info.get("asr_result"),
                    diarization_result=file_info.get("diarization_result"),
                    pitch_result=file_info.get("pitch_result"),
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"speech_analysis | Error processing file_id={file_id}: {e}")
                results.append(self._create_result(
                    success=False,
                    error=str(e),
                ))
        
        return results

    def _validate_input(self, input_uri: str) -> bool:
        if not super()._validate_input(input_uri):
            return False
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        return True
