"""
Экстрактор оценки темпа (BPM) на базе librosa.
"""

# NOTE: This package replaces the historical `tempo_extractor.py` module.
# Backward compatible import path:
#   from src.extractors.tempo_extractor import TempoExtractor

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import capture_tempo_resource_profile, is_tempo_resource_profile_enabled

logger = logging.getLogger(__name__)

TEMPO_CONTRACT_VERSION = "tempo_contract_v1"


class TempoExtractor(BaseExtractor):
    """Экстрактор темпа (BPM) с использованием onset-энергии и beat tracking.

    Лёгкий по зависимостям (librosa) и подходит для CPU/GPU пайплайна.
    """

    name = "tempo"
    version = "2.0.1"
    description = "Оценка темпа (BPM) на основе onset-энергии"
    category = "rhythm"
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
        aggregate: str = "median",
        average_channels: bool = True,
        windowed_bpm: bool = False,
        window_sec: float = 15.0,
        step_sec: float = 5.0,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.aggregate = aggregate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.windowed_bpm = bool(windowed_bpm)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)

    def _estimate_from_np(self, waveform_np: np.ndarray, sr: int) -> Dict[str, Any]:
        waveform_np = np.asarray(waveform_np, dtype=np.float32).reshape(-1)
        if waveform_np.size == 0:
            raise ValueError("Пустой аудиосигнал")

        import librosa  # локальный импорт

        onset_env = librosa.onset.onset_strength(y=waveform_np, sr=sr, hop_length=self.hop_length)
        try:
            from librosa.feature.rhythm import tempo as rhythm_tempo

            tempo_all = rhythm_tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, aggregate=None)
        except (ImportError, AttributeError):
            tempo_all = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, aggregate=None)

        if tempo_all is None or len(tempo_all) == 0:
            raise RuntimeError("Не удалось оценить темп")

        tempo_all = np.asarray(tempo_all, dtype=np.float32).reshape(-1)
        tempo_median = float(np.median(tempo_all))
        tempo_mean = float(np.mean(tempo_all))
        tempo_std = float(np.std(tempo_all))
        confidence = float(1.0 / (1.0 + (tempo_std / (tempo_mean + 1e-6))))

        warnings: list[str] = []
        if tempo_median < 40 or tempo_median > 220:
            warnings.append("tempo_out_of_range")
        if confidence < 0.3:
            warnings.append("low_confidence")
        if float(np.max(np.abs(waveform_np))) < 0.02:
            warnings.append("signal_too_quiet")

        return {
            "tempo_estimates": tempo_all.astype(np.float32),
            "tempo_bpm": tempo_median if self.aggregate == "median" else tempo_mean,
            "tempo_bpm_mean": tempo_mean,
            "tempo_bpm_median": tempo_median,
            "tempo_bpm_std": tempo_std,
            "confidence": confidence,
            "warnings": warnings,
            "sample_rate": int(sr),
            "hop_length": int(self.hop_length),
            "duration": float(waveform_np.shape[-1] / sr),
            "tempo_contract_version": TEMPO_CONTRACT_VERSION,
        }

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: list[dict],
        *,
        segment_parallelism: int = 1,
        max_inflight: Optional[int] = None,
    ) -> ExtractorResult:
        """
        Compute tempo over Segmenter-provided windows (tempo family) and aggregate.
        Produces:
          - windowed_times_sec / windowed_bpm sequences (per window)
          - global tempo_bpm_* from full track for backward compatibility
        """
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        tempo_resource_profile: Optional[Dict[str, Any]] = None
        if is_tempo_resource_profile_enabled():
            tempo_resource_profile = {
                "at_start": capture_tempo_resource_profile(stage="at_start"),
            }
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            self._log_extraction_start(input_uri)
            
            total_segments = len(segments)
            
            # Начальное сообщение о начале обработки
            if self.progress_callback and total_segments > 0:
                self.progress_callback("tempo", 0, total_segments, f"Starting tempo estimation: 0/{total_segments} segments (0%)")

            seg_p = max(1, int(segment_parallelism or 1))
            inflight = int(max_inflight) if max_inflight is not None else seg_p
            inflight = max(1, int(inflight))

            def _one(i: int, seg: dict):
                """Process one segment; on failure return (i, NaN, mask=False)."""
                try:
                    ss = int(seg.get("start_sample"))
                    es = int(seg.get("end_sample"))
                    start_sec = float(seg.get("start_sec", 0.0))
                    end_sec = float(seg.get("end_sec", 0.0))
                    c = float(seg.get("center_sec"))
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
                    m = self._estimate_from_np(waveform_np, int(sr))
                    bpm = float(m["tempo_bpm_median"])
                    warns = [str(w) for w in (m.get("warnings") or [])]
                    return i, start_sec, end_sec, float(c), bpm, True, warns
                except Exception as e:
                    logger.error(f"Error processing tempo segment {i}: {e}")
                    start_sec = float(seg.get("start_sec", 0.0))
                    end_sec = float(seg.get("end_sec", 0.0))
                    c = float(seg.get("center_sec", 0.0))
                    return i, start_sec, end_sec, c, float("nan"), False, []

            bpms: list[float] = [0.0] * int(len(segments))
            times: list[float] = [0.0] * int(len(segments))
            segment_start_sec: list[float] = []
            segment_end_sec: list[float] = []
            segment_mask: list[bool] = []
            warnings_all: set[str] = set()
            last_reported_pct = -1

            if seg_p <= 1:
                t0_proc = time.perf_counter()
                for i, seg in enumerate(segments):
                    _, s0, s1, c, bpm, ok, warns = _one(i, seg)
                    bpms[i] = bpm
                    times[i] = c
                    segment_start_sec.append(s0)
                    segment_end_sec.append(s1)
                    segment_mask.append(ok)
                    for w in warns:
                        warnings_all.add(w)
                    
                    # Progress reporting: каждые 10% сегментов или на первом/последнем
                    if self.progress_callback and total_segments > 0:
                        current = i + 1
                        pct = int(current * 100 / total_segments)
                        if current == 1 or current == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                            self.progress_callback("tempo", current, total_segments, f"Processing segments: {current}/{total_segments} ({pct}%)")
                            last_reported_pct = pct
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_proc) * 1000.0
            else:
                # Best-effort concurrency: cap workers by inflight.
                t0_proc = time.perf_counter()
                workers = max(1, min(int(seg_p), int(inflight)))
                completed_count = 0
                results: list[tuple] = [None] * len(segments)  # type: ignore
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_one, i, seg) for i, seg in enumerate(segments)]
                    for fut in as_completed(futs):
                        try:
                            r = fut.result()
                            i, s0, s1, c, bpm, ok, warns = r
                            idx = int(i)
                            bpms[idx] = float(bpm)
                            times[idx] = float(c)
                            results[idx] = (s0, s1, ok)
                            for w in warns:
                                warnings_all.add(w)
                            
                            completed_count += 1
                            
                            # Progress reporting
                            if self.progress_callback and total_segments > 0:
                                pct = int(completed_count * 100 / total_segments)
                                if completed_count == 1 or completed_count == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                                    self.progress_callback("tempo", completed_count, total_segments, f"Processing segments: {completed_count}/{total_segments} ({pct}%)")
                                    last_reported_pct = pct
                        except Exception as e:
                            logger.error(f"Error processing tempo segment in parallel: {e}")
                            raise
                for idx in range(len(segments)):
                    r = results[idx]
                    if r is not None:
                        s0, s1, ok = r
                        segment_start_sec.append(s0)
                        segment_end_sec.append(s1)
                        segment_mask.append(ok)
                    else:
                        seg = segments[idx]
                        segment_start_sec.append(float(seg.get("start_sec", 0.0)))
                        segment_end_sec.append(float(seg.get("end_sec", 0.0)))
                        segment_mask.append(False)
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t0_proc) * 1000.0

            # Full-track metrics (same as run())
            full = {}
            try:
                t0_full = time.perf_counter()
                waveform_t_full, sr_full = self.audio_utils.load_audio(input_uri, self.sample_rate)
                waveform_np_full = self.audio_utils.to_numpy(waveform_t_full)
                if waveform_np_full.ndim == 2:
                    waveform_np_full = np.mean(waveform_np_full, axis=0) if self.average_channels else waveform_np_full[0]
                full = self._estimate_from_np(waveform_np_full, int(sr_full))
                stage_timings_ms["full_track_ms"] = (time.perf_counter() - t0_full) * 1000.0
            except Exception as e:
                logger.warning(f"tempo | full-track estimation failed: {e}")
                full = {
                    "tempo_estimates": np.zeros((0,), dtype=np.float32),
                    "tempo_bpm": float("nan"),
                    "tempo_bpm_mean": float("nan"),
                    "tempo_bpm_median": float("nan"),
                    "tempo_bpm_std": float("nan"),
                    "confidence": 0.0,
                    "warnings": ["full_track_failed"],
                    "sample_rate": self.sample_rate,
                    "hop_length": self.hop_length,
                    "duration": 0.0,
                    "tempo_contract_version": TEMPO_CONTRACT_VERSION,
                }

            # Audit v3: empty when all segments failed
            n_ok = sum(1 for m in segment_mask if m)
            if n_ok == 0:
                t0 = time.perf_counter()
                payload_empty: Dict[str, Any] = {
                    **full,
                    "status": "empty",
                    "empty_reason": "tempo_all_segments_failed",
                    "segment_start_sec": segment_start_sec,
                    "segment_end_sec": segment_end_sec,
                    "segment_center_sec": times,
                    "segment_mask": segment_mask,
                    "bpm_by_segment": bpms,
                    "segments_count": int(len(segments)),
                    "warnings": sorted(set(full.get("warnings") or []) | warnings_all),
                    "device_used": self.device,
                }
                stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0
                stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                payload_empty["stage_timings_ms"] = stage_timings_ms
                if tempo_resource_profile is not None:
                    tempo_resource_profile["at_end"] = capture_tempo_resource_profile(stage="at_end")
                    payload_empty["tempo_resource_profile"] = tempo_resource_profile
                processing_time = time.time() - start_time
                return self._create_result(True, payload=payload_empty, processing_time=processing_time)

            t0 = time.perf_counter()
            payload: Dict[str, Any] = {
                **full,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": times,
                "segment_mask": segment_mask,
                "bpm_by_segment": bpms,
                "segments_count": int(len(segments)),
                "warnings": sorted(set(full.get("warnings") or []) | warnings_all),
                "device_used": self.device,
            }
            stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload["stage_timings_ms"] = stage_timings_ms
            if tempo_resource_profile is not None:
                tempo_resource_profile["at_end"] = capture_tempo_resource_profile(stage="at_end")
                payload["tempo_resource_profile"] = tempo_resource_profile
            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)
        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        tempo_resource_profile: Optional[Dict[str, Any]] = None
        if is_tempo_resource_profile_enabled():
            tempo_resource_profile = {
                "at_start": capture_tempo_resource_profile(stage="at_start"),
            }
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            t0 = time.perf_counter()
            waveform_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            waveform_np = self.audio_utils.to_numpy(waveform_t)
            stage_timings_ms["to_numpy_ms"] = (time.perf_counter() - t0) * 1000.0
            if waveform_np.ndim == 2:
                waveform_np = np.mean(waveform_np, axis=0) if self.average_channels else waveform_np[0]
            t0 = time.perf_counter()
            base = self._estimate_from_np(waveform_np, int(sr))
            stage_timings_ms["estimate_ms"] = (time.perf_counter() - t0) * 1000.0

            processing_time = time.time() - start_time

            # Опциональная оценка BPM по окнам для лупов/диджейских треков
            window_series = None
            try:
                if self.windowed_bpm:
                    t0 = time.perf_counter()
                    win = max(1, int(self.window_sec * int(sr)))
                    step = max(1, int(self.step_sec * int(sr)))
                    if len(waveform_np) >= win:
                        bpms = []
                        times = []
                        for start in range(0, len(waveform_np) - win + 1, step):
                            segment = waveform_np[start:start+win]
                            m_seg = self._estimate_from_np(segment, int(sr))
                            bpms.append(float(m_seg["tempo_bpm_median"]))
                            times.append(float(start / float(sr)))
                        if bpms:
                            window_series = {
                                "times_sec": times,
                                "bpm": bpms,
                                "bpm_mean": float(np.mean(bpms)),
                                "bpm_median": float(np.median(bpms)),
                                "bpm_std": float(np.std(bpms)),
                            }
                    stage_timings_ms["windowed_bpm_ms"] = (time.perf_counter() - t0) * 1000.0
            except Exception:
                window_series = None

            t0 = time.perf_counter()
            payload: Dict[str, Any] = {
                **base,
                "windowed_bpm": window_series,
                "device_used": self.device,
            }
            stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload["stage_timings_ms"] = stage_timings_ms
            if tempo_resource_profile is not None:
                tempo_resource_profile["at_end"] = capture_tempo_resource_profile(stage="at_end")
                payload["tempo_resource_profile"] = tempo_resource_profile

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)


