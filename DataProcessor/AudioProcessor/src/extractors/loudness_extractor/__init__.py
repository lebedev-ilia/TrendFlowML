"""
Экстрактор громкости: RMS, peak, dBFS и опционально LUFS (если доступен pyloudnorm).
"""

# NOTE: This package replaces the historical `loudness_extractor.py` module.
# Backward compatible import path:
#   from src.extractors.loudness_extractor import LoudnessExtractor

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)


class LoudnessExtractor(BaseExtractor):
    """Извлекает метрики громкости.

    - RMS и peak по всему треку
    - dBFS (20*log10(rms + eps))
    - LUFS при наличии `pyloudnorm` (иначе аккуратно пропускается)
    - frame-wise RMS статистики (mean/std/median/p10/p90) для short-term dynamics
    """

    name = "loudness"
    version = "2.1.1"
    description = "Метрики громкости (RMS, peak, dBFS, опционально LUFS)"
    category = "loudness"
    dependencies = ["numpy", "pyloudnorm"]
    estimated_duration = 0.5

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        mix_to_mono: bool = True,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.mix_to_mono = bool(mix_to_mono)

    def _compute_from_np(self, x: np.ndarray, sr: int) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            raise ValueError("Пустой аудиосигнал")

        eps = 1e-12
        rms = float(np.sqrt(float(np.mean(x * x)) + eps))
        peak = float(np.max(np.abs(x)) + eps)
        dbfs = float(20.0 * np.log10(rms + eps))

        if x.size >= self.frame_length:
            sq = x * x
            window = np.ones(self.frame_length, dtype=np.float32)
            window_sums = np.convolve(sq, window, mode="valid")
            rms_frames = np.sqrt(window_sums / float(self.frame_length) + eps)
            if self.hop_length > 0:
                rms_frames = rms_frames[:: self.hop_length]
        else:
            rms_frames = np.array([rms], dtype=np.float32)

        f_mean = float(np.mean(rms_frames))
        f_std = float(np.std(rms_frames))
        f_median = float(np.median(rms_frames))
        f_p10 = float(np.percentile(rms_frames, 10))
        f_p90 = float(np.percentile(rms_frames, 90))
        frames_count = int(rms_frames.shape[0])

        lufs_value: Optional[float] = None
        try:
            import pyloudnorm as pyln  # type: ignore

            try:
                meter = pyln.Meter(sr)
                lufs_value = float(meter.integrated_loudness(x))
            except Exception as e:
                logger.warning(f"Loudness: pyloudnorm present but failed to compute LUFS: {e}")
                lufs_value = None
        except Exception:
            lufs_value = None

        return {
            "rms": rms,
            "peak": peak,
            "dbfs": dbfs,
            "lufs": lufs_value,
            "sample_rate": int(sr),
            "duration": float(x.shape[-1] / sr),
            "frame_length": int(self.frame_length),
            "hop_length": int(self.hop_length),
            "frames_count": frames_count,
            "frame_rms_mean": f_mean,
            "frame_rms_std": f_std,
            "frame_rms_median": f_median,
            "frame_rms_p10": f_p10,
            "frame_rms_p90": f_p90,
            "frame_rms_stats_vector": [f_mean, f_std, f_median, f_p10, f_p90],
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
        Compute loudness over Segmenter-provided audio windows and aggregate.
        Produces:
          - per-segment rms/dbfs/peak (+ optional lufs) sequences
          - aggregated stats over segment RMS (mean/std/median/p10/p90)
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        loudness_resource_profile: Optional[Dict[str, Any]] = None
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

            if resource_profile_enabled():
                try:
                    loudness_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    loudness_resource_profile = None
            
            # Начальное сообщение о начале обработки
            if self.progress_callback and total_segments > 0:
                self.progress_callback("loudness", 0, total_segments, f"Starting loudness computation: 0/{total_segments} segments (0%)")

            seg_p = max(1, int(segment_parallelism or 1))
            inflight = int(max_inflight) if max_inflight is not None else seg_p
            inflight = max(1, int(inflight))

            # Strict alignment (Audit v3): pre-allocate arrays, no skipping
            n = int(len(segments))
            segment_start_sec = np.zeros(n, dtype=np.float32)
            segment_end_sec = np.zeros(n, dtype=np.float32)
            segment_center_sec = np.zeros(n, dtype=np.float32)
            segment_mask = np.zeros(n, dtype=bool)

            seg_rms_arr = np.full(n, np.nan, dtype=np.float32)
            seg_peak_arr = np.full(n, np.nan, dtype=np.float32)
            seg_dbfs_arr = np.full(n, np.nan, dtype=np.float32)
            seg_lufs_arr = np.full(n, np.nan, dtype=np.float32)
            last_reported_pct = -1

            def _process_one(i: int, seg: dict) -> None:
                # Always populate time axis.
                st = float(seg.get("start_sec", 0.0))
                en = float(seg.get("end_sec", 0.0))
                c = float(seg.get("center_sec", (st + en) * 0.5))
                segment_start_sec[i] = st
                segment_end_sec[i] = en
                segment_center_sec[i] = c

                # Basic validation: keep masked if invalid or empty window.
                if not np.isfinite(st) or not np.isfinite(en) or en <= st:
                    return

                try:
                    ss = int(seg.get("start_sample", 0))
                    es = int(seg.get("end_sample", 0))
                    waveform_t, sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=ss,
                        end_sample=es,
                        target_sr=self.sample_rate,
                        mix_to_mono=self.mix_to_mono,
                    )
                    x = self.audio_utils.to_numpy(waveform_t)
                    if x.ndim == 2:
                        x = np.mean(x, axis=0) if self.mix_to_mono else x[0]
                    m = self._compute_from_np(x, int(sr))
                    seg_rms_arr[i] = float(m["rms"])
                    seg_peak_arr[i] = float(m["peak"])
                    seg_dbfs_arr[i] = float(m["dbfs"])
                    if m.get("lufs") is not None and np.isfinite(float(m["lufs"])):
                        seg_lufs_arr[i] = float(m["lufs"])
                    segment_mask[i] = True
                except Exception as e:
                    # No-fallback on segments: keep strict alignment, mark as failed.
                    logger.warning(f"loudness | Segment {i} failed: {e}")
                    return

            if seg_p <= 1:
                t_seg0 = time.perf_counter()
                for i, seg in enumerate(segments):
                    _process_one(i, seg)
                    
                    # Progress reporting: каждые 10% сегментов или на первом/последнем
                    if self.progress_callback and total_segments > 0:
                        current = i + 1
                        pct = int(current * 100 / total_segments)
                        if current == 1 or current == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                            self.progress_callback("loudness", current, total_segments, f"Processing segments: {current}/{total_segments} ({pct}%)")
                            last_reported_pct = pct
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t_seg0) * 1000.0
            else:
                workers = max(1, min(int(seg_p), int(inflight)))
                completed_count = 0
                t_seg0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_process_one, i, seg) for i, seg in enumerate(segments)]
                    for fut in as_completed(futs):
                        try:
                            fut.result()
                            
                            completed_count += 1
                            
                            # Progress reporting: каждые 10% сегментов или на первом/последнем
                            if self.progress_callback and total_segments > 0:
                                pct = int(completed_count * 100 / total_segments)
                                if completed_count == 1 or completed_count == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                                    self.progress_callback("loudness", completed_count, total_segments, f"Processing segments: {completed_count}/{total_segments} ({pct}%)")
                                    last_reported_pct = pct
                        except Exception as e:
                            logger.error(f"Error processing loudness segment in parallel: {e}")
                            raise
                stage_timings_ms["process_segments_ms"] = (time.perf_counter() - t_seg0) * 1000.0

            t_agg0 = time.perf_counter()
            valid = segment_mask & np.isfinite(seg_rms_arr)
            valid_rms = seg_rms_arr[valid]
            # Aggregate segment RMS stats over valid segments only
            agg = {
                "segment_rms_mean": float(np.mean(valid_rms)) if valid_rms.size else float("nan"),
                "segment_rms_std": float(np.std(valid_rms)) if valid_rms.size else float("nan"),
                "segment_rms_median": float(np.median(valid_rms)) if valid_rms.size else float("nan"),
                "segment_rms_p10": float(np.percentile(valid_rms, 10)) if valid_rms.size else float("nan"),
                "segment_rms_p90": float(np.percentile(valid_rms, 90)) if valid_rms.size else float("nan"),
            }
            stage_timings_ms["aggregate_segments_ms"] = (time.perf_counter() - t_agg0) * 1000.0

            # Keep backward-compatible globals by also computing full-track metrics.
            t_full0 = time.perf_counter()
            wav_full_t, sr_full = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x_full = self.audio_utils.to_numpy(wav_full_t)
            if x_full.ndim == 2:
                x_full = np.mean(x_full, axis=0) if self.mix_to_mono else x_full[0]
            full = self._compute_from_np(x_full, int(sr_full))
            stage_timings_ms["compute_full_track_ms"] = (time.perf_counter() - t_full0) * 1000.0
            full_lufs = full.get("lufs")
            lufs_present = bool(
                (full_lufs is not None and np.isfinite(float(full_lufs)))
                or bool(np.any(np.isfinite(seg_lufs_arr)))
            )

            payload: Dict[str, Any] = {
                **full,
                **agg,
                "segments_count": int(len(segments)),
                "lufs_present": bool(lufs_present),
                "stage_timings_ms": stage_timings_ms,
                "loudness_resource_profile": loudness_resource_profile,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "segment_center_sec": segment_center_sec,
                "segment_mask": segment_mask,
                "segment_rms": seg_rms_arr,
                "segment_peak": seg_peak_arr,
                "segment_dbfs": seg_dbfs_arr,
                "segment_lufs": seg_lufs_arr,
                "device_used": self.device,
            }

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0

            if loudness_resource_profile is not None:
                try:
                    payload["loudness_resource_profile"] = {
                        **(loudness_resource_profile or {}),
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass
            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)
        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            waveform_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x = self.audio_utils.to_numpy(waveform_t)
            if x.ndim == 2:
                x = np.mean(x, axis=0) if self.mix_to_mono else x[0]

            metrics = self._compute_from_np(x, int(sr))

            processing_time = time.time() - start_time

            payload: Dict[str, Any] = {
                **metrics,
                "device_used": self.device,
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)


