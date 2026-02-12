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

logger = logging.getLogger(__name__)


class TempoExtractor(BaseExtractor):
    """Экстрактор темпа (BPM) с использованием onset-энергии и beat tracking.

    Лёгкий по зависимостям (librosa) и подходит для CPU/GPU пайплайна.
    """

    name = "tempo"
    version = "1.1.0"
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
                try:
                    ss = int(seg.get("start_sample"))
                    es = int(seg.get("end_sample"))
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
                    return i, float(c), bpm, warns
                except Exception as e:
                    logger.error(f"Error processing tempo segment {i}: {e}")
                    raise

            bpms: list[float] = [0.0] * int(len(segments))
            times: list[float] = [0.0] * int(len(segments))
            warnings_all: set[str] = set()
            last_reported_pct = -1

            if seg_p <= 1:
                for i, seg in enumerate(segments):
                    _, c, bpm, warns = _one(i, seg)
                    bpms[i] = bpm
                    times[i] = c
                    for w in warns:
                        warnings_all.add(w)
                    
                    # Progress reporting: каждые 10% сегментов или на первом/последнем
                    if self.progress_callback and total_segments > 0:
                        current = i + 1
                        pct = int(current * 100 / total_segments)
                        if current == 1 or current == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                            self.progress_callback("tempo", current, total_segments, f"Processing segments: {current}/{total_segments} ({pct}%)")
                            last_reported_pct = pct
            else:
                # Best-effort concurrency: cap workers by inflight.
                workers = max(1, min(int(seg_p), int(inflight)))
                completed_count = 0
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_one, i, seg) for i, seg in enumerate(segments)]
                    for fut in as_completed(futs):
                        try:
                            i, c, bpm, warns = fut.result()
                            bpms[int(i)] = float(bpm)
                            times[int(i)] = float(c)
                            for w in warns:
                                warnings_all.add(w)
                            
                            completed_count += 1
                            
                            # Progress reporting: каждые 10% сегментов или на первом/последнем
                            if self.progress_callback and total_segments > 0:
                                pct = int(completed_count * 100 / total_segments)
                                if completed_count == 1 or completed_count == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                                    self.progress_callback("tempo", completed_count, total_segments, f"Processing segments: {completed_count}/{total_segments} ({pct}%)")
                                    last_reported_pct = pct
                        except Exception as e:
                            logger.error(f"Error processing tempo segment in parallel: {e}")
                            raise

            # Full-track metrics (same as run())
            waveform_t_full, sr_full = self.audio_utils.load_audio(input_uri, self.sample_rate)
            waveform_np_full = self.audio_utils.to_numpy(waveform_t_full)
            if waveform_np_full.ndim == 2:
                waveform_np_full = np.mean(waveform_np_full, axis=0) if self.average_channels else waveform_np_full[0]
            full = self._estimate_from_np(waveform_np_full, int(sr_full))

            payload: Dict[str, Any] = {
                **full,
                "windowed_bpm": {
                    "times_sec": times,
                    "bpm": bpms,
                    "bpm_mean": float(np.mean(bpms)) if bpms else float("nan"),
                    "bpm_median": float(np.median(bpms)) if bpms else float("nan"),
                    "bpm_std": float(np.std(bpms)) if bpms else float("nan"),
                },
                "segments_count": int(len(segments)),
                "warnings": sorted(set(full.get("warnings") or []) | warnings_all),
                "device_used": self.device,
            }
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
            waveform_np = self.audio_utils.to_numpy(waveform_t)
            if waveform_np.ndim == 2:
                waveform_np = np.mean(waveform_np, axis=0) if self.average_channels else waveform_np[0]
            base = self._estimate_from_np(waveform_np, int(sr))

            processing_time = time.time() - start_time

            # Опциональная оценка BPM по окнам для лупов/диджейских треков
            window_series = None
            try:
                if self.windowed_bpm:
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
            except Exception:
                window_series = None

            payload: Dict[str, Any] = {
                **base,
                "windowed_bpm": window_series,
                "device_used": self.device,
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)


