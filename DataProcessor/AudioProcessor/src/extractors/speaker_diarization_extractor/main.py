#!/usr/bin/env python3
"""
Speaker diarization via pyannote.audio Pipeline (Audit v3).

Audit v3 policy:
- ModelManager-only (dp_models), no-network runtime (no HF fallback)
- diarization-only (no ASR/whisperx, no transcripts/word alignment)
- Segmenter-owned sampling: required family `diarization`; windows are collapsed to one hull [min(start), max(end)] for a single whole-span diarization pass
- Token-ready outputs: turn arrays (start/end/speaker_id/mask) + stable tabular aggregates
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import warnings

import numpy as np
import torch
from packaging import version

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    capture_speaker_diarization_resource_profile,
    is_speaker_diarization_resource_profile_enabled,
)

# До импорта pyannote: waveform через soundfile, torchcodec не используем — иначе огромный warning в логах.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

logger = logging.getLogger(__name__)
DIARIZATION_CONTRACT_VERSION = "diarization_contract_v1"


def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        # Temporarily disable logging error reporting to prevent traceback output
        old_raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            logger_instance.warning(message, *args, **kwargs)
        finally:
            # Restore original setting
            logging.raiseExceptions = old_raise_exceptions
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        # This is expected behavior during cleanup/shutdown phases
        pass


class SpeakerDiarizationExtractor(BaseExtractor):
    name = "speaker_diarization_extractor"
    version = "3.1.1"
    description = "Speaker diarization via pyannote.audio (local bundle pyannote/speaker-diarization-community-1)"
    category = "speech"
    dependencies = ["numpy", "torch", "pyannote.audio", "dp_models"]
    estimated_duration = 10.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 2.0

    def __init__(
        self,
        device: str = "auto",
        whisper_model_size: str = "small",
        huggingface_token: Optional[str] = None,
        sample_rate: int = 16000,
        enable_speaker_segments: bool = False,
        enable_speaker_embeddings: bool = False,
        enable_speaker_stats: bool = False,
        enable_speaker_durations: bool = False,
        enable_transcript: bool = False,
        enable_word_segments: bool = False,
        silence_peak_threshold: float = 1e-3,
        silence_rms_threshold: float = 1e-4,
        enable_silence_detection: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        super().__init__(device=device)

        # Audit v3: strict offline-only mode (no-network), always on for audited contract.
        self.offline_only = True

        # Legacy params retained for backward compatibility (no-op in Audit v3 diarization-only mode).
        self.whisper_model_size = str(whisper_model_size or "small").lower()

        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")

        self.sample_rate = int(sample_rate)
        # Audit v3: turns are always-on (used downstream by speech_analysis).
        self.enable_speaker_segments = True
        # Audit v3: embeddings/transcript/words are not part of audited contract.
        self.enable_speaker_embeddings = False
        # Legacy flags retained as no-ops for backward compatibility (strict arrays contract).
        self.enable_speaker_stats = False
        self.enable_speaker_durations = False
        self.enable_transcript = False
        self.enable_word_segments = False

        self.silence_peak_threshold = float(silence_peak_threshold)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.enable_silence_detection = bool(enable_silence_detection)
        self.progress_callback = progress_callback

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # device selection
        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        # ModelManager
        try:
            from dp_models import get_global_model_manager  # type: ignore
            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"ModelManager init failed: {e}") from e

        # Try to mitigate PyTorch 2.6+ safe-unpickle issues by allowing omegaconf classes
        try:
            torch_version = version.parse(torch.__version__.split("+")[0])
            if torch_version >= version.parse("2.6.0"):
                try:
                    import omegaconf  # type: ignore
                    # add common omegaconf types to allowed safe globals (best-effort)
                    safe_list = []
                    for name in ("listconfig", "dictconfig", "base"):
                        mod = getattr(omegaconf, name, None)
                        if mod is not None:
                            # try to collect classes we know appear in checkpoints
                            for cls_name in ("ListConfig", "DictConfig", "ContainerMetadata"):
                                cls = getattr(getattr(omegaconf, name, omegaconf), cls_name, None)
                                if cls is not None:
                                    safe_list.append(cls)
                    # fallback generic attempt:
                    if not safe_list:
                        if hasattr(omegaconf, "ListConfig"):
                            safe_list.append(omegaconf.ListConfig)
                        if hasattr(omegaconf, "DictConfig"):
                            safe_list.append(omegaconf.DictConfig)
                    if safe_list:
                        try:
                            torch.serialization.add_safe_globals(safe_list)
                            safe_log_warning(logger, "Added OmegaConf classes to torch safe globals (runtime workaround).")
                        except Exception:
                            # Some torch versions use the context manager API
                            try:
                                torch.serialization.safe_globals(safe_list)
                                safe_log_warning(logger, "Called torch.serialization.safe_globals(...) (context manager).")
                            except Exception:
                                safe_log_warning(logger, "Couldn't add safe globals to torch; if load fails consider downgrading torch to 2.5.x.")
                except Exception:
                    logger.debug("OmegaConf not available or failed to register safe globals; downgrading torch to 2.5.x recommended for full compatibility.")
        except Exception:
            pass  # if packaging/version isn't available, continue

        # Models will be loaded lazily on first use (lazy loading)
        # This allows initialization in main venv even if dependencies are only in isolated venv
        self.diarization_pipeline = None
        self.diarization_model_name = None
        self.diarization_weights_digest = None
        self._models_loaded = False

        # Whisper is not used in audited diarization-only mode.
        self.whisper_model = None
        self.whisper_model_name = None
        self.whisper_weights_digest = None

    # -------------------------
    # Loading helpers
    # -------------------------
    def _progress(self, step: int, total: int, msg: str):
        if self.progress_callback:
            try:
                self.progress_callback(step, total, msg)
            except Exception:
                logger.debug("progress_callback failed", exc_info=True)

    def _load_pyannote_pipeline(self):
        self._progress(1, 3, "Loading pyannote pipeline...")
        logger.info("SpeakerDiarization | Starting pyannote pipeline load...")
        try:
            # Try to import torch_audiomentations first to provide better error message
            try:
                import torch_audiomentations  # type: ignore
                logger.info("SpeakerDiarization | torch_audiomentations imported successfully")
            except ImportError:
                safe_log_warning(logger, "torch_audiomentations not found. Installing it may be required for pyannote.audio. Attempting to continue...")
            
            logger.info("SpeakerDiarization | Importing pyannote.audio Pipeline...")
            from pyannote.audio import Pipeline  # type: ignore
            logger.info("SpeakerDiarization | pyannote.audio Pipeline imported successfully")
        except ImportError as e:
            if "torch_audiomentations" in str(e):
                raise RuntimeError(
                    f"pyannote.audio import failed: {e}. "
                    "Please install torch-audiomentations: pip install torch-audiomentations"
                ) from e
            raise RuntimeError(f"pyannote.audio import failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"pyannote.audio import failed: {e}") from e

        # Audit v3: ModelManager-only (offline).
        try:
            spec_name = "pyannote_speaker_diarization"
            spec = self._mm.get_spec(model_name=spec_name)
            dev, prec, rt, eng, wd, arts = self._mm.resolve(spec)
            if str(rt) == "inprocess":
                resolved = self._mm.get(model_name=spec_name)
                self.diarization_pipeline = resolved.handle
                self.diarization_model_name = str(spec.model_name)
                self.diarization_weights_digest = str(wd)
                logger.info("Loaded pyannote pipeline from ModelManager (inprocess).")
            else:
                raise RuntimeError(f"pyannote spec runtime must be inprocess, got {rt}")
        except Exception as e:
            raise RuntimeError(
                "SpeakerDiarization | ModelManager could not load pyannote_speaker_diarization. "
                "Audit v3 requires this extractor to be fully offline via dp_models."
            ) from e

        if self.diarization_pipeline is not None and self.device_str == "cuda":
            try:
                self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("SpeakerDiarization | Pipeline moved to CUDA.")
            except Exception:
                safe_log_warning(logger, "Failed to move pyannote pipeline to CUDA; continuing on CPU.")

    def _load_whisperx_model(self):
        raise RuntimeError("SpeakerDiarization | whisperx is disabled in Audit v3 (diarization-only)")

    # -------------------------
    # Utility: waveform loading
    # -------------------------
    def _load_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio with soundfile and return waveform as (channels, time) torch.float32 tensor."""
        import soundfile as sf
        waveform_np, sr = sf.read(audio_path, dtype="float32")
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[None, :]  # mono -> (1, time)
        else:
            waveform_np = waveform_np.T  # (time, channels) -> (channels, time)
        waveform = torch.from_numpy(waveform_np)
        if waveform.dtype != torch.float32:
            waveform = waveform.to(dtype=torch.float32)
        return waveform, int(sr)

    # -------------------------
    # Utility: audio stats
    # -------------------------
    def _rms_and_peak(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    # -------------------------
    # Speaker attribution helpers
    # -------------------------
    @staticmethod
    def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        """Return overlap length between intervals a and b"""
        left = max(a_start, b_start)
        right = min(a_end, b_end)
        return max(0.0, right - left)

    def _assign_speaker_for_interval(
        self,
        w_start: float,
        w_end: float,
        diarization_segments: List[Tuple[float, float, str]],
    ) -> str:
        """
        Prefer speaker with the largest overlap with [w_start, w_end].
        Fall back to midpoint search and then 'unknown'.
        """
        best_label = "unknown"
        best_overlap = 0.0
        for s, e, label in diarization_segments:
            ov = self._overlap(w_start, w_end, s, e)
            if ov > best_overlap:
                best_overlap = ov
                best_label = label
        if best_overlap > 0:
            return best_label
        # midpoint fallback
        mid = (w_start + w_end) / 2.0
        for s, e, label in diarization_segments:
            if s <= mid <= e:
                return label
        return "unknown"

    # -------------------------
    # Main entry
    # -------------------------
    def _ensure_models_loaded(self):
        """Lazy load models on first use."""
        if not self._models_loaded:
            logger.info("SpeakerDiarization | Loading models (lazy load)...")
            self._load_pyannote_pipeline()
            logger.info("SpeakerDiarization | pyannote pipeline loaded (diarization-only)")
            self._models_loaded = True

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        t0_total = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}

        speaker_diarization_resource_profile: Optional[Dict[str, Any]] = None
        if is_speaker_diarization_resource_profile_enabled():
            speaker_diarization_resource_profile = {
                "at_start": capture_speaker_diarization_resource_profile(stage="at_start"),
            }
        logger.info(f"SpeakerDiarization | Starting run() for {input_uri}")
        try:
            # Lazy load models on first use
            logger.info("SpeakerDiarization | Ensuring models are loaded...")
            t0 = time.perf_counter()
            self._ensure_models_loaded()
            stage_timings_ms["load_models_ms"] = (time.perf_counter() - t0) * 1000.0
            logger.info("SpeakerDiarization | Models loaded, validating input...")
            
            if not self._validate_input(input_uri):
                return self._create_result(False, error="invalid input", processing_time=time.time() - start_time)

            self._progress(1, 5, "Loading audio")
            logger.info("SpeakerDiarization | Loading audio waveform...")
            t0 = time.perf_counter()
            waveform, sr = self._load_waveform(input_uri)
            stage_timings_ms["load_audio_ms"] = (time.perf_counter() - t0) * 1000.0
            logger.info(f"SpeakerDiarization | Audio loaded: shape={waveform.shape}, sr={sr}")
            
            # Resample if needed
            if sr != self.sample_rate:
                import librosa
                waveform_np = waveform.numpy()
                if waveform_np.ndim == 2:
                    # Resample each channel
                    resampled_channels = []
                    for ch in range(waveform_np.shape[0]):
                        resampled = librosa.resample(waveform_np[ch], orig_sr=sr, target_sr=self.sample_rate)
                        resampled_channels.append(resampled)
                    waveform_np = np.stack(resampled_channels)
                else:
                    waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=self.sample_rate)
                waveform = torch.from_numpy(waveform_np).float()
                sr = self.sample_rate
            
            # Convert to numpy for silence detection and duration calculation
            t0 = time.perf_counter()
            audio_np = waveform.numpy()
            if audio_np.ndim == 2:
                audio_np_mono = np.mean(audio_np, axis=0)  # mixdown to mono for stats
            else:
                audio_np_mono = audio_np
            audio_np_mono = np.asarray(audio_np_mono, dtype=np.float32).reshape(-1)
            duration = float(len(audio_np_mono) / self.sample_rate)
            stage_timings_ms["to_numpy_ms"] = (time.perf_counter() - t0) * 1000.0

            # silence detection
            t0 = time.perf_counter()
            if self.enable_silence_detection:
                rms, peak = self._rms_and_peak(audio_np_mono)
                if peak < self.silence_peak_threshold and rms < self.silence_rms_threshold:
                    payload = {
                        "status": "empty",
                        "empty_reason": "audio_silent",
                        "speaker_segments": [],
                        "speaker_count": 0,
                        "speaker_ids": [],
                        "duration": duration,
                        "sample_rate": self.sample_rate,
                        "rms": float(rms),
                        "peak": float(peak),
                        "model_name": self.diarization_model_name,
                        "diarization_contract_version": DIARIZATION_CONTRACT_VERSION,
                    }
                    stage_timings_ms["silence_detection_ms"] = (time.perf_counter() - t0) * 1000.0
                    stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
                    payload["stage_timings_ms"] = stage_timings_ms
                    if speaker_diarization_resource_profile is not None:
                        speaker_diarization_resource_profile["at_end"] = capture_speaker_diarization_resource_profile(stage="at_end")
                        payload["speaker_diarization_resource_profile"] = speaker_diarization_resource_profile
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
                rms_val, peak_val = rms, peak
            else:
                rms_val, peak_val = self._rms_and_peak(audio_np_mono)
            stage_timings_ms["silence_detection_ms"] = (time.perf_counter() - t0) * 1000.0

            # diarization
            self._progress(2, 3, "Running diarization")
            # Освобождаем GPU память перед запуском диаризации, чтобы избежать OOM
            if torch.cuda.is_available() and self.device_str == "cuda":
                torch.cuda.empty_cache()
                logger.info("SpeakerDiarization | Cleared GPU cache before diarization")
            
            _dev = torch.device(self.device_str)
            t0 = time.perf_counter()
            waveform_on_dev = waveform.to(_dev)
            try:
                diarization = self.diarization_pipeline(
                    {"waveform": waveform_on_dev, "sample_rate": self.sample_rate}
                )
            except RuntimeError as runtime_error:
                # Проверяем, является ли это OOM ошибкой
                error_msg = str(runtime_error).lower()
                if "out of memory" in error_msg or "cuda" in error_msg and "memory" in error_msg:
                    # Если OOM на GPU, пробуем на CPU
                    safe_log_warning(logger, "SpeakerDiarization | CUDA OOM error detected, trying CPU fallback: %s", runtime_error)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Перемещаем pipeline и волну на CPU
                    try:
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cpu"))
                        self.device_str = "cpu"
                        logger.info("SpeakerDiarization | Moved pipeline to CPU for fallback")
                        waveform_cpu = waveform.to(torch.device("cpu"))
                        diarization = self.diarization_pipeline(
                            {"waveform": waveform_cpu, "sample_rate": self.sample_rate}
                        )
                    except Exception as cpu_error:
                        raise RuntimeError(f"SpeakerDiarization failed on both GPU (OOM) and CPU: {cpu_error}") from cpu_error
                else:
                    # Это не OOM ошибка, пробрасываем дальше
                    raise
            stage_timings_ms["diarize_ms"] = (time.perf_counter() - t0) * 1000.0

            # build segments from speaker_diarization (as in example.py)
            diarization_segments: List[Tuple[float, float, str]] = []
            for turn, speaker in diarization.speaker_diarization:
                diarization_segments.append((float(turn.start), float(turn.end), str(speaker)))
            diarization_segments.sort(key=lambda x: (x[0], x[1]))

            unique_speakers_str = sorted({lab for _, _, lab in diarization_segments})
            speaker_count = len(unique_speakers_str)
            
            # Create mapping from string IDs to numeric IDs (0, 1, 2, ...)
            speaker_id_map = {sp_str: idx for idx, sp_str in enumerate(unique_speakers_str)}
            unique_speakers = list(range(speaker_count))  # Numeric IDs for npz_saver

            # prepare payload
            t0 = time.perf_counter()
            payload: Dict[str, Any] = {
                "speaker_count": speaker_count,
                "speaker_ids": unique_speakers,  # Numeric IDs [0, 1, 2, ...]
                "duration": duration,
                "sample_rate": int(self.sample_rate),
                "device_used": self.device_str,
                "rms": float(rms_val),
                "peak": float(peak_val),
                "model_name": self.diarization_model_name,
                "weights_digest": self.diarization_weights_digest,
                "diarization_contract_version": DIARIZATION_CONTRACT_VERSION,
            }

            # Token-ready turn arrays (always-on for audited preset)
            turn_start_sec = np.asarray([s for s, _, _ in diarization_segments], dtype=np.float32)
            turn_end_sec = np.asarray([e for _, e, _ in diarization_segments], dtype=np.float32)
            turn_speaker_id = np.asarray([speaker_id_map[sp] for _, _, sp in diarization_segments], dtype=np.int32)
            turn_mask = np.ones((turn_start_sec.shape[0],), dtype=bool)
            payload["turn_start_sec"] = turn_start_sec
            payload["turn_end_sec"] = turn_end_sec
            payload["turn_speaker_id"] = turn_speaker_id
            payload["turn_mask"] = turn_mask

            # Durations/ratios (structured arrays)
            speaker_duration_sec = np.zeros((speaker_count,), dtype=np.float32)
            speaker_turns_count_by_speaker = np.zeros((speaker_count,), dtype=np.int32)
            for s, e, sp in diarization_segments:
                sp_id = int(speaker_id_map[sp])
                speaker_duration_sec[sp_id] += float(e - s)
                speaker_turns_count_by_speaker[sp_id] += 1
            speaker_time_ratio = (speaker_duration_sec / float(duration)).astype(np.float32) if duration > 0 else np.zeros((speaker_count,), dtype=np.float32)
            payload["speaker_duration_sec"] = speaker_duration_sec
            payload["speaker_time_ratio"] = speaker_time_ratio
            payload["speaker_turns_count_by_speaker"] = speaker_turns_count_by_speaker

            # Aggregates (always computed)
            payload["speaker_balance_score"] = float(1.0 - np.std(speaker_time_ratio)) if speaker_time_ratio.size > 1 else 1.0
            payload["dominant_speaker_id"] = int(np.argmax(speaker_time_ratio)) if speaker_time_ratio.size > 0 else -1
            payload["speaker_turns_count"] = int(turn_start_sec.shape[0])
            payload["speaker_turns_density"] = float(turn_start_sec.shape[0] / (duration + 1e-9))
            payload["speaker_transitions_count"] = int(np.sum(turn_speaker_id[1:] != turn_speaker_id[:-1])) if turn_speaker_id.size > 1 else 0
            # speech_analysis / legacy consumers: число диаризационных «сегментов» = число тёрнов
            payload["segments_count"] = int(turn_start_sec.shape[0])

            payload["_features_enabled"] = [
                f for f, enabled in [
                    ("turns", True),
                ] if enabled
            ]

            stage_timings_ms["build_payload_ms"] = (time.perf_counter() - t0) * 1000.0
            stage_timings_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000.0
            payload["stage_timings_ms"] = stage_timings_ms
            if speaker_diarization_resource_profile is not None:
                speaker_diarization_resource_profile["at_end"] = capture_speaker_diarization_resource_profile(stage="at_end")
                payload["speaker_diarization_resource_profile"] = speaker_diarization_resource_profile
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except Exception as e:
            logger.exception("speaker_diarization failed")
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    @staticmethod
    def _hull_diarization_window(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One pyannote pass over the union of Segmenter diarization windows."""
        if len(segments) <= 1:
            return segments
        starts: List[float] = []
        ends: List[float] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            try:
                starts.append(float(seg.get("start_sec", 0.0)))
                ends.append(float(seg.get("end_sec", 0.0)))
            except (TypeError, ValueError):
                continue
        if not starts or not ends:
            raise ValueError("speaker_diarization | could not derive time span from families.diarization.segments")
        st, en = min(starts), max(ends)
        logger.info(
            "speaker_diarization | collapsing %d diarization windows to hull [%.3f, %.3f]",
            len(segments),
            st,
            en,
        )
        return [{"start_sec": st, "end_sec": en}]

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        if not isinstance(segments, list) or not segments:
            raise ValueError("speaker_diarization | segments missing (no-fallback): require families.diarization.segments")
        segments = self._hull_diarization_window(segments)
        if len(segments) != 1:
            raise ValueError(f"speaker_diarization | expected 1 diarization window after normalization, got {len(segments)}")
        # Lazy load models on first use
        self._ensure_models_loaded()
        r = self.run(input_uri, tmp_path)
        if r.payload is None:
            return r
        seg = segments[0] or {}
        st = float(seg.get("start_sec", 0.0))
        en = float(seg.get("end_sec", float(r.payload.get("duration", 0.0) or 0.0)))
        r.payload["segment_start_sec"] = [st]
        r.payload["segment_end_sec"] = [en]
        r.payload["segment_center_sec"] = [0.5 * (st + en)]
        r.payload["segment_mask"] = [True] if r.payload.get("status", "ok") == "ok" else [False]
        return r

    def _validate_input(self, input_uri: str) -> bool:
        if not super()._validate_input(input_uri):
            return False
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            logger.error("unsupported input file type: %s", input_uri)
            return False
        return True

    @property
    def supports_batch(self) -> bool:
        return False