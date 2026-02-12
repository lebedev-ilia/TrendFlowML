#!/usr/bin/env python3
"""
Speaker diarization via pyannote.audio Pipeline + whisperx for transcription and word-level alignment.
Improved robustness:
 - safe handling of PyTorch >= 2.6 safe-unpickle changes (try to allow omegaconf types or recommend downgrade)
 - robust HuggingFace authentication (huggingface_hub.login)
 - fallback loading from ModelManager or HF repo
 - better speaker attribution using overlap (not only midpoint)
 - configurable progress_callback and helpful logging
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from packaging import version

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)
DIARIZATION_CONTRACT_VERSION = "diarization_contract_v1"


class SpeakerDiarizationExtractor(BaseExtractor):
    name = "speaker_diarization_extractor"
    version = "3.1.0"
    description = "Speaker diarization via pyannote.audio + whisperx (improved robustness)"
    category = "speech"
    dependencies = ["numpy", "torch", "pyannote.audio", "whisperx", "dp_models"]
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

        self.whisper_model_size = str(whisper_model_size or "small").lower()
        if self.whisper_model_size not in ("tiny", "base", "small", "medium", "large"):
            raise ValueError("unsupported whisper_model_size")

        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.huggingface_token:
            raise ValueError("huggingface_token is required (pass arg or set HUGGINGFACE_TOKEN)")

        self.sample_rate = int(sample_rate)
        self.enable_speaker_segments = bool(enable_speaker_segments)
        self.enable_speaker_embeddings = bool(enable_speaker_embeddings)
        self.enable_speaker_stats = bool(enable_speaker_stats)
        self.enable_speaker_durations = bool(enable_speaker_durations)
        self.enable_transcript = bool(enable_transcript)
        self.enable_word_segments = bool(enable_word_segments)

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
                            logger.warning("Added OmegaConf classes to torch safe globals (runtime workaround).")
                        except Exception:
                            # Some torch versions use the context manager API
                            try:
                                torch.serialization.safe_globals(safe_list)
                                logger.warning("Called torch.serialization.safe_globals(...) (context manager).")
                            except Exception:
                                logger.warning("Couldn't add safe globals to torch; if load fails consider downgrading torch to 2.5.x.")
                except Exception:
                    logger.debug("OmegaConf not available or failed to register safe globals; downgrading torch to 2.5.x recommended for full compatibility.")
        except Exception:
            pass  # if packaging/version isn't available, continue

        # Load pyannote pipeline
        self.diarization_pipeline = None
        self.diarization_model_name = None
        self.diarization_weights_digest = None
        self._load_pyannote_pipeline()

        # Load whisperx model
        self.whisper_model = None
        self.whisper_model_name = None
        self.whisper_weights_digest = None
        self._load_whisperx_model()

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
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pyannote.audio import failed: {e}") from e

        # Try ModelManager first (offline)
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
                return
            else:
                logger.warning("pyannote spec runtime is not inprocess; falling back to HF.")
        except Exception as e:
            logger.debug("ModelManager load failed for pyannote: %s", e)

        # Ensure HF creds are set
        try:
            from huggingface_hub import login  # type: ignore
            try:
                login(token=self.huggingface_token, add_to_git_credential=False)
                logger.info("Logged to HF via huggingface_hub.login()")
            except Exception as ex_login:
                os.environ["HUGGINGFACE_TOKEN"] = self.huggingface_token
                os.environ["HF_TOKEN"] = self.huggingface_token
                logger.warning("huggingface_hub.login failed, exported token to env: %s", ex_login)
        except Exception:
            # best-effort: set env var
            os.environ["HUGGINGFACE_TOKEN"] = self.huggingface_token
            os.environ["HF_TOKEN"] = self.huggingface_token

        # Try loading pipeline from HF repo with a couple of fallbacks
        try:
            # preferred: no explicit token argument (HF client picks it up from login/env)
            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            self.diarization_model_name = "pyannote/speaker-diarization"
            self.diarization_weights_digest = "unknown"
            logger.info("Loaded pyannote pipeline from HF (no-token variant).")
        except TypeError as e_type:
            # some pyannote versions expect use_auth_token
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.huggingface_token
                )
                self.diarization_model_name = "pyannote/speaker-diarization"
                self.diarization_weights_digest = "unknown"
                logger.info("Loaded pyannote pipeline from HF (use_auth_token fallback).")
            except Exception as e_final:
                raise RuntimeError(f"Failed to load pyannote pipeline: {e_final}") from e_final
        except Exception as e_other:
            raise RuntimeError(f"Failed to load pyannote pipeline: {e_other}") from e_other

        # push pipeline to GPU if requested
        if self.diarization_pipeline is not None and self.device_str == "cuda":
            try:
                self.diarization_pipeline.to(torch.device("cuda"))
            except Exception:
                logger.warning("Failed to move pyannote pipeline to CUDA; continuing on CPU.")

    def _load_whisperx_model(self):
        self._progress(2, 3, f"Loading whisperx ({self.whisper_model_size})...")
        try:
            import whisperx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"whisperx import failed: {e}") from e

        # ModelManager offline attempt
        try:
            spec_name = f"whisper_{self.whisper_model_size}_inprocess"
            spec = self._mm.get_spec(model_name=spec_name)
            dev, prec, rt, eng, wd, arts = self._mm.resolve(spec)
            if str(rt) == "inprocess":
                resolved = self._mm.get(model_name=spec_name)
                self.whisper_model = resolved.handle
                self.whisper_model_name = str(spec.model_name)
                self.whisper_weights_digest = str(wd)
                logger.info("Loaded whisper model from ModelManager.")
                return
        except Exception:
            logger.debug("whisper ModelManager load failed; falling back to direct load.")

        # Direct load
        try:
            self.whisper_model = whisperx.load_model(self.whisper_model_size, device=self.device_str)
            self.whisper_model_name = f"whisperx-{self.whisper_model_size}"
            self.whisper_weights_digest = "unknown"
            logger.info("Loaded whisperx model directly.")
        except Exception as e:
            raise RuntimeError(f"Failed to load whisperx model: {e}") from e

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
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="invalid input", processing_time=time.time() - start_time)

            self._progress(1, 5, "Loading audio")
            audio_tensor, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            audio_np = self.audio_utils.to_numpy(audio_tensor)
            if audio_np.ndim == 2:
                audio_np = np.mean(audio_np, axis=0)  # mixdown to mono
            audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
            duration = float(len(audio_np) / self.sample_rate)

            # silence detection
            if self.enable_silence_detection:
                rms, peak = self._rms_and_peak(audio_np)
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
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
                rms_val, peak_val = rms, peak
            else:
                rms_val, peak_val = self._rms_and_peak(audio_np)

            # diarization
            self._progress(2, 5, "Running diarization")
            diarization = self.diarization_pipeline(input_uri)

            # build segments
            diarization_segments: List[Tuple[float, float, str]] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_segments.append((float(turn.start), float(turn.end), str(speaker)))
            diarization_segments.sort(key=lambda x: (x[0], x[1]))

            unique_speakers = sorted({lab for _, _, lab in diarization_segments})
            speaker_count = len(unique_speakers)

            # ASR
            self._progress(3, 5, "Running ASR transcription")
            result = self.whisper_model.transcribe(input_uri)

            # alignment
            self._progress(4, 5, "Word alignment")
            try:
                import whisperx  # type: ignore
                model_a, metadata = whisperx.load_align_model(language_code=result.get("language", "en"), device=self.device_str)
                alignment = whisperx.align(result["segments"], model_a, metadata, input_uri, device=self.device_str)
                word_segments = alignment.get("word_segments", []) or []
            except Exception as e:
                logger.warning("whisperx alignment failed: %s; falling back to segment-level splitting", e)
                # fallback: chop each segment uniformly into words
                word_segments = []
                for seg in result.get("segments", []):
                    s = float(seg.get("start", 0.0))
                    e = float(seg.get("end", 0.0))
                    text = seg.get("text", "").strip()
                    words = text.split()
                    if not words:
                        continue
                    wd = (e - s) / len(words)
                    for i, w in enumerate(words):
                        word_segments.append({"start": s + i * wd, "end": s + (i + 1) * wd, "word": w})

            # attribute words to speakers using max-overlap
            self._progress(5, 5, "Attributing words to speakers")
            speaker_attributed_words = []
            for w in word_segments:
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", w_start))
                w_word = w.get("word", "").strip()
                speaker = self._assign_speaker_for_interval(w_start, w_end, diarization_segments)
                speaker_attributed_words.append({"start": w_start, "end": w_end, "word": w_word, "speaker": speaker})

            # group into turns
            transcript = []
            current = None
            for w in speaker_attributed_words:
                if current is None:
                    current = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": [w["word"]]}
                    continue
                if w["speaker"] == current["speaker"] and w["start"] - current["end"] < 1.0:
                    current["end"] = w["end"]
                    current["text"].append(w["word"])
                else:
                    transcript.append(current)
                    current = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": [w["word"]]}
            if current:
                transcript.append(current)

            # prepare payload
            payload: Dict[str, Any] = {
                "speaker_count": speaker_count,
                "speaker_ids": unique_speakers,
                "duration": duration,
                "sample_rate": int(self.sample_rate),
                "device_used": self.device_str,
                "rms": float(rms_val),
                "peak": float(peak_val),
                "model_name": self.diarization_model_name,
                "whisper_model_name": self.whisper_model_name,
                "diarization_contract_version": DIARIZATION_CONTRACT_VERSION,
            }
            if self.enable_speaker_segments:
                payload["speaker_segments"] = [
                    {"start": float(s), "end": float(e), "duration": float(e - s), "speaker_id": sp}
                    for s, e, sp in diarization_segments
                ]
            if self.enable_transcript:
                payload["transcript"] = [
                    {"start": float(t["start"]), "end": float(t["end"]), "speaker": t["speaker"], "text": " ".join(t["text"])}
                    for t in transcript
                ]
            if self.enable_word_segments:
                payload["word_segments"] = speaker_attributed_words
            if self.enable_speaker_stats:
                stats = {}
                for sp in unique_speakers:
                    segs = [(s, e) for s, e, lab in diarization_segments if lab == sp]
                    total = sum(e - s for s, e in segs)
                    stats[sp] = {"segments_count": len(segs), "total_duration": float(total)}
                payload["speaker_stats"] = stats
            if self.enable_speaker_durations:
                stats = payload.get("speaker_stats", {})
                ratios = {}
                if duration > 0:
                    for sp, st in stats.items():
                        ratios[sp] = float(st.get("total_duration", 0.0) / duration)
                payload["speaker_time_ratios"] = ratios
                if ratios:
                    vals = list(ratios.values())
                    payload["speaker_balance_score"] = float(1.0 - np.std(vals)) if len(vals) > 1 else 1.0
                    payload["dominant_speaker_id"] = max(ratios.items(), key=lambda x: x[1])[0]

            payload["_features_enabled"] = [
                f for f, enabled in [
                    ("speaker_segments", self.enable_speaker_segments),
                    ("transcript", self.enable_transcript),
                    ("word_segments", self.enable_word_segments),
                    ("speaker_stats", self.enable_speaker_stats),
                    ("speaker_durations", self.enable_speaker_durations),
                ] if enabled
            ]

            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except Exception as e:
            logger.exception("speaker_diarization failed")
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        # pyannote works best on full audio by default
        return self.run(input_uri, tmp_path)

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