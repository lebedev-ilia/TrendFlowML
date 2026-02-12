"""
ASR extractor (Whisper) — inprocess, no-network, token-IDs output (no raw text).
"""
import time
import logging
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import Counter
from contextlib import nullcontext

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for TextProcessor compatibility validation
ASR_TEXT_CONTRACT_VERSION = "asr_text_contract_v1"

# Special token IDs (Whisper tokenizer vocabulary size is typically 51865)
# These are approximate ranges; actual validation should use tokenizer vocab size
WHISPER_VOCAB_SIZE = 51865
WHISPER_TOKEN_ID_MIN = 0
WHISPER_TOKEN_ID_MAX = WHISPER_VOCAB_SIZE - 1

# Special tokens (approximate, should be validated against actual tokenizer)
SPECIAL_TOKEN_IDS = {
    50257,  # <|endoftext|>
    50258,  # <|startoftranscript|>
    50259,  # <|translate|>
    50260,  # <|notimestamps|>
    50261,  # <|nospeech|>
    50262,  # <|transcribe|>
}


class ASRExtractor(BaseExtractor):
    """
    Whisper ASR via inprocess model. Output is token IDs from a shared tokenizer (dp_models),
    so TextProcessor can decode without storing raw transcript text in artifacts.
    """
    
    name = "asr_extractor"
    version = "2.2.0"
    description = "Whisper ASR via inprocess model (token IDs, no raw text)"
    category = "speech"
    dependencies = ["numpy", "torch", "whisper", "dp_models"]
    estimated_duration = 8.0
    
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 2.0  # Model runs in-process, requires GPU memory
    
    def __init__(
        self, 
        device: str = "auto",
        model_size: str = "small",
        sample_rate: int = 16000,
        # Decode controls
        language: str = "auto",  # "auto" | "ru" | "en" | ...
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 1,
        # Fallback decode (only used when enabled)
        enable_fallback_decode: bool = False,
        fallback_temperature: float = 0.4,
        fallback_avg_logprob_threshold: float = -1.0,
        # Optional: persist raw text per segment (for downstream TextProcessor / debugging)
        save_segment_text: bool = False,
        # Feature gating flags (per-feature control, default: all False)
        enable_token_sequences: bool = False,
        enable_token_counts: bool = False,
        enable_token_total: bool = False,
        enable_token_density: bool = False,
        enable_speech_rate: bool = False,
        enable_lang_distribution: bool = False,
        enable_segments_with_speech: bool = False,
        enable_avg_segment_duration: bool = False,
        enable_token_variance: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Инициализация ASR экстрактора.
        
        Args:
            device: Устройство для обработки
            model_size: Whisper size: small|medium|large (inprocess model selection via ModelManager)
            sample_rate: Частота дискретизации
            enable_token_sequences: Включить token_ids_by_segment (sequences)
            enable_token_counts: Включить token_counts (per-segment counts)
            enable_token_total: Включить token_total (aggregate)
            enable_token_density: Включить token_density_per_sec
            enable_speech_rate: Включить speech_rate_wpm (words per minute estimate)
            enable_lang_distribution: Включить lang_distribution
            enable_segments_with_speech: Включить segments_with_speech count
            enable_avg_segment_duration: Включить avg_segment_duration_sec
            enable_token_variance: Включить token_variance (statistical variance)
            progress_callback: Callback для прогресса (segment_index, total_segments, message)
        """
        super().__init__(device=device)
        
        # Set up logger consistently
        self.logger = globals().get("logger", None) or logging.getLogger(__name__)
        
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "medium", "large"):
            raise ValueError(f"ASR | unsupported model_size={self.model_size}. Expected: small|medium|large")
        self.sample_rate = int(sample_rate)

        # Decode controls
        self.language = str(language or "auto").strip().lower()
        if self.language in ("", "none", "null"):
            self.language = "auto"
        self.temperature = float(temperature)
        self.beam_size = int(beam_size)
        self.best_of = int(best_of)
        if self.temperature < 0:
            raise ValueError("ASR | temperature must be >= 0")
        if self.beam_size <= 0:
            raise ValueError("ASR | beam_size must be > 0")
        if self.best_of <= 0:
            raise ValueError("ASR | best_of must be > 0")

        self.enable_fallback_decode = bool(enable_fallback_decode)
        self.fallback_temperature = float(fallback_temperature)
        self.fallback_avg_logprob_threshold = float(fallback_avg_logprob_threshold)
        if self.fallback_temperature < 0:
            raise ValueError("ASR | fallback_temperature must be >= 0")

        self.save_segment_text = bool(save_segment_text)
        
        # Feature gating flags
        self.enable_token_sequences = bool(enable_token_sequences)
        self.enable_token_counts = bool(enable_token_counts)
        self.enable_token_total = bool(enable_token_total)
        self.enable_token_density = bool(enable_token_density)
        self.enable_speech_rate = bool(enable_speech_rate)
        self.enable_lang_distribution = bool(enable_lang_distribution)
        self.enable_segments_with_speech = bool(enable_segments_with_speech)
        self.enable_avg_segment_duration = bool(enable_avg_segment_duration)
        self.enable_token_variance = bool(enable_token_variance)
        
        # Progress callback
        self.progress_callback = progress_callback
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

        # Resolve models via ModelManager (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore
            from dp_models.errors import ModelManagerError  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"ASR | ModelManager is required but failed to init: {e}") from e

        # Shared tokenizer must exist locally (B: shared tokenizer contract).
        try:
            tok_spec = self._mm.get_spec(model_name="shared_tokenizer_v1")
            _d, _p, _rt, _eng, tok_digest, tok_artifacts = self._mm.resolve(tok_spec)
            self.tokenizer_model_name = str(tok_spec.model_name)
            self.tokenizer_weights_digest = str(tok_digest)
            self.tokenizer_artifact_path = list(tok_artifacts.values())[0] if tok_artifacts else None
            if not self.tokenizer_artifact_path:
                raise RuntimeError("ASR | shared_tokenizer_v1 has empty artifacts")
        except Exception as e:
            raise RuntimeError(f"ASR | shared tokenizer is missing/invalid: {e}") from e
        
        # Load vocab size and special tokens from tokenizer artifact if available
        self.vocab_size = None
        self.special_token_ids = set()
        try:
            tok_path = self.tokenizer_artifact_path
            if tok_path and os.path.exists(tok_path):
                # Try to load tokenizer JSON (HuggingFace tokenizers or similar)
                import json
                # Check if it's a directory or file
                if os.path.isdir(tok_path):
                    # Common locations: tokenizer.json, vocab.json, tokenizer_config.json
                    for fname in ("tokenizer.json", "vocab.json", "tokenizer_config.json"):
                        f = os.path.join(tok_path, fname)
                        if os.path.exists(f):
                            try:
                                with open(f, "r", encoding="utf-8") as fh:
                                    tok_meta = json.load(fh)
                                # Heuristics to extract vocab size / special tokens
                                if "model" in tok_meta and isinstance(tok_meta["model"], dict):
                                    self.vocab_size = tok_meta["model"].get("vocab_size", self.vocab_size)
                                if "special_tokens" in tok_meta:
                                    for k, v in tok_meta["special_tokens"].items():
                                        if isinstance(v, (int, dict)):
                                            if isinstance(v, int):
                                                self.special_token_ids.add(int(v))
                                            elif isinstance(v, dict) and "id" in v:
                                                self.special_token_ids.add(int(v["id"]))
                            except Exception:
                                pass
                elif os.path.isfile(tok_path) and tok_path.endswith(".json"):
                    # Single JSON file
                    try:
                        with open(tok_path, "r", encoding="utf-8") as fh:
                            tok_meta = json.load(fh)
                        if "model" in tok_meta and isinstance(tok_meta["model"], dict):
                            self.vocab_size = tok_meta["model"].get("vocab_size", self.vocab_size)
                        if "special_tokens" in tok_meta:
                            for k, v in tok_meta["special_tokens"].items():
                                if isinstance(v, (int, dict)):
                                    if isinstance(v, int):
                                        self.special_token_ids.add(int(v))
                                    elif isinstance(v, dict) and "id" in v:
                                        self.special_token_ids.add(int(v["id"]))
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Fallbacks
        if self.vocab_size is None:
            self.vocab_size = WHISPER_VOCAB_SIZE
        if not self.special_token_ids:
            self.special_token_ids = SPECIAL_TOKEN_IDS.copy()
        
        # Update token ID range based on actual vocab size
        self.token_id_min = 0
        self.token_id_max = self.vocab_size - 1

        # Load shared tokenizer instance (best-effort) for converting decoded text -> shared token IDs.
        # This keeps artifacts privacy-safe (we do NOT store raw text), but ensures token ids match shared_tokenizer_v1.
        self._shared_tokenizer = None
        try:
            tok_path = self.tokenizer_artifact_path
            if tok_path:
                if os.path.isfile(tok_path):
                    tok_path = os.path.dirname(tok_path)
                if os.path.isdir(tok_path):
                    from transformers import AutoTokenizer  # type: ignore
                    self._shared_tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
                    # Prefer tokenizer's vocab size if available
                    try:
                        vs = int(getattr(self._shared_tokenizer, "vocab_size", 0) or 0)
                        if vs > 0:
                            self.vocab_size = vs
                            self.token_id_min = 0
                            self.token_id_max = self.vocab_size - 1
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"ASR | failed to load shared tokenizer for encode(): {e}")

        # Whisper inprocess spec selection by size.
        whisper_spec_name = f"whisper_{self.model_size}_inprocess"
        try:
            self.whisper_spec = self._mm.get_spec(model_name=whisper_spec_name)
            dev, prec, rt, eng, wd, _art = self._mm.resolve(self.whisper_spec)
            if str(rt) != "inprocess":
                raise RuntimeError(f"ASR | expected runtime=inprocess in spec {whisper_spec_name}, got {rt}")
            self.whisper_model_name = str(self.whisper_spec.model_name)
            self.whisper_weights_digest = str(wd)
            
            # Load model via ModelManager
            resolved_model = self._mm.get(
                model_name=whisper_spec_name,
            )
            self.whisper_model = resolved_model.handle
            self.whisper_device = str(dev)
            
            # Ensure model is in eval mode
            try:
                self.whisper_model.eval()
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"ASR | failed to resolve/load whisper inprocess model via ModelManager: {e}") from e

    def _prepare_audio_mel(self, audio_1d: np.ndarray) -> torch.Tensor:
        """
        Prepare audio for Whisper: convert to mel spectrogram and pad/trim
        to the model's expected number of mel frames (Whisper audio contract).

        Args:
            audio_1d: Audio array (samples, float32, 16000 Hz)

        Returns:
            Mel spectrogram tensor [80, n_frames] (moved to device if CUDA available)
        """
        try:
            import whisper  # type: ignore
        except Exception as e:
            raise RuntimeError(f"openai-whisper is not installed: {e}") from e

        # Convert numpy to torch tensor (1D)
        audio_tensor = torch.from_numpy(audio_1d.astype(np.float32))

        # Compute mel spectrogram (shape: [n_mels, n_frames])
        mel = whisper.audio.log_mel_spectrogram(audio_tensor, n_mels=80)

        # Defensive: ensure mel is 2D [n_mels, n_frames]
        if mel.ndim != 2:
            raise RuntimeError(f"ASR | computed mel spectrogram has wrong ndim={mel.ndim}, expected 2")

        # Whisper expects mel frames padded/trimmed to a fixed length (typically 3000),
        # so that after the stride-2 conv the time axis matches encoder positional embeddings.
        expected_frames = None
        try:
            expected_frames = int(getattr(whisper.audio, "N_FRAMES", None) or 0)  # type: ignore[attr-defined]
        except Exception:
            expected_frames = None

        if not expected_frames:
            # Fallback: derive from model dims if present (mel frames ~= 2 * n_audio_ctx due to stride=2)
            try:
                n_audio_ctx = int(getattr(getattr(self.whisper_model, "dims", None), "n_audio_ctx", 0) or 0)
                expected_frames = 2 * n_audio_ctx if n_audio_ctx > 0 else None
            except Exception:
                expected_frames = None

        if expected_frames is not None:
            n_mels, n_frames = mel.shape
            if n_mels != 80:
                # Warn but continue (Whisper expects 80)
                self.logger.warning(f"ASR | mel n_mels={n_mels} != 80 (expected). Proceeding anyway.")

            if n_frames < expected_frames:
                pad_amount = int(expected_frames - n_frames)
                pad_tensor = torch.zeros((n_mels, pad_amount), dtype=mel.dtype)
                mel = torch.cat([mel, pad_tensor], dim=1)
                self.logger.debug(f"ASR | padded mel frames: {n_frames} -> {expected_frames}")
            elif n_frames > expected_frames:
                mel = mel[:, : int(expected_frames)]
                self.logger.debug(f"ASR | trimmed mel frames: {n_frames} -> {expected_frames}")

        # Move mel to device if appropriate
        try:
            if getattr(self, "whisper_device", "").startswith("cuda") and torch.cuda.is_available():
                mel = mel.to(self.whisper_device)
        except Exception:
            # fallback: keep on CPU
            pass

        return mel

    def _infer_token_ids_from_mel(self, mel: torch.Tensor) -> Tuple[np.ndarray, int, str, Dict[str, Any]]:
        """
        Run Whisper inference on mel spectrogram to get token IDs and language ID.
        
        Args:
            mel: Mel spectrogram tensor [80, n_frames]
        
        Returns:
            (token_ids, lang_id) where lang_id is a language index (>= -1), not a token ID
        """
        import whisper  # type: ignore
        
        # --- compute torch.device robustly ---
        try:
            # If whisper_device is like 'cuda:0' this will create a proper torch.device
            torch_device = torch.device(self.whisper_device) if getattr(self, "whisper_device", None) else torch.device("cpu")
            if torch_device.type == "cuda" and not torch.cuda.is_available():
                # system doesn't have CUDA available
                torch_device = torch.device("cpu")
        except Exception:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device_type = "cuda" if torch_device.type == "cuda" else "cpu"

        with torch.inference_mode():
            # Prepare input: add batch dimension [1, 80, n_frames]
            mel_input = mel.unsqueeze(0).to(torch_device)
            
            # Debug logging: log shapes for troubleshooting
            pos_emb_shape = None
            if hasattr(self.whisper_model, "positional_embedding"):
                pos_emb = self.whisper_model.positional_embedding
                if hasattr(pos_emb, "shape"):
                    pos_emb_shape = pos_emb.shape
            self.logger.debug(f"ASR | mel_input.shape={mel_input.shape}, pos_emb.shape={pos_emb_shape}")

            # Detect language and decode inside autocast (only if cuda)
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                # language detection
                lang_id = -1  # default unknown
                try:
                    detect_out = self.whisper_model.detect_language(mel_input)
                    # whisper version compatibility:
                    # - often returns (lang_token_ids: Tensor[B], probs: List[Dict[str,float]] or Tensor)
                    if isinstance(detect_out, tuple) and len(detect_out) >= 2:
                        lang_tok, probs = detect_out[0], detect_out[1]
                        # Prefer explicit language token id if provided
                        try:
                            if hasattr(lang_tok, "numel") and int(lang_tok.numel()) > 0:
                                lang_id = int(lang_tok.reshape(-1)[0].item())
                        except Exception:
                            pass
                        # Fallback: if probs is tensor-like
                        if lang_id == -1 and hasattr(probs, "argmax"):
                            try:
                                lang_id = int(probs.argmax(dim=-1).reshape(-1)[0].item())
                            except Exception:
                                lang_id = -1
                        # Fallback: if probs is list-of-dicts, keep -1 (we can store code separately later)
                    else:
                        probs = detect_out
                        if hasattr(probs, "argmax"):
                            lang_id = int(probs.argmax(dim=-1).reshape(-1)[0].item()) if getattr(probs, "numel", lambda: 0)() > 0 else -1
                except Exception as e:
                    self.logger.warning(f"ASR | language detection failed: {e}")
                    lang_id = -1

                forced_language = None if self.language == "auto" else self.language

                def _decode_once(temp: float):
                    # Notes:
                    # - beam_size/best_of are supported by openai-whisper DecodingOptions
                    # - for deterministic runs, use temperature=0.0
                    temp_f = float(temp)
                    # openai-whisper constraint: beam_size and best_of can't be given together.
                    # We pick a mode automatically:
                    # - temperature == 0.0 -> beam search (beam_size), no best_of
                    # - temperature > 0.0 -> sampling (best_of), no beam_size
                    use_beam = (temp_f == 0.0)
                    beam_size = int(self.beam_size) if use_beam else None
                    best_of = int(self.best_of) if (not use_beam) else None
                    return self.whisper_model.decode(
                        mel_input,
                        whisper.DecodingOptions(
                            language=forced_language,
                            task="transcribe",
                            fp16=(device_type == "cuda"),
                            without_timestamps=True,
                            temperature=temp_f,
                            beam_size=beam_size,
                            best_of=best_of,
                        ),
                    )

                # First pass decode
                result = _decode_once(self.temperature)
                
                # Defensive extraction of token IDs + text + quality metrics from decode result
                token_ids = np.array([], dtype=np.int32)
                decoded_text = ""
                quality: Dict[str, Any] = {}
                try:
                    # whisper.decode() may return a single DecodingResult OR a list[DecodingResult] (newer versions)
                    if isinstance(result, list):
                        result_obj = result[0] if result else None
                    else:
                        result_obj = result

                    if result_obj is not None and hasattr(result_obj, "text"):
                        try:
                            decoded_text = str(getattr(result_obj, "text") or "")
                        except Exception:
                            decoded_text = ""
                    for k in ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature"):
                        if result_obj is not None and hasattr(result_obj, k):
                            try:
                                v = getattr(result_obj, k)
                                if isinstance(v, (int, float)):
                                    quality[k] = float(v)
                            except Exception:
                                pass

                    # Optional fallback decode for difficult audio (explicit logging)
                    try:
                        if self.enable_fallback_decode:
                            avg_lp = quality.get("avg_logprob")
                            if avg_lp is not None and avg_lp < float(self.fallback_avg_logprob_threshold):
                                self.logger.warning(
                                    f"ASR | low avg_logprob={avg_lp:.3f} < {self.fallback_avg_logprob_threshold:.3f}; "
                                    f"retry decode with temperature={self.fallback_temperature}"
                                )
                                result_fb = _decode_once(self.fallback_temperature)
                                result_fb_obj = result_fb[0] if isinstance(result_fb, list) and result_fb else result_fb
                                if result_fb_obj is not None and hasattr(result_fb_obj, "text"):
                                    try:
                                        decoded_text = str(getattr(result_fb_obj, "text") or "")
                                    except Exception:
                                        pass
                                # overwrite quality with fallback metrics if present
                                for k in ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature"):
                                    if result_fb_obj is not None and hasattr(result_fb_obj, k):
                                        try:
                                            v = getattr(result_fb_obj, k)
                                            if isinstance(v, (int, float)):
                                                quality[k] = float(v)
                                        except Exception:
                                            pass
                                result_obj = result_fb_obj
                    except Exception as e:
                        self.logger.warning(f"ASR | fallback decode failed: {e}")

                    # Preferred path: convert decoded text -> shared tokenizer ids (privacy-safe: do not persist text)
                    try:
                        if (
                            result_obj is not None
                            and self._shared_tokenizer is not None
                            and hasattr(result_obj, "text")
                            and isinstance(getattr(result_obj, "text"), str)
                        ):
                            txt = str(getattr(result_obj, "text") or "")
                            if txt.strip():
                                ids = self._shared_tokenizer.encode(txt, add_special_tokens=False)
                                token_ids = np.asarray([int(i) for i in ids], dtype=np.int32).reshape(-1)
                    except Exception as e:
                        self.logger.debug(f"ASR | shared-tokenizer encode() failed, fallback to whisper tokens: {e}")

                    if result_obj is not None and hasattr(result_obj, "tokens"):
                        tokens = result_obj.tokens
                        if isinstance(tokens, (list, tuple, np.ndarray)):
                            if token_ids.size == 0:
                                token_ids = np.asarray([int(t) for t in tokens], dtype=np.int32)
                        else:
                            # tensor-like
                            if token_ids.size == 0:
                                token_ids = np.asarray(tokens.cpu().numpy(), dtype=np.int32).reshape(-1)
                    elif result_obj is not None and hasattr(result_obj, "segments") and result_obj.segments:
                        # Some versions provide segments with text; try to extract token ids per segment if available
                        # Fallback: empty token list
                        token_ids = np.array([], dtype=np.int32)
                    else:
                        token_ids = np.array([], dtype=np.int32)
                except Exception as e:
                    self.logger.warning(f"ASR | token extraction failed: {e}")
                    token_ids = np.array([], dtype=np.int32)
                    # Don't reset lang_id here - it was already detected
        
        return token_ids, lang_id, decoded_text, quality

    def _validate_token_ids(self, token_ids: np.ndarray, lang_id: int) -> tuple[bool, Optional[str]]:
        """
        Валидация token IDs: проверка диапазонов, special tokens, согласованность с lang_id.
        
        Args:
            token_ids: массив token IDs (int32)
            lang_id: языковой ID
        
        Returns:
            (is_valid, error_message)
        """
        if token_ids.size == 0:
            return True, None  # Empty is valid (no speech)
        
        # Check dtype
        if token_ids.dtype != np.int32:
            return False, f"ASR | token_ids dtype must be int32, got {token_ids.dtype}"
        
        # Check range using actual vocab size
        if np.any(token_ids < self.token_id_min) or np.any(token_ids > self.token_id_max):
            min_val = int(np.min(token_ids))
            max_val = int(np.max(token_ids))
            return False, f"ASR | token_ids out of range [{self.token_id_min}, {self.token_id_max}]: min={min_val}, max={max_val}"
        
        # Check for special tokens (should be present in valid transcripts)
        # Note: This is a soft check - special tokens are expected but not required
        has_special = any(tid in self.special_token_ids for tid in token_ids.flatten())
        
        # Check lang_id range (more permissive - accept any int >= -1)
        if not (isinstance(lang_id, int) and lang_id >= -1):
            return False, f"ASR | lang_id must be int >= -1: got {lang_id}"
        
        return True, None

    def _infer_segment_token_ids(self, audio_1d: np.ndarray) -> tuple[np.ndarray, int, str, Dict[str, Any]]:
        """
        Выполнить inference для одного сегмента через inprocess Whisper model.
        
        Returns:
            (token_ids, lang_id)
        
        Raises:
            RuntimeError при ошибках inference
        """
        try:
            # Debug logging: log original audio length
            self.logger.debug(f"ASR | audio_samples={audio_1d.shape[0]}")
            
            # Prepare mel spectrogram
            mel = self._prepare_audio_mel(audio_1d)
            
            # Run inference
            token_ids, lang_id, decoded_text, quality = self._infer_token_ids_from_mel(mel)
            
            # Validate token IDs
            is_valid, error_msg = self._validate_token_ids(token_ids, lang_id)
            if not is_valid:
                raise ValueError(f"ASR | token validation failed: {error_msg}")
            
            return token_ids, lang_id, decoded_text, quality
        except Exception as e:
            raise RuntimeError(f"ASR | inference failed: {e}") from e

    def _infer_batch_token_ids(self, audio_batch: List[np.ndarray]) -> List[tuple[np.ndarray, int, str, Dict[str, Any]]]:
        """
        Выполнить batch inference для нескольких сегментов через inprocess Whisper model.
        Note: Whisper doesn't natively support batching, so we process sequentially.
        
        Args:
            audio_batch: список аудио массивов (каждый shape [samples])
        
        Returns:
            список (token_ids, lang_id) для каждого сегмента
        """
        results = []
        for audio_1d in audio_batch:
            try:
                tok, lang_id, txt, q = self._infer_segment_token_ids(audio_1d)
                results.append((tok, lang_id, txt, q))
            except Exception as e:
                # On error, add empty result
                self.logger.warning(f"ASR | batch inference failed for one segment: {e}")
                results.append((np.array([], dtype=np.int32), -1, "", {}))
        return results

    def run_segments(
        self, 
        input_uri: str, 
        tmp_path: str, 
        segments: List[Dict[str, Any]]
    ) -> ExtractorResult:
        """
        Run ASR on Segmenter-provided long windows (families.asr) and return token ids per segment.
        No raw transcript is stored.
        
        Progress reporting: каждые 10% сегментов (если progress_callback установлен).
        """
        start_time = time.time()
        try:
            self.logger.info(f"ASR | run_segments: input_uri={input_uri}, segments_count={len(segments) if isinstance(segments, list) else 0}")
            if not self._validate_input(input_uri):
                self.logger.error(f"ASR | run_segments: input validation failed for {input_uri}")
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                self.logger.error(f"ASR | run_segments: segments is empty or not a list (type={type(segments)}, len={len(segments) if hasattr(segments, '__len__') else 'N/A'})")
                raise ValueError("segments is empty (no-fallback)")

            total_segments = len(segments)
            token_ids_by_segment: list[np.ndarray] = []
            lang_id_by_segment: list[int] = []
            segment_texts_by_segment: list[str] = []
            segment_quality_by_segment: list[Dict[str, Any]] = []
            seg_st: list[float] = []
            seg_en: list[float] = []
            seg_center: list[float] = []
            segment_durations: list[float] = []

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Load all audio segments first
            audio_segments: List[np.ndarray] = []
            for seg_idx, seg in enumerate(segments):
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                st = float(seg.get("start_sec"))
                en = float(seg.get("end_sec"))
                c = float(seg.get("center_sec"))

                wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)
                wav_np = self.audio_utils.to_numpy(wav_t)
                if wav_np.ndim == 2:
                    wav_np = wav_np[0]
                wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
                if int(sr) != int(self.sample_rate):
                    raise RuntimeError(f"ASR | segment SR mismatch: got {sr} expected {self.sample_rate}")
                
                audio_segments.append(wav_np)
                seg_st.append(float(st))
                seg_en.append(float(en))
                seg_center.append(float(c))
                segment_durations.append(float(en - st))

            # Process segments sequentially (Whisper doesn't support native batching)
            for seg_idx, audio_1d in enumerate(audio_segments):
                tok, lang_id, txt, q = self._infer_segment_token_ids(audio_1d)
                token_ids_by_segment.append(tok.astype(np.int32))
                lang_id_by_segment.append(int(lang_id))
                if self.save_segment_text:
                    segment_texts_by_segment.append(str(txt or ""))
                    # Keep quality metrics privacy-safe (numbers only)
                    segment_quality_by_segment.append(q if isinstance(q, dict) else {})
                
                # Progress reporting
                if self.progress_callback and seg_idx % progress_report_interval == 0:
                    pct = int((seg_idx + 1) * 100 / total_segments)
                    if pct != last_reported_pct:
                        self.progress_callback(seg_idx + 1, total_segments, f"Processed {seg_idx + 1}/{total_segments} segments ({pct}%)")
                        last_reported_pct = pct

            # Calculate aggregates and statistics
            payload: Dict[str, Any] = {
                "segments_count": int(len(token_ids_by_segment)),
                "sample_rate": int(self.sample_rate),
                "whisper_model_name": self.whisper_model_name,
                "tokenizer_model_name": self.tokenizer_model_name,
                "tokenizer_weights_digest": self.tokenizer_weights_digest,
                "device_used": self.whisper_device,
                "asr_text_contract_version": ASR_TEXT_CONTRACT_VERSION,  # For TextProcessor compatibility
                "decode_language": (None if self.language == "auto" else self.language),
                "decode_temperature": float(self.temperature),
                "decode_beam_size": int(self.beam_size),
                "decode_best_of": int(self.best_of),
                "decode_enable_fallback": bool(self.enable_fallback_decode),
            }

            if self.save_segment_text:
                payload["segment_texts_by_segment"] = segment_texts_by_segment
                payload["segment_quality_by_segment"] = segment_quality_by_segment
            
            # Feature gating: token sequences
            if self.enable_token_sequences:
                payload["token_ids_by_segment"] = [t.tolist() for t in token_ids_by_segment]
            
            # Feature gating: segment timings (always included, needed for downstream)
            payload["segment_start_sec"] = seg_st
            payload["segment_end_sec"] = seg_en
            payload["segment_center_sec"] = seg_center
            payload["lang_id_by_segment"] = lang_id_by_segment
            
            # Calculate token counts (if enabled)
            token_counts: List[int] = []
            if self.enable_token_counts or self.enable_token_total or self.enable_token_density or self.enable_speech_rate or self.enable_token_variance:
                token_counts = [int(tok.size) for tok in token_ids_by_segment]
                if self.enable_token_counts:
                    payload["token_counts"] = token_counts
            
            # Calculate aggregates
            if token_counts:
                # Token total
                if self.enable_token_total:
                    payload["token_total"] = int(sum(token_counts))
                
                # Token density (tokens per second)
                if self.enable_token_density:
                    total_duration = sum(segment_durations) if segment_durations else 1.0
                    payload["token_density_per_sec"] = float(sum(token_counts) / total_duration) if total_duration > 0 else 0.0
                
                # Speech rate (words per minute estimate: ~1.3 tokens per word average)
                if self.enable_speech_rate:
                    total_duration_min = sum(segment_durations) / 60.0 if segment_durations else 1.0
                    total_words_estimate = sum(token_counts) / 1.3  # Approximate tokens-to-words ratio
                    payload["speech_rate_wpm"] = float(total_words_estimate / total_duration_min) if total_duration_min > 0 else 0.0
                
                # Token variance (statistical variance of token counts across segments)
                if self.enable_token_variance:
                    if len(token_counts) > 1:
                        payload["token_variance"] = float(np.var(token_counts))
                    else:
                        payload["token_variance"] = 0.0
            
            # Language distribution
            if self.enable_lang_distribution:
                lang_counter = Counter(lang_id_by_segment)
                payload["lang_distribution"] = {str(k): int(v) for k, v in lang_counter.items()}
            
            # Segments with speech (non-empty token sequences)
            if self.enable_segments_with_speech:
                segments_with_speech = sum(1 for tok in token_ids_by_segment if tok.size > 0)
                payload["segments_with_speech"] = int(segments_with_speech)
            
            # Average segment duration
            if self.enable_avg_segment_duration:
                if segment_durations:
                    payload["avg_segment_duration_sec"] = float(np.mean(segment_durations))
                else:
                    payload["avg_segment_duration_sec"] = 0.0
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_token_sequences:
                enabled_features.append("token_sequences")
            if self.enable_token_counts:
                enabled_features.append("token_counts")
            if self.enable_token_total:
                enabled_features.append("token_total")
            if self.enable_token_density:
                enabled_features.append("token_density")
            if self.enable_speech_rate:
                enabled_features.append("speech_rate")
            if self.enable_lang_distribution:
                enabled_features.append("lang_distribution")
            if self.enable_segments_with_speech:
                enabled_features.append("segments_with_speech")
            if self.enable_avg_segment_duration:
                enabled_features.append("avg_segment_duration")
            if self.enable_token_variance:
                enabled_features.append("token_variance")
            
            payload["_features_enabled"] = enabled_features
            
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"ASR | run_segments failed: {error_msg}", exc_info=True)
            return self._create_result(False, error=error_msg, processing_time=time.time() - start_time)
    
    @property
    def supports_batch(self) -> bool:
        """ASR extractor поддерживает batch processing для сегментов (последовательная обработка)."""
        return True
    
    def extract_batch_segments(
        self,
        audio_files_with_segments: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
        max_segments_per_batch: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких видео с гибридным подходом.
        
        Гибридный подход (вариант C):
        - Собирает сегменты из всех видео
        - Группирует в батчи по max_segments_per_batch (если задан) или использует triton_batch_size
        - Обрабатывает батчи через Triton
        - Распределяет результаты обратно по видео
        
        Args:
            audio_files_with_segments: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'segments': Список сегментов для обработки
                - 'file_id': Идентификатор файла (для распределения результатов)
            max_workers: Не используется для GPU extractors (оставлено для совместимости)
            max_segments_per_batch: Максимальное количество сегментов в одном батче (None = использует triton_batch_size)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        start_time = time.time()
        
        if not audio_files_with_segments:
            return []
        
        try:
            # Этап 1: Сбор всех сегментов с привязкой к файлам
            all_segments_with_metadata: List[Dict[str, Any]] = []
            file_segment_ranges: Dict[str, Tuple[int, int]] = {}  # file_id -> (start_idx, end_idx)
            
            for file_info in audio_files_with_segments:
                file_id = file_info.get("file_id", "unknown")
                segments = file_info.get("segments", [])
                input_uri = file_info.get("input_uri")
                tmp_path = file_info.get("tmp_path")
                
                if not input_uri or not tmp_path or not segments:
                    continue
                
                start_idx = len(all_segments_with_metadata)
                for seg in segments:
                    all_segments_with_metadata.append({
                        "segment": seg,
                        "file_id": file_id,
                        "input_uri": input_uri,
                        "tmp_path": tmp_path,
                    })
                end_idx = len(all_segments_with_metadata)
                file_segment_ranges[file_id] = (start_idx, end_idx)
            
            if not all_segments_with_metadata:
                # Нет сегментов для обработки
                return [
                    self._create_result(
                        success=False,
                        error="No segments provided",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            # Этап 2: Загрузка и предобработка всех сегментов
            audio_segments: List[Dict[str, Any]] = []
            seg_st: List[float] = []
            seg_en: List[float] = []
            seg_center: List[float] = []
            segment_durations: List[float] = []
            
            for seg_meta in all_segments_with_metadata:
                seg = seg_meta["segment"]
                input_uri = seg_meta["input_uri"]
                
                try:
                    ss = int(seg.get("start_sample"))
                    es = int(seg.get("end_sample"))
                    st = float(seg.get("start_sec", 0.0))
                    en = float(seg.get("end_sec", 0.0))
                    c = float(seg.get("center_sec", 0.0))
                    
                    wav_t, sr = self.audio_utils.load_audio_segment(
                        input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate
                    )
                    wav_np = self.audio_utils.to_numpy(wav_t)
                    if wav_np.ndim == 2:
                        wav_np = wav_np[0]
                    wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
                    
                    if int(sr) != int(self.sample_rate):
                        raise RuntimeError(f"ASR | segment SR mismatch: got {sr} expected {self.sample_rate}")
                    
                    audio_segments.append({
                        "audio": wav_np,
                        "file_id": seg_meta["file_id"],
                    })
                    seg_st.append(float(st))
                    seg_en.append(float(en))
                    seg_center.append(float(c))
                    segment_durations.append(float(en - st))
                except Exception as e:
                    self.logger.error(f"Error preprocessing segment for file_id={seg_meta['file_id']}: {e}")
                    # Продолжаем обработку остальных сегментов
                    continue
            
            if not audio_segments:
                return [
                    self._create_result(
                        success=False,
                        error="Failed to preprocess any segments",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            # Этап 3: Определение размера батча
            # Используем max_segments_per_batch если задан, иначе 1 (последовательная обработка)
            effective_batch_size = max_segments_per_batch if max_segments_per_batch and max_segments_per_batch > 1 else 1
            
            # Этап 4: Обработка батчей через inprocess модель
            all_token_ids: List[np.ndarray] = []
            all_lang_ids: List[int] = []
            
            total_segments = len(audio_segments)
            for batch_start in range(0, total_segments, effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, total_segments)
                batch_audio = [item["audio"] for item in audio_segments[batch_start:batch_end]]
                
                try:
                    batch_results = self._infer_batch_token_ids(batch_audio)
                    for tok, lang_id in batch_results:
                        all_token_ids.append(tok.astype(np.int32))
                        all_lang_ids.append(int(lang_id))
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_start // effective_batch_size}: {e}")
                    # Добавляем пустые результаты для неудачных сегментов
                    for _ in range(batch_end - batch_start):
                        all_token_ids.append(np.array([], dtype=np.int32))
                        all_lang_ids.append(-1)
            
            # Этап 5: Распределение результатов обратно по файлам
            results: List[ExtractorResult] = []
            
            for file_info in audio_files_with_segments:
                file_id = file_info.get("file_id", "unknown")
                file_start, file_end = file_segment_ranges.get(file_id, (0, 0))
                
                # Извлекаем результаты для этого файла
                file_token_ids: List[np.ndarray] = []
                file_lang_ids: List[int] = []
                file_seg_st: List[float] = []
                file_seg_en: List[float] = []
                file_seg_center: List[float] = []
                file_seg_durations: List[float] = []
                
                for idx in range(len(audio_segments)):
                    if audio_segments[idx]["file_id"] == file_id:
                        file_token_ids.append(all_token_ids[idx])
                        file_lang_ids.append(all_lang_ids[idx])
                        # Находим соответствующие временные метки
                        seg_idx_in_all = idx
                        if seg_idx_in_all < len(seg_st):
                            file_seg_st.append(seg_st[seg_idx_in_all])
                            file_seg_en.append(seg_en[seg_idx_in_all])
                            file_seg_center.append(seg_center[seg_idx_in_all])
                            file_seg_durations.append(segment_durations[seg_idx_in_all])
                
                if not file_token_ids:
                    # Нет результатов для этого файла
                    results.append(self._create_result(
                        success=False,
                        error="No results generated for this file",
                        processing_time=time.time() - start_time,
                    ))
                    continue
                
                # Формируем payload для файла (аналогично run_segments)
                payload: Dict[str, Any] = {
                    "segments_count": int(len(file_token_ids)),
                    "sample_rate": int(self.sample_rate),
                    "whisper_model_name": self.whisper_model_name,
                    "tokenizer_model_name": self.tokenizer_model_name,
                    "tokenizer_weights_digest": self.tokenizer_weights_digest,
                    "device_used": self.whisper_device,
                    "asr_text_contract_version": ASR_TEXT_CONTRACT_VERSION,
                }
                
                # Feature gating: token sequences
                if self.enable_token_sequences:
                    payload["token_ids_by_segment"] = [t.tolist() for t in file_token_ids]
                
                # Feature gating: segment timings (always included)
                payload["segment_start_sec"] = file_seg_st
                payload["segment_end_sec"] = file_seg_en
                payload["segment_center_sec"] = file_seg_center
                payload["lang_id_by_segment"] = file_lang_ids
                
                # Calculate token counts (if enabled)
                token_counts: List[int] = []
                if self.enable_token_counts or self.enable_token_total or self.enable_token_density or self.enable_speech_rate or self.enable_token_variance:
                    token_counts = [int(tok.size) for tok in file_token_ids]
                    if self.enable_token_counts:
                        payload["token_counts"] = token_counts
                
                # Calculate aggregates
                if token_counts:
                    if self.enable_token_total:
                        payload["token_total"] = int(sum(token_counts))
                    
                    if self.enable_token_density:
                        total_duration = sum(file_seg_durations) if file_seg_durations else 1.0
                        payload["token_density_per_sec"] = float(sum(token_counts) / total_duration) if total_duration > 0 else 0.0
                    
                    if self.enable_speech_rate:
                        total_duration_min = sum(file_seg_durations) / 60.0 if file_seg_durations else 1.0
                        total_words_estimate = sum(token_counts) / 1.3
                        payload["speech_rate_wpm"] = float(total_words_estimate / total_duration_min) if total_duration_min > 0 else 0.0
                    
                    if self.enable_token_variance:
                        if len(token_counts) > 1:
                            payload["token_variance"] = float(np.var(token_counts))
                        else:
                            payload["token_variance"] = 0.0
                
                # Language distribution
                if self.enable_lang_distribution:
                    from collections import Counter
                    lang_counter = Counter(file_lang_ids)
                    payload["lang_distribution"] = {str(k): int(v) for k, v in lang_counter.items()}
                
                # Segments with speech
                if self.enable_segments_with_speech:
                    segments_with_speech = sum(1 for tok in file_token_ids if tok.size > 0)
                    payload["segments_with_speech"] = int(segments_with_speech)
                
                # Average segment duration
                if self.enable_avg_segment_duration:
                    if file_seg_durations:
                        payload["avg_segment_duration_sec"] = float(np.mean(file_seg_durations))
                    else:
                        payload["avg_segment_duration_sec"] = 0.0
                
                # Track enabled features
                enabled_features = []
                if self.enable_token_sequences:
                    enabled_features.append("token_sequences")
                if self.enable_token_counts:
                    enabled_features.append("token_counts")
                if self.enable_token_total:
                    enabled_features.append("token_total")
                if self.enable_token_density:
                    enabled_features.append("token_density")
                if self.enable_speech_rate:
                    enabled_features.append("speech_rate")
                if self.enable_lang_distribution:
                    enabled_features.append("lang_distribution")
                if self.enable_segments_with_speech:
                    enabled_features.append("segments_with_speech")
                if self.enable_avg_segment_duration:
                    enabled_features.append("avg_segment_duration")
                if self.enable_token_variance:
                    enabled_features.append("token_variance")
                
                payload["_features_enabled"] = enabled_features
                
                results.append(self._create_result(
                    success=True,
                    payload=payload,
                    processing_time=time.time() - start_time,
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in extract_batch_segments: {e}")
            return [
                self._create_result(
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение транскрипции речи.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult with token IDs (requires segments mode in production).
        """
        return self._create_result(
            success=False,
            error="ASRExtractor | run() is not supported in production. Use run_segments() with Segmenter-provided families.asr windows.",
            processing_time=0.0,
        )
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем, что это аудио файл
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели Whisper."""
        return {
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "whisper_model_name": self.whisper_model_name,
            "tokenizer_model_name": self.tokenizer_model_name,
            "whisper_device": self.whisper_device,
            "tokenizer_vocab_size": self.vocab_size,
            "special_token_ids": sorted(list(self.special_token_ids)),
            "token_id_range": [self.token_id_min, self.token_id_max],
        }