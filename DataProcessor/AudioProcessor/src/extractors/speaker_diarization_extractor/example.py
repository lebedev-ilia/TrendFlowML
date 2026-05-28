#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust speaker diarization + whisperx transcription + word-level alignment CLI.

Features:
 - CLI arguments: audio path, output dir, whisper model size, device override, enable alignment
 - Uses waveform input (no torchcodec/FFmpeg required)
 - HF login (HUGGINGFACE_TOKEN or --huggingface-token)
 - Best-effort mitigation for PyTorch 2.6+ safe-unpickle (logs warning)
 - Fallback: when whisperx.word-align fails, uses uniform splitting of segments
 - Outputs: RTTM (optional) and JSON with speaker segments, embeddings, transcript
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger("diarize_whisperx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DIARIZATION_CONTRACT_VERSION = "diarization_contract_v1"


# ---------------------------
# Utilities
# ---------------------------
def ensure_hf_login(hf_token: Optional[str]):
    """Try huggingface_hub.login, fallback to env var export."""
    if not hf_token:
        logger.info("No HuggingFace token provided; attempting unauthenticated load (may fail for gated models).")
        return
    try:
        from huggingface_hub import login  # type: ignore
        login(token=hf_token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face via huggingface_hub.login()")
    except Exception as e:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        logger.warning("huggingface_hub.login() failed; exported token to env. Error: %s", e)


def allow_omegaconf_for_torch_if_possible():
    """Best-effort: register OmegaConf classes as safe globals for torch (only for >=2.6)."""
    try:
        from packaging import version
        import torch as _t
        if version.parse(_t.__version__.split("+")[0]) >= version.parse("2.6.0"):
            try:
                import omegaconf  # type: ignore
                safe_list = []
                for cls_name in ("ListConfig", "DictConfig", "ContainerMetadata"):
                    cls = getattr(omegaconf, cls_name, None)
                    if cls is not None:
                        safe_list.append(cls)
                if safe_list:
                    try:
                        _t.serialization.add_safe_globals(safe_list)
                        logger.warning("Added OmegaConf classes to torch safe globals (unsafe but may allow loading).")
                    except Exception:
                        try:
                            _t.serialization.safe_globals(safe_list)
                            logger.warning("Called torch.serialization.safe_globals(...) (context API).")
                        except Exception:
                            logger.warning("Could not register safe globals; consider downgrading torch to <2.6.")
            except Exception:
                logger.debug("omegaconf not available; can't register safe globals.")
    except Exception as e:
        logger.debug("Failed to check/register torch safe globals: %s", e)


def load_waveform(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio with soundfile and return waveform as (channels, time) torch.float32 tensor."""
    waveform_np, sr = sf.read(audio_path, dtype="float32")
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[None, :]  # mono -> (1, time)
    else:
        waveform_np = waveform_np.T  # (time, channels) -> (channels, time)
    waveform = torch.from_numpy(waveform_np)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(dtype=torch.float32)
    return waveform, int(sr)


# ---------------------------
# Main pipeline helpers
# ---------------------------
def load_pyannote_pipeline(model_name: str = "pyannote/speaker-diarization-3.1", hf_token: Optional[str] = None, device: str = "cpu"):
    """Load pyannote pipeline with robust fallbacks."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pyannote.audio import failed: {e}") from e

    # Try without token (hf login or env should handle auth). If TypeError occurs, try use_auth_token fallback.
    try:
        pipeline = Pipeline.from_pretrained(model_name)
    except TypeError:
        # older pyannote expects use_auth_token param
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as e:
        # re-raise with context
        raise RuntimeError(f"Failed to load pyannote pipeline '{model_name}': {e}") from e

    # move to cuda if asked
    if device == "cuda":
        try:
            pipeline.to(torch.device("cuda"))
        except Exception as e:
            logger.warning("Failed to move pyannote pipeline to CUDA (continuing on CPU): %s", e)
    
    return pipeline


def load_whisperx_model(size: str = "small", device: str = "cpu"):
    try:
        import whisperx  # type: ignore
    except Exception as e:
        raise RuntimeError(f"whisperx import failed: {e}") from e
    try:
        model = whisperx.load_model(size, device=device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load whisperx model '{size}': {e}") from e


def run_diarization_on_waveform(pipeline, waveform: torch.Tensor, sample_rate: int):
    """
    pipeline(...) accepts dict with 'waveform' and 'sample_rate'.
    Returns pipeline result (dataclass with speaker_diarization, speaker_embeddings, etc.)
    """
    try:
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        return result
    except Exception as e:
        raise RuntimeError(f"pyannote pipeline failed on waveform: {e}") from e


def transcribe_with_whisperx(whisper_model, audio_path: str):
    """Run whisperx.transcribe on the file path (returns result dict)."""
    try:
        result = whisper_model.transcribe(audio_path)
        return result
    except Exception as e:
        raise RuntimeError(f"whisperx.transcribe failed: {e}") from e


def align_words_whisperx(result_segments: List[dict], audio_path: str, device: str = "cpu") -> List[dict]:
    """Try whisperx align; fallback to segment-level uniform split on failure."""
    try:
        import whisperx  # type: ignore
        model_a, metadata = whisperx.load_align_model(language_code=result_segments[0].get("language", "en") if result_segments else "en", device=device)
        alignment = whisperx.align(result_segments, model_a, metadata, audio_path, device=device)
        word_segments = alignment.get("word_segments", []) or []
        logger.info("whisperx.align -> %d word segments", len(word_segments))
        return word_segments
    except Exception as e:
        logger.warning("whisperx alignment failed (%s). Falling back to uniform split of segments.", e)
        # fallback
        word_segments = []
        for seg in result_segments:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            text = seg.get("text", "").strip()
            words = text.split()
            if not words:
                continue
            wd = (e - s) / max(1, len(words))
            for i, w in enumerate(words):
                word_segments.append({"start": s + i * wd, "end": s + (i + 1) * wd, "word": w})
        logger.info("Fallback produced %d pseudo-word segments", len(word_segments))
        return word_segments


def assign_speaker_to_words(word_segments: List[dict], diarization_segments: List[Tuple[float, float, str]]):
    """Assign speaker label to each word by max-overlap, fallback to midpoint."""
    attributed = []
    for w in word_segments:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        best_label = "unknown"
        best_ov = 0.0
        for s, e, lab in diarization_segments:
            ov = max(0.0, min(we, e) - max(ws, s))
            if ov > best_ov:
                best_ov = ov
                best_label = lab
        if best_ov == 0.0:
            mid = 0.5 * (ws + we)
            for s, e, lab in diarization_segments:
                if s <= mid <= e:
                    best_label = lab
                    break
        attributed.append({"start": ws, "end": we, "word": w.get("word", w.get("text", "")).strip(), "speaker": best_label})
    return attributed


def group_words_to_turns(attributed_words: List[dict], max_gap: float = 1.0):
    """Group attributed consecutive words with same speaker into turns."""
    turns = []
    cur = None
    for w in attributed_words:
        if cur is None:
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": [w["word"]]}
            continue
        if w["speaker"] == cur["speaker"] and (w["start"] - cur["end"] < max_gap):
            cur["end"] = w["end"]
            cur["text"].append(w["word"])
        else:
            turns.append(cur)
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": [w["word"]]}
    if cur:
        turns.append(cur)
    # normalize text to strings
    for t in turns:
        t["text"] = " ".join(t["text"])
    return turns


def extract_diarization_segments(annotation) -> List[Tuple[float, float, str]]:
    segments = []
    for seg, _, sp in annotation.itertracks(yield_label=True):
        segments.append((float(seg.start), float(seg.end), str(sp)))
    return segments


# ---------------------------
# CLI / main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Diarize audio + whisperx transcription (waveform mode)")
    p.add_argument("--audio", "-i", required=True, help="Path to audio file (wav/flac/...)")
    p.add_argument("--output-dir", "-o", default="diarization_output", help="Output directory")
    p.add_argument("--whisper-size", default="small", help="Whisper model size")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Device selection")
    p.add_argument("--hf-token", default=None, help="HuggingFace token (or use HUGGINGFACE_TOKEN env var)")
    p.add_argument("--save-rttm", action="store_true", help="Save RTTM file")
    p.add_argument("--save-json", action="store_true", help="Save JSON summary")
    p.add_argument("--align", action="store_true", help="Attempt word alignment with whisperx (may require additional deps)")
    return p.parse_args()


def main_cli():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    ensure_hf_login(hf_token)
    allow_omegaconf_for_torch_if_possible()

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Device chosen: %s", device)

    # prepare output dir
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # load models
    start_time = time.time()
    pipeline = load_pyannote_pipeline("pyannote/speaker-diarization-community-1", hf_token=hf_token, device=device)
    whisper_model = load_whisperx_model(args.whisper_size, device=device)
    logger.info("Models loaded (elapsed %.1fs)", time.time() - start_time)

    # load waveform
    waveform, sr = load_waveform(args.audio)
    logger.info("Loaded audio: %s, sr=%d, waveform.shape=%s", args.audio, sr, tuple(waveform.shape))

    # diarize
    start_time = time.time()
    diarization = run_diarization_on_waveform(pipeline, waveform, sr)
    logger.info("Diarization finished (elapsed %.1fs)", time.time() - start_time)

    print(diarization.speaker_diarization)

    for turn, speaker in diarization.speaker_diarization:
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")

    print()

    for turn, speaker in diarization.exclusive_speaker_diarization:
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")

    # extract segments and embeddings
    dia_segments = extract_diarization_segments(diarization.speaker_diarization)
    unique_speakers = sorted({lab for _, _, lab in dia_segments})
    logger.info("Detected %d segments, %d unique speakers", len(dia_segments), len(unique_speakers))

    # (optional) save RTTM
    if args.save_rttm:
        rttm_path = os.path.join(outdir, "output.rttm")
        with open(rttm_path, "w", encoding="utf8") as f:
            diarization.speaker_diarization.write_rttm(f)
        logger.info("Saved RTTM -> %s", rttm_path)

    # whisperx transcribe (file-based)
    start_time = time.time()
    whisper_result = transcribe_with_whisperx(whisper_model, args.audio)
    logger.info("Transcription finished (%.1fs); segments=%d", time.time() - start_time, len(whisper_result.get("segments", [])))

    # align words (if requested)
    if args.align:
        word_segments = align_words_whisperx(whisper_result.get("segments", []), args.audio, device=device)
    else:
        # fallback to uniform split
        word_segments = align_words_whisperx(whisper_result.get("segments", []), args.audio, device=device)

    # attribute words
    attributed_words = assign_speaker_to_words(word_segments, dia_segments)

    # group turns
    turns = group_words_to_turns(attributed_words, max_gap=1.0)

    # prepare summary JSON
    out = {
        "audio_path": args.audio,
        "duration": float(sum((e - s) for s, e, _ in dia_segments)) if dia_segments else None,
        "sample_rate": sr,
        "speakers": unique_speakers,
        "speaker_segments": [{"start": s, "end": e, "speaker": sp} for s, e, sp in dia_segments],
        "word_segments": word_segments,
        "attributed_words": attributed_words,
        "turns": turns,
        "diarization_contract_version": DIARIZATION_CONTRACT_VERSION,
        "pipeline_model": getattr(pipeline, "__repr__", lambda: str(pipeline))(),
        "whisper_model": f"whisperx-{args.whisper_size}",
        "processing_time_s": time.time() - start_time,
    }

    if args.save_json:
        json_path = os.path.join(outdir, "diarization_result.json")
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("Saved JSON -> %s", json_path)
    else:
        logger.info("Not saving JSON (use --save-json to persist results)")

    logger.info("Done.")


if __name__ == "__main__":
    main_cli()
