#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_model_metadata(args: Any) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Resolve model metadata via ModelManager for reproducibility.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with model metadata for:
        - clap_model_used
        - asr_model_used
        - tokenizer_model_used
        - diar_model_used
        - emo_model_used
        - sep_model_used
    """
    result: Dict[str, Optional[Dict[str, str]]] = {
        "clap_model_used": None,
        "asr_model_used": None,
        "tokenizer_model_used": None,
        "diar_model_used": None,
        "emo_model_used": None,
        "sep_model_used": None,
    }
    
    # Resolve CLAP model meta via ModelManager (if available) for reproducibility.
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        spec = mm.get_spec(model_name="laion_clap")
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(spec)
        result["clap_model_used"] = {
            "model_name": str(spec.model_name),
            "model_version": str(getattr(spec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            # device in ModelManager is auto-picked; we still report actual extractor device for execution.
            "device": str(device),
        }
    except Exception:
        result["clap_model_used"] = None

    # Resolve Whisper(triton) + shared tokenizer meta for ASR reproducibility.
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        whisper_spec_name = f"whisper_{str(args.asr_model_size).strip().lower()}_triton"
        wspec = mm.get_spec(model_name=whisper_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(wspec)
        result["asr_model_used"] = {
            "model_name": str(wspec.model_name),
            "model_version": str(getattr(wspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
        tspec = mm.get_spec(model_name="shared_tokenizer_v1")
        _d2, _p2, rt2, eng2, wd2, _arts = mm.resolve(tspec)
        result["tokenizer_model_used"] = {
            "model_name": str(tspec.model_name),
            "model_version": str(getattr(tspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(wd2 or "unknown"),
            "runtime": str(rt2),
            "engine": str(eng2),
            "precision": str(getattr(tspec, "precision", "unknown") or "unknown"),
            "device": "cpu",
        }
    except Exception:
        result["asr_model_used"] = None
        result["tokenizer_model_used"] = None

    # Resolve speaker diarization model meta via ModelManager.
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        diar_spec_name = f"speaker_diarization_{str(args.diarization_model_size).strip().lower()}_triton"
        dspec = mm.get_spec(model_name=diar_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(dspec)
        result["diar_model_used"] = {
            "model_name": str(dspec.model_name),
            "model_version": str(getattr(dspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        result["diar_model_used"] = None

    # Resolve emotion diarization model meta via ModelManager.
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        emo_spec_name = f"emotion_diarization_{str(args.emotion_model_size).strip().lower()}_triton"
        espec = mm.get_spec(model_name=emo_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(espec)
        result["emo_model_used"] = {
            "model_name": str(espec.model_name),
            "model_version": str(getattr(espec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        result["emo_model_used"] = None

    # Resolve source separation model meta via ModelManager.
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        sep_spec_name = f"source_separation_{str(args.source_separation_model_size).strip().lower()}_inprocess"
        sspec = mm.get_spec(model_name=sep_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(sspec)
        result["sep_model_used"] = {
            "model_name": str(sspec.model_name),
            "model_version": str(getattr(sspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        result["sep_model_used"] = None

    return result

