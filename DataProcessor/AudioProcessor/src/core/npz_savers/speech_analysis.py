"""
NPZ савер для speech_analysis_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_speech_analysis_npz(
    *,
    out_path: str,
    payload: Dict[str, Any],
    status: str,
    error: Optional[str],
    empty_reason: Optional[str],
    producer_version: str,
    schema_version: str,
    extra_meta: Optional[Dict[str, Any]],
    run_rs_path: str,
    feature_names: list,
    feature_values: list,
    add: Callable[[str, Any], None],
    _arr: Callable[[str, Any], np.ndarray],
    build_meta: Callable[..., np.ndarray],
) -> str:
    """
    Сохраняет NPZ артефакт для speech_analysis_extractor.
    """
    add("duration_sec", payload.get("duration_sec"))
    add("sample_rate", payload.get("sample_rate"))
    
    # Feature-gated fields
    features_enabled = payload.get("_features_enabled", [])
    if not isinstance(features_enabled, list):
        features_enabled = []
    
    # Audit v3: missing → NaN (no zero-placeholders)
    if "asr_metrics" in features_enabled:
        add("asr_segments_count", payload.get("asr_segments_count"))
        add("asr_token_total", payload.get("asr_token_total"))
        add("asr_token_mean", payload.get("asr_token_mean"))
        add("asr_token_std", payload.get("asr_token_std"))
        add("asr_token_density_per_sec", payload.get("asr_token_density_per_sec"))
        add("asr_speech_rate_wpm", payload.get("asr_speech_rate_wpm"))
    
    if "diarization_metrics" in features_enabled:
        add("speaker_count", payload.get("speaker_count"))
        add("dominant_speaker_share", payload.get("dominant_speaker_share"))
        add("speaker_balance_score", payload.get("speaker_balance_score"))
        add("speaker_transitions_count", payload.get("speaker_transitions_count"))
        add("diar_segments_count", payload.get("diar_segments_count"))
    
    if "pitch_metrics" in features_enabled:
        add("pitch_enabled", payload.get("pitch_enabled"))
        add("pitch_f0_mean", payload.get("pitch_f0_mean"))
        add("pitch_f0_std", payload.get("pitch_f0_std"))
        add("pitch_f0_min", payload.get("pitch_f0_min"))
        add("pitch_f0_max", payload.get("pitch_f0_max"))
        add("pitch_f0_range", payload.get("pitch_f0_range"))
        add("pitch_stability", payload.get("pitch_stability"))

    # Arrays (feature-gated)
    asr_lang_id_by_segment = payload.get("asr_lang_id_by_segment") if "asr_metrics" in features_enabled else None
    speaker_ids = payload.get("speaker_ids") if "diarization_metrics" in features_enabled else None
    asr_lang_distribution = payload.get("asr_lang_distribution") if "asr_metrics" in features_enabled else None
    pitch_distribution = payload.get("pitch_distribution") if "pitch_metrics" in features_enabled else None

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        asr_lang_id_by_segment=_arr("asr_lang_id_by_segment", dtype=np.int32) if asr_lang_id_by_segment is not None else np.zeros((0,), dtype=np.int32),
        speaker_ids=_arr("speaker_ids", dtype=np.int32) if speaker_ids is not None else np.zeros((0,), dtype=np.int32),
        asr_lang_distribution=np.asarray(asr_lang_distribution, dtype=object) if asr_lang_distribution is not None else np.asarray({}, dtype=object),
        pitch_distribution=np.asarray(pitch_distribution, dtype=object) if pitch_distribution is not None else np.asarray({}, dtype=object),
        meta=build_meta(
            producer="speech_analysis_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "speech_analysis_contract_version": payload.get("speech_analysis_contract_version"),
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "speech_analysis_resource_profile": payload.get("speech_analysis_resource_profile"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

