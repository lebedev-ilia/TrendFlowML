"""
NPZ saver for speaker_diarization_extractor (Audit v3).

Contract goals:
- no `payload` key (NPZ is source-of-truth; strict per-extractor schema)
- no object dict/list for core fields (turn arrays + structured per-speaker arrays)
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ...utils.cli_utils import atomic_save_npz


def save_speaker_diarization_npz(
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
    features_enabled = payload.get("_features_enabled", [])
    if not isinstance(features_enabled, list):
        features_enabled = []

    # Model-facing scalars (feature vector)
    add("speaker_count", payload.get("speaker_count"))
    add("duration_sec", payload.get("duration"))
    add("sample_rate", payload.get("sample_rate"))
    add("rms", payload.get("rms"))
    add("peak", payload.get("peak"))
    add("speaker_balance_score", payload.get("speaker_balance_score"))
    add("dominant_speaker_id", payload.get("dominant_speaker_id"))
    add("speaker_turns_count", payload.get("speaker_turns_count"))
    add("speaker_turns_density", payload.get("speaker_turns_density"))
    add("speaker_transitions_count", payload.get("speaker_transitions_count"))

    # Structured arrays
    speaker_ids = _arr("speaker_ids", dtype=np.int32)
    turn_start_sec = _arr("turn_start_sec", dtype=np.float32)
    turn_end_sec = _arr("turn_end_sec", dtype=np.float32)
    turn_speaker_id = _arr("turn_speaker_id", dtype=np.int32)
    tm = payload.get("turn_mask")
    if tm is None:
        tm = []
    turn_mask = np.asarray(tm, dtype=bool).reshape(-1)

    speaker_duration_sec = _arr("speaker_duration_sec", dtype=np.float32)
    speaker_time_ratio = _arr("speaker_time_ratio", dtype=np.float32)
    speaker_turns_count_by_speaker = _arr("speaker_turns_count_by_speaker", dtype=np.int32)

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        # Segmenter-owned sampling axis (single full-audio window)
        segment_start_sec=_arr("segment_start_sec", dtype=np.float32),
        segment_end_sec=_arr("segment_end_sec", dtype=np.float32),
        segment_center_sec=_arr("segment_center_sec", dtype=np.float32),
        segment_mask=np.asarray(payload.get("segment_mask") if payload.get("segment_mask") is not None else [], dtype=bool).reshape(-1),
        # Speaker turns (token-ready)
        turn_start_sec=turn_start_sec,
        turn_end_sec=turn_end_sec,
        turn_speaker_id=turn_speaker_id,
        turn_mask=turn_mask,
        # Per-speaker structured stats
        speaker_ids=speaker_ids,
        speaker_duration_sec=speaker_duration_sec,
        speaker_time_ratio=speaker_time_ratio,
        speaker_turns_count_by_speaker=speaker_turns_count_by_speaker,
        meta=build_meta(
            producer="speaker_diarization_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "speaker_diarization_resource_profile": payload.get("speaker_diarization_resource_profile"),
                "diarization_contract_version": payload.get("diarization_contract_version"),
                "features_enabled": features_enabled,
                "model_name": payload.get("model_name"),
                "weights_digest": payload.get("weights_digest") or payload.get("diarization_weights_digest"),
            },
        ),
    )
    return out_path

