"""
NPZ савер для emotion_diarization_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_emotion_diarization_npz(
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
    Сохраняет NPZ артефакт для emotion_diarization_extractor.
    """
    # Feature-gated fields
    features_enabled = payload.get("_features_enabled", [])
    if not isinstance(features_enabled, list):
        features_enabled = []

    # Audit v3: minimal model-facing tabular subset (frozen within schema_version).
    add("segments_count", payload.get("segments_count"))
    add("emotion_entropy", payload.get("emotion_entropy"))
    add("dominant_emotion_id", payload.get("dominant_emotion_id"))
    add("dominant_emotion_prob", payload.get("dominant_emotion_prob"))
    add("emotion_transitions_count", payload.get("emotion_transitions_count"))
    add("emotion_stability_score", payload.get("emotion_stability_score"))
    add("emotion_diversity_score", payload.get("emotion_diversity_score"))

    labels = payload.get("emotion_labels")
    if labels is None:
        labels = []
    if not isinstance(labels, list):
        labels = []

    seg_mask_v = payload.get("segment_mask")
    if seg_mask_v is None:
        seg_mask_v = []

    arrays: Dict[str, Any] = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        # Strict-aligned time axis + mask
        "segment_start_sec": _arr("segment_start_sec", dtype=np.float32),
        "segment_end_sec": _arr("segment_end_sec", dtype=np.float32),
        "segment_center_sec": _arr("segment_center_sec", dtype=np.float32),
        "segment_mask": np.asarray(seg_mask_v, dtype=bool).reshape(-1),
        # Model-facing per-segment sequences (always present)
        "emotion_id": _arr("emotion_id", dtype=np.int32),
        "emotion_confidence": _arr("emotion_confidence", dtype=np.float32),
        "emotion_labels": np.asarray(labels, dtype=object).reshape(-1),
    }

    # Optional heavy arrays/objects: omit keys if feature disabled or missing.
    if "probs" in features_enabled and payload.get("emotion_probs") is not None:
        probs_arr = np.asarray(payload.get("emotion_probs"), dtype=np.float32)
        if probs_arr.ndim != 2:
            probs_arr = probs_arr.reshape(probs_arr.shape[0], -1) if probs_arr.size else np.zeros((0, 0), dtype=np.float32)
        arrays["emotion_probs"] = probs_arr
    if "mean_probs" in features_enabled and payload.get("emotion_mean_probs") is not None:
        arrays["emotion_mean_probs"] = np.asarray(payload.get("emotion_mean_probs"), dtype=np.float32).reshape(-1)

    if "dominant" in features_enabled:
        for k in ["emotion_distribution", "emotion_segments_per_emotion", "emotion_duration_per_emotion"]:
            v = payload.get(k)
            if isinstance(v, dict) and v:
                arrays[k] = np.asarray(v, dtype=object)

    if "quality_metrics" in features_enabled and isinstance(payload.get("emotion_quality_metrics"), dict):
        arrays["emotion_quality_metrics"] = np.asarray(payload.get("emotion_quality_metrics"), dtype=object)

    atomic_save_npz(
        out_path,
        **arrays,
        meta=build_meta(
            producer="emotion_diarization_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "emotion_contract_version": payload.get("emotion_contract_version"),
                "features_enabled": features_enabled,
                # Audit v4.2: observability
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "emotion_diarization_resource_profile": payload.get("emotion_diarization_resource_profile"),
                # Debug/meta-only (do not put into feature vector)
                "sample_rate": payload.get("sample_rate"),
                "segments_total": payload.get("segments_total"),
                "model_name": payload.get("model_name"),
                "weights_digest": payload.get("weights_digest"),
                "silence_peak_threshold": payload.get("silence_peak_threshold"),
                "silence_rms_threshold": payload.get("silence_rms_threshold"),
            },
        ),
    )
    return out_path

