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
    probs = payload.get("emotion_probs") if features_enabled and "probs" in features_enabled else None
    if probs is None:
        probs_arr = np.zeros((0, 0), dtype=np.float32)
    else:
        probs_arr = np.asarray(probs, dtype=np.float32)
        if probs_arr.ndim != 2:
            probs_arr = probs_arr.reshape(probs_arr.shape[0], -1) if probs_arr.size else np.zeros((0, 0), dtype=np.float32)

    emo_id = _arr("emotion_id", dtype=np.int32) if features_enabled and "ids" in features_enabled else np.zeros((0,), dtype=np.int32)
    emo_conf = _arr("emotion_confidence", dtype=np.float32) if features_enabled and "confidence" in features_enabled else np.zeros((0,), dtype=np.float32)
    mean_probs = _arr("emotion_mean_probs", dtype=np.float32) if features_enabled and "mean_probs" in features_enabled else np.zeros((0,), dtype=np.float32)
    labels = payload.get("emotion_labels")
    if labels is None:
        labels = []
    if not isinstance(labels, list):
        labels = []

    add("segments_count", payload.get("segments_count"))
    add("sample_rate", payload.get("sample_rate"))
    add("rms", payload.get("rms"))
    add("peak", payload.get("peak"))
    add("model_name", payload.get("model_name"))
    # Aggregates (feature-gated)
    add("emotion_entropy", payload.get("emotion_entropy"))
    add("dominant_emotion_id", payload.get("dominant_emotion_id"))
    add("dominant_emotion_prob", payload.get("dominant_emotion_prob"))
    add("emotion_transitions_count", payload.get("emotion_transitions_count"))
    add("emotion_stability_score", payload.get("emotion_stability_score"))
    add("emotion_diversity_score", payload.get("emotion_diversity_score"))

    # Emotion distribution (feature-gated)
    emotion_distribution = payload.get("emotion_distribution")
    if emotion_distribution is None:
        emotion_distribution = {}
    if not isinstance(emotion_distribution, dict):
        emotion_distribution = {}
    
    emotion_segments_per_emotion = payload.get("emotion_segments_per_emotion")
    if emotion_segments_per_emotion is None:
        emotion_segments_per_emotion = {}
    if not isinstance(emotion_segments_per_emotion, dict):
        emotion_segments_per_emotion = {}
    
    emotion_duration_per_emotion = payload.get("emotion_duration_per_emotion")
    if emotion_duration_per_emotion is None:
        emotion_duration_per_emotion = {}
    if not isinstance(emotion_duration_per_emotion, dict):
        emotion_duration_per_emotion = {}

    # Quality metrics (feature-gated)
    quality_metrics = payload.get("emotion_quality_metrics")
    if quality_metrics is None:
        quality_metrics = {}
    if not isinstance(quality_metrics, dict):
        quality_metrics = {}

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        emotion_probs=probs_arr if probs_arr.size > 0 else np.zeros((0, 0), dtype=np.float32),
        emotion_id=emo_id,
        emotion_confidence=emo_conf,
        emotion_mean_probs=mean_probs,
        emotion_labels=np.asarray(labels, dtype=object),
        emotion_distribution=np.asarray(emotion_distribution, dtype=object),
        emotion_segments_per_emotion=np.asarray(emotion_segments_per_emotion, dtype=object),
        emotion_duration_per_emotion=np.asarray(emotion_duration_per_emotion, dtype=object),
        emotion_quality_metrics=np.asarray(quality_metrics, dtype=object) if quality_metrics else np.asarray({}, dtype=object),
        segment_start_sec=_arr("segment_start_sec", dtype=np.float32),
        segment_end_sec=_arr("segment_end_sec", dtype=np.float32),
        segment_center_sec=_arr("segment_center_sec", dtype=np.float32),
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
            },
        ),
    )
    return out_path

