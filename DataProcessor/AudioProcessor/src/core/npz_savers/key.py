"""
NPZ савер для key_extractor.
Audit v3: segment_start_sec, segment_end_sec, segment_center_sec, segment_mask;
key_id_by_segment, key_confidence_by_segment; key_id in meta when n_valid > 0;
chroma_reused in meta.extra; omit optional keys when features disabled.
"""
import numpy as np
from typing import Any, Callable, Dict, Optional

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
KEY_CONTRACT_VERSION = "key_contract_v1"


def save_key_npz(
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
    Сохраняет NPZ артефакт для key_extractor.
    Audit v3: canonical keys segment_start_sec, segment_end_sec, segment_center_sec,
    segment_mask, key_id_by_segment, key_confidence_by_segment.
    """
    # Tabular: floats only (strings/bool → meta; avoids as_float NaN on categorical fields)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("key_id", payload.get("key_id"))
    add("key_confidence", payload.get("key_confidence"))

    features_enabled = payload.get("_features_enabled", [])

    # Canonical segment arrays (Audit v3: always present)
    segment_start_sec = _arr("segment_start_sec", dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool)
    key_id_by_segment = _arr("key_id_by_segment", dtype=np.int32)
    key_confidence_by_segment = _arr("key_confidence_by_segment", dtype=np.float32)

    # Optional: detailed scores (feature-gated)
    if "detailed_scores" in features_enabled:
        key_scores_arr = _arr("key_scores", dtype=np.float32)
    else:
        key_scores_arr = np.zeros((24,), dtype=np.float32)

    # Optional: top-K (feature-gated)
    key_top_k = payload.get("key_top_k", []) if "top_k" in features_enabled else []

    # Optional: time series sequences (feature-gated)
    time_series_enabled = "time_series" in features_enabled
    if time_series_enabled:
        key_names_sequence = payload.get("key_names_sequence", [])
        key_modes_sequence = payload.get("key_modes_sequence", [])
        key_confidences_sequence = _arr("key_confidences_sequence", dtype=np.float32)
    else:
        key_names_sequence = []
        key_modes_sequence = []
        key_confidences_sequence = np.zeros((0,), dtype=np.float32)

    # Optional: key changes (feature-gated)
    if "key_changes" in features_enabled:
        key_transitions = payload.get("key_transitions", [])
        key_transitions_count = payload.get("key_transitions_count", 0)
        key_transitions_rate = payload.get("key_transitions_rate", 0.0)
    else:
        key_transitions = []
        key_transitions_count = 0
        key_transitions_rate = 0.0

    # Optional: stability metrics (feature-gated)
    if "stability_metrics" in features_enabled:
        key_stability_score = payload.get("key_stability_score", 0.0)
        key_confidence_mean = payload.get("key_confidence_mean", 0.0)
        key_confidence_std = payload.get("key_confidence_std", 0.0)
        key_confidence_min = payload.get("key_confidence_min", 0.0)
        key_confidence_max = payload.get("key_confidence_max", 0.0)
        key_distribution = payload.get("key_distribution", {})
        key_diversity = payload.get("key_diversity", 0)
        key_detection_quality = payload.get("key_detection_quality", 0.0)
    else:
        key_stability_score = 0.0
        key_confidence_mean = 0.0
        key_confidence_std = 0.0
        key_confidence_min = 0.0
        key_confidence_max = 0.0
        key_distribution = {}
        key_diversity = 0
        key_detection_quality = 0.0

    # Meta extra: key_id when n_valid > 0, chroma_reused
    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        "empty_reason": empty_reason,
        "key_contract_version": payload.get("key_contract_version", KEY_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        # Audit v4.2: observability
        "stage_timings_ms": payload.get("stage_timings_ms"),
        "key_resource_profile": payload.get("key_resource_profile"),
        "key_method": payload.get("method"),
        "chroma_reused": payload.get("chroma_reused", False),
        "key_name": payload.get("key_name"),
        "key_mode": payload.get("key_mode"),
        "key_confidence_category": payload.get("key_confidence_category"),
        "key_confidence_reason": payload.get("key_confidence_reason"),
        "key_low_confidence_warning": payload.get("key_low_confidence_warning"),
    }
    if error:
        meta_extra["error"] = error

    key_id = payload.get("key_id")
    if key_id is not None and key_id >= 0:
        meta_extra["key_id"] = int(key_id)

    if "top_k" in features_enabled and key_top_k:
        meta_extra["key_top_k"] = key_top_k
    if "key_changes" in features_enabled:
        meta_extra["key_transitions"] = key_transitions
        meta_extra["key_transitions_count"] = key_transitions_count
        meta_extra["key_transitions_rate"] = key_transitions_rate
    if "stability_metrics" in features_enabled:
        meta_extra["key_stability_score"] = key_stability_score
        meta_extra["key_confidence_mean"] = key_confidence_mean
        meta_extra["key_confidence_std"] = key_confidence_std
        meta_extra["key_confidence_min"] = key_confidence_min
        meta_extra["key_confidence_max"] = key_confidence_max
        meta_extra["key_distribution"] = key_distribution
        meta_extra["key_diversity"] = key_diversity
        meta_extra["key_detection_quality"] = key_detection_quality

    arrays: Dict[str, Any] = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "segment_start_sec": segment_start_sec,
        "segment_end_sec": segment_end_sec,
        "segment_center_sec": segment_center_sec,
        "segment_mask": segment_mask,
        "key_id_by_segment": key_id_by_segment,
        "key_confidence_by_segment": key_confidence_by_segment,
        "key_scores": key_scores_arr,
        "meta": build_meta(
            producer="key_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    }
    # Schema: sequence keys bind dim N; omit entirely when time_series is off (empty arrays are not N-aligned).
    if time_series_enabled:
        arrays["key_names_sequence"] = np.asarray(key_names_sequence, dtype=object)
        arrays["key_modes_sequence"] = np.asarray(key_modes_sequence, dtype=object)
        arrays["key_confidences_sequence"] = key_confidences_sequence

    atomic_save_npz(out_path, **arrays)
    return out_path
