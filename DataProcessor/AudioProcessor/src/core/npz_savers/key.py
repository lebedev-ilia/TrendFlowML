"""
NPZ савер для key_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

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
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("key_name", payload.get("key_name"))
    add("key_mode", payload.get("key_mode"))
    add("key_confidence", payload.get("key_confidence"))
    add("method", payload.get("method"))
    add("key_confidence_category", payload.get("key_confidence_category"))
    add("key_low_confidence_warning", payload.get("key_low_confidence_warning"))
    add("key_confidence_reason", payload.get("key_confidence_reason"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"key_npz_saver | features_enabled from payload: {features_enabled}")
    logger.info(f"key_npz_saver | payload keys: {list(payload.keys())}")
    
    # Detailed scores (feature-gated)
    key_scores_arr = _arr("key_scores", dtype=np.float32) if "detailed_scores" in features_enabled else np.zeros((24,), dtype=np.float32)
    
    # Top-K keys (feature-gated)
    key_top_k = payload.get("key_top_k") if "top_k" in features_enabled else []
    logger.info(f"key_npz_saver | top_k in features_enabled: {'top_k' in features_enabled}, key_top_k={key_top_k}")
    
    # Time series (feature-gated)
    # Note: segment_centers_sec and segment_durations_sec are always saved (not feature-gated)
    # because they're needed for understanding segment structure
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32)
    logger.info(f"key_npz_saver | segment_centers_sec shape: {segment_centers_sec.shape}, size: {segment_centers_sec.size}")
    
    # Time series sequences (feature-gated)
    time_series_enabled = "time_series" in features_enabled
    logger.info(f"key_npz_saver | time_series in features_enabled: {time_series_enabled}")
    key_names_sequence = payload.get("key_names_sequence", []) if time_series_enabled else []
    key_modes_sequence = payload.get("key_modes_sequence", []) if time_series_enabled else []
    key_confidences_sequence = _arr("key_confidences_sequence", dtype=np.float32) if time_series_enabled else np.zeros((0,), dtype=np.float32)
    logger.info(f"key_npz_saver | key_names_sequence from payload: {len(key_names_sequence) if isinstance(key_names_sequence, list) else 'not a list'}, type: {type(key_names_sequence)}")
    logger.info(f"key_npz_saver | key_confidences_sequence shape: {key_confidences_sequence.shape}, size: {key_confidences_sequence.size}")
    
    # Key changes (feature-gated)
    key_transitions = payload.get("key_transitions", []) if "key_changes" in features_enabled else []
    key_transitions_count = payload.get("key_transitions_count", 0) if "key_changes" in features_enabled else 0
    key_transitions_rate = payload.get("key_transitions_rate", 0.0) if "key_changes" in features_enabled else 0.0
    
    # Stability metrics (feature-gated)
    key_stability_score = payload.get("key_stability_score", 0.0) if "stability_metrics" in features_enabled else 0.0
    key_confidence_mean = payload.get("key_confidence_mean", 0.0) if "stability_metrics" in features_enabled else 0.0
    key_confidence_std = payload.get("key_confidence_std", 0.0) if "stability_metrics" in features_enabled else 0.0
    key_confidence_min = payload.get("key_confidence_min", 0.0) if "stability_metrics" in features_enabled else 0.0
    key_confidence_max = payload.get("key_confidence_max", 0.0) if "stability_metrics" in features_enabled else 0.0
    key_distribution = payload.get("key_distribution", {}) if "stability_metrics" in features_enabled else {}
    key_diversity = payload.get("key_diversity", 0) if "stability_metrics" in features_enabled else 0
    key_detection_quality = payload.get("key_detection_quality", 0.0) if "stability_metrics" in features_enabled else 0.0
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),  # Save payload for string values (key_name, key_mode, method, etc.)
        key_scores=key_scores_arr,
        segment_centers_sec=segment_centers_sec,
        key_names_sequence=np.asarray(key_names_sequence, dtype=object) if key_names_sequence else np.zeros((0,), dtype=object),
        key_modes_sequence=np.asarray(key_modes_sequence, dtype=object) if key_modes_sequence else np.zeros((0,), dtype=object),
        key_confidences_sequence=key_confidences_sequence,
        segment_durations=segment_durations_sec,
        meta=build_meta(
            producer="key_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "key_contract_version": payload.get("key_contract_version", KEY_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "key_method": payload.get("method"),
                "key_top_k": key_top_k,
                "key_transitions": key_transitions,
                "key_transitions_count": key_transitions_count,
                "key_transitions_rate": key_transitions_rate,
                "key_stability_score": key_stability_score,
                "key_confidence_mean": key_confidence_mean,
                "key_confidence_std": key_confidence_std,
                "key_confidence_min": key_confidence_min,
                "key_confidence_max": key_confidence_max,
                "key_distribution": key_distribution,
                "key_diversity": key_diversity,
                "key_detection_quality": key_detection_quality,
            },
        ),
    )
    return out_path

