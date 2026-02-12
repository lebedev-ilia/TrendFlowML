"""
NPZ савер для mfcc_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
MFCC_CONTRACT_VERSION = "mfcc_contract_v1"


def save_mfcc_npz(
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
    Сохраняет NPZ артефакт для mfcc_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("n_mfcc", payload.get("n_mfcc"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("n_mels", payload.get("n_mels"))
    add("fmin", payload.get("fmin", 0.0))
    add("fmax", payload.get("fmax"))
    add("duration", payload.get("duration"))
    add("device_used", payload.get("device_used", "cpu"))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic features (feature-gated)
    if "basic_features" in features_enabled:
        # MFCC statistics
        mfcc_stats = payload.get("mfcc_statistics", {})
        if isinstance(mfcc_stats, dict):
            add("mfcc_mean", mfcc_stats.get("mean", 0.0))
            add("mfcc_std", mfcc_stats.get("std", 0.0))
            add("mfcc_min", mfcc_stats.get("min", 0.0))
            add("mfcc_max", mfcc_stats.get("max", 0.0))
            if "median" in mfcc_stats:
                add("mfcc_median", mfcc_stats.get("median", 0.0))
        
        # Additional metrics
        add("mfcc_energy", payload.get("mfcc_energy", 0.0))
        add("mfcc_centroid", payload.get("mfcc_centroid", 0.0))
        add("mfcc_bandwidth", payload.get("mfcc_bandwidth", 0.0))
        add("mfcc_skewness", payload.get("mfcc_skewness", 0.0))
        add("mfcc_kurtosis", payload.get("mfcc_kurtosis", 0.0))
        add("mfcc_correlation", payload.get("mfcc_correlation", 0.0))
        add("mfcc_stability", payload.get("mfcc_stability", 0.0))
        
        # Deltas (if enabled)
        if "deltas" in features_enabled:
            if "delta_mean" in mfcc_stats:
                add("delta_mean", mfcc_stats.get("delta_mean", 0.0))
            if "delta_std" in mfcc_stats:
                add("delta_std", mfcc_stats.get("delta_std", 0.0))
            if "delta_delta_mean" in mfcc_stats:
                add("delta_delta_mean", mfcc_stats.get("delta_delta_mean", 0.0))
            if "delta_delta_std" in mfcc_stats:
                add("delta_delta_std", mfcc_stats.get("delta_delta_std", 0.0))
    
    # MFCC features array (feature-gated, can be large)
    mfcc_features = payload.get("mfcc_features")
    if mfcc_features is not None and "basic_features" in features_enabled:
        if isinstance(mfcc_features, np.ndarray):
            mfcc_features_arr = mfcc_features.astype(np.float32)
        else:
            mfcc_features_arr = np.asarray(mfcc_features, dtype=np.float32)
    else:
        mfcc_features_arr = np.zeros((0, 0), dtype=np.float32)
    
    # Time series (feature-gated)
    mfcc_series = _arr("mfcc_series", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0, 0), dtype=np.float32)
    delta_series = _arr("delta_series", dtype=np.float32) if "time_series" in features_enabled and "deltas" in features_enabled else np.zeros((0, 0), dtype=np.float32)
    delta_delta_series = _arr("delta_delta_series", dtype=np.float32) if "time_series" in features_enabled and "deltas" in features_enabled else np.zeros((0, 0), dtype=np.float32)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        mfcc_features=mfcc_features_arr,
        mfcc_series=mfcc_series,
        delta_series=delta_series,
        delta_delta_series=delta_delta_series,
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="mfcc_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "mfcc_contract_version": payload.get("mfcc_contract_version", MFCC_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
            },
        ),
    )
    return out_path

