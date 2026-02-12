"""
NPZ савер для spectral_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
SPECTRAL_CONTRACT_VERSION = "spectral_contract_v1"


def save_spectral_npz(
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
    Сохраняет NPZ артефакт для spectral_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("n_fft", payload.get("n_fft"))
    add("duration", payload.get("duration"))
    add("device_used", payload.get("device_used", "cpu"))
    add("average_channels", payload.get("average_channels", True))
    add("keep_contrast_bands", payload.get("keep_contrast_bands", True))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic features stats (feature-gated)
    if "basic_features" in features_enabled:
        add("spectral_centroid_mean", payload.get("spectral_centroid_stats", {}).get("mean", 0.0))
        add("spectral_centroid_std", payload.get("spectral_centroid_stats", {}).get("std", 0.0))
        add("spectral_centroid_min", payload.get("spectral_centroid_stats", {}).get("min", 0.0))
        add("spectral_centroid_max", payload.get("spectral_centroid_stats", {}).get("max", 0.0))
        add("spectral_centroid_median", payload.get("spectral_centroid_stats", {}).get("median", 0.0))
        
        add("spectral_bandwidth_mean", payload.get("spectral_bandwidth_stats", {}).get("mean", 0.0))
        add("spectral_bandwidth_std", payload.get("spectral_bandwidth_stats", {}).get("std", 0.0))
        add("spectral_bandwidth_min", payload.get("spectral_bandwidth_stats", {}).get("min", 0.0))
        add("spectral_bandwidth_max", payload.get("spectral_bandwidth_stats", {}).get("max", 0.0))
        add("spectral_bandwidth_median", payload.get("spectral_bandwidth_stats", {}).get("median", 0.0))
        
        add("spectral_flatness_mean", payload.get("spectral_flatness_stats", {}).get("mean", 0.0))
        add("spectral_flatness_std", payload.get("spectral_flatness_stats", {}).get("std", 0.0))
        add("spectral_flatness_min", payload.get("spectral_flatness_stats", {}).get("min", 0.0))
        add("spectral_flatness_max", payload.get("spectral_flatness_stats", {}).get("max", 0.0))
        add("spectral_flatness_median", payload.get("spectral_flatness_stats", {}).get("median", 0.0))
        
        add("spectral_rolloff_mean", payload.get("spectral_rolloff_stats", {}).get("mean", 0.0))
        add("spectral_rolloff_std", payload.get("spectral_rolloff_stats", {}).get("std", 0.0))
        add("spectral_rolloff_min", payload.get("spectral_rolloff_stats", {}).get("min", 0.0))
        add("spectral_rolloff_max", payload.get("spectral_rolloff_stats", {}).get("max", 0.0))
        add("spectral_rolloff_median", payload.get("spectral_rolloff_stats", {}).get("median", 0.0))
        
        add("zcr_mean", payload.get("zcr_stats", {}).get("mean", 0.0))
        add("zcr_std", payload.get("zcr_stats", {}).get("std", 0.0))
        add("zcr_min", payload.get("zcr_stats", {}).get("min", 0.0))
        add("zcr_max", payload.get("zcr_stats", {}).get("max", 0.0))
        add("zcr_median", payload.get("zcr_stats", {}).get("median", 0.0))
        
        # Additional ML/analytics metrics
        add("spectral_centroid_median_metric", payload.get("spectral_centroid_median", 0.0))
        add("spectral_bandwidth_ratio", payload.get("spectral_bandwidth_ratio", 0.0))
        add("spectral_rolloff_ratio", payload.get("spectral_rolloff_ratio", 0.0))
        add("spectral_flatness_entropy", payload.get("spectral_flatness_entropy", 0.0))
    
    # Contrast stats (feature-gated)
    if "contrast" in features_enabled:
        add("spectral_contrast_mean", payload.get("spectral_contrast_stats", {}).get("mean", 0.0))
        add("spectral_contrast_std", payload.get("spectral_contrast_stats", {}).get("std", 0.0))
        add("spectral_contrast_min", payload.get("spectral_contrast_stats", {}).get("min", 0.0))
        add("spectral_contrast_max", payload.get("spectral_contrast_stats", {}).get("max", 0.0))
        add("spectral_contrast_median", payload.get("spectral_contrast_stats", {}).get("median", 0.0))
        add("spectral_contrast_variance", payload.get("spectral_contrast_variance", 0.0))
    
    # Advanced features stats (feature-gated)
    if "advanced_features" in features_enabled:
        add("spectral_slope_mean", payload.get("spectral_slope_stats", {}).get("mean", 0.0))
        add("spectral_slope_std", payload.get("spectral_slope_stats", {}).get("std", 0.0))
        add("spectral_slope_min", payload.get("spectral_slope_stats", {}).get("min", 0.0))
        add("spectral_slope_max", payload.get("spectral_slope_stats", {}).get("max", 0.0))
        add("spectral_slope_median", payload.get("spectral_slope_stats", {}).get("median", 0.0))
        add("spectral_slope_stability", payload.get("spectral_slope_stability", 0.0))
        
        if "spectral_flatness_db_stats" in payload:
            add("spectral_flatness_db_mean", payload.get("spectral_flatness_db_stats", {}).get("mean", 0.0))
            add("spectral_flatness_db_std", payload.get("spectral_flatness_db_stats", {}).get("std", 0.0))
            add("spectral_flatness_db_min", payload.get("spectral_flatness_db_stats", {}).get("min", 0.0))
            add("spectral_flatness_db_max", payload.get("spectral_flatness_db_stats", {}).get("max", 0.0))
            add("spectral_flatness_db_median", payload.get("spectral_flatness_db_stats", {}).get("median", 0.0))
    
    # Time series (feature-gated)
    time_series_keys = ["centroid_series", "bandwidth_series", "flatness_series", "rolloff_series", "zcr_series", "contrast_series", "slope_series"]
    segment_keys = ["segment_centers_sec", "segment_durations_sec"]
    
    time_series_arrays = {}
    if "time_series" in features_enabled:
        for key in time_series_keys:
            if key in payload:
                series = payload.get(key)
                if isinstance(series, list):
                    time_series_arrays[key] = np.asarray(series, dtype=np.float32)
                elif isinstance(series, np.ndarray):
                    time_series_arrays[key] = series.astype(np.float32)
        
        for key in segment_keys:
            if key in payload:
                series = payload.get(key)
                if isinstance(series, list):
                    time_series_arrays[key] = np.asarray(series, dtype=np.float32)
                elif isinstance(series, np.ndarray):
                    time_series_arrays[key] = series.astype(np.float32)
    
    # Spectral features correlation (if basic_features enabled)
    spectral_features_correlation = payload.get("spectral_features_correlation", {}) if "basic_features" in features_enabled else {}
    
    # Contrast bands (if contrast enabled and keep_contrast_bands=True)
    contrast_bands = None
    if "contrast" in features_enabled and payload.get("keep_contrast_bands", True):
        contrast_bands_data = payload.get("spectral_contrast_bands")
        if contrast_bands_data:
            if isinstance(contrast_bands_data, list):
                contrast_bands = np.asarray(contrast_bands_data, dtype=np.float32)
            elif isinstance(contrast_bands_data, np.ndarray):
                contrast_bands = contrast_bands_data.astype(np.float32)
    
    # Prepare NPZ data
    npz_data = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "payload": np.asarray(payload, dtype=object),
    }
    
    # Add time series arrays
    for key, arr in time_series_arrays.items():
        npz_data[key] = arr
    
    # Add contrast bands if present
    if contrast_bands is not None:
        npz_data["spectral_contrast_bands"] = contrast_bands
    
    # Build meta
    meta_extra = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "spectral_contract_version": payload.get("spectral_contract_version", SPECTRAL_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        "device_used": payload.get("device_used", "cpu"),
        "sample_rate": payload.get("sample_rate", 22050),
        "hop_length": payload.get("hop_length", 512),
        "n_fft": payload.get("n_fft", 2048),
        "average_channels": payload.get("average_channels", True),
        "keep_contrast_bands": payload.get("keep_contrast_bands", True),
    }
    
    # Add spectral features correlation if present
    if spectral_features_correlation:
        meta_extra["spectral_features_correlation"] = spectral_features_correlation
    
    # Add stage timings if present
    if "stage_timings_ms" in payload:
        meta_extra["stage_timings_ms"] = payload.get("stage_timings_ms")
    
    npz_data["meta"] = build_meta(
        producer="spectral_extractor",
        producer_version=producer_version,
        schema_version=schema_version,
        status=status,
        extra=meta_extra,
    )
    
    atomic_save_npz(out_path, **npz_data)
    return out_path

