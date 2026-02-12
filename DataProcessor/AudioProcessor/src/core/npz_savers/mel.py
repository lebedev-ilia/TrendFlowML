"""
NPZ савер для mel_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
MEL_CONTRACT_VERSION = "mel_contract_v1"


def save_mel_npz(
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
    Сохраняет NPZ артефакт для mel_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("n_mels", payload.get("n_mels"))
    add("fmin", payload.get("fmin", 0.0))
    add("fmax", payload.get("fmax"))
    add("power", payload.get("power", 2.0))
    add("duration", payload.get("duration"))
    add("device_used", payload.get("device_used", "cpu"))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic features (feature-gated)
    if "basic_features" in features_enabled:
        # Mel shape and elements
        mel_shape = payload.get("mel_shape")
        if mel_shape:
            add("mel_shape_0", mel_shape[0] if len(mel_shape) > 0 else 0)
            add("mel_shape_1", mel_shape[1] if len(mel_shape) > 1 else 0)
        add("mel_elements", payload.get("mel_elements", 0))
    
    # Statistics (feature-gated)
    if "statistics" in features_enabled:
        # Try to get arrays directly from payload (they should be added in _build_payload_from_segments)
        mel_mean = payload.get("mel_mean")
        mel_std = payload.get("mel_std")
        mel_min = payload.get("mel_min")
        mel_max = payload.get("mel_max")
        freq_mean = payload.get("freq_mean")
        freq_std = payload.get("freq_std")
        
        # Convert to numpy arrays if needed
        if mel_mean is not None and not isinstance(mel_mean, np.ndarray):
            mel_mean = np.array(mel_mean, dtype=np.float32) if mel_mean is not None else np.zeros((0,), dtype=np.float32)
        elif mel_mean is None:
            mel_mean = np.zeros((0,), dtype=np.float32)
        
        if mel_std is not None and not isinstance(mel_std, np.ndarray):
            mel_std = np.array(mel_std, dtype=np.float32) if mel_std is not None else np.zeros((0,), dtype=np.float32)
        elif mel_std is None:
            mel_std = np.zeros((0,), dtype=np.float32)
        
        if mel_min is not None and not isinstance(mel_min, np.ndarray):
            mel_min = np.array(mel_min, dtype=np.float32) if mel_min is not None else np.zeros((0,), dtype=np.float32)
        elif mel_min is None:
            mel_min = np.zeros((0,), dtype=np.float32)
        
        if mel_max is not None and not isinstance(mel_max, np.ndarray):
            mel_max = np.array(mel_max, dtype=np.float32) if mel_max is not None else np.zeros((0,), dtype=np.float32)
        elif mel_max is None:
            mel_max = np.zeros((0,), dtype=np.float32)
        
        if freq_mean is not None and not isinstance(freq_mean, np.ndarray):
            freq_mean = np.array(freq_mean, dtype=np.float32) if freq_mean is not None else np.zeros((0,), dtype=np.float32)
        elif freq_mean is None:
            freq_mean = np.zeros((0,), dtype=np.float32)
        
        if freq_std is not None and not isinstance(freq_std, np.ndarray):
            freq_std = np.array(freq_std, dtype=np.float32) if freq_std is not None else np.zeros((0,), dtype=np.float32)
        elif freq_std is None:
            freq_std = np.zeros((0,), dtype=np.float32)
        
        # Stats vector (if enabled)
        if "stats_vector" in features_enabled:
            mel_stats_vector = payload.get("mel_stats_vector")
            if mel_stats_vector is not None and not isinstance(mel_stats_vector, np.ndarray):
                mel_stats_vector = np.array(mel_stats_vector, dtype=np.float32)
            elif mel_stats_vector is None:
                mel_stats_vector = np.zeros((0,), dtype=np.float32)
        else:
            mel_stats_vector = np.zeros((0,), dtype=np.float32)
    else:
        mel_mean = np.zeros((0,), dtype=np.float32)
        mel_std = np.zeros((0,), dtype=np.float32)
        mel_min = np.zeros((0,), dtype=np.float32)
        mel_max = np.zeros((0,), dtype=np.float32)
        freq_mean = np.zeros((0,), dtype=np.float32)
        freq_std = np.zeros((0,), dtype=np.float32)
        mel_stats_vector = np.zeros((0,), dtype=np.float32)
    
    # Spectral features (feature-gated)
    if "spectral_features" in features_enabled:
        # Try to get arrays directly from payload (they should be added in _build_payload_from_segments)
        spectral_centroid = payload.get("spectral_centroid")
        spectral_bandwidth = payload.get("spectral_bandwidth")
        
        # Convert to numpy arrays if needed
        if spectral_centroid is not None and not isinstance(spectral_centroid, np.ndarray):
            spectral_centroid = np.array(spectral_centroid, dtype=np.float32)
        elif spectral_centroid is None:
            spectral_centroid = np.zeros((0,), dtype=np.float32)
        
        if spectral_bandwidth is not None and not isinstance(spectral_bandwidth, np.ndarray):
            spectral_bandwidth = np.array(spectral_bandwidth, dtype=np.float32)
        elif spectral_bandwidth is None:
            spectral_bandwidth = np.zeros((0,), dtype=np.float32)
    else:
        spectral_centroid = np.zeros((0,), dtype=np.float32)
        spectral_bandwidth = np.zeros((0,), dtype=np.float32)
    
    # Additional metrics
    add("mel_energy", payload.get("mel_energy", 0.0))
    add("mel_centroid", payload.get("mel_centroid", 0.0))
    add("mel_bandwidth", payload.get("mel_bandwidth", 0.0))
    add("mel_rolloff", payload.get("mel_rolloff", 0.0))
    add("mel_flatness", payload.get("mel_flatness", 0.0))
    add("mel_stability", payload.get("mel_stability", 0.0))
    
    # Time series (feature-gated)
    if "time_series" in features_enabled:
        mel_series = payload.get("mel_series")
        if mel_series is not None and not isinstance(mel_series, np.ndarray):
            mel_series = np.array(mel_series, dtype=np.float32)
        elif mel_series is None:
            mel_series = np.zeros((0, 0), dtype=np.float32)
        
        segment_centers_sec = payload.get("segment_centers_sec")
        if segment_centers_sec is not None and not isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = np.array(segment_centers_sec, dtype=np.float32)
        elif segment_centers_sec is None:
            segment_centers_sec = np.zeros((0,), dtype=np.float32)
        
        segment_durations_sec = payload.get("segment_durations_sec")
        if segment_durations_sec is not None and not isinstance(segment_durations_sec, np.ndarray):
            segment_durations_sec = np.array(segment_durations_sec, dtype=np.float32)
        elif segment_durations_sec is None:
            segment_durations_sec = np.zeros((0,), dtype=np.float32)
    else:
        mel_series = np.zeros((0, 0), dtype=np.float32)
        segment_centers_sec = np.zeros((0,), dtype=np.float32)
        segment_durations_sec = np.zeros((0,), dtype=np.float32)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        mel_mean=mel_mean,
        mel_std=mel_std,
        mel_min=mel_min,
        mel_max=mel_max,
        freq_mean=freq_mean,
        freq_std=freq_std,
        mel_stats_vector=mel_stats_vector,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        mel_series=mel_series,
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="mel_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "mel_contract_version": payload.get("mel_contract_version", MEL_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
            },
        ),
    )
    return out_path

