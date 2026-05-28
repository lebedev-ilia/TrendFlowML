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
    # Metadata (always saved into feature_names/feature_values)
    add("sample_rate", payload.get("sample_rate"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("n_mels", payload.get("n_mels"))
    add("fmin", payload.get("fmin", 0.0))
    add("fmax", payload.get("fmax"))
    add("power", payload.get("power", 2.0))
    add("duration_sec", payload.get("duration"))
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
    
    # Model-facing scalars (stable order; expanded set by audit decision)
    add("mel_energy", payload.get("mel_energy", 0.0))
    add("mel_centroid_mean", payload.get("mel_centroid_mean", 0.0))
    add("mel_centroid_std", payload.get("mel_centroid_std", 0.0))
    add("mel_bandwidth_mean", payload.get("mel_bandwidth_mean", 0.0))
    add("mel_bandwidth_std", payload.get("mel_bandwidth_std", 0.0))
    add("mel_spectrogram_entropy", payload.get("mel_spectrogram_entropy", 0.0))
    add("mel_spectrogram_contrast", payload.get("mel_spectrogram_contrast", 0.0))
    add("mel_rolloff", payload.get("mel_rolloff", 0.0))
    add("mel_flatness", payload.get("mel_flatness", 0.0))
    add("mel_stability", payload.get("mel_stability", 0.0))

    arrays: Dict[str, np.ndarray] = {}

    # Canonical segment time axis (Audit v3): always present (may be empty for run()).
    arrays["segment_start_sec"] = _arr("segment_start_sec", dtype=np.float32)
    arrays["segment_end_sec"] = _arr("segment_end_sec", dtype=np.float32)
    arrays["segment_center_sec"] = _arr("segment_center_sec", dtype=np.float32)
    arrays["segment_mask"] = _arr("segment_mask", dtype=bool)

    # Statistics arrays (optional; omit when disabled)
    if "statistics" in features_enabled:
        arrays["mel_mean"] = _arr("mel_mean", dtype=np.float32)
        arrays["mel_std"] = _arr("mel_std", dtype=np.float32)
        arrays["mel_min"] = _arr("mel_min", dtype=np.float32)
        arrays["mel_max"] = _arr("mel_max", dtype=np.float32)
    if "stats_vector" in features_enabled:
        arrays["mel_stats_vector"] = _arr("mel_stats_vector", dtype=np.float32)

    # Segment-aligned sequences (optional; omit when disabled)
    if "time_series" in features_enabled:
        mm = payload.get("mel_mean_by_segment")
        if mm is not None:
            arrays["mel_mean_by_segment"] = np.asarray(mm, dtype=np.float32)
        arrays["mel_energy_by_segment"] = _arr("mel_energy_by_segment", dtype=np.float32)
        arrays["mel_centroid_mean_by_segment"] = _arr("mel_centroid_mean_by_segment", dtype=np.float32)
        arrays["mel_bandwidth_mean_by_segment"] = _arr("mel_bandwidth_mean_by_segment", dtype=np.float32)

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "mel_contract_version": payload.get("mel_contract_version", MEL_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        "device_used": payload.get("device_used"),
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "mel_resource_profile": payload.get("mel_resource_profile"),
    }
    # Debug-only artifact pointers (offline; no large arrays in NPZ)
    if isinstance(payload.get("mel_spectrogram_npy"), str) and payload.get("mel_spectrogram_npy"):
        meta_extra["mel_spectrogram_npy"] = payload.get("mel_spectrogram_npy")
    if isinstance(payload.get("mel_series_npy"), str) and payload.get("mel_series_npy"):
        meta_extra["mel_series_npy"] = payload.get("mel_series_npy")

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        **arrays,
        meta=build_meta(
            producer="mel_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path

