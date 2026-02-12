"""
NPZ савер для onset_extractor.
"""
import os
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_onset_npz(
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
    Сохраняет NPZ артефакт для onset_extractor.
    """
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic features (feature-gated)
    if features_enabled and "basic_features" in features_enabled:
        add("onset_count", payload.get("onset_count"))
        add("onset_density_per_sec", payload.get("onset_density_per_sec"))
        add("insufficient_onsets", payload.get("insufficient_onsets") if payload.get("insufficient_onsets") is not None else False)
    
    # Interval stats (feature-gated)
    if features_enabled and "interval_stats" in features_enabled:
        add("avg_interval_sec", payload.get("avg_interval_sec"))
        add("interval_std", payload.get("interval_std"))
        add("interval_min", payload.get("interval_min"))
        add("interval_max", payload.get("interval_max"))
        add("interval_median", payload.get("interval_median"))
    
    # Rhythmic metrics (feature-gated)
    if features_enabled and "rhythmic_metrics" in features_enabled:
        add("onset_regularity_score", payload.get("onset_regularity_score"))
        add("onset_clustering_score", payload.get("onset_clustering_score"))
        add("onset_tempo_estimate", payload.get("onset_tempo_estimate"))
        add("onset_syncopation_score", payload.get("onset_syncopation_score"))
        add("onset_strength_mean", payload.get("onset_strength_mean"))
        add("onset_strength_std", payload.get("onset_strength_std"))
        add("onset_density_variance", payload.get("onset_density_variance"))
        add("onset_tempo_consistency", payload.get("onset_tempo_consistency"))  # Optional integration with tempo_extractor
    
    # Time series (feature-gated)
    onset_times = None
    if features_enabled and "time_series" in features_enabled:
        onset_times = payload.get("onset_times")
        if onset_times is None:
            onset_times = payload.get("onset_times_npy")
            if isinstance(onset_times, str) and onset_times.startswith("_artifacts/"):
                # Load from .npy file
                npy_path = os.path.join(run_rs_path, "onset_extractor", onset_times)
                if os.path.exists(npy_path):
                    try:
                        onset_times = np.load(npy_path)
                    except Exception:
                        onset_times = np.zeros((0,), dtype=np.float32)
                else:
                    onset_times = np.zeros((0,), dtype=np.float32)
    
    if onset_times is None:
        onset_times = np.zeros((0,), dtype=np.float32)
    else:
        onset_times = np.asarray(onset_times, dtype=np.float32).reshape(-1)
    
    # Per-segment data (for run_segments)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32)
    segments_count = payload.get("segments_count")
    
    # Metadata
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("segments_count", segments_count)
    add("backend", payload.get("backend"))
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        onset_times=onset_times,
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="onset_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "onset_contract_version": payload.get("onset_contract_version"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

