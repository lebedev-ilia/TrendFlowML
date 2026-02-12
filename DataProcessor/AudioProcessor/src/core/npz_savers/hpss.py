"""
NPZ савер для hpss_extractor.
"""
import os
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_hpss_npz(
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
    Сохраняет NPZ артефакт для hpss_extractor.
    """
    # Feature-gated metrics
    features_enabled = payload.get("_features_enabled", [])
    
    # Energy metrics (feature-gated)
    if "energy_metrics" in features_enabled:
        add("hpss_harmonic_share", payload.get("hpss_harmonic_share"))
        add("hpss_percussive_share", payload.get("hpss_percussive_share"))
        add("hpss_energy_total", payload.get("hpss_energy_total"))
        add("hpss_energy_harmonic", payload.get("hpss_energy_harmonic"))
        add("hpss_energy_percussive", payload.get("hpss_energy_percussive"))
        add("hpss_harmonic_stability", payload.get("hpss_harmonic_stability"))
        add("hpss_percussive_stability", payload.get("hpss_percussive_stability"))
        add("hpss_separation_quality", payload.get("hpss_separation_quality"))
        add("hpss_balance_score", payload.get("hpss_balance_score"))
        # Segment-level aggregates (if run_segments was used)
        if "hpss_harmonic_share_mean" in payload:
            add("hpss_harmonic_share_mean", payload.get("hpss_harmonic_share_mean"))
            add("hpss_harmonic_share_std", payload.get("hpss_harmonic_share_std"))
            add("hpss_percussive_share_mean", payload.get("hpss_percussive_share_mean"))
            add("hpss_percussive_share_std", payload.get("hpss_percussive_share_std"))
    
    # Spectral features (feature-gated)
    if "spectral_features" in features_enabled:
        add("hpss_harmonic_centroid_mean", payload.get("hpss_harmonic_centroid_mean"))
        add("hpss_harmonic_centroid_std", payload.get("hpss_harmonic_centroid_std"))
        add("hpss_harmonic_bandwidth_mean", payload.get("hpss_harmonic_bandwidth_mean"))
        add("hpss_harmonic_bandwidth_std", payload.get("hpss_harmonic_bandwidth_std"))
        add("hpss_harmonic_rolloff_mean", payload.get("hpss_harmonic_rolloff_mean"))
        add("hpss_harmonic_rolloff_std", payload.get("hpss_harmonic_rolloff_std"))
        add("hpss_percussive_centroid_mean", payload.get("hpss_percussive_centroid_mean"))
        add("hpss_percussive_centroid_std", payload.get("hpss_percussive_centroid_std"))
        add("hpss_percussive_bandwidth_mean", payload.get("hpss_percussive_bandwidth_mean"))
        add("hpss_percussive_bandwidth_std", payload.get("hpss_percussive_bandwidth_std"))
        add("hpss_percussive_rolloff_mean", payload.get("hpss_percussive_rolloff_mean"))
        add("hpss_percussive_rolloff_std", payload.get("hpss_percussive_rolloff_std"))
    
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("hpss_frames", payload.get("hpss_frames"))
    add("hpss_kernel_size", payload.get("hpss_kernel_size"))
    add("hpss_margin", payload.get("hpss_margin"))
    add("hpss_power", payload.get("hpss_power"))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    
    # Time series (feature-gated, saved to .npy if large)
    harmonic_share_series = None
    percussive_share_series = None
    if "time_series" in features_enabled:
        # Check if time series are saved as .npy files
        harmonic_share_series_npy = payload.get("hpss_harmonic_share_series_npy")
        percussive_share_series_npy = payload.get("hpss_percussive_share_series_npy")
        if harmonic_share_series_npy:
            # Load from .npy file
            npy_path = os.path.join(run_rs_path, "hpss_extractor", harmonic_share_series_npy)
            if os.path.exists(npy_path):
                harmonic_share_series = np.load(npy_path)
        elif "hpss_harmonic_share_series" in payload:
            harmonic_share_series = np.asarray(payload.get("hpss_harmonic_share_series"), dtype=np.float32)
        
        if percussive_share_series_npy:
            npy_path = os.path.join(run_rs_path, "hpss_extractor", percussive_share_series_npy)
            if os.path.exists(npy_path):
                percussive_share_series = np.load(npy_path)
        elif "hpss_percussive_share_series" in payload:
            percussive_share_series = np.asarray(payload.get("hpss_percussive_share_series"), dtype=np.float32)
    
    # Segment-level data (if run_segments was used)
    segment_centers_sec = None
    segment_durations_sec = None
    if "segment_centers_sec" in payload:
        segment_centers_sec = np.asarray(payload.get("segment_centers_sec"), dtype=np.float32)
    if "segment_durations_sec" in payload:
        segment_durations_sec = np.asarray(payload.get("segment_durations_sec"), dtype=np.float32)
    
    # Waveforms (feature-gated, saved to .npy)
    harmonic_npy = None
    percussive_npy = None
    if "waveforms" in features_enabled:
        harmonic_npy_path = payload.get("hpss_harmonic_npy")
        percussive_npy_path = payload.get("hpss_percussive_npy")
        if harmonic_npy_path:
            npy_path = os.path.join(run_rs_path, "hpss_extractor", harmonic_npy_path)
            if os.path.exists(npy_path):
                harmonic_npy = np.load(npy_path)
        if percussive_npy_path:
            npy_path = os.path.join(run_rs_path, "hpss_extractor", percussive_npy_path)
            if os.path.exists(npy_path):
                percussive_npy = np.load(npy_path)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        harmonic_share_series=harmonic_share_series if harmonic_share_series is not None else np.zeros((0,), dtype=np.float32),
        percussive_share_series=percussive_share_series if percussive_share_series is not None else np.zeros((0,), dtype=np.float32),
        segment_centers_sec=segment_centers_sec if segment_centers_sec is not None else np.zeros((0,), dtype=np.float32),
        segment_durations_sec=segment_durations_sec if segment_durations_sec is not None else np.zeros((0,), dtype=np.float32),
        harmonic_npy=harmonic_npy if harmonic_npy is not None else np.zeros((0,), dtype=np.float32),
        percussive_npy=percussive_npy if percussive_npy is not None else np.zeros((0,), dtype=np.float32),
        meta=build_meta(
            producer="hpss_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "hpss_contract_version": payload.get("hpss_contract_version"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

