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
    
    # Energy metrics (feature-gated, omit when disabled)
    if "energy_metrics" in features_enabled:
        add("hpss_harmonic_share", payload.get("hpss_harmonic_share"))
        add("hpss_percussive_share", payload.get("hpss_percussive_share"))
        for _k in (
            "hpss_energy_total",
            "hpss_energy_harmonic",
            "hpss_energy_percussive",
            "hpss_harmonic_stability",
            "hpss_percussive_stability",
        ):
            _v = payload.get(_k)
            if _v is not None:
                add(_k, _v)
        add("hpss_separation_quality", payload.get("hpss_separation_quality"))
        add("hpss_balance_score", payload.get("hpss_balance_score"))
        if "hpss_harmonic_share_mean" in payload:
            add("hpss_harmonic_share_mean", payload.get("hpss_harmonic_share_mean"))
            add("hpss_harmonic_share_std", payload.get("hpss_harmonic_share_std"))
            add("hpss_percussive_share_mean", payload.get("hpss_percussive_share_mean"))
            add("hpss_percussive_share_std", payload.get("hpss_percussive_share_std"))
    
    # Spectral features (feature-gated; omit tabular row if missing — e.g. failed spectral sub-step)
    if "spectral_features" in features_enabled:
        for _sk in (
            "hpss_harmonic_centroid_mean",
            "hpss_harmonic_centroid_std",
            "hpss_harmonic_bandwidth_mean",
            "hpss_harmonic_bandwidth_std",
            "hpss_harmonic_rolloff_mean",
            "hpss_harmonic_rolloff_std",
            "hpss_percussive_centroid_mean",
            "hpss_percussive_centroid_std",
            "hpss_percussive_bandwidth_mean",
            "hpss_percussive_bandwidth_std",
            "hpss_percussive_rolloff_mean",
            "hpss_percussive_rolloff_std",
        ):
            _sv = payload.get(_sk)
            if _sv is not None:
                add(_sk, _sv)
    
    # Metadata — only when values are present (empty-path payload is empty → skip to keep F=0)
    for _mk, _mv in [
        ("sample_rate", payload.get("sample_rate")),
        ("n_fft", payload.get("n_fft")),
        ("hop_length", payload.get("hop_length")),
        ("duration", payload.get("duration")),
        ("hpss_frames", payload.get("hpss_frames")),
        ("hpss_kernel_size", payload.get("hpss_kernel_size")),
        ("hpss_margin", payload.get("hpss_margin")),
        ("hpss_power", payload.get("hpss_power")),
    ]:
        if _mv is not None:
            add(_mk, _mv)
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
    
    # Segment-level data (run_segments: strict alignment, Audit v3)
    segment_start_sec = _arr("segment_start_sec", dtype=np.float32) if "segment_start_sec" in payload else np.zeros((0,), dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32) if "segment_end_sec" in payload else np.zeros((0,), dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32) if "segment_center_sec" in payload else np.zeros((0,), dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool) if "segment_mask" in payload else np.zeros((0,), dtype=bool)
    hpss_harmonic_share_by_segment = _arr("hpss_harmonic_share_by_segment", dtype=np.float32) if "hpss_harmonic_share_by_segment" in payload else np.zeros((0,), dtype=np.float32)
    hpss_percussive_share_by_segment = _arr("hpss_percussive_share_by_segment", dtype=np.float32) if "hpss_percussive_share_by_segment" in payload else np.zeros((0,), dtype=np.float32)

    # Waveform paths in meta.extra only (no arrays in NPZ, Audit v3)
    meta_extra = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "hpss_contract_version": payload.get("hpss_contract_version"),
        "features_enabled": features_enabled,
        # Audit v4.2: observability
        "stage_timings_ms": payload.get("stage_timings_ms"),
        "hpss_resource_profile": payload.get("hpss_resource_profile"),
        "device_used": payload.get("device_used"),
    }
    if payload.get("hpss_dominance"):
        meta_extra["hpss_dominance"] = payload["hpss_dominance"]
    if "waveforms" in features_enabled:
        h_path = payload.get("hpss_harmonic_npy")
        p_path = payload.get("hpss_percussive_npy")
        if h_path:
            meta_extra["hpss_harmonic_npy_path"] = h_path
        if p_path:
            meta_extra["hpss_percussive_npy_path"] = p_path

    npz_arrays = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "segment_start_sec": segment_start_sec,
        "segment_end_sec": segment_end_sec,
        "segment_center_sec": segment_center_sec,
        "segment_mask": segment_mask,
        "hpss_harmonic_share_by_segment": hpss_harmonic_share_by_segment,
        "hpss_percussive_share_by_segment": hpss_percussive_share_by_segment,
        "meta": build_meta(
            producer="hpss_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    }
    if "time_series" in features_enabled:
        npz_arrays["hpss_harmonic_share_series"] = harmonic_share_series if harmonic_share_series is not None else np.zeros((0,), dtype=np.float32)
        npz_arrays["hpss_percussive_share_series"] = percussive_share_series if percussive_share_series is not None else np.zeros((0,), dtype=np.float32)

    atomic_save_npz(out_path, **npz_arrays)
    return out_path

