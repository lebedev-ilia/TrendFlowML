"""
NPZ савер для voice_quality_extractor.
"""
import os
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_voice_quality_npz(
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
    Сохраняет NPZ артефакт для voice_quality_extractor.
    """
    # Feature-gated metrics
    features_enabled = payload.get("_features_enabled", [])
    
    # Jitter metrics (feature-gated)
    if "jitter" in features_enabled:
        add("vq_jitter", payload.get("vq_jitter"))
        add("vq_jitter_mean", payload.get("vq_jitter_mean"))
        add("vq_jitter_std", payload.get("vq_jitter_std"))
        add("vq_jitter_min", payload.get("vq_jitter_min"))
        add("vq_jitter_max", payload.get("vq_jitter_max"))
    
    # Shimmer metrics (feature-gated)
    if "shimmer" in features_enabled:
        add("vq_shimmer", payload.get("vq_shimmer"))
        add("vq_shimmer_mean", payload.get("vq_shimmer_mean"))
        add("vq_shimmer_std", payload.get("vq_shimmer_std"))
        add("vq_shimmer_min", payload.get("vq_shimmer_min"))
        add("vq_shimmer_max", payload.get("vq_shimmer_max"))
    
    # HNR metrics (feature-gated)
    if "hnr" in features_enabled:
        add("vq_hnr_like_db", payload.get("vq_hnr_like_db"))
        add("vq_hnr_mean", payload.get("vq_hnr_mean"))
        add("vq_hnr_std", payload.get("vq_hnr_std"))
        add("vq_hnr_min", payload.get("vq_hnr_min"))
        add("vq_hnr_max", payload.get("vq_hnr_max"))
    
    # F0 stats (feature-gated)
    if "f0_stats" in features_enabled:
        add("vq_f0_mean", payload.get("vq_f0_mean"))
        add("vq_f0_std", payload.get("vq_f0_std"))
        add("vq_f0_min", payload.get("vq_f0_min"))
        add("vq_f0_max", payload.get("vq_f0_max"))
        add("vq_f0_median", payload.get("vq_f0_median"))
        add("vq_f0_stability", payload.get("vq_f0_stability"))
        add("vq_voice_presence_ratio", payload.get("vq_voice_presence_ratio"))
    
    # Quality scores (if all three metrics enabled)
    if "jitter" in features_enabled and "shimmer" in features_enabled and "hnr" in features_enabled:
        add("vq_voice_quality_score", payload.get("vq_voice_quality_score"))
        add("vq_breathiness_score", payload.get("vq_breathiness_score"))
    
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("duration", payload.get("duration"))
    add("f0_method", payload.get("f0_method"))
    add("f0_fmin", payload.get("f0_fmin"))
    add("f0_fmax", payload.get("f0_fmax"))
    add("segments_count", payload.get("segments_count"))
    
    # Time series (feature-gated)
    f0 = None
    amps = None
    hnr_vals = None
    if "time_series" in features_enabled:
        f0 = payload.get("f0")
        if f0 is None:
            f0_npy = payload.get("f0_npy")
            if isinstance(f0_npy, str) and f0_npy.startswith("_artifacts/"):
                npy_path = os.path.join(run_rs_path, "voice_quality_extractor", f0_npy)
                if os.path.exists(npy_path):
                    try:
                        f0 = np.load(npy_path)
                    except Exception:
                        f0 = np.zeros((0,), dtype=np.float32)
                else:
                    f0 = np.zeros((0,), dtype=np.float32)
        
        amps = payload.get("amps")
        if amps is None:
            amps_npy = payload.get("amps_npy")
            if isinstance(amps_npy, str) and amps_npy.startswith("_artifacts/"):
                npy_path = os.path.join(run_rs_path, "voice_quality_extractor", amps_npy)
                if os.path.exists(npy_path):
                    try:
                        amps = np.load(npy_path)
                    except Exception:
                        amps = np.zeros((0,), dtype=np.float32)
                else:
                    amps = np.zeros((0,), dtype=np.float32)
        
        hnr_vals = payload.get("hnr_vals")
        if hnr_vals is None:
            hnr_vals_npy = payload.get("hnr_vals_npy")
            if isinstance(hnr_vals_npy, str) and hnr_vals_npy.startswith("_artifacts/"):
                npy_path = os.path.join(run_rs_path, "voice_quality_extractor", hnr_vals_npy)
                if os.path.exists(npy_path):
                    try:
                        hnr_vals = np.load(npy_path)
                    except Exception:
                        hnr_vals = np.zeros((0,), dtype=np.float32)
                else:
                    hnr_vals = np.zeros((0,), dtype=np.float32)
    
    if f0 is None:
        f0 = np.zeros((0,), dtype=np.float32)
    else:
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    
    if amps is None:
        amps = np.zeros((0,), dtype=np.float32)
    else:
        amps = np.asarray(amps, dtype=np.float32).reshape(-1)
    
    if hnr_vals is None:
        hnr_vals = np.zeros((0,), dtype=np.float32)
    else:
        hnr_vals = np.asarray(hnr_vals, dtype=np.float32).reshape(-1)
    
    # Per-segment data (for run_segments)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        f0=f0 if f0.size > 0 else np.zeros((0,), dtype=np.float32),
        amps=amps if amps.size > 0 else np.zeros((0,), dtype=np.float32),
        hnr_vals=hnr_vals if hnr_vals.size > 0 else np.zeros((0,), dtype=np.float32),
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="voice_quality_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "voice_quality_contract_version": payload.get("voice_quality_contract_version"),
                "features_enabled": features_enabled,
                "f0_method": payload.get("f0_method"),
            },
        ),
    )
    return out_path

