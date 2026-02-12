"""
NPZ савер для chroma_extractor.
"""
import os
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_chroma_npz(
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
    Сохраняет NPZ артефакт для chroma_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("n_fft", payload.get("n_fft"))
    add("duration", payload.get("duration"))
    add("chroma_frames", payload.get("chroma_frames"))
    add("n_chroma", payload.get("n_chroma", 12))
    add("tuning_estimate", payload.get("tuning_estimate"))
    add("chroma_dominant_class", payload.get("chroma_dominant_class"))
    add("chroma_dominant_energy", payload.get("chroma_dominant_energy"))
    add("chroma_harmonic_stability", payload.get("chroma_harmonic_stability"))
    add("chroma_entropy", payload.get("chroma_entropy"))
    add("chroma_contrast", payload.get("chroma_contrast"))
    add("chroma_centroid", payload.get("chroma_centroid"))
    add("chroma_rolloff", payload.get("chroma_rolloff"))
    add("segments_count", payload.get("segments_count"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    n_chroma = int(payload.get("n_chroma", 12))
    
    # Basic stats (feature-gated)
    chroma_mean_arr = _arr("chroma_mean", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    chroma_std_arr = _arr("chroma_std", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    chroma_min_arr = _arr("chroma_min", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    chroma_max_arr = _arr("chroma_max", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    
    # Extended stats (feature-gated)
    chroma_median_arr = _arr("chroma_median", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    chroma_p25_arr = _arr("chroma_p25", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    chroma_p75_arr = _arr("chroma_p75", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((n_chroma,), dtype=np.float32)
    
    # Stats vector (feature-gated)
    chroma_stats_vector_arr = _arr("chroma_stats_vector", dtype=np.float32) if "stats_vector" in features_enabled else np.zeros((0,), dtype=np.float32)
    
    # Time series (feature-gated)
    chroma = None
    if "time_series" in features_enabled:
        chroma = payload.get("chroma")
        if chroma is None:
            chroma_npy = payload.get("chroma_npy")
            if isinstance(chroma_npy, str) and chroma_npy.startswith("_artifacts/"):
                # Load from .npy file
                npy_path = os.path.join(run_rs_path, "chroma_extractor", chroma_npy)
                if os.path.exists(npy_path):
                    try:
                        chroma = np.load(npy_path)
                    except Exception:
                        chroma = np.zeros((n_chroma, 0), dtype=np.float32)
                else:
                    chroma = np.zeros((n_chroma, 0), dtype=np.float32)
    
    if chroma is None:
        chroma = np.zeros((n_chroma, 0), dtype=np.float32)
    else:
        chroma = np.asarray(chroma, dtype=np.float32)
        if chroma.ndim != 2:
            chroma = chroma.reshape(chroma.shape[0], -1) if chroma.size else np.zeros((n_chroma, 0), dtype=np.float32)
    
    # Per-segment data (for run_segments)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        chroma=chroma if chroma.size > 0 else np.zeros((n_chroma, 0), dtype=np.float32),
        chroma_mean=chroma_mean_arr,
        chroma_std=chroma_std_arr,
        chroma_min=chroma_min_arr,
        chroma_max=chroma_max_arr,
        chroma_median=chroma_median_arr,
        chroma_p25=chroma_p25_arr,
        chroma_p75=chroma_p75_arr,
        chroma_stats_vector=chroma_stats_vector_arr,
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="chroma_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "chroma_contract_version": payload.get("chroma_contract_version"),
                "features_enabled": features_enabled,
                "chroma_type": payload.get("chroma_type"),
                "normalize": payload.get("normalize"),
            },
        ),
    )
    return out_path

