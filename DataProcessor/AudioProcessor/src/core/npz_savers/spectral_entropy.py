"""
NPZ савер для spectral_entropy_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
SPECTRAL_ENTROPY_CONTRACT_VERSION = "spectral_entropy_contract_v1"


def save_spectral_entropy_npz(
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
    Сохраняет NPZ артефакт для spectral_entropy_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("use_mel", payload.get("use_mel"))
    add("n_mels", payload.get("n_mels"))
    add("smoothing_window", payload.get("smoothing_window"))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Entropy stats (feature-gated)
    entropy_stats = payload.get("spectral_entropy_stats", {})
    if "basic_stats" in features_enabled:
        add("spectral_entropy_mean", entropy_stats.get("mean", 0.0))
        add("spectral_entropy_std", entropy_stats.get("std", 0.0))
        if "extended_stats" in features_enabled:
            add("spectral_entropy_min", entropy_stats.get("min", 0.0))
            add("spectral_entropy_max", entropy_stats.get("max", 0.0))
            add("spectral_entropy_p25", entropy_stats.get("p25", 0.0))
            add("spectral_entropy_p75", entropy_stats.get("p75", 0.0))
    
    # Flatness stats (feature-gated)
    flatness_stats = payload.get("spectral_flatness_stats", {})
    if "flatness" in features_enabled:
        add("spectral_flatness_mean", flatness_stats.get("mean", 0.0))
        add("spectral_flatness_std", flatness_stats.get("std", 0.0))
        if "extended_stats" in features_enabled:
            add("spectral_flatness_min", flatness_stats.get("min", 0.0))
            add("spectral_flatness_max", flatness_stats.get("max", 0.0))
            add("spectral_flatness_p25", flatness_stats.get("p25", 0.0))
            add("spectral_flatness_p75", flatness_stats.get("p75", 0.0))
    
    # Spread stats (feature-gated)
    spread_stats = payload.get("spectral_spread_stats", {})
    if "spread" in features_enabled:
        add("spectral_spread_mean", spread_stats.get("mean", 0.0))
        add("spectral_spread_std", spread_stats.get("std", 0.0))
        if "extended_stats" in features_enabled:
            add("spectral_spread_min", spread_stats.get("min", 0.0))
            add("spectral_spread_max", spread_stats.get("max", 0.0))
            add("spectral_spread_p25", spread_stats.get("p25", 0.0))
            add("spectral_spread_p75", spread_stats.get("p75", 0.0))
    
    # Time series (feature-gated)
    entropy_series = None
    flatness_series = None
    spread_series = None
    if "time_series" in features_enabled:
        entropy_series_data = payload.get("spectral_entropy_series", [])
        if entropy_series_data:
            if isinstance(entropy_series_data, list):
                entropy_series = np.array(entropy_series_data, dtype=np.float32)
        
        if "flatness" in features_enabled:
            flatness_series_data = payload.get("spectral_flatness_series", [])
            if flatness_series_data:
                if isinstance(flatness_series_data, list):
                    flatness_series = np.array(flatness_series_data, dtype=np.float32)
        
        if "spread" in features_enabled:
            spread_series_data = payload.get("spectral_spread_series", [])
            if spread_series_data:
                if isinstance(spread_series_data, list):
                    spread_series = np.array(spread_series_data, dtype=np.float32)
    
    # Additional metrics
    entropy_variance = payload.get("spectral_entropy_variance", 0.0)
    entropy_min = payload.get("spectral_entropy_min", 0.0)
    entropy_max = payload.get("spectral_entropy_max", 0.0)
    flatness_variance = payload.get("spectral_flatness_variance", 0.0) if "flatness" in features_enabled else 0.0
    flatness_min = payload.get("spectral_flatness_min", 0.0) if "flatness" in features_enabled else 0.0
    flatness_max = payload.get("spectral_flatness_max", 0.0) if "flatness" in features_enabled else 0.0
    spread_variance = payload.get("spectral_spread_variance", 0.0) if "spread" in features_enabled else 0.0
    spread_min = payload.get("spectral_spread_min", 0.0) if "spread" in features_enabled else 0.0
    spread_max = payload.get("spectral_spread_max", 0.0) if "spread" in features_enabled else 0.0
    
    # Dynamics metrics (feature-gated)
    entropy_stability = payload.get("spectral_entropy_stability", 0.0) if "dynamics" in features_enabled else 0.0
    entropy_transitions_count = payload.get("spectral_entropy_transitions_count", 0) if "dynamics" in features_enabled else 0
    entropy_transitions_rate = payload.get("spectral_entropy_transitions_rate", 0.0) if "dynamics" in features_enabled else 0.0
    entropy_distribution = payload.get("spectral_entropy_distribution", {}) if "dynamics" in features_enabled else {}
    entropy_diversity = payload.get("spectral_entropy_diversity", 0.0) if "dynamics" in features_enabled else 0.0
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),  # Save payload for render.py compatibility
        spectral_entropy_series=entropy_series if entropy_series is not None and entropy_series.size > 0 else np.zeros((0,), dtype=np.float32),
        spectral_flatness_series=flatness_series if flatness_series is not None and flatness_series.size > 0 else np.zeros((0,), dtype=np.float32),
        spectral_spread_series=spread_series if spread_series is not None and spread_series.size > 0 else np.zeros((0,), dtype=np.float32),
        meta=build_meta(
            producer="spectral_entropy_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "spectral_entropy_contract_version": payload.get("spectral_entropy_contract_version", SPECTRAL_ENTROPY_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "spectral_entropy_variance": entropy_variance,
                "spectral_entropy_min": entropy_min,
                "spectral_entropy_max": entropy_max,
                "spectral_flatness_variance": flatness_variance,
                "spectral_flatness_min": flatness_min,
                "spectral_flatness_max": flatness_max,
                "spectral_spread_variance": spread_variance,
                "spectral_spread_min": spread_min,
                "spectral_spread_max": spread_max,
                "spectral_entropy_stability": entropy_stability,
                "spectral_entropy_transitions_count": entropy_transitions_count,
                "spectral_entropy_transitions_rate": entropy_transitions_rate,
                "spectral_entropy_distribution": entropy_distribution,
                "spectral_entropy_diversity": entropy_diversity,
            },
        ),
    )
    return out_path

