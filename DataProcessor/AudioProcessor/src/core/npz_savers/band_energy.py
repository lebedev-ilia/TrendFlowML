"""
NPZ савер для band_energy_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
BAND_ENERGY_CONTRACT_VERSION = "band_energy_contract_v1"


def save_band_energy_npz(
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
    Сохраняет NPZ артефакт для band_energy_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("total_energy", payload.get("total_energy"))
    add("method", payload.get("method"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Band edges and energies (always saved)
    band_edges = payload.get("band_edges", [])
    band_energies_arr = _arr("band_energies", dtype=np.float32)
    band_shares_arr = _arr("band_energy_shares", dtype=np.float32)
    
    # Basic stats (feature-gated)
    band_energy_mean_arr = _arr("band_energy_mean", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    band_energy_std_arr = _arr("band_energy_std", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    band_energy_median_arr = _arr("band_energy_median", dtype=np.float32) if "basic_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    
    # Extended stats (feature-gated)
    band_energy_min_arr = _arr("band_energy_min", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    band_energy_max_arr = _arr("band_energy_max", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    band_energy_p25_arr = _arr("band_energy_p25", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    band_energy_p75_arr = _arr("band_energy_p75", dtype=np.float32) if "extended_stats" in features_enabled else np.zeros((len(band_edges) if band_edges else 0,), dtype=np.float32)
    
    # Time series (feature-gated)
    band_energy_ts = None
    if "time_series" in features_enabled:
        band_energy_ts_data = payload.get("band_energy_ts", [])
        if band_energy_ts_data:
            if isinstance(band_energy_ts_data, list):
                band_energy_ts = np.array(band_energy_ts_data, dtype=np.float32)
                if band_energy_ts.ndim == 2:
                    band_energy_ts = band_energy_ts.T  # (num_bands, frames)
    
    if band_energy_ts is None:
        band_energy_ts = np.zeros((len(band_edges) if band_edges else 0, 0), dtype=np.float32)
    
    # Per-segment data (for run_segments)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    segment_durations = _arr("segment_durations", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    
    # Balance metrics (feature-gated)
    band_balance_score = payload.get("band_balance_score", 0.0) if "balance_metrics" in features_enabled else 0.0
    band_dominance = payload.get("band_dominance", 0) if "balance_metrics" in features_enabled else 0
    band_contrast = payload.get("band_contrast", 0.0) if "balance_metrics" in features_enabled else 0.0
    
    # Dynamics metrics (feature-gated)
    band_energy_stability = payload.get("band_energy_stability", 0.0) if "dynamics" in features_enabled else 0.0
    band_transitions = payload.get("band_transitions", []) if "dynamics" in features_enabled else []
    band_transitions_count = payload.get("band_transitions_count", 0) if "dynamics" in features_enabled else 0
    band_transitions_rate = payload.get("band_transitions_rate", 0.0) if "dynamics" in features_enabled else 0.0
    band_distribution = payload.get("band_distribution", {}) if "dynamics" in features_enabled else {}
    band_diversity = payload.get("band_diversity", 0) if "dynamics" in features_enabled else 0
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),  # Save payload for render.py compatibility
        band_energies=band_energies_arr,
        band_energy_shares=band_shares_arr,
        band_energy_mean=band_energy_mean_arr,
        band_energy_std=band_energy_std_arr,
        band_energy_median=band_energy_median_arr,
        band_energy_min=band_energy_min_arr,
        band_energy_max=band_energy_max_arr,
        band_energy_p25=band_energy_p25_arr,
        band_energy_p75=band_energy_p75_arr,
        band_energy_ts=band_energy_ts if band_energy_ts.size > 0 else np.zeros((len(band_edges) if band_edges else 0, 0), dtype=np.float32),
        segment_centers_sec=segment_centers_sec,
        segment_durations=segment_durations,
        meta=build_meta(
            producer="band_energy_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "band_energy_contract_version": payload.get("band_energy_contract_version", BAND_ENERGY_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "band_edges": band_edges,
                "method": payload.get("method"),
                "band_balance_score": band_balance_score,
                "band_dominance": band_dominance,
                "band_contrast": band_contrast,
                "band_energy_stability": band_energy_stability,
                "band_transitions": band_transitions,
                "band_transitions_count": band_transitions_count,
                "band_transitions_rate": band_transitions_rate,
                "band_distribution": band_distribution,
                "band_diversity": band_diversity,
            },
        ),
    )
    return out_path

