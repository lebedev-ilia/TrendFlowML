"""
NPZ савер для rhythmic_extractor.
"""
import os
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_rhythmic_npz(
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
    Сохраняет NPZ артефакт для rhythmic_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("backend", payload.get("backend"))
    add("segments_count", payload.get("segments_count"))
    
    # Feature-gated metrics
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic metrics (feature-gated)
    if "basic_metrics" in features_enabled:
        add("rhythm_tempo_bpm", payload.get("rhythm_tempo_bpm"))
        add("rhythm_beats_count", payload.get("rhythm_beats_count"))
        add("rhythm_beat_density", payload.get("rhythm_beat_density"))
    
    # Interval stats (feature-gated)
    if "interval_stats" in features_enabled:
        add("rhythm_avg_period_sec", payload.get("rhythm_avg_period_sec"))
        add("rhythm_period_std_sec", payload.get("rhythm_period_std_sec"))
        add("rhythm_median_period_sec", payload.get("rhythm_median_period_sec"))
        add("rhythm_min_period_sec", payload.get("rhythm_min_period_sec"))
        add("rhythm_max_period_sec", payload.get("rhythm_max_period_sec"))
    
    # Regularity metrics (feature-gated)
    if "regularity_metrics" in features_enabled:
        add("rhythm_regularity", payload.get("rhythm_regularity"))
        add("rhythm_syncopation_score", payload.get("rhythm_syncopation_score"))
        add("rhythm_polyrhythm_score", payload.get("rhythm_polyrhythm_score"))
        add("rhythm_beat_strength_mean", payload.get("rhythm_beat_strength_mean"))
        add("rhythm_beat_strength_std", payload.get("rhythm_beat_strength_std"))
        add("rhythm_metrical_stability", payload.get("rhythm_metrical_stability"))
    
    # Tempo metrics (feature-gated)
    if "tempo_metrics" in features_enabled:
        add("rhythm_median_bpm", payload.get("rhythm_median_bpm"))
        add("rhythm_tempo_variation", payload.get("rhythm_tempo_variation"))
        add("rhythm_beat_consistency", payload.get("rhythm_beat_consistency"))
        add("rhythm_tempo_mean", payload.get("rhythm_tempo_mean"))
        add("rhythm_tempo_std", payload.get("rhythm_tempo_std"))
        add("rhythm_tempo_min", payload.get("rhythm_tempo_min"))
        add("rhythm_tempo_max", payload.get("rhythm_tempo_max"))
    
    # Beat times (feature-gated)
    beat_times = None
    if "beat_times" in features_enabled:
        beat_times = payload.get("beat_times")
        if beat_times is None:
            beat_times_npy = payload.get("beat_times_npy")
            if isinstance(beat_times_npy, str) and beat_times_npy.startswith("_artifacts/"):
                # Load from .npy file
                npy_path = os.path.join(run_rs_path, "rhythmic_extractor", beat_times_npy)
                if os.path.exists(npy_path):
                    try:
                        beat_times = np.load(npy_path)
                    except Exception:
                        beat_times = np.zeros((0,), dtype=np.float32)
                else:
                    beat_times = np.zeros((0,), dtype=np.float32)
    
    if beat_times is None:
        beat_times = np.zeros((0,), dtype=np.float32)
    else:
        beat_times = np.asarray(beat_times, dtype=np.float32).reshape(-1)
    
    # Per-segment data (for run_segments)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32)
    segment_beat_times = payload.get("segment_beat_times")
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),
        beat_times=beat_times if beat_times.size > 0 else np.zeros((0,), dtype=np.float32),
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        segment_beat_times=segment_beat_times if segment_beat_times else [],
        meta=build_meta(
            producer="rhythmic_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "rhythmic_contract_version": payload.get("rhythmic_contract_version"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

