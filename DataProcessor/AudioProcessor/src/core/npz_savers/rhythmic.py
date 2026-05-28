"""
NPZ saver for rhythmic_extractor.
Audit v3: strict keys (no payload), canonical segment axis, beat events are token-ready flat arrays,
optional `.npy` sub-artifacts for large beats arrays (paths in meta).
"""

import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

RHYTHMIC_CONTRACT_VERSION = "rhythmic_contract_v1"


def _as_arr(v: Any, *, dtype: Any) -> np.ndarray:
    if v is None:
        return np.asarray([], dtype=dtype).reshape(-1)
    return np.asarray(v, dtype=dtype).reshape(-1)


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
    Save NPZ artifact for rhythmic_extractor (Audit v3).
    """
    features_enabled = payload.get("_features_enabled", [])

    # Frozen model-facing tabular subset (Audit v3):
    # Note: we always publish these names in a fixed order; missing values are NaN.
    add("rhythm_tempo_bpm", payload.get("rhythm_tempo_bpm"))
    add("rhythm_beats_count", payload.get("rhythm_beats_count"))
    add("rhythm_beat_density", payload.get("rhythm_beat_density"))
    add("rhythm_regularity", payload.get("rhythm_regularity"))
    add("rhythm_tempo_variation", payload.get("rhythm_tempo_variation"))
    add("rhythm_beat_consistency", payload.get("rhythm_beat_consistency"))

    # Basic context scalars (still tabular-friendly)
    add("duration_sec", payload.get("duration"))
    add("sample_rate", payload.get("sample_rate"))
    add("segments_count", payload.get("segments_count"))

    # Canonical axis (run_segments). Optional for run() (may be empty).
    segment_start_sec = _arr("segment_start_sec", dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool)

    # Beat events (token-ready). Optional, may be omitted entirely if disabled or saved to .npy.
    beat_times_sec = _as_arr(payload.get("beat_times_sec"), dtype=np.float32)
    beat_segment_index = _as_arr(payload.get("beat_segment_index"), dtype=np.int32)

    # Additional analytics scalars (optional keys)
    extra_fields: Dict[str, Any] = {}
    for k in [
        # interval stats
        "rhythm_avg_period_sec",
        "rhythm_period_std_sec",
        "rhythm_median_period_sec",
        "rhythm_min_period_sec",
        "rhythm_max_period_sec",
        # tempo metrics
        "rhythm_median_bpm",
        "rhythm_ibi_tempo_bpm",
        "rhythm_tempo_mean",
        "rhythm_tempo_std",
        "rhythm_tempo_min",
        "rhythm_tempo_max",
        # regularity extras
        "rhythm_syncopation_score",
        "rhythm_polyrhythm_score",
        "rhythm_beat_strength_mean",
        "rhythm_beat_strength_std",
        "rhythm_metrical_stability",
    ]:
        if k in payload:
            extra_fields[k] = np.asarray(payload.get(k), dtype=np.float32).reshape(())  # scalar

    meta_extra = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "rhythmic_contract_version": payload.get("rhythmic_contract_version", RHYTHMIC_CONTRACT_VERSION),
        "stage_timings_ms": payload.get("stage_timings_ms"),
        "rhythmic_resource_profile": payload.get("rhythmic_resource_profile"),
        "features_enabled": features_enabled,
        "backend": payload.get("backend"),
        "hop_length": payload.get("hop_length"),
        "sampling_family_used": payload.get("sampling_family_used"),
        # Optional `.npy` sub-artifacts (paths)
        "beat_times_sec_npy": payload.get("beat_times_sec_npy"),
        "beat_segment_index_npy": payload.get("beat_segment_index_npy"),
    }

    # If beats were saved to .npy, do not duplicate them into NPZ.
    has_beats_npy = isinstance(meta_extra.get("beat_times_sec_npy"), str) and bool(meta_extra.get("beat_times_sec_npy"))
    if has_beats_npy:
        beat_times_sec = np.asarray([], dtype=np.float32)
        beat_segment_index = np.asarray([], dtype=np.int32)

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        segment_center_sec=segment_center_sec,
        segment_mask=segment_mask,
        **({"beat_times_sec": beat_times_sec} if beat_times_sec.size > 0 else {}),
        **({"beat_segment_index": beat_segment_index} if beat_segment_index.size > 0 else {}),
        **extra_fields,
        meta=build_meta(
            producer="rhythmic_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path

