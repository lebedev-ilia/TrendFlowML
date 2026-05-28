"""
NPZ савер для onset_extractor.
Audit v3: canonical segment axis, omit disabled keys, onset_times only in meta.extra (debug .npy).
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
    Сохраняет NPZ артефакт для onset_extractor (Audit v3).
    onset_times не в NPZ — только в debug .npy, путь в meta.extra.onset_times_npy.
    """
    features_enabled = payload.get("_features_enabled", [])

    # Basic features (omit when disabled)
    if "basic_features" in features_enabled:
        add("onset_count", payload.get("onset_count"))
        add("onset_density_per_sec", payload.get("onset_density_per_sec"))
        add(
            "insufficient_onsets",
            payload.get("insufficient_onsets") if payload.get("insufficient_onsets") is not None else False,
        )

    # Interval stats (omit when disabled)
    if "interval_stats" in features_enabled:
        add("avg_interval_sec", payload.get("avg_interval_sec"))
        add("interval_std", payload.get("interval_std"))
        add("interval_min", payload.get("interval_min"))
        add("interval_max", payload.get("interval_max"))
        add("interval_median", payload.get("interval_median"))

    # Rhythmic metrics (omit when disabled; no onset_clustering_score — removed as redundant)
    if "rhythmic_metrics" in features_enabled:
        add("onset_regularity_score", payload.get("onset_regularity_score"))
        add("onset_tempo_estimate", payload.get("onset_tempo_estimate"))
        add("onset_syncopation_score", payload.get("onset_syncopation_score"))
        add("onset_strength_mean", payload.get("onset_strength_mean"))
        add("onset_strength_std", payload.get("onset_strength_std"))
        add("onset_density_variance", payload.get("onset_density_variance"))
        add("onset_tempo_consistency", payload.get("onset_tempo_consistency"))

    # Canonical segment axis (Audit v3)
    segment_start_sec = _arr("segment_start_sec", dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool)

    # Metadata (always)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("duration", payload.get("duration"))
    add("segments_count", payload.get("segments_count"))

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "onset_contract_version": payload.get("onset_contract_version"),
        "features_enabled": features_enabled,
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "onset_resource_profile": payload.get("onset_resource_profile"),
    }
    b = payload.get("backend")
    if b is not None:
        meta_extra["backend"] = str(b)
    onset_times_npy = payload.get("onset_times_npy")
    if isinstance(onset_times_npy, str) and onset_times_npy:
        meta_extra["onset_times_npy"] = onset_times_npy

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        segment_center_sec=segment_center_sec,
        segment_mask=segment_mask,
        meta=build_meta(
            producer="onset_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path
