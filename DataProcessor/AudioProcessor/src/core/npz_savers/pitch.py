"""
NPZ савер для pitch_extractor.
Audit v3: canonical segment axis, omit disabled keys, f0_series only in meta.extra (debug .npy).
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_pitch_npz(
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
    Сохраняет NPZ артефакт для pitch_extractor (Audit v3).
    f0_series не в NPZ — только в debug .npy, путь в meta.extra.f0_series_npy.
    """
    features_enabled = payload.get("_features_enabled", [])

    # Metadata (always)
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("frame_length", payload.get("frame_length"))
    add("fmin", payload.get("fmin"))
    add("fmax", payload.get("fmax"))
    add("duration", payload.get("duration"))
    add("segments_count", payload.get("segments_count"))

    # Basic stats (omit when disabled)
    if "basic_stats" in features_enabled:
        add("f0_mean", payload.get("f0_mean"))
        add("f0_std", payload.get("f0_std"))
        add("f0_min", payload.get("f0_min"))
        add("f0_max", payload.get("f0_max"))
        add("f0_median", payload.get("f0_median"))

    # Stability metrics (omit when disabled)
    if "stability_metrics" in features_enabled:
        add("pitch_variation", payload.get("pitch_variation"))
        add("pitch_stability", payload.get("pitch_stability"))
        add("pitch_range", payload.get("pitch_range"))

    # Delta features (omit when disabled)
    if "delta_features" in features_enabled:
        add("f0_delta_mean", payload.get("f0_delta_mean"))
        add("f0_delta_std", payload.get("f0_delta_std"))
        add("f0_delta_abs_mean", payload.get("f0_delta_abs_mean"))

    # Analytics from basic_stats (omit when disabled)
    if "basic_stats" in features_enabled:
        add("pitch_contour_smoothness", payload.get("pitch_contour_smoothness"))
        add("pitch_jump_count", payload.get("pitch_jump_count"))
        add("pitch_skewness", payload.get("pitch_skewness"))
        add("pitch_kurtosis", payload.get("pitch_kurtosis"))

    # Method stats (omit when disabled)
    if "method_stats" in features_enabled:
        for prefix in ["pyin", "yin", "torchcrepe"]:
            for key in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_median", "f0_count"]:
                add(f"{key}_{prefix}", payload.get(f"{key}_{prefix}"))
        add("voiced_fraction_pyin", payload.get("voiced_fraction_pyin"))
        add("voiced_probability_mean_pyin", payload.get("voiced_probability_mean_pyin"))

    # Canonical segment axis (Audit v3)
    segment_start_sec = _arr("segment_start_sec", dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool)

    # pitch_octave_distribution (dict, store as object; omit when disabled)
    pitch_octave_dist = payload.get("pitch_octave_distribution")
    if pitch_octave_dist is not None and isinstance(pitch_octave_dist, dict) and "basic_stats" in features_enabled:
        octave_arr = np.empty((), dtype=object)
        octave_arr[()] = pitch_octave_dist
    else:
        octave_arr = None

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "pitch_contract_version": payload.get("pitch_contract_version"),
        "features_enabled": features_enabled,
        "f0_method": payload.get("f0_method"),
        "backend": payload.get("backend"),
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "pitch_resource_profile": payload.get("pitch_resource_profile"),
    }
    f0_series_npy = payload.get("f0_series_npy")
    if isinstance(f0_series_npy, str) and f0_series_npy:
        meta_extra["f0_series_npy"] = f0_series_npy
    if isinstance(payload.get("f0_series_torchcrepe_npy"), str):
        meta_extra["f0_series_torchcrepe_npy"] = payload.get("f0_series_torchcrepe_npy")

    arrays: Dict[str, np.ndarray] = {
        "segment_start_sec": segment_start_sec,
        "segment_end_sec": segment_end_sec,
        "segment_center_sec": segment_center_sec,
        "segment_mask": segment_mask,
    }
    if octave_arr is not None:
        arrays["pitch_octave_distribution"] = octave_arr

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        **arrays,
        meta=build_meta(
            producer="pitch_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path
