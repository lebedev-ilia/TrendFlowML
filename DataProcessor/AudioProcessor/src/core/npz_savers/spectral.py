"""
NPZ saver for spectral_extractor (Audit v3).

Contract goals:
- no `payload` key (NPZ is source-of-truth; strict per-extractor schema)
- canonical segment axis + per-segment arrays (no concatenated time series)
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ...utils.cli_utils import atomic_save_npz

SPECTRAL_CONTRACT_VERSION = "spectral_contract_v1"


def _opt_arr(payload: Dict[str, Any], key: str, *, dtype: Any) -> Optional[np.ndarray]:
    v = payload.get(key)
    if v is None:
        return None
    return np.asarray(v, dtype=dtype).reshape(-1)


def save_spectral_npz(
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
    Сохраняет NPZ артефакт для spectral_extractor (Audit v3).
    No payload; flat keys; canonical axis; per-segment arrays.
    """
    features_enabled = payload.get("_features_enabled", [])
    if not isinstance(features_enabled, list):
        features_enabled = []

    # Metadata
    add("sample_rate", payload.get("sample_rate"))
    add("hop_length", payload.get("hop_length"))
    add("n_fft", payload.get("n_fft"))
    add("duration", payload.get("duration"))
    add("segments_count", payload.get("segments_count", 0))

    # Model-facing scalars (from aggregate stats)
    def _s(key: str) -> Dict[str, float]:
        d = payload.get(key) or {}
        return d if isinstance(d, dict) else {}

    if "basic_features" in features_enabled:
        sc = _s("spectral_centroid_stats")
        add("spectral_centroid_mean", sc.get("mean"))
        add("spectral_centroid_std", sc.get("std"))
        add("spectral_centroid_min", sc.get("min"))
        add("spectral_centroid_max", sc.get("max"))
        add("spectral_centroid_median", sc.get("median"))
        sb = _s("spectral_bandwidth_stats")
        add("spectral_bandwidth_mean", sb.get("mean"))
        add("spectral_bandwidth_std", sb.get("std"))
        add("spectral_bandwidth_min", sb.get("min"))
        add("spectral_bandwidth_max", sb.get("max"))
        add("spectral_bandwidth_median", sb.get("median"))
        sf = _s("spectral_flatness_stats")
        add("spectral_flatness_mean", sf.get("mean"))
        add("spectral_flatness_std", sf.get("std"))
        add("spectral_flatness_min", sf.get("min"))
        add("spectral_flatness_max", sf.get("max"))
        add("spectral_flatness_median", sf.get("median"))
        sr = _s("spectral_rolloff_stats")
        add("spectral_rolloff_mean", sr.get("mean"))
        add("spectral_rolloff_std", sr.get("std"))
        add("spectral_rolloff_min", sr.get("min"))
        add("spectral_rolloff_max", sr.get("max"))
        add("spectral_rolloff_median", sr.get("median"))
        zcr = _s("zcr_stats")
        add("zcr_mean", zcr.get("mean"))
        add("zcr_std", zcr.get("std"))
        add("zcr_min", zcr.get("min"))
        add("zcr_max", zcr.get("max"))
        add("zcr_median", zcr.get("median"))
        add("spectral_centroid_median_metric", payload.get("spectral_centroid_median"))
        add("spectral_bandwidth_ratio", payload.get("spectral_bandwidth_ratio"))
        add("spectral_rolloff_ratio", payload.get("spectral_rolloff_ratio"))
        add("spectral_flatness_entropy", payload.get("spectral_flatness_entropy"))

    if "contrast" in features_enabled:
        sc = _s("spectral_contrast_stats")
        add("spectral_contrast_mean", sc.get("mean"))
        add("spectral_contrast_std", sc.get("std"))
        add("spectral_contrast_min", sc.get("min"))
        add("spectral_contrast_max", sc.get("max"))
        add("spectral_contrast_median", sc.get("median"))
        add("spectral_contrast_variance", payload.get("spectral_contrast_variance"))

    if "advanced_features" in features_enabled:
        ss = _s("spectral_slope_stats")
        add("spectral_slope_mean", ss.get("mean"))
        add("spectral_slope_std", ss.get("std"))
        add("spectral_slope_min", ss.get("min"))
        add("spectral_slope_max", ss.get("max"))
        add("spectral_slope_median", ss.get("median"))
        add("spectral_slope_stability", payload.get("spectral_slope_stability"))
        sfd = _s("spectral_flatness_db_stats")
        if sfd:
            add("spectral_flatness_db_mean", sfd.get("mean"))
            add("spectral_flatness_db_std", sfd.get("std"))
            add("spectral_flatness_db_min", sfd.get("min"))
            add("spectral_flatness_db_max", sfd.get("max"))
            add("spectral_flatness_db_median", sfd.get("median"))

    # Canonical axis (required)
    seg_start = _arr("segment_start_sec", dtype=np.float32)
    seg_end = _arr("segment_end_sec", dtype=np.float32)
    seg_center = _arr("segment_center_sec", dtype=np.float32)
    seg_mask_raw = payload.get("segment_mask")
    seg_mask = np.asarray(seg_mask_raw if seg_mask_raw is not None else [], dtype=bool).reshape(-1)

    # Fallback for run() full-audio path (no segments): synthetic single window
    if seg_start.size == 0:
        dur = float(payload.get("duration", 0.0) or 0.0)
        seg_start = np.array([0.0], dtype=np.float32)
        seg_end = np.array([dur], dtype=np.float32)
        seg_center = np.array([0.5 * dur], dtype=np.float32)
        seg_mask = np.array([True], dtype=bool)

    # Per-segment arrays (feature-gated)
    npz_kw: Dict[str, Any] = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "segment_start_sec": seg_start,
        "segment_end_sec": seg_end,
        "segment_center_sec": seg_center,
        "segment_mask": seg_mask,
    }

    if "basic_features" in features_enabled:
        for key in ["centroid_mean_by_segment", "bandwidth_mean_by_segment", "flatness_mean_by_segment", "rolloff_mean_by_segment", "zcr_mean_by_segment"]:
            arr = _opt_arr(payload, key, dtype=np.float32)
            if arr is not None:
                npz_kw[key] = arr

    if "contrast" in features_enabled:
        arr = _opt_arr(payload, "contrast_mean_by_segment", dtype=np.float32)
        if arr is not None:
            npz_kw["contrast_mean_by_segment"] = arr

    if "advanced_features" in features_enabled:
        arr = _opt_arr(payload, "slope_mean_by_segment", dtype=np.float32)
        if arr is not None:
            npz_kw["slope_mean_by_segment"] = arr

    # Contrast bands (analytics, optional)
    if "contrast" in features_enabled and payload.get("keep_contrast_bands", True):
        cb = payload.get("spectral_contrast_bands")
        if cb is not None:
            npz_kw["spectral_contrast_bands"] = np.asarray(cb, dtype=np.float32)

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "spectral_contract_version": payload.get("spectral_contract_version", SPECTRAL_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        # Categorical / debug: not tabular (avoids as_float NaN on strings)
        "device_used": payload.get("device_used"),
        "sample_rate": payload.get("sample_rate"),
        "hop_length": payload.get("hop_length"),
        "n_fft": payload.get("n_fft"),
        "average_channels": payload.get("average_channels", True),
        "keep_contrast_bands": payload.get("keep_contrast_bands", True),
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "spectral_resource_profile": payload.get("spectral_resource_profile"),
    }
    if payload.get("spectral_features_correlation"):
        meta_extra["spectral_features_correlation"] = payload["spectral_features_correlation"]

    npz_kw["meta"] = build_meta(
        producer="spectral_extractor",
        producer_version=producer_version,
        schema_version=schema_version,
        status=status,
        extra=meta_extra,
    )

    atomic_save_npz(out_path, **npz_kw)
    return out_path
