"""
NPZ савер для mfcc_extractor.
Audit v3: canonical keys, omit disabled, segment axis, segment-aligned sequences.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

MFCC_CONTRACT_VERSION = "mfcc_contract_v1"


def save_mfcc_npz(
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
    Сохраняет NPZ артефакт для mfcc_extractor (Audit v3).
    """
    add("sample_rate", payload.get("sample_rate"))
    add("n_mfcc", payload.get("n_mfcc"))
    add("n_fft", payload.get("n_fft"))
    add("hop_length", payload.get("hop_length"))
    add("n_mels", payload.get("n_mels"))
    add("fmin", payload.get("fmin", 0.0))
    add("fmax", payload.get("fmax"))
    add("duration_sec", payload.get("duration"))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))

    features_enabled = payload.get("_features_enabled", [])

    # Model-facing scalars (always)
    add("mfcc_energy", payload.get("mfcc_energy", 0.0))
    add("mfcc_centroid", payload.get("mfcc_centroid", 0.0))
    add("mfcc_bandwidth", payload.get("mfcc_bandwidth", 0.0))
    add("mfcc_stability", payload.get("mfcc_stability", 0.0))

    arrays: Dict[str, np.ndarray] = {}

    # Canonical segment axis (Audit v3)
    arrays["segment_start_sec"] = _arr("segment_start_sec", dtype=np.float32)
    arrays["segment_end_sec"] = _arr("segment_end_sec", dtype=np.float32)
    arrays["segment_center_sec"] = _arr("segment_center_sec", dtype=np.float32)
    arrays["segment_mask"] = _arr("segment_mask", dtype=bool)

    # Basic statistics (omit when disabled)
    if "basic_features" in features_enabled:
        mfcc_stats = payload.get("mfcc_statistics", {})
        if isinstance(mfcc_stats, dict):
            mfcc_mean = mfcc_stats.get("mfcc_mean")
            mfcc_std = mfcc_stats.get("mfcc_std")
            mfcc_min = mfcc_stats.get("mfcc_min")
            mfcc_max = mfcc_stats.get("mfcc_max")
            if mfcc_mean is not None:
                arrays["mfcc_mean"] = np.asarray(mfcc_mean, dtype=np.float32)
            if mfcc_std is not None:
                arrays["mfcc_std"] = np.asarray(mfcc_std, dtype=np.float32)
            if mfcc_min is not None:
                arrays["mfcc_min"] = np.asarray(mfcc_min, dtype=np.float32)
            if mfcc_max is not None:
                arrays["mfcc_max"] = np.asarray(mfcc_max, dtype=np.float32)

        if "deltas" in features_enabled:
            delta_mean = mfcc_stats.get("delta_mean")
            delta_std = mfcc_stats.get("delta_std")
            delta_delta_mean = mfcc_stats.get("delta_delta_mean")
            delta_delta_std = mfcc_stats.get("delta_delta_std")
            if delta_mean is not None:
                arrays["delta_mean"] = np.asarray(delta_mean, dtype=np.float32)
            if delta_std is not None:
                arrays["delta_std"] = np.asarray(delta_std, dtype=np.float32)
            if delta_delta_mean is not None:
                arrays["delta_delta_mean"] = np.asarray(delta_delta_mean, dtype=np.float32)
            if delta_delta_std is not None:
                arrays["delta_delta_std"] = np.asarray(delta_delta_std, dtype=np.float32)

    # Segment-aligned sequences (omit when disabled)
    if "time_series" in features_enabled:
        mm = payload.get("mfcc_mean_by_segment")
        if mm is not None:
            arrays["mfcc_mean_by_segment"] = np.asarray(mm, dtype=np.float32)
        arrays["mfcc_energy_by_segment"] = _arr("mfcc_energy_by_segment", dtype=np.float32)
        dm = payload.get("delta_mean_by_segment")
        if dm is not None and "deltas" in features_enabled:
            arrays["delta_mean_by_segment"] = np.asarray(dm, dtype=np.float32)

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "mfcc_contract_version": payload.get("mfcc_contract_version", MFCC_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        "device_used": payload.get("device_used"),
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "mfcc_resource_profile": payload.get("mfcc_resource_profile"),
    }
    if isinstance(payload.get("mfcc_npy"), str) and payload.get("mfcc_npy"):
        meta_extra["mfcc_npy"] = payload.get("mfcc_npy")

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        **arrays,
        meta=build_meta(
            producer="mfcc_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path
