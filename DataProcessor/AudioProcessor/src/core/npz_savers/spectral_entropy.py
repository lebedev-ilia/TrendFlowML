"""
NPZ saver for spectral_entropy_extractor (Audit v3).

Contract goals:
- no `payload` key (NPZ is source-of-truth; strict per-extractor schema)
- Segmenter-owned axis + per-segment arrays (no concatenated time series)
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from ...utils.cli_utils import atomic_save_npz

SPECTRAL_ENTROPY_CONTRACT_VERSION = "spectral_entropy_contract_v1"


def _opt_arr(payload: Dict[str, Any], key: str, *, dtype: Any) -> Optional[np.ndarray]:
    v = payload.get(key)
    if v is None:
        return None
    return np.asarray(v, dtype=dtype).reshape(-1)


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
    features_enabled = payload.get("_features_enabled", [])
    if not isinstance(features_enabled, list):
        features_enabled = []

    # Model-facing scalars (stable subset)
    add("spectral_entropy_mean", payload.get("spectral_entropy_mean"))
    add("spectral_entropy_std", payload.get("spectral_entropy_std"))

    # Required canonical axis + required per-segment arrays
    seg_start = _arr("segment_start_sec", dtype=np.float32)
    seg_end = _arr("segment_end_sec", dtype=np.float32)
    seg_center = _arr("segment_center_sec", dtype=np.float32)
    seg_mask = np.asarray(payload.get("segment_mask") if payload.get("segment_mask") is not None else [], dtype=bool).reshape(-1)

    entropy_mean_by_segment = _arr("entropy_mean_by_segment", dtype=np.float32)
    entropy_std_by_segment = _arr("entropy_std_by_segment", dtype=np.float32)

    # Optional per-segment arrays
    entropy_min_by_segment = _opt_arr(payload, "entropy_min_by_segment", dtype=np.float32)
    entropy_max_by_segment = _opt_arr(payload, "entropy_max_by_segment", dtype=np.float32)

    flatness_mean_by_segment = _opt_arr(payload, "flatness_mean_by_segment", dtype=np.float32)
    flatness_std_by_segment = _opt_arr(payload, "flatness_std_by_segment", dtype=np.float32)

    spread_mean_by_segment = _opt_arr(payload, "spread_mean_by_segment", dtype=np.float32)
    spread_std_by_segment = _opt_arr(payload, "spread_std_by_segment", dtype=np.float32)

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        segment_start_sec=seg_start,
        segment_end_sec=seg_end,
        segment_center_sec=seg_center,
        segment_mask=seg_mask,
        entropy_mean_by_segment=entropy_mean_by_segment,
        entropy_std_by_segment=entropy_std_by_segment,
        **({"entropy_min_by_segment": entropy_min_by_segment} if entropy_min_by_segment is not None else {}),
        **({"entropy_max_by_segment": entropy_max_by_segment} if entropy_max_by_segment is not None else {}),
        **({"flatness_mean_by_segment": flatness_mean_by_segment} if flatness_mean_by_segment is not None else {}),
        **({"flatness_std_by_segment": flatness_std_by_segment} if flatness_std_by_segment is not None else {}),
        **({"spread_mean_by_segment": spread_mean_by_segment} if spread_mean_by_segment is not None else {}),
        **({"spread_std_by_segment": spread_std_by_segment} if spread_std_by_segment is not None else {}),
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
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "spectral_entropy_resource_profile": payload.get("spectral_entropy_resource_profile"),
                "features_enabled": features_enabled,
                # parameter echo (debug/analytics only)
                "device_used": payload.get("device_used"),
                "sample_rate": payload.get("sample_rate"),
                "n_fft": payload.get("n_fft"),
                "hop_length": payload.get("hop_length"),
                "use_mel": payload.get("use_mel"),
                "n_mels": payload.get("n_mels"),
                "smoothing_window": payload.get("smoothing_window"),
                "average_channels": payload.get("average_channels"),
                "duration": payload.get("duration"),
                "segments_count": payload.get("segments_count"),
            },
        ),
    )
    return out_path

