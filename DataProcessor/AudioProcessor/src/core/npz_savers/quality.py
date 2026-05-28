"""
NPZ савер для quality_extractor.
Audit v3: canonical segment axis, omit disabled keys, time series only in meta.extra (.npy paths).
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

QUALITY_CONTRACT_VERSION = "quality_contract_v1"


def save_quality_npz(
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
    Сохраняет NPZ артефакт для quality_extractor (Audit v3).
    Time series не в NPZ — только в .npy, пути в meta.extra.
    """
    features_enabled = payload.get("_features_enabled", [])

    add("sample_rate", payload.get("sample_rate"))
    add("average_channels", payload.get("average_channels", True))
    add("frame_len_ms", payload.get("frame_len_ms", 50.0))
    add("hop_ms", payload.get("hop_ms", 25.0))
    add("clip_threshold", payload.get("clip_threshold", 0.999))
    add("duration", payload.get("duration"))
    add("segments_count", payload.get("segments_count", 0))

    if "basic_metrics" in features_enabled:
        add("dc_offset", payload.get("dc_offset", 0.0))
        add("clipping_ratio", payload.get("clipping_ratio", 0.0))
        add("crest_factor_db", payload.get("crest_factor_db", 0.0))
        add("clipping_segments_count", payload.get("clipping_segments_count", 0))
        add("quality_score", payload.get("quality_score", 0.0))
        add("crest_factor_median", payload.get("crest_factor_median", 0.0))

    if "dynamic_metrics" in features_enabled:
        add("dynamic_range_db", payload.get("dynamic_range_db", 0.0))
        add("dynamic_range_stability", payload.get("dynamic_range_stability", 0.0))

    if "frame_analysis" in features_enabled:
        dist = payload.get("frame_levels_distribution")
        if isinstance(dist, dict):
            add("frame_levels_mean", dist.get("mean", 0.0))
            add("frame_levels_std", dist.get("std", 0.0))
            add("frame_levels_min", dist.get("min", 0.0))
            add("frame_levels_max", dist.get("max", 0.0))
            add("frame_levels_median", dist.get("median", 0.0))

    segment_start_sec = _arr("segment_start_sec", dtype=np.float32)
    segment_end_sec = _arr("segment_end_sec", dtype=np.float32)
    segment_center_sec = _arr("segment_center_sec", dtype=np.float32)
    segment_mask = _arr("segment_mask", dtype=bool)

    meta_extra: Dict[str, Any] = {
        **(extra_meta or {}),
        **({"error": error} if error else {}),
        "empty_reason": empty_reason,
        "quality_contract_version": payload.get("quality_contract_version", QUALITY_CONTRACT_VERSION),
        "features_enabled": features_enabled,
        "device_used": payload.get("device_used"),
        "stage_timings_ms": payload.get("stage_timings_ms", {}),
        "quality_resource_profile": payload.get("quality_resource_profile"),
    }
    for key in ["dc_offset_series_npy", "clipping_ratio_series_npy", "crest_factor_db_series_npy", "dynamic_range_db_series_npy", "frame_levels_db_series_npy", "frame_rms_series_npy", "clipping_segments_series_npy"]:
        path = payload.get(key)
        if isinstance(path, str) and path:
            meta_extra[key] = path

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        segment_start_sec=segment_start_sec,
        segment_end_sec=segment_end_sec,
        segment_center_sec=segment_center_sec,
        segment_mask=segment_mask,
        meta=build_meta(
            producer="quality_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra=meta_extra,
        ),
    )
    return out_path
