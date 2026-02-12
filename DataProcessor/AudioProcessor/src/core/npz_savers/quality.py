"""
NPZ савер для quality_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
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
    Сохраняет NPZ артефакт для quality_extractor.
    """
    # Metadata (always saved)
    add("sample_rate", payload.get("sample_rate"))
    add("device_used", payload.get("device_used", "cpu"))
    add("average_channels", payload.get("average_channels", True))
    add("frame_len_ms", payload.get("frame_len_ms", 50.0))
    add("hop_ms", payload.get("hop_ms", 25.0))
    add("clip_threshold", payload.get("clip_threshold", 0.999))
    if "segments_count" in payload:
        add("segments_count", payload.get("segments_count"))
    if "duration" in payload:
        add("duration", payload.get("duration"))
    
    # Feature-gated arrays
    features_enabled = payload.get("_features_enabled", [])
    
    # Basic metrics (feature-gated)
    if "basic_metrics" in features_enabled:
        add("dc_offset", payload.get("dc_offset", 0.0))
        add("dc_offset_abs", payload.get("dc_offset_abs", 0.0))
        add("clipping_ratio", payload.get("clipping_ratio", 0.0))
        add("crest_factor_db", payload.get("crest_factor_db", 0.0))
        add("clipping_segments_count", payload.get("clipping_segments_count", 0))
        
        # Additional ML/analytics metrics
        add("quality_score", payload.get("quality_score", 0.0))
        add("dc_offset_stability", payload.get("dc_offset_stability", 0.0))
        add("clipping_severity", payload.get("clipping_severity", 0.0))
        add("crest_factor_stability", payload.get("crest_factor_stability", 0.0))
    
    # Dynamic metrics (feature-gated)
    if "dynamic_metrics" in features_enabled:
        add("dynamic_range_db", payload.get("dynamic_range_db", 0.0))
        add("snr_db", payload.get("snr_db", 0.0))
        add("dynamic_range_stability", payload.get("dynamic_range_stability", 0.0))
        add("snr_stability", payload.get("snr_stability", 0.0))
    
    # Frame analysis (feature-gated)
    if "frame_analysis" in features_enabled:
        add("frame_levels_mean", payload.get("frame_levels_mean", 0.0))
        add("frame_levels_std", payload.get("frame_levels_std", 0.0))
        add("frame_levels_min", payload.get("frame_levels_min", 0.0))
        add("frame_levels_max", payload.get("frame_levels_max", 0.0))
        add("frame_levels_median", payload.get("frame_levels_median", 0.0))
        
        # Frame levels distribution (if present)
        frame_levels_distribution = payload.get("frame_levels_distribution")
        if frame_levels_distribution is not None:
            if isinstance(frame_levels_distribution, dict):
                add("frame_levels_distribution", frame_levels_distribution)
    
    # Time series (feature-gated)
    dc_offset_series = _arr("dc_offset_series", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    clipping_ratio_series = _arr("clipping_ratio_series", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    crest_factor_db_series = _arr("crest_factor_db_series", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    dynamic_range_db_series = _arr("dynamic_range_db_series", dtype=np.float32) if "time_series" in features_enabled and "dynamic_metrics" in features_enabled else np.zeros((0,), dtype=np.float32)
    snr_db_series = _arr("snr_db_series", dtype=np.float32) if "time_series" in features_enabled and "dynamic_metrics" in features_enabled else np.zeros((0,), dtype=np.float32)
    segment_centers_sec = _arr("segment_centers_sec", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    segment_durations_sec = _arr("segment_durations_sec", dtype=np.float32) if "time_series" in features_enabled else np.zeros((0,), dtype=np.float32)
    
    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        dc_offset_series=dc_offset_series,
        clipping_ratio_series=clipping_ratio_series,
        crest_factor_db_series=crest_factor_db_series,
        dynamic_range_db_series=dynamic_range_db_series,
        snr_db_series=snr_db_series,
        segment_centers_sec=segment_centers_sec,
        segment_durations_sec=segment_durations_sec,
        meta=build_meta(
            producer="quality_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "quality_contract_version": payload.get("quality_contract_version", QUALITY_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
            },
        ),
    )
    return out_path

