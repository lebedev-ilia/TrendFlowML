"""
NPZ савер для loudness_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_loudness_npz(
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
    Сохраняет NPZ артефакт для loudness_extractor.
    """
    # LUFS is optional. Store NaN + a flag.
    lufs = payload.get("lufs")
    lufs_present = bool(isinstance(lufs, (int, float)) and np.isfinite(float(lufs)))
    add("loudness_rms", payload.get("rms"))
    add("loudness_peak", payload.get("peak"))
    add("loudness_dbfs", payload.get("dbfs"))
    add("loudness_lufs", lufs if lufs_present else float("nan"))
    add("duration_sec", payload.get("duration"))
    add("sample_rate", payload.get("sample_rate"))
    add("frame_rms_mean", payload.get("frame_rms_mean"))
    add("frame_rms_std", payload.get("frame_rms_std"))
    add("frame_rms_median", payload.get("frame_rms_median"))
    add("frame_rms_p10", payload.get("frame_rms_p10"))
    add("frame_rms_p90", payload.get("frame_rms_p90"))
    add("frames_count", payload.get("frames_count"))
    add("segments_count", payload.get("segments_count"))
    add("segment_rms_mean", payload.get("segment_rms_mean"))
    add("segment_rms_std", payload.get("segment_rms_std"))
    add("segment_rms_median", payload.get("segment_rms_median"))
    add("segment_rms_p10", payload.get("segment_rms_p10"))
    add("segment_rms_p90", payload.get("segment_rms_p90"))

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        lufs_present=np.asarray(lufs_present, dtype=np.bool_),
        segment_centers_sec=_arr("segment_centers_sec", dtype=np.float32),
        segment_rms=_arr("segment_rms", dtype=np.float32),
        segment_peak=_arr("segment_peak", dtype=np.float32),
        segment_dbfs=_arr("segment_dbfs", dtype=np.float32),
        segment_lufs=_arr("segment_lufs", dtype=np.float32),
        meta=build_meta(
            producer="loudness_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
            },
        ),
    )
    return out_path

