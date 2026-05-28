"""
NPZ савер для source_separation_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_source_separation_npz(
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
    Сохраняет NPZ артефакт для source_separation_extractor.
    """
    # Audit v3: optional keys must be absent when disabled (no placeholders).
    features_enabled = payload.get("_features_enabled", [])

    # Canonical order is frozen in contract.
    src_order = ["vocals", "drums", "bass", "other"]

    def _vec4(key: str, *, dtype: Any, fill: Any) -> np.ndarray:
        v = payload.get(key)
        if v is None:
            return np.asarray(fill, dtype=dtype).reshape(4)
        a = np.asarray(v, dtype=dtype).reshape(-1)
        if a.size != 4:
            return np.asarray(fill, dtype=dtype).reshape(4)
        return a.reshape(4)

    def _seg_arr(key: str, *, dtype: Any) -> np.ndarray:
        v = payload.get(key)
        if v is None:
            return np.asarray([], dtype=dtype).reshape(-1)
        return np.asarray(v, dtype=dtype).reshape(-1)

    share_mean = _arr("share_mean", dtype=np.float32)
    if share_mean.size != 4:
        # In audited contract this key is expected; if missing, encode as NaN vector.
        share_mean = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)

    # Frozen model-facing tabular subset (order fixed within schema_version)
    add("share_vocals_mean", float(share_mean[0]) if share_mean.size >= 1 else None)
    add("share_drums_mean", float(share_mean[1]) if share_mean.size >= 2 else None)
    add("share_bass_mean", float(share_mean[2]) if share_mean.size >= 3 else None)
    add("share_other_mean", float(share_mean[3]) if share_mean.size >= 4 else None)
    add("dominant_source_id", payload.get("dominant_source_id"))
    add("dominant_source_share", payload.get("dominant_source_share"))
    add("source_balance_score", payload.get("source_balance_score"))
    add("source_transitions_count", payload.get("source_transitions_count"))
    add("source_stability_score", payload.get("source_stability_score"))
    add("segments_count", payload.get("segments_count"))
    add("sample_rate", payload.get("sample_rate"))

    # Feature-gated arrays
    share_seq_arr = None
    if isinstance(payload.get("share_sequence"), np.ndarray):
        share_seq_arr = payload.get("share_sequence")
    elif "share_sequence" in features_enabled and payload.get("share_sequence") is not None:
        share_seq_arr = np.asarray(payload.get("share_sequence"), dtype=np.float32)
    if share_seq_arr is not None:
        share_seq_arr = np.asarray(share_seq_arr, dtype=np.float32)

    energy_seq_arr = None
    if isinstance(payload.get("energy_sequence"), np.ndarray):
        energy_seq_arr = payload.get("energy_sequence")
    elif "energy_sequence" in features_enabled and payload.get("energy_sequence") is not None:
        energy_seq_arr = np.asarray(payload.get("energy_sequence"), dtype=np.float32)
    if energy_seq_arr is not None:
        energy_seq_arr = np.asarray(energy_seq_arr, dtype=np.float32)

    share_std = None
    if "share_std" in features_enabled and payload.get("share_std") is not None:
        share_std = _arr("share_std", dtype=np.float32)
        if share_std.size != 4:
            share_std = None

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        # Canonical axis + mask (always present for run_segments in audited contract)
        segment_start_sec=_seg_arr("segment_start_sec", dtype=np.float32),
        segment_end_sec=_seg_arr("segment_end_sec", dtype=np.float32),
        segment_center_sec=_seg_arr("segment_center_sec", dtype=np.float32),
        segment_mask=_seg_arr("segment_mask", dtype=bool),
        # Small vectors
        share_mean=share_mean.astype(np.float32).reshape(4),
        **({"share_std": share_std.astype(np.float32).reshape(4)} if share_std is not None else {}),
        # Structured per-source stats (no dicts)
        source_distribution_ratio=_vec4("source_distribution_ratio", dtype=np.float32, fill=[0.0, 0.0, 0.0, 0.0]),
        source_segments_count=_vec4("source_segments_count", dtype=np.int32, fill=[0, 0, 0, 0]),
        source_duration_sec=_vec4("source_duration_sec", dtype=np.float32, fill=[0.0, 0.0, 0.0, 0.0]),
        source_order=np.asarray(src_order, dtype=object),
        # Optional sequences
        **({"share_sequence": share_seq_arr.astype(np.float32)} if share_seq_arr is not None else {}),
        **({"energy_sequence": energy_seq_arr.astype(np.float32)} if energy_seq_arr is not None else {}),
        # Optional analytics scalars (only if present in payload)
        **{
            k: np.asarray(payload.get(k), dtype=np.float32).reshape(())
            for k in [
                # advanced features
                "source_entropy_mean",
                "source_entropy_std",
                "energy_balance_mean",
                "vocals_presence_ratio",
                "drums_flux",
                "bass_floor_p20",
                "vocals_delta_mean",
                "vocals_delta_std",
                "vocals_delta_max",
                "drums_delta_mean",
                "drums_delta_std",
                "drums_delta_max",
                "bass_delta_mean",
                "bass_delta_std",
                "bass_delta_max",
                "other_delta_mean",
                "other_delta_std",
                "other_delta_max",
                "vocals_stability",
                "drums_stability",
                "bass_stability",
                "other_stability",
                "vocals_mean_share",
                "drums_mean_share",
                "bass_mean_share",
                "other_mean_share",
                "vocals_dominance_ratio",
                "drums_dominance_ratio",
                "bass_dominance_ratio",
                "other_dominance_ratio",
                # quality metrics (scalar keys)
                "quality_share_mean_min",
                "quality_share_mean_max",
                "quality_share_mean_std",
                "quality_share_std_mean",
                "quality_share_std_max",
                "quality_share_sequence_min",
                "quality_share_sequence_max",
                "quality_share_sequence_mean",
                "quality_energy_sequence_min",
                "quality_energy_sequence_max",
                "quality_energy_sequence_mean",
            ]
            if (k in payload and payload.get(k) is not None)
        },
        meta=build_meta(
            producer="source_separation_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "source_separation_resource_profile": payload.get("source_separation_resource_profile"),
                "source_separation_contract_version": payload.get("source_separation_contract_version"),
                "features_enabled": features_enabled,
                "model_name": payload.get("model_name"),
                "weights_digest": payload.get("weights_digest"),
            },
        ),
    )
    return out_path

