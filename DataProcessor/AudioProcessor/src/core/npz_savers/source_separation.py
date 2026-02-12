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
    # Feature-gated fields
    features_enabled = payload.get("_features_enabled", [])
    share_seq = payload.get("share_sequence") if features_enabled and "share_sequence" in features_enabled else None
    if share_seq is None:
        share_seq_arr = np.zeros((0, 4), dtype=np.float32)
    else:
        share_seq_arr = np.asarray(share_seq, dtype=np.float32)
        if share_seq_arr.ndim != 2:
            share_seq_arr = share_seq_arr.reshape(share_seq_arr.shape[0], -1) if share_seq_arr.size else np.zeros((0, 4), dtype=np.float32)
    
    energy_seq = payload.get("energy_sequence") if features_enabled and "energy_sequence" in features_enabled else None
    if energy_seq is None:
        energy_seq_arr = np.zeros((0, 4), dtype=np.float32)
    else:
        energy_seq_arr = np.asarray(energy_seq, dtype=np.float32)
        if energy_seq_arr.ndim != 2:
            energy_seq_arr = energy_seq_arr.reshape(energy_seq_arr.shape[0], -1) if energy_seq_arr.size else np.zeros((0, 4), dtype=np.float32)

    share_mean = _arr("share_mean", dtype=np.float32) if features_enabled and "share_mean" in features_enabled else np.zeros((4,), dtype=np.float32)
    share_std = _arr("share_std", dtype=np.float32) if features_enabled and "share_std" in features_enabled else np.zeros((4,), dtype=np.float32)
    src_order = payload.get("source_order") or ["vocals", "drums", "bass", "other"]
    if not isinstance(src_order, list):
        src_order = ["vocals", "drums", "bass", "other"]

    # Flatten mean shares into feature vector for compatibility
    if share_mean.size >= 4:
        add("share_vocals_mean", float(share_mean[0]))
        add("share_drums_mean", float(share_mean[1]))
        add("share_bass_mean", float(share_mean[2]))
        add("share_other_mean", float(share_mean[3]))
    add("segments_count", payload.get("segments_count"))
    add("sample_rate", payload.get("sample_rate"))
    add("model_name", payload.get("model_name"))
    # Aggregates (feature-gated)
    add("dominant_source_id", payload.get("dominant_source_id"))
    add("dominant_source_share", payload.get("dominant_source_share"))
    add("source_balance_score", payload.get("source_balance_score"))
    add("source_transitions_count", payload.get("source_transitions_count"))
    add("source_stability_score", payload.get("source_stability_score"))
    
    # Advanced features (automatically computed if share_sequence is enabled)
    # Transition features
    for source_name in src_order:
        add(f"{source_name}_delta_mean", payload.get(f"{source_name}_delta_mean"))
        add(f"{source_name}_delta_std", payload.get(f"{source_name}_delta_std"))
        add(f"{source_name}_delta_max", payload.get(f"{source_name}_delta_max"))
    
    # Stability features
    for source_name in src_order:
        add(f"{source_name}_stability", payload.get(f"{source_name}_stability"))
    
    # Distribution features
    for source_name in src_order:
        add(f"{source_name}_mean_share", payload.get(f"{source_name}_mean_share"))
        add(f"{source_name}_dominance_ratio", payload.get(f"{source_name}_dominance_ratio"))
    
    # Energy balance
    add("source_entropy_mean", payload.get("source_entropy_mean"))
    add("source_entropy_std", payload.get("source_entropy_std"))
    add("energy_balance_mean", payload.get("energy_balance_mean"))
    
    # Musical heuristics
    add("vocals_presence_ratio", payload.get("vocals_presence_ratio"))
    add("drums_flux", payload.get("drums_flux"))
    add("bass_floor_p20", payload.get("bass_floor_p20"))

    # Source distribution (feature-gated)
    source_distribution = payload.get("source_distribution")
    if source_distribution is None:
        source_distribution = {}
    if not isinstance(source_distribution, dict):
        source_distribution = {}
    
    source_segments_per_source = payload.get("source_segments_per_source")
    if source_segments_per_source is None:
        source_segments_per_source = {}
    if not isinstance(source_segments_per_source, dict):
        source_segments_per_source = {}
    
    source_duration_per_source = payload.get("source_duration_per_source")
    if source_duration_per_source is None:
        source_duration_per_source = {}
    if not isinstance(source_duration_per_source, dict):
        source_duration_per_source = {}

    # Quality metrics (feature-gated)
    quality_metrics = payload.get("source_quality_metrics")
    if quality_metrics is None:
        quality_metrics = {}
    if not isinstance(quality_metrics, dict):
        quality_metrics = {}

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        share_sequence=share_seq_arr if share_seq_arr.size > 0 else np.zeros((0, 4), dtype=np.float32),
        energy_sequence=energy_seq_arr if energy_seq_arr.size > 0 else np.zeros((0, 4), dtype=np.float32),
        share_mean=share_mean,
        share_std=share_std,
        source_order=np.asarray(src_order, dtype=object),
        source_distribution=np.asarray(source_distribution, dtype=object),
        source_segments_per_source=np.asarray(source_segments_per_source, dtype=object),
        source_duration_per_source=np.asarray(source_duration_per_source, dtype=object),
        source_quality_metrics=np.asarray(quality_metrics, dtype=object) if quality_metrics else np.asarray({}, dtype=object),
        segment_start_sec=_arr("segment_start_sec", dtype=np.float32),
        segment_end_sec=_arr("segment_end_sec", dtype=np.float32),
        segment_center_sec=_arr("segment_center_sec", dtype=np.float32),
        meta=build_meta(
            producer="source_separation_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "source_separation_contract_version": payload.get("source_separation_contract_version"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

