"""
NPZ савер для speaker_diarization_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_speaker_diarization_npz(
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
    Сохраняет NPZ артефакт для speaker_diarization_extractor.
    """
    # Feature-gated fields
    features_enabled = payload.get("_features_enabled", [])
    speaker_segments = payload.get("speaker_segments") if features_enabled and "speaker_segments" in features_enabled else []
    if speaker_segments is None:
        speaker_segments = []
    if not isinstance(speaker_segments, list):
        speaker_segments = []
    
    speaker_ids = _arr("speaker_ids", dtype=np.int32)
    
    emb = payload.get("speaker_embeddings_mean") if features_enabled and "speaker_embeddings" in features_enabled else None
    if emb is None:
        emb_arr = np.zeros((0, 0), dtype=np.float32)
    else:
        emb_arr = np.asarray(emb, dtype=np.float32)
        if emb_arr.ndim != 2:
            emb_arr = emb_arr.reshape(emb_arr.shape[0], -1) if emb_arr.size else np.zeros((0, 0), dtype=np.float32)
    
    # Segment embeddings (feature-gated)
    segment_embeddings = payload.get("segment_embeddings")
    if segment_embeddings is not None and features_enabled and "segment_embeddings" in features_enabled:
        seg_emb_arr = np.asarray([np.asarray(x, dtype=np.float32) for x in segment_embeddings], dtype=object)
    else:
        seg_emb_arr = np.asarray([], dtype=object)

    add("speaker_count", payload.get("speaker_count"))
    add("duration_sec", payload.get("duration"))
    add("segments_count", payload.get("segments_count"))
    add("sample_rate", payload.get("sample_rate"))
    add("rms", payload.get("rms"))
    add("peak", payload.get("peak"))
    add("model_name", payload.get("model_name"))
    # Aggregates (feature-gated)
    add("speaker_balance_score", payload.get("speaker_balance_score"))
    add("speaker_transitions_count", payload.get("speaker_transitions_count"))
    add("speaker_segments_density", payload.get("speaker_segments_density"))
    add("dominant_speaker_id", payload.get("dominant_speaker_id"))

    # Clustering metrics (feature-gated)
    clustering_metrics = payload.get("clustering_metrics")
    if clustering_metrics is None:
        clustering_metrics = {}
    if not isinstance(clustering_metrics, dict):
        clustering_metrics = {}

    # Speaker time ratios (feature-gated)
    speaker_time_ratios = payload.get("speaker_time_ratios")
    if speaker_time_ratios is None:
        speaker_time_ratios = {}
    if not isinstance(speaker_time_ratios, dict):
        speaker_time_ratios = {}

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        speaker_segments=np.asarray(speaker_segments, dtype=object) if speaker_segments else np.asarray([], dtype=object),
        speaker_ids=speaker_ids,
        speaker_embeddings_mean=emb_arr,
        speaker_stats=np.asarray(payload.get("speaker_stats") or {}, dtype=object) if features_enabled and "speaker_stats" in features_enabled else np.asarray({}, dtype=object),
        segment_embeddings=seg_emb_arr,
        speaker_time_ratios=np.asarray(speaker_time_ratios, dtype=object),
        clustering_metrics=np.asarray(clustering_metrics, dtype=object) if clustering_metrics else np.asarray({}, dtype=object),
        segment_start_sec=_arr("segment_start_sec", dtype=np.float32),
        segment_end_sec=_arr("segment_end_sec", dtype=np.float32),
        segment_center_sec=_arr("segment_center_sec", dtype=np.float32),
        meta=build_meta(
            producer="speaker_diarization_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "diarization_contract_version": payload.get("diarization_contract_version"),
                "features_enabled": features_enabled,
            },
        ),
    )
    return out_path

