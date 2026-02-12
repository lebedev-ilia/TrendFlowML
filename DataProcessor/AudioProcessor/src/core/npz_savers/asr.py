"""
NPZ савер для asr_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_asr_npz(
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
    Сохраняет NPZ артефакт для asr_extractor.
    """
    # Token IDs from a shared tokenizer (no raw text stored).
    token_ids_by_segment = payload.get("token_ids_by_segment")
    if token_ids_by_segment is None:
        token_ids_by_segment = []
    if not isinstance(token_ids_by_segment, list):
        token_ids_by_segment = []
    # Store as object array of int32 vectors (variable lengths).
    tok_obj = np.asarray(
        [np.asarray(x, dtype=np.int32).reshape(-1) for x in token_ids_by_segment],
        dtype=object,
    )
    seg_st = _arr("segment_start_sec", dtype=np.float32)
    seg_en = _arr("segment_end_sec", dtype=np.float32)
    seg_center = _arr("segment_center_sec", dtype=np.float32)
    lang_ids = _arr("lang_id_by_segment", dtype=np.int32)
    token_counts = _arr("token_counts", dtype=np.int32)

    add("segments_count", payload.get("segments_count"))
    add("tokenizer_model", payload.get("tokenizer_model_name"))
    add("whisper_model", payload.get("whisper_model_name"))
    add("sample_rate", payload.get("sample_rate"))
    # Aggregates and statistics (feature-gated)
    add("token_total", payload.get("token_total"))
    add("token_density_per_sec", payload.get("token_density_per_sec"))
    add("speech_rate_wpm", payload.get("speech_rate_wpm"))
    add("segments_with_speech", payload.get("segments_with_speech"))
    add("avg_segment_duration_sec", payload.get("avg_segment_duration_sec"))
    add("token_variance", payload.get("token_variance"))

    # Language distribution (dict -> object array)
    lang_dist = payload.get("lang_distribution")
    if lang_dist is None:
        lang_dist = {}
    if not isinstance(lang_dist, dict):
        lang_dist = {}

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        token_ids_by_segment=tok_obj if token_ids_by_segment else np.asarray([], dtype=object),
        segment_start_sec=seg_st,
        segment_end_sec=seg_en,
        segment_center_sec=seg_center,
        lang_id_by_segment=lang_ids,
        token_counts=token_counts if token_counts.size > 0 else np.asarray([], dtype=np.int32),
        lang_distribution=np.asarray(lang_dist, dtype=object),
        meta=build_meta(
            producer="asr_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "asr_text_contract_version": payload.get("asr_text_contract_version"),
                "features_enabled": payload.get("_features_enabled", []),
            },
        ),
    )
    return out_path

