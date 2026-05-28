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
    # Store as 1D object array of int32 vectors (variable lengths).
    # IMPORTANT: avoid np.asarray(list_of_lists, dtype=object) -> 2D object array.
    tok_obj = np.empty((len(token_ids_by_segment),), dtype=object)
    for i, x in enumerate(token_ids_by_segment):
        tok_obj[i] = np.asarray(x, dtype=np.int32).reshape(-1)
    seg_st = _arr("segment_start_sec", dtype=np.float32)
    seg_en = _arr("segment_end_sec", dtype=np.float32)
    seg_center = _arr("segment_center_sec", dtype=np.float32)
    lang_ids = _arr("lang_id_by_segment", dtype=np.int32)
    # Preferred language info (analytics)
    lang_codes = payload.get("lang_code_by_segment")
    if lang_codes is None:
        lang_codes = []
    if not isinstance(lang_codes, list):
        lang_codes = []
    lang_codes_obj = np.asarray([str(x or "") for x in lang_codes], dtype=object)
    lang_conf = _arr("lang_conf_by_segment", dtype=np.float32)
    token_counts = _arr("token_counts", dtype=np.int32)
    # Per-segment quality metrics (privacy-safe numeric signals); stored as object array of dicts.
    seg_quality = payload.get("segment_quality_by_segment")
    if seg_quality is None:
        seg_quality = []
    if not isinstance(seg_quality, list):
        seg_quality = []
    seg_quality_obj = np.asarray([x if isinstance(x, dict) else {} for x in seg_quality], dtype=object)

    # Schema v2: segmenter-owned audio duration + sampling params (best-effort; required keys filled with NaN/""/-1)
    audio_duration_sec = payload.get("audio_duration_sec")
    asr_sampling_profile = payload.get("asr_sampling_profile")
    asr_window_sec = payload.get("asr_window_sec")
    asr_stride_sec = payload.get("asr_stride_sec")
    asr_max_windows = payload.get("asr_max_windows")

    def _f32_scalar(v: Any) -> np.ndarray:
        try:
            return np.asarray(float(v), dtype=np.float32)
        except Exception:
            return np.asarray(float("nan"), dtype=np.float32)

    def _i32_scalar(v: Any) -> np.ndarray:
        try:
            return np.asarray(int(v), dtype=np.int32)
        except Exception:
            return np.asarray(int(-1), dtype=np.int32)

    def _obj_scalar(v: Any) -> np.ndarray:
        return np.asarray(str(v or ""), dtype=object)

    add("segments_count", payload.get("segments_count"))
    add("sample_rate", payload.get("sample_rate"))
    # Aggregates and statistics (feature-gated)
    add("token_total", payload.get("token_total"))
    add("token_density_per_sec", payload.get("token_density_per_sec"))
    add("speech_rate_wpm", payload.get("speech_rate_wpm"))
    add("segments_with_speech", payload.get("segments_with_speech"))
    add("avg_segment_duration_sec", payload.get("avg_segment_duration_sec"))
    add("token_variance", payload.get("token_variance"))

    # Quality aggregates (analytics-only intent, but stored in tabular for baseline/UI).
    # Keep naming stable and prefixed.
    if isinstance(seg_quality, list) and seg_quality:
        def _vals(key: str) -> np.ndarray:
            out = []
            for d in seg_quality:
                if not isinstance(d, dict):
                    continue
                v = d.get(key)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    out.append(float(v))
            return np.asarray(out, dtype=np.float32)

        for k in ("avg_logprob", "compression_ratio", "no_speech_prob"):
            arr = _vals(k)
            if arr.size:
                add(f"asr_quality__{k}_mean", float(np.mean(arr)))
                add(f"asr_quality__{k}_p50", float(np.percentile(arr, 50)))
                add(f"asr_quality__{k}_p90", float(np.percentile(arr, 90)))
                add(f"asr_quality__{k}_present_rate", float(arr.size / max(1, len(seg_quality))))
            else:
                add(f"asr_quality__{k}_mean", float("nan"))
                add(f"asr_quality__{k}_p50", float("nan"))
                add(f"asr_quality__{k}_p90", float("nan"))
                add(f"asr_quality__{k}_present_rate", 0.0)

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
        # v2: lang_id is legacy/best-effort -> optional; keep writing if present.
        lang_id_by_segment=lang_ids if lang_ids.size > 0 else np.asarray([], dtype=np.int32),
        lang_code_by_segment=lang_codes_obj if lang_codes else np.asarray([], dtype=object),
        lang_conf_by_segment=lang_conf,
        token_counts=token_counts if token_counts.size > 0 else np.asarray([], dtype=np.int32),
        lang_distribution=np.asarray(lang_dist, dtype=object),
        segment_quality_by_segment=seg_quality_obj if seg_quality else np.asarray([], dtype=object),
        audio_duration_sec=_f32_scalar(audio_duration_sec),
        asr_sampling_profile=_obj_scalar(asr_sampling_profile),
        asr_window_sec=_f32_scalar(asr_window_sec),
        asr_stride_sec=_f32_scalar(asr_stride_sec),
        asr_max_windows=_i32_scalar(asr_max_windows),
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
                "asr_stage_timings_ms": payload.get("asr_stage_timings_ms") or {},
                "asr_resource_profile": payload.get("asr_resource_profile") or {},
            },
        ),
    )
    return out_path

