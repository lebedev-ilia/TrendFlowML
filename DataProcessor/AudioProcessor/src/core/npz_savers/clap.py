"""
NPZ савер для clap_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_clap_npz(
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
    Сохраняет NPZ артефакт для clap_extractor.
    """
    # Embeddings are expected to be provided directly in audited contract.
    emb_present = False
    emb = np.zeros((0,), dtype=np.float32)
    emb_seq = np.zeros((0, 0), dtype=np.float32)
    seg_start = np.zeros((0,), dtype=np.float32)
    seg_end = np.zeros((0,), dtype=np.float32)
    seg_centers = np.zeros((0,), dtype=np.float32)
    seg_mask = np.zeros((0,), dtype=np.bool_)
    seg_norm = np.zeros((0,), dtype=np.float32)

    if payload.get("embedding") is not None:
        try:
            emb = np.asarray(payload.get("embedding"), dtype=np.float32).reshape(-1)
            emb_present = emb.size > 0
        except Exception:
            emb_present = False
            emb = np.zeros((0,), dtype=np.float32)
    else:
        # Legacy path support was removed in audited contract.
        emb_present = False
        emb = np.zeros((0,), dtype=np.float32)

    if payload.get("embedding_sequence") is not None:
        try:
            emb_seq = np.asarray(payload.get("embedding_sequence"), dtype=np.float32)
            if emb_seq.ndim != 2:
                emb_seq = emb_seq.reshape(emb_seq.shape[0], -1) if emb_seq.size else np.zeros((0, 0), dtype=np.float32)
        except Exception:
            emb_seq = np.zeros((0, 0), dtype=np.float32)

    if payload.get("segment_start_sec") is not None:
        try:
            seg_start = np.asarray(payload.get("segment_start_sec"), dtype=np.float32).reshape(-1)
        except Exception:
            seg_start = np.zeros((0,), dtype=np.float32)
    if payload.get("segment_end_sec") is not None:
        try:
            seg_end = np.asarray(payload.get("segment_end_sec"), dtype=np.float32).reshape(-1)
        except Exception:
            seg_end = np.zeros((0,), dtype=np.float32)
    if payload.get("segment_center_sec") is not None:
        try:
            seg_centers = np.asarray(payload.get("segment_center_sec"), dtype=np.float32).reshape(-1)
        except Exception:
            seg_centers = np.zeros((0,), dtype=np.float32)
    if payload.get("segment_mask") is not None:
        try:
            seg_mask = np.asarray(payload.get("segment_mask"), dtype=np.bool_).reshape(-1)
        except Exception:
            seg_mask = np.zeros((0,), dtype=np.bool_)
    if payload.get("segment_embedding_norm") is not None:
        try:
            seg_norm = np.asarray(payload.get("segment_embedding_norm"), dtype=np.float32).reshape(-1)
        except Exception:
            seg_norm = np.zeros((0,), dtype=np.float32)

    # Audit v3: minimal model-facing tabular subset (frozen within schema_version).
    add("embedding_dim", payload.get("embedding_dim"))
    add("clap_norm", payload.get("clap_norm"))
    add("clap_magnitude_mean", payload.get("clap_magnitude_mean"))
    add("clap_magnitude_std", payload.get("clap_magnitude_std"))
    add("segments_count", payload.get("segments_count"))

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        embedding=emb,
        embedding_present=np.asarray(emb_present, dtype=np.bool_),
        embedding_sequence=emb_seq,
        segment_start_sec=seg_start,
        segment_end_sec=seg_end,
        segment_center_sec=seg_centers,
        segment_mask=seg_mask,
        segment_embedding_norm=seg_norm,
        meta=build_meta(
            producer="clap_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                # Audit v3: extractor-specific debug meta (not tabular)
                # Audit v4.2: observability
                "stage_timings_ms": payload.get("stage_timings_ms"),
                "clap_resource_profile": payload.get("clap_resource_profile"),
                "sample_rate": payload.get("sample_rate"),
                "device_used": payload.get("device_used"),
                "embedding_dim": payload.get("embedding_dim"),
                "max_audio_length_sec": payload.get("max_audio_length_sec"),
                "trimmed_segments_count": payload.get("trimmed_segments_count"),
                "trimmed_ratio": payload.get("trimmed_ratio"),
            },
        ),
    )
    return out_path

