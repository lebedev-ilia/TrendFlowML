"""
NPZ савер для clap_extractor.
"""
import os
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
    # Embeddings can be provided directly (preferred) or as a .npy path (legacy).
    emb_present = False
    emb = np.zeros((0,), dtype=np.float32)
    emb_seq = np.zeros((0, 0), dtype=np.float32)
    seg_centers = np.zeros((0,), dtype=np.float32)

    if payload.get("embedding") is not None:
        try:
            emb = np.asarray(payload.get("embedding"), dtype=np.float32).reshape(-1)
            emb_present = emb.size > 0
        except Exception:
            emb_present = False
            emb = np.zeros((0,), dtype=np.float32)
    else:
        emb_path = payload.get("clap_embeddings_npy")
        if isinstance(emb_path, str) and emb_path and os.path.exists(emb_path):
            try:
                emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
                emb_present = emb.size > 0
            except Exception:
                emb_present = False
                emb = np.zeros((0,), dtype=np.float32)

    if payload.get("embedding_sequence") is not None:
        try:
            emb_seq = np.asarray(payload.get("embedding_sequence"), dtype=np.float32)
            if emb_seq.ndim != 2:
                emb_seq = emb_seq.reshape(emb_seq.shape[0], -1) if emb_seq.size else np.zeros((0, 0), dtype=np.float32)
        except Exception:
            emb_seq = np.zeros((0, 0), dtype=np.float32)
    if payload.get("segment_centers_sec") is not None:
        try:
            seg_centers = np.asarray(payload.get("segment_centers_sec"), dtype=np.float32).reshape(-1)
        except Exception:
            seg_centers = np.zeros((0,), dtype=np.float32)

    add("embedding_dim", payload.get("embedding_dim"))
    add("sample_rate", payload.get("sample_rate"))
    add("clap_norm", payload.get("clap_norm"))
    add("clap_magnitude_mean", payload.get("clap_magnitude_mean"))
    add("clap_magnitude_std", payload.get("clap_magnitude_std"))
    add("clap_non_zero_count", payload.get("clap_non_zero_count"))
    add("segments_count", payload.get("segments_count"))

    atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        embedding=emb,
        embedding_present=np.asarray(emb_present, dtype=np.bool_),
        embedding_sequence=emb_seq,
        segment_centers_sec=seg_centers,
        meta=build_meta(
            producer="clap_extractor",
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

