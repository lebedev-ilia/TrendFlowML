"""
NPZ савер для chroma_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz


def save_chroma_npz(
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
    Сохраняет NPZ артефакт для chroma_extractor.
    """
    features_enabled = payload.get("_features_enabled") or []

    # Canonical arrays (Audit v3)
    chroma_mean = np.asarray(payload.get("chroma_mean"), dtype=np.float32).reshape(-1)
    chroma_entropy = np.asarray(payload.get("chroma_entropy"), dtype=np.float32).reshape(())
    chroma_harmonic_stability = np.asarray(payload.get("chroma_harmonic_stability"), dtype=np.float32).reshape(())
    chroma_contrast = np.asarray(payload.get("chroma_contrast"), dtype=np.float32).reshape(())
    _dc = payload.get("chroma_dominant_class")
    chroma_dominant_class = np.asarray(int(_dc) if _dc is not None else -1, dtype=np.int32).reshape(())
    chroma_dominant_energy = np.asarray(payload.get("chroma_dominant_energy"), dtype=np.float32).reshape(())
    tuning_estimate = np.asarray(payload.get("tuning_estimate"), dtype=np.float32).reshape(())

    # Tabular model_facing: stable names
    chroma_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, n in enumerate(chroma_names):
        add(f"chroma_mean_{n}", float(chroma_mean[i]) if i < chroma_mean.size else np.nan)
    add("chroma_entropy", float(chroma_entropy))
    add("chroma_harmonic_stability", float(chroma_harmonic_stability))
    add("chroma_contrast", float(chroma_contrast))
    add("chroma_dominant_energy", float(chroma_dominant_energy))

    arrays: Dict[str, Any] = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "chroma_mean": chroma_mean.astype(np.float32),
        "chroma_entropy": chroma_entropy.astype(np.float32),
        "chroma_harmonic_stability": chroma_harmonic_stability.astype(np.float32),
        "chroma_contrast": chroma_contrast.astype(np.float32),
        "chroma_dominant_class": chroma_dominant_class.astype(np.int32),
        "chroma_dominant_energy": chroma_dominant_energy.astype(np.float32),
        "tuning_estimate": tuning_estimate.astype(np.float32),
        "meta": build_meta(
            producer="chroma_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "chroma_contract_version": payload.get("chroma_contract_version"),
                "features_enabled": features_enabled,
                # Observability (Audit v3+): stage timings are produced by the extractor.
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
                # Optional per-extractor profiling (env-gated; best-effort).
                "chroma_resource_profile": payload.get("chroma_resource_profile"),
                "device_used": payload.get("device_used"),
                "chroma_type": payload.get("chroma_type"),
                "normalize": payload.get("normalize"),
                "tuning_failed": bool(payload.get("tuning_failed", False)),
                "chroma_time_series_omitted": bool(payload.get("chroma_time_series_omitted", False)),
                "sample_rate": payload.get("sample_rate"),
                "hop_length": payload.get("hop_length"),
                "n_fft": payload.get("n_fft"),
                "duration_sec": payload.get("duration"),
                "segments_count": payload.get("segments_count"),
            },
        ),
    }

    # Optional debug chroma time series (only if present)
    if "time_series" in features_enabled and isinstance(payload.get("chroma"), np.ndarray):
        arrays["chroma"] = np.asarray(payload.get("chroma"), dtype=np.float32)

    # Optional segment-level sequence
    if "time_series" in features_enabled:
        if payload.get("segment_centers_sec") is not None:
            arrays["segment_centers_sec"] = np.asarray(payload.get("segment_centers_sec"), dtype=np.float32).reshape(-1)
        if payload.get("segment_durations_sec") is not None:
            arrays["segment_durations_sec"] = np.asarray(payload.get("segment_durations_sec"), dtype=np.float32).reshape(-1)
        if payload.get("segment_mask") is not None:
            arrays["segment_mask"] = np.asarray(payload.get("segment_mask"), dtype=bool).reshape(-1)
        if payload.get("chroma_mean_by_segment") is not None:
            arrays["chroma_mean_by_segment"] = np.asarray(payload.get("chroma_mean_by_segment"), dtype=np.float32)

    atomic_save_npz(out_path, **arrays)
    return out_path

