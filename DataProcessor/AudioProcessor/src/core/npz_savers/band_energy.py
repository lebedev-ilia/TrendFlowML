"""
NPZ савер для band_energy_extractor.
"""
import numpy as np
from typing import Any, Dict, Optional, Callable

from ...utils.cli_utils import atomic_save_npz

# Contract version constant
BAND_ENERGY_CONTRACT_VERSION = "band_energy_contract_v1"


def _dominant_band_tabular(v: Any) -> Any:
    """Float index {0,1,2} for tabular; None → None; non-numeric → NaN (defensive vs stray strings)."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip().isdigit():
            return float(int(v.strip(), 10))
        return float(int(v))
    except (TypeError, ValueError):
        return float("nan")


def save_band_energy_npz(
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
    Сохраняет NPZ артефакт для band_energy_extractor.
    """
    features_enabled = payload.get("_features_enabled") or []

    # Canonical bands (Audit v3): fixed 3 bands.
    band_edges = payload.get("band_edges") or []
    band_edges_hz = np.asarray(band_edges, dtype=np.float32).reshape(-1, 2)

    band_shares = payload.get("band_energy_shares") or []
    band_energy_shares = np.asarray(band_shares, dtype=np.float32).reshape(-1)

    # Tabular features (model_facing subset): shares only.
    if band_energy_shares.size == 3:
        add("band_share_low", float(band_energy_shares[0]))
        add("band_share_mid", float(band_energy_shares[1]))
        add("band_share_high", float(band_energy_shares[2]))
    else:
        # Keep vector length stable via NaNs (do not inject 0 placeholders).
        add("band_share_low", np.nan)
        add("band_share_mid", np.nan)
        add("band_share_high", np.nan)

    # Optional analytics scalars (feature-gated)
    if "balance_metrics" in features_enabled:
        add("band_balance_score", payload.get("band_balance_score"))
        add("band_contrast", payload.get("band_contrast"))
        # Tabular: float band index; coerce int/bool/str-digit; unknown types → NaN.
        add("band_dominant_band", _dominant_band_tabular(payload.get("band_dominance")))

    arrays: Dict[str, Any] = {
        "feature_names": np.asarray(feature_names, dtype=object),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
        "band_edges_hz": band_edges_hz.astype(np.float32),
        "band_energy_shares": band_energy_shares.astype(np.float32),
        "meta": build_meta(
            producer="band_energy_extractor",
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
                "band_energy_contract_version": payload.get("band_energy_contract_version", BAND_ENERGY_CONTRACT_VERSION),
                "features_enabled": features_enabled,
                # Observability (Audit v3+): stage timings are produced by the extractor.
                "stage_timings_ms": payload.get("stage_timings_ms", {}),
                # Optional per-extractor profiling (env-gated; best-effort).
                "band_energy_resource_profile": payload.get("band_energy_resource_profile"),
                "method": payload.get("method"),
                "sample_rate": payload.get("sample_rate"),
                "n_fft": payload.get("n_fft"),
                "hop_length": payload.get("hop_length"),
                "duration": payload.get("duration"),
                # Balance metrics (debug)
                **({"band_balance_score": payload.get("band_balance_score")} if "balance_metrics" in features_enabled else {}),
                **({"band_dominance": payload.get("band_dominance")} if "balance_metrics" in features_enabled else {}),
                **({"band_contrast": payload.get("band_contrast")} if "balance_metrics" in features_enabled else {}),
            },
        ),
    }

    # Optional segment-aligned sequence (Audit v3)
    if "time_series" in features_enabled:
        segment_centers_sec = np.asarray(payload.get("segment_centers_sec") or [], dtype=np.float32).reshape(-1)
        segment_durations_sec = np.asarray(payload.get("segment_durations") or [], dtype=np.float32).reshape(-1)
        segment_mask = np.asarray(payload.get("segment_mask") or [], dtype=bool).reshape(-1)
        band_shares_by_segment = np.asarray(payload.get("band_shares_by_segment") or [], dtype=np.float32)
        if band_shares_by_segment.ndim != 2:
            band_shares_by_segment = np.zeros((0, 3), dtype=np.float32)
        arrays.update(
            {
                "segment_centers_sec": segment_centers_sec,
                "segment_durations_sec": segment_durations_sec,
                "segment_mask": segment_mask,
                "band_shares_by_segment": band_shares_by_segment.astype(np.float32),
            }
        )

    atomic_save_npz(out_path, **arrays)
    return out_path

