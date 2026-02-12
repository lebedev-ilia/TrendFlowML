from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _meta_from_npz(npz: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "meta" not in npz.files:
        return {}
    arr = npz["meta"]
    try:
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.shape == ():
                v = arr.item()
            else:
                first = arr.flat[0]
                v = first.item() if hasattr(first, "item") else first
            return dict(v) if isinstance(v, dict) else {}
    except Exception:
        return {}
    return {}


def _summarize_numeric_array(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if arr.size == 0:
        out[f"{prefix}__empty"] = 1.0
        return out

    x = arr
    if x.dtype == bool:
        out[f"{prefix}__mean"] = float(np.mean(x.astype(np.float32)))
        return out

    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32, copy=False).reshape(-1)
        xf = xf[np.isfinite(xf)]
        if xf.size == 0:
            out[f"{prefix}__all_non_finite"] = 1.0
            return out
        out[f"{prefix}__mean"] = float(np.mean(xf))
        out[f"{prefix}__std"] = float(np.std(xf))
        out[f"{prefix}__min"] = float(np.min(xf))
        out[f"{prefix}__max"] = float(np.max(xf))
        out[f"{prefix}__p50"] = float(np.percentile(xf, 50))
        out[f"{prefix}__p90"] = float(np.percentile(xf, 90))
        out[f"{prefix}__count"] = float(xf.size)
    return out


def _flatten_obj_dict(prefix: str, d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            out[key] = _safe_float(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    continue
                out.update(_summarize_numeric_array(key, arr))
            except Exception:
                continue
        # ignore strings / nested dicts
    return out


def extract_features_from_npz(component: str, path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Extract tabular-friendly features from a component NPZ artifact.

    Preference order:
      1) feature_names + feature_values
      2) features dict
      3) summarize numeric arrays/scalars
    """
    feats: Dict[str, float] = {}
    meta: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as npz:
        meta = _meta_from_npz(npz)

        if "feature_names" in npz.files and "feature_values" in npz.files:
            try:
                names = npz["feature_names"]
                vals = npz["feature_values"]
                names_list = [str(x) for x in (names.tolist() if isinstance(names, np.ndarray) else list(names))]
                vals_arr = np.asarray(vals, dtype=np.float32).reshape(-1)
                for n, v in zip(names_list, vals_arr):
                    feats[f"{component}__{n}"] = float(v)
                return feats, meta
            except Exception:
                pass

        if "features" in npz.files:
            try:
                f_arr = npz["features"]
                if isinstance(f_arr, np.ndarray) and f_arr.dtype == object:
                    f_dict = f_arr.item() if f_arr.shape == () else f_arr.flat[0].item()
                    if isinstance(f_dict, dict):
                        feats.update(_flatten_obj_dict(component, f_dict))
            except Exception:
                pass

        for k in npz.files:
            if k in ("meta", "features", "feature_names", "feature_values"):
                continue
            arr = npz[k]
            if k in ("embeddings", "embedding", "ocr_raw", "ocr_unique_elements"):
                continue
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    continue
                if arr.ndim == 0:
                    feats[f"{component}__{k}"] = _safe_float(arr.item())
                else:
                    feats.update(_summarize_numeric_array(f"{component}__{k}", arr))
    return feats, meta


def find_first_npz_artifact(component_entry: Dict[str, Any]) -> str | None:
    arts = component_entry.get("artifacts") or []
    if not isinstance(arts, list):
        return None
    for a in arts:
        if isinstance(a, dict) and isinstance(a.get("path"), str):
            p = str(a["path"])
            if p.lower().endswith(".npz") and os.path.exists(p):
                return p
    return None


