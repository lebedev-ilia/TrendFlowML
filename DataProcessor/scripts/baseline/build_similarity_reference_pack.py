#!/usr/bin/env python3
"""
Build dp_models reference pack for similarity_metrics (wide baseline v1).

Writes to:
  DataProcessor/dp_models/bundled_models/similarity/reference_sets/<reference_set_id>/

Input:
  A list of per-run result_store directories:
    result_store/<platform_id>/<video_id>/<run_id>/

This script is deterministic given the same inputs and artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, dict):
        raise RuntimeError(f"invalid json dict: {path}")
    return x


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def _unbox_dict(x: Any) -> Optional[Dict[str, Any]]:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            x = x.item()
        except Exception:
            return None
    return x if isinstance(x, dict) else None


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + 1e-9)
    return (v / n).astype(np.float32)


def _core_clip_centroid(run_dir: str) -> np.ndarray:
    p = os.path.join(run_dir, "core_clip", "embeddings.npz")
    d = _load_npz(p)
    fe = d.get("frame_embeddings")
    if fe is None:
        raise RuntimeError(f"core_clip missing frame_embeddings: {p}")
    emb = np.asarray(fe, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise RuntimeError(f"core_clip invalid frame_embeddings shape: {emb.shape}")
    emb_n = _normalize_rows(emb)
    cen = np.mean(emb_n, axis=0)
    return _normalize_vec(cen)


def _clap_embedding(run_dir: str) -> np.ndarray:
    p = os.path.join(run_dir, "clap_extractor", "clap_extractor_features.npz")
    d = _load_npz(p)
    e = d.get("embedding")
    if e is None:
        raise RuntimeError(f"clap_extractor missing embedding: {p}")
    return _normalize_vec(np.asarray(e, dtype=np.float32))


def _text_primary_embedding(run_dir: str) -> Tuple[Optional[np.ndarray], bool]:
    p = os.path.join(run_dir, "text_processor", "text_features.npz")
    d = _load_npz(p)
    present = bool(np.asarray(d.get("primary_embedding_present") or False).item())
    if not present:
        return None, False
    e = d.get("primary_embedding")
    if e is None:
        return None, False
    v = np.asarray(e, dtype=np.float32).reshape(-1)
    if v.size == 0 or np.any(np.isnan(v)):
        return None, False
    return _normalize_vec(v), True


def _pacing_vector(run_dir: str, keys: List[str]) -> np.ndarray:
    p = os.path.join(run_dir, "video_pacing", "video_pacing_features.npz")
    d = _load_npz(p)
    fobj = _unbox_dict(d.get("features"))
    if fobj is None:
        raise RuntimeError(f"video_pacing missing features dict: {p}")
    v = np.asarray([float(fobj.get(k, float("nan"))) for k in keys], dtype=np.float32)
    # Missing keys must not poison normalization. Treat missing as 0 for reference-pack stability.
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize_vec(v)


def _shot_quality_vector(run_dir: str) -> Tuple[np.ndarray, List[str]]:
    p = os.path.join(run_dir, "shot_quality", "shot_quality.npz")
    d = _load_npz(p)
    names = d.get("feature_names")
    means = d.get("shot_features_mean")
    if names is None or means is None:
        raise RuntimeError(f"shot_quality missing feature_names/shot_features_mean: {p}")
    fn = [str(x) for x in np.asarray(names, dtype=object).tolist()]
    m = np.asarray(means, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != len(fn):
        raise RuntimeError(f"shot_quality invalid shot_features_mean shape: {m.shape} vs names={len(fn)}")
    v = np.nanmean(m, axis=0)
    return _normalize_vec(v), fn


def _emotion_vector(run_dir: str, keys: List[str]) -> np.ndarray:
    p = os.path.join(run_dir, "micro_emotion", "micro_emotion.npz")
    d = _load_npz(p)
    fobj = _unbox_dict(d.get("features"))
    meta = _unbox_dict(d.get("meta"))
    status = (meta.get("status") if isinstance(meta, dict) else None)
    if status == "empty":
        return np.full((len(keys),), np.nan, dtype=np.float32)
    if fobj is None:
        raise RuntimeError(f"micro_emotion missing features dict: {p}")
    v = np.asarray([float(fobj.get(k, float("nan"))) for k in keys], dtype=np.float32)
    # Treat missing keys as 0 to avoid NaN vectors in the pack.
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize_vec(v)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--reference-set-id", required=True, help="Reference set id (folder name)")
    ap.add_argument("--dp-models-root", default=None, help="Path to dp_models (defaults to DataProcessor/dp_models)")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="List of per-run result_store directories")
    ap.add_argument("--pacing-max-keys", type=int, default=64, help="Max pacing scalar keys")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dp_models_root = Path(args.dp_models_root).resolve() if args.dp_models_root else (repo_root / "dp_models").resolve()
    out_dir = dp_models_root / "bundled_models" / "similarity" / "reference_sets" / str(args.reference_set_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [str(Path(p).resolve()) for p in args.run_dirs]
    ref_ids: List[str] = []

    # Determine pacing feature keys from first run (deterministic).
    first_pacing = _load_npz(os.path.join(run_dirs[0], "video_pacing", "video_pacing_features.npz"))
    fobj = _unbox_dict(first_pacing.get("features"))
    if fobj is None:
        raise RuntimeError("video_pacing first run missing features dict")
    scalar_keys = sorted([str(k) for k, v in fobj.items() if isinstance(v, (int, float, np.floating, np.integer)) and (str(k).startswith("pacing_") or str(k).startswith("shot_") or str(k).startswith("motion_"))])
    pacing_keys = scalar_keys[: max(1, int(args.pacing_max_keys))]

    emotion_keys = [
        "smile_ratio",
        "eye_contact_ratio",
        "blink_rate_per_min",
        "pose_stability_score",
        "face_presence_ratio",
        "au_quality_overall",
        "AU06_mean",
        "AU12_mean",
        "AU04_mean",
        "AU25_mean",
        "AU26_mean",
        "AU07_mean",
        "AU15_mean",
        "AU06_peak_count",
        "AU12_peak_count",
        "AU04_peak_count",
        "AU25_peak_count",
        "AU26_peak_count",
        "AU07_peak_count",
        "AU15_peak_count",
    ]

    clip_rows: List[np.ndarray] = []
    clap_rows: List[np.ndarray] = []
    text_rows: List[np.ndarray] = []
    pacing_rows: List[np.ndarray] = []
    quality_rows: List[np.ndarray] = []
    emo_rows: List[np.ndarray] = []

    shot_quality_feature_names: Optional[List[str]] = None
    text_dim: Optional[int] = None

    for rd in run_dirs:
        # reference_video_id: use video_id from metadata if available, else folder name
        meta_path = os.path.join(rd, "manifest.json")
        vid = None
        if os.path.isfile(meta_path):
            try:
                m = _read_json(meta_path)
                run_meta = m.get("run") if isinstance(m.get("run"), dict) else {}
                vid = run_meta.get("video_id") if isinstance(run_meta, dict) else None
            except Exception:
                vid = None
        ref_id = str(vid or Path(rd).name)
        ref_ids.append(ref_id)

        clip_rows.append(_core_clip_centroid(rd))
        clap_rows.append(_clap_embedding(rd))

        tvec, tp = _text_primary_embedding(rd)
        if tp and tvec is not None:
            if text_dim is None:
                text_dim = int(tvec.size)
            if int(tvec.size) != int(text_dim):
                raise RuntimeError(f"text embedding dim mismatch for {rd}: {tvec.size} vs {text_dim}")
            text_rows.append(tvec)
        else:
            # fill later after text_dim known
            text_rows.append(np.asarray([], dtype=np.float32))

        pacing_rows.append(_pacing_vector(rd, pacing_keys))
        qvec, qnames = _shot_quality_vector(rd)
        if shot_quality_feature_names is None:
            shot_quality_feature_names = list(qnames)
        if list(qnames) != list(shot_quality_feature_names):
            raise RuntimeError(f"shot_quality feature_names mismatch for {rd} (strict)")
        quality_rows.append(qvec)

        emo_rows.append(_emotion_vector(rd, emotion_keys))

    if text_dim is None:
        raise RuntimeError("No reference video has TextProcessor primary_embedding_present=true; cannot infer text_dim")
    # Fill missing text rows with NaNs
    fixed_text_rows: List[np.ndarray] = []
    for v in text_rows:
        if v.size == 0:
            fixed_text_rows.append(np.full((int(text_dim),), np.nan, dtype=np.float32))
        else:
            fixed_text_rows.append(v)
    text_rows = fixed_text_rows

    clip_m = np.stack(clip_rows, axis=0).astype(np.float32)
    clap_m = np.stack(clap_rows, axis=0).astype(np.float32)
    text_m = np.stack(text_rows, axis=0).astype(np.float32)
    pacing_m = np.stack(pacing_rows, axis=0).astype(np.float32)
    quality_m = np.stack(quality_rows, axis=0).astype(np.float32)
    emo_m = np.stack(emo_rows, axis=0).astype(np.float32)

    np.save(out_dir / "clip_video_embeddings.npy", clip_m)
    np.save(out_dir / "clap_audio_embeddings.npy", clap_m)
    np.save(out_dir / "text_primary_embeddings.npy", text_m)
    np.save(out_dir / "pacing_features.npy", pacing_m)
    np.save(out_dir / "shot_quality_features.npy", quality_m)
    np.save(out_dir / "emotion_embeddings.npy", emo_m)

    manifest = {
        "schema_version": "similarity_reference_pack_v1",
        "reference_set_id": str(args.reference_set_id),
        "reference_video_ids": ref_ids,
        "dims": {
            "clip": int(clip_m.shape[1]),
            "clap": int(clap_m.shape[1]),
            "text": int(text_m.shape[1]),
            "pacing": int(pacing_m.shape[1]),
            "quality": int(quality_m.shape[1]),
            "emotion": int(emo_m.shape[1]),
        },
        "pacing_feature_keys": pacing_keys,
        "emotion_feature_keys": emotion_keys,
        "shot_quality_feature_names": list(shot_quality_feature_names or []),
        "source_runs": run_dirs,
    }
    _atomic_write_json(str(out_dir / "manifest.json"), manifest)
    print(f"[ok] wrote reference pack: {out_dir}")
    print(f"[ok] M={len(ref_ids)} dims={manifest['dims']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


