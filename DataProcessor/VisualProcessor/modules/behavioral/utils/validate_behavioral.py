#!/usr/bin/env python3
"""Валидатор behavioral NPZ: схема, --struct, --qa (view_csv_feature_qa.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "behavioral_npz_v1"
_ARTIFACT = "behavioral_features.npz"

# Синхрон с utils/behavior_analyzer._pack_npz_results + GestureClassifier.gesture_types
_SEQ_INNER: Tuple[str, ...] = (
    "num_hands",
    "hands_visibility",
    "hand_motion_energy",
    "arm_openness",
    "pose_expansion",
    "body_lean_angle",
    "balance_offset",
    "shoulder_angle",
    "shoulder_angle_velocity",
    "head_position_x_norm",
    "head_position_y_norm",
    "head_motion_energy",
    "head_stability",
    "mouth_width_norm",
    "mouth_height_norm",
    "mouth_area_norm",
    "mouth_velocity",
    "mouth_open_ratio",
    "speech_activity_proxy",
    "blink_flag",
    "blink_rate_short",
    "self_touch_flag",
    "fidgeting_energy",
    "timestamp_norm",
)

_GESTURE_NAMES: Tuple[str, ...] = (
    "pointing",
    "open_palm",
    "hands_on_hips",
    "self_touch",
    "fist",
    "thumbs_up",
    "thumbs_down",
    "victory",
    "ok",
    "rock",
    "call_me",
    "love",
)

_REQUIRED: Tuple[str, ...] = (
    "frame_indices",
    "times_s",
    "landmarks_present",
    "hand_gestures",
    "frame_results",
    "aggregated",
) + tuple(f"seq_{k}" for k in _SEQ_INNER) + tuple(f"seq_gesture_prob_{g}" for g in _GESTURE_NAMES) + ("meta",)


def load_npz(npz_path: str) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    try:
        out: Dict[str, Any] = {}
        for k in z.files:
            v = z[k]
            if isinstance(v, np.ndarray) and v.dtype == object and getattr(v, "shape", None) == ():
                try:
                    out[k] = v.item()
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out
    finally:
        try:
            z.close()
        except Exception:
            pass


def extract_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    m = d.get("meta")
    if m is None:
        return {}
    if isinstance(m, np.ndarray) and m.dtype == object and m.shape == ():
        m = m.item()
    return m if isinstance(m, dict) else {}


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        for k in _REQUIRED:
            if k not in d:
                return False
        meta = extract_meta(d)
        sv = str(meta.get("schema_version", ""))
        return _SCHEMA in sv
    except Exception:
        return False


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    meta = extract_meta(d)
    st = str(meta.get("status", ""))
    if st and st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int64).ravel()
    ts = np.asarray(d["times_s"], dtype=np.float64).ravel()
    n = int(fi.size)
    if n == 0:
        out.append("frame_indices пуст")
    if n > 1 and not np.all(np.diff(fi) > 0):
        out.append("frame_indices должен строго возрастать")
    if int(ts.size) != n:
        out.append(f"len(times_s) != N")
    if n > 1 and np.any(np.diff(ts) < -1e-6):
        out.append("times_s не неубывает")

    lp = np.asarray(d["landmarks_present"])
    if lp.shape != (n,):
        out.append(f"landmarks_present shape {lp.shape}, ож. ({n},)")

    hg = d.get("hand_gestures")
    fr = d.get("frame_results")
    for name, arr in (("hand_gestures", hg), ("frame_results", fr)):
        a = np.asarray(arr, dtype=object)
        if a.ndim != 1 or int(a.size) != n:
            out.append(f"{name}: ож. 1D длины N={n}")

    agg = d.get("aggregated")
    if isinstance(agg, np.ndarray) and agg.dtype == object and getattr(agg, "shape", None) == ():
        agg = agg.item()
    if not isinstance(agg, dict):
        out.append("aggregated: ож. dict (или 0d object → dict)")

    for k in _SEQ_INNER:
        key = f"seq_{k}"
        a = np.asarray(d[key], dtype=np.float64).ravel()
        if int(a.size) != n:
            out.append(f"{key}: len {a.size} != N={n}")

    for g in _GESTURE_NAMES:
        key = f"seq_gesture_prob_{g}"
        a = np.asarray(d[key], dtype=np.float64).ravel()
        if int(a.size) != n:
            out.append(f"{key}: len {a.size} != N={n}")

    if not meta:
        out.append("meta пустой")
    return out


def _dataprocessor_on_path() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_qa_config() -> Tuple[Any, Path]:
    dp = _dataprocessor_on_path()
    r = str(dp)
    if r not in sys.path:
        sys.path.insert(0, r)
    from qa.component_feature_qa import find_repo_root_from_path, load_qa_config

    root = find_repo_root_from_path(Path(__file__))
    if root is None:
        raise FileNotFoundError("view_csv_feature_qa.json (repo root not found)")
    path = root / "storage" / "result_store" / "view_csv_feature_qa.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return load_qa_config(path), path


def validate_qa_rows(npz_path: str, qa: Any) -> List[str]:
    from qa.component_feature_qa import flatten_meta

    d = load_npz(npz_path)
    meta = extract_meta(d)
    flat: Dict[str, Any] = dict(flatten_meta(meta, prefix="meta_"))
    warnings: List[str] = []
    comp = "behavioral"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate behavioral NPZ (artifact {_ARTIFACT}, VisualProcessor)"
    )
    p.add_argument("npz_path")
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
    args = p.parse_args()
    ok = validate_schema(args.npz_path)
    print("✅ VALID schema" if ok else "❌ INVALID schema")
    if not ok:
        return 1
    ex = 0
    if args.struct:
        st = validate_structure(args.npz_path)
        if st:
            print("⚠️  structure:")
            for s in st:
                print("  -", s)
            ex = max(ex, 2)
    if args.qa:
        try:
            qa, path = _load_qa_config()
        except Exception as e:
            print(f"QA: пропуск ({e})", flush=True)
            return ex or 0
        warns = validate_qa_rows(args.npz_path, qa)
        if warns:
            print(f"⚠️  QA warnings ({path}):")
            for w in warns:
                print("  -", w)
            ex = max(ex, 2)
        else:
            print(f"✅ QA OK (rules {path})")
    return ex


if __name__ == "__main__":
    sys.exit(main())
