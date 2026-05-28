#!/usr/bin/env python3
"""Валидатор story_structure NPZ: схема, --struct, --qa (view_csv_feature_qa.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "story_structure_npz_v3"
_ARTIFACT = "story_structure.npz"

# Синхрон с utils/story_structure.py — _FEATURE_NAMES_V1
_FEATURE_NAMES_V1: Tuple[str, ...] = (
    "n_frames",
    "video_length_seconds",
    "hook_visual_surprise_score",
    "hook_visual_surprise_std",
    "hook_motion_intensity",
    "hook_cut_rate",
    "hook_motion_spikes",
    "hook_rhythm_score",
    "hook_face_presence",
    "climax_frame_index",
    "climax_time_sec",
    "climax_position_normalized",
    "climax_strength",
    "climax_strength_normalized",
    "number_of_peaks",
    "time_from_hook_to_climax",
    "hook_to_avg_energy_ratio",
    "main_character_screen_time",
    "speaker_switch_rate",
    "speaker_switches_per_minute",
    "topic_shift_curve_present",
    "topic_shift_peaks_count",
)

_REQUIRED = (
    "frame_indices",
    "times_s",
    "story_energy_curve",
    "frame_feature_present_ratio",
    "motion_norm_per_sec_mean",
    "embedding_change_rate_per_sec",
    "any_face_present",
    "topic_shift_curve",
    "topic_shift_curve_present",
    "topic_shift_peaks_idx",
    "embedding_sim_next",
    "embedding_diff_next",
    "story_energy_curve_downsampled_128",
    "story_energy_peaks_idx",
    "story_energy_peaks_times_s",
    "story_energy_peaks_values_z",
    "feature_names",
    "feature_values",
    "meta",
)


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


def _as_f32_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).ravel()


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    meta = extract_meta(d)
    st = meta.get("status", "")
    if st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int32).ravel()
    N = int(fi.size)
    if N == 0:
        out.append("frame_indices пуст")
        return out
    if N > 1 and not np.all(np.diff(fi.astype(np.int64)) > 0):
        out.append("frame_indices должен строго возрастать")
    if np.any(fi < 0):
        out.append("frame_indices: отрицательные значения")

    ts = _as_f32_1d(d["times_s"])
    if int(ts.size) != N:
        out.append(f"len(times_s) != N ({ts.size} != {N})")
    elif ts.size > 1 and np.any(np.diff(ts) < -1e-3):
        out.append("times_s не неубывает")

    for name in (
        "story_energy_curve",
        "motion_norm_per_sec_mean",
        "embedding_change_rate_per_sec",
    ):
        a = _as_f32_1d(d[name])
        if int(a.size) != N:
            out.append(f"{name}: len != N ({a.size} != {N})")

    af = np.asarray(d["any_face_present"]).ravel()
    if int(af.size) != N:
        out.append(f"any_face_present: len != N ({af.size} != {N})")

    tc = _as_f32_1d(d["topic_shift_curve"])
    if int(tc.size) != N:
        out.append(f"topic_shift_curve: len != N ({tc.size} != {N})")

    tp = d["topic_shift_curve_present"]
    if isinstance(tp, np.ndarray) and tp.shape == ():
        tp = bool(tp.item())
    elif isinstance(tp, np.ndarray):
        try:
            tp = bool(tp.reshape(-1)[0])
        except Exception:
            out.append("topic_shift_curve_present: невалидный скаляр")
    elif not isinstance(tp, (bool, np.bool_)):
        try:
            tp = bool(tp)
        except Exception:
            out.append("topic_shift_curve_present: невалидный тип")

    expected_pairs = max(N - 1, 0)
    for name in ("embedding_sim_next", "embedding_diff_next"):
        a = _as_f32_1d(d[name])
        if int(a.size) != expected_pairs:
            out.append(f"{name}: ож. длина N-1={expected_pairs}, факт {a.size}")

    eds = _as_f32_1d(d["story_energy_curve_downsampled_128"])
    if eds.shape != (128,):
        out.append("story_energy_curve_downsampled_128: ож. (128,)")

    epi = np.asarray(d["story_energy_peaks_idx"]).ravel()
    ept = _as_f32_1d(d["story_energy_peaks_times_s"])
    epz = _as_f32_1d(d["story_energy_peaks_values_z"])
    for nm, a in (("story_energy_peaks_idx", epi), ("story_energy_peaks_times_s", ept), ("story_energy_peaks_values_z", epz)):
        if a.ndim != 1:
            out.append(f"{nm}: ож. 1D")
    if epi.size == ept.size == epz.size:
        pass
    else:
        out.append("story_energy_peaks_*: разная длина")

    tpk = np.asarray(d["topic_shift_peaks_idx"], dtype=np.int32).ravel()
    if tpk.ndim != 1:
        out.append("topic_shift_peaks_idx: ож. 1D")

    fpr = _as_f32_1d(d["frame_feature_present_ratio"])
    if int(fpr.size) != N:
        out.append("frame_feature_present_ratio: len != N")

    fn = np.asarray(d["feature_names"], dtype=object).ravel()
    fv = np.asarray(d["feature_values"], dtype=np.float32).ravel()
    if int(fn.size) != int(fv.size):
        out.append("feature_names / feature_values: разная длина")
    else:
        got = [str(x) for x in fn.tolist()]
        if got != list(_FEATURE_NAMES_V1):
            out.append("feature_names: порядок/состав не совпадает с _FEATURE_NAMES_V1")

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
    comp = "story_structure"
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
        description=f"validate story_structure NPZ (artifact {_ARTIFACT}, VisualProcessor)"
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
