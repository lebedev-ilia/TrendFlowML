#!/usr/bin/env python3
"""Валидатор video_pacing NPZ: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "video_pacing_npz_v3"
_ARTIFACT = "video_pacing_features.npz"

# Синхрон с utils/video_pacing.py — _FEATURE_NAMES_V1
_FEATURE_NAMES_V1: Tuple[str, ...] = (
    "video_length_seconds",
    "shots_count",
    "shot_duration_mean",
    "shot_duration_median",
    "shot_duration_min",
    "shot_duration_max",
    "shot_duration_std",
    "shot_duration_mean_normalized",
    "cuts_variance",
    "cuts_per_10s",
    "cuts_per_10s_max",
    "cuts_per_10s_median",
    "shot_duration_entropy",
    "shot_length_gini",
    "short_shot_fraction",
    "quick_cut_burst_count",
    "tempo_entropy",
    "shot_length_histogram_5bins_0",
    "shot_length_histogram_5bins_1",
    "shot_length_histogram_5bins_2",
    "shot_length_histogram_5bins_3",
    "shot_length_histogram_5bins_4",
    "cut_density_map_8bins_0",
    "cut_density_map_8bins_1",
    "cut_density_map_8bins_2",
    "cut_density_map_8bins_3",
    "cut_density_map_8bins_4",
    "cut_density_map_8bins_5",
    "cut_density_map_8bins_6",
    "cut_density_map_8bins_7",
    "pace_curve_slope",
    "pace_curve_slope_normalized",
    "pace_curve_peaks_mean_prominence",
    "pace_curve_dominant_period_sec",
    "pace_curve_power_at_period",
    "mean_motion_speed_per_shot",
    "motion_speed_median",
    "motion_speed_variance",
    "motion_speed_90perc",
    "share_of_high_motion_frames",
    "share_of_high_motion_shots",
    "motion_shot_corr",
    "frame_embedding_diff_mean",
    "frame_embedding_diff_std",
    "high_change_frames_ratio",
    "scene_embedding_jumps",
    "semantic_change_burst_count",
    "color_change_rate_mean",
    "color_change_rate_std",
    "color_change_bursts",
    "saturation_change_rate",
    "brightness_change_rate",
    "luminance_spikes_per_minute",
    "intro_speed",
    "main_speed",
    "climax_speed",
    "pacing_symmetry",
)

_REQUIRED = (
    "frame_indices",
    "times_s",
    "shot_boundary_frame_indices",
    "motion_norm_per_sec_mean",
    "semantic_change_rate_per_sec",
    "color_change_rate_per_sec",
    "feature_names",
    "feature_values",
    "meta",
)

_RATIO_0_1 = (
    "short_shot_fraction",
    "share_of_high_motion_frames",
    "share_of_high_motion_shots",
    "high_change_frames_ratio",
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
        return _SCHEMA in sv or "video_pacing" in sv
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
    st = meta.get("status", "")
    if st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    # При status=empty структурные проверки payload не применяются
    if st == "empty":
        if not meta.get("empty_reason"):
            out.append("meta.empty_reason обязателен при status=empty")
        return out

    fi = np.asarray(d["frame_indices"], dtype=np.int64).ravel()
    ts = np.asarray(d["times_s"], dtype=np.float64).ravel()
    n = int(fi.size)
    if n == 0:
        out.append("frame_indices пуст")
    if n > 1 and not np.all(np.diff(fi) > 0):
        out.append("frame_indices должен строго возрастать")
    if int(ts.size) != n:
        out.append(f"len(times_s) != N ({ts.size} != {n})")
    if n > 1 and np.any(np.diff(ts) < -1e-6):
        out.append("times_s не неубывает")

    sb = np.asarray(d["shot_boundary_frame_indices"], dtype=np.int64).ravel()
    if sb.ndim != 1:
        out.append("shot_boundary_frame_indices: ож. 1D")
    elif np.any(sb < 0):
        out.append("shot_boundary_frame_indices: отрицательные индексы")

    for name in (
        "motion_norm_per_sec_mean",
        "semantic_change_rate_per_sec",
        "color_change_rate_per_sec",
    ):
        a = np.asarray(d[name], dtype=np.float64).ravel()
        if int(a.size) != n:
            out.append(f"{name}: len != N ({a.size} != {n})")

    fn = np.asarray(d["feature_names"], dtype=object).ravel()
    fv = np.asarray(d["feature_values"], dtype=np.float64).ravel()
    if int(fn.size) != int(fv.size):
        out.append("feature_names / feature_values: разная длина")
    else:
        got = [str(x) for x in fn.tolist()]
        if got != list(_FEATURE_NAMES_V1):
            out.append("feature_names: порядок/состав не совпадает с _FEATURE_NAMES_V1")

    if not meta:
        out.append("meta пустой")
    return out


def _load_qa_config() -> Tuple[Any, Path]:
    from qa.component_feature_qa import find_repo_root_from_path, load_qa_config

    root = find_repo_root_from_path(Path(__file__))
    if root is None:
        raise FileNotFoundError("view_csv_feature_qa.json (repo root not found)")
    dp = root / "DataProcessor"
    r = str(dp)
    if r not in sys.path:
        sys.path.insert(0, r)
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
    comp = "video_pacing"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _feature_tab(d: Dict[str, Any]) -> Dict[str, float]:
    fn = np.asarray(d["feature_names"], dtype=object).ravel()
    fv = np.asarray(d["feature_values"], dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    for i in range(int(min(fn.size, fv.size))):
        v = float(fv[i])
        if np.isfinite(v):
            out[str(fn[i])] = v
    return out


def validate_ranges(npz_path: str) -> List[str]:
    """См. docs/FEATURE_DESCRIPTION.md; часть проверок дублирует view_csv_feature_qa по tabular-полям."""
    out: List[str] = []
    d = load_npz(npz_path)
    n = int(np.asarray(d["frame_indices"], dtype=np.int64).ravel().size)
    if n:
        mot = np.asarray(d["motion_norm_per_sec_mean"], dtype=np.float64).ravel()
        if int(mot.size) == n and n:
            m = np.isfinite(mot) & (mot < -1e-6)
            if m.any():
                out.append("motion_norm_per_sec_mean: отрицательные finite (не ожидается)")

    tab = _feature_tab(d)
    for k in _RATIO_0_1:
        v = tab.get(k)
        if v is not None and np.isfinite(v) and (v < -1e-4 or v > 1.0 + 1e-4):
            out.append(f"{k}: вне [0, 1] (finite)")

    meta = extract_meta(d)
    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            tfi, pfi = int(tf), int(pf)
            if tfi >= 0 and pfi > tfi:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass

    return out


def _run_batch(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    c = 0
    for npz in sorted(root.rglob(f"video_pacing/{_ARTIFACT}")):
        c += 1
        ok = validate_schema(str(npz))
        st = validate_structure(str(npz)) if ok else ["INVALID schema"]
        if not ok or st:
            ex = max(ex, 2)
        status = "OK" if ok and not st else "ISSUES"
        print(f"[{status}] {npz}", flush=True)
        for line in st:
            print(f"    - {line}", flush=True)
    print(f"Проверено файлов: {c}", flush=True)
    return ex if c else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate video_pacing NPZ (artifact {_ARTIFACT}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Доли, гистограммы ≈1, motion≥0, processed≤total (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/video_pacing/{_ARTIFACT}",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
    args = p.parse_args()

    if args.results_base:
        return _run_batch(results_base=args.results_base, platform_id=args.platform_id or "youtube")

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

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
        else:
            print("✅ Structure OK (N, _FEATURE_NAMES_V1)")

    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (motion≥0, доли [0,1], processed≤total)")

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
