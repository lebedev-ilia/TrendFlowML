#!/usr/bin/env python3
"""Валидатор optical_flow/optical_flow.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_FIXED_FEATURE_NAMES = (
    "motion_curve_mean",
    "motion_curve_median",
    "motion_curve_p90",
    "motion_curve_variance",
    "missing_frame_ratio",
    "cam_shake_std_mean",
    "cam_rotation_abs_mean",
    "cam_translation_abs_mean",
    "flow_consistency_mean",
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
        need = (
            "frame_indices",
            "times_s",
            "motion_norm_per_sec_mean",
            "frame_feature_names",
            "frame_feature_values",
            "feature_names",
            "feature_values",
            "meta",
        )
        if any(k not in d for k in need):
            return False
        meta = extract_meta(d)
        sv = str(meta.get("schema_version", ""))
        return "optical_flow_npz_v3" in sv or "optical_flow" in sv
    except Exception:
        return False


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
    fn, fv = d.get("feature_names"), d.get("feature_values")
    if fn is not None and fv is not None:
        try:
            names = [str(x) for x in np.asarray(fn, dtype=object).ravel()]
            vals = np.asarray(fv, dtype=np.float64).ravel()
            for n, v in zip(names, vals):
                flat[str(n)] = v
        except Exception:
            pass
    warnings: List[str] = []
    comp = "optical_flow"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _validate_data_dict(d: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    need = (
        "frame_indices",
        "times_s",
        "motion_norm_per_sec_mean",
        "frame_feature_names",
        "frame_feature_values",
        "feature_names",
        "feature_values",
    )
    for k in need:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out
    st = str(meta.get("status", "") or "")
    if st and st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int32).reshape(-1)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    mot = np.asarray(d["motion_norm_per_sec_mean"], dtype=np.float32).reshape(-1)
    n = int(fi.size)
    if n != int(ts.size) or n != int(mot.size):
        out.append(f"N: frame_indices={fi.size} times_s={ts.size} motion={mot.size}")
    ffn = np.asarray(d["frame_feature_names"], dtype=object).ravel()
    ffv = np.asarray(d["frame_feature_values"], dtype=np.float32)
    if ffv.ndim != 2:
        out.append("frame_feature_values: ожидается 2D (N,D)")
    elif int(ffv.shape[0]) != n:
        out.append(f"frame_feature_values: строк {ffv.shape[0]} != N={n}")
    elif int(ffv.shape[1]) != int(ffn.size):
        out.append("frame_feature_values: D != len(frame_feature_names)")
    fna = np.asarray(d["feature_names"], dtype=object).ravel()
    fva = np.asarray(d["feature_values"], dtype=np.float32).ravel()
    if fna.size != fva.size:
        out.append("feature_names / feature_values: разная длина")
    names = [str(x) for x in fna.tolist()]
    miss = [x for x in _FIXED_FEATURE_NAMES if x not in names]
    if miss:
        out.append(f"feature_names: нет ожидаемых: {miss[:5]}{'...' if len(miss) > 5 else ''}")
    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    if m.get("status") == "error":
        return [
            f"meta.status=error: {m.get('empty_reason')!r} (struct не валидирует payload)"
        ]
    return _validate_data_dict(d, m)


def _tabular(d: Dict[str, Any]) -> Dict[str, float]:
    fna = np.asarray(d.get("feature_names"), dtype=object).ravel()
    fva = np.asarray(d.get("feature_values"), dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    for i in range(int(min(fna.size, fva.size))):
        v = fva[i]
        if np.isfinite(v):
            out[str(fna[i])] = float(v)
    return out


def validate_ranges(npz_path: str) -> List[str]:
    """См. docs/FEATURE_DESCRIPTION.md и view_csv_feature_qa → optical_flow."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).size)

    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if int(ts.size) == n and n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    mot = np.asarray(d["motion_norm_per_sec_mean"], dtype=np.float64).reshape(-1)
    if int(mot.size) == n and n:
        m = np.isfinite(mot) & (mot < -1e-6)
        if m.any():
            out.append("motion_norm_per_sec_mean: отрицательные finite")

    tab = _tabular(d)
    mfr = tab.get("missing_frame_ratio")
    if mfr is not None and (mfr < -1e-6 or mfr > 1.0 + 1e-6):
        out.append("missing_frame_ratio: вне [0, 1]")

    fc = tab.get("flow_consistency_mean")
    if fc is not None and (fc < -1e-6 or fc > 1.0 + 1e-6):
        out.append("flow_consistency_mean: вне [0, 1] (finite)")

    for key in ("motion_curve_mean", "motion_curve_median", "motion_curve_p90", "motion_curve_variance"):
        v = tab.get(key)
        if v is not None and v < -1e-6:
            out.append(f"{key}: отрицательное значение")
    for key in ("cam_shake_std_mean", "cam_rotation_abs_mean", "cam_translation_abs_mean"):
        v = tab.get(key)
        if v is not None and v < -1e-6:
            out.append(f"{key}: отрицательное значение")

    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            tfi, pfi = int(tf), int(pf)
            if tfi >= 0 and pfi < 0:
                out.append("meta.processed_frames отрицательно")
            if tfi >= 0 and pfi > tfi:
                out.append("meta.processed_frames > meta.total_frames (не ожидается)")
        except (TypeError, ValueError):
            pass
    if tf is not None and n:
        try:
            if n > int(tf) >= 0:
                out.append("len(frame_indices) > meta.total_frames")
        except (TypeError, ValueError):
            pass

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("optical_flow/optical_flow.npz")):
        n += 1
        d = load_npz(str(npz))
        m = extract_meta(d)
        ok = validate_schema(str(npz))
        stl: List[str] = []
        if not ok:
            stl = ["INVALID schema"]
        elif m.get("status") == "error":
            stl = [f"meta.status=error: {m.get('empty_reason')!r}"]
        else:
            stl = _validate_data_dict(d, m)
        if not ok or stl:
            ex = max(ex, 2)
        status = "OK" if ok and not stl else "ISSUES"
        print(f"[{status}] {npz}", flush=True)
        for line in stl:
            print(f"    - {line}", flush=True)
    print(f"Проверено файлов: {n}", flush=True)
    return ex if n else 1


def main() -> int:
    p = argparse.ArgumentParser(description="validate optical_flow/optical_flow.npz")
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Плоский meta + tabular (view_csv_feature_qa → optical_flow).",
    )
    p.add_argument(
        "--struct",
        action="store_true",
        help="N, frame table, feature_names (фикс. набор).",
    )
    p.add_argument(
        "--ranges",
        action="store_true",
        help="times_s, motion, video-level агрегаты, processed≤total (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/optical_flow/optical_flow.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube", help="Субкаталог платформы (батч)")
    args = p.parse_args()

    if args.results_base:
        return _run_batch_rglob(
            results_base=args.results_base, platform_id=args.platform_id or "youtube"
        )

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    ok = validate_schema(args.npz_path)
    print("✅ VALID schema" if ok else "❌ INVALID schema")
    if not ok:
        return 1
    ex = 0
    d_once: Dict[str, Any] | None = None
    if args.struct or args.ranges:
        d_once = load_npz(args.npz_path)
    if args.struct:
        st = validate_structure(args.npz_path)
        if st:
            print("⚠️  structure:")
            for s in st:
                print("  -", s)
            ex = max(ex, 2)
        else:
            d0 = d_once if d_once is not None else load_npz(args.npz_path)
            n = int(np.asarray(d0["frame_indices"], dtype=np.int32).ravel().size)
            ffv = np.asarray(d0["frame_feature_values"], dtype=np.float32)
            dcol = int(ffv.shape[1]) if ffv.ndim == 2 else 0
            print(
                f"✅ Structure OK (N={n}, D={dcol}, fixed tabular, optical_flow_npz_v3)"
            )

    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (times/motion, tabular, meta frames, N≤total)")

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
