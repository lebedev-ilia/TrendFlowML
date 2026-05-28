#!/usr/bin/env python3
"""Валидатор uniqueness/uniqueness.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "uniqueness_npz_v4"
_ARTIFACT = "uniqueness.npz"

_FEATURE_NAMES_V1: Tuple[str, ...] = (
    "repeat_threshold_is_otsu",
    "repeat_threshold_used",
    "repeat_threshold_raw",
    "repeat_threshold_quality",
    "repeat_threshold_min",
    "repeat_threshold_max",
    "repeat_threshold_bins",
    "max_frames",
    "repetition_ratio",
    "max_sim_to_other_mean",
    "max_sim_to_other_p95",
    "pairwise_sim_mean",
    "pairwise_sim_p95",
    "cos_dist_next_mean",
    "cos_dist_next_p95",
    "temporal_change_mean",
    "diversity_score",
    "effective_unique_frames",
    "effective_unique_ratio",
    "n_frames",
)

_REQUIRED = (
    "frame_indices",
    "times_s",
    "max_sim_to_other",
    "cos_dist_next",
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
        return _SCHEMA in sv or (
            "uniqueness" in sv.lower() and "npz" in sv.lower()
        )
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
    comp = "uniqueness"
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
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    stv = meta.get("status")
    if stv is not None:
        s = str(stv)
        if s not in ("ok", "empty", "error"):
            out.append(f"meta.status неожидан: {s!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int64).ravel()
    ts = np.asarray(d["times_s"], dtype=np.float64).ravel()
    n = int(fi.size)
    if n == 0:
        out.append("frame_indices пуст")
    if n > 1 and not np.all(np.diff(fi) > 0):
        out.append("frame_indices должен строго возрастать")
    if int(ts.size) != n:
        out.append(f"len(times_s)={ts.size} != N={n}")
    if n > 1 and np.any(np.diff(ts) < -1e-6):
        out.append("times_s не неубывает")

    ms = np.asarray(d["max_sim_to_other"], dtype=np.float64).ravel()
    if int(ms.size) != n:
        out.append(f"len(max_sim_to_other) != N ({ms.size} != {n})")

    cd = np.asarray(d["cos_dist_next"], dtype=np.float64).ravel()
    exp = max(n - 1, 0)
    if int(cd.size) != exp:
        out.append(f"len(cos_dist_next) ож. N-1={exp}, факт {cd.size}")

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


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return [
            f"meta.status=error: {meta.get('empty_reason')!r} (struct не валидирует payload)"
        ]
    return _validate_data_dict(d, meta)


def validate_ranges(npz_path: str) -> List[str]:
    """max_sim = cos sim ∈ [0,1]; cos_dist_next = 1−cos ∈ [0,2] (см. uniqueness.py)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int64).ravel().size)
    ms = np.asarray(d["max_sim_to_other"], dtype=np.float64).ravel()
    if n >= 1 and int(ms.size) == n:
        m = np.isfinite(ms)
        if m.any():
            lo, hi = float(np.min(ms[m])), float(np.max(ms[m]))
            if lo < -1e-3 or hi > 1.0 + 1e-3:
                out.append(f"max_sim_to_other: finite вне [0,1] (min={lo}, max={hi})")

    cd = np.asarray(d["cos_dist_next"], dtype=np.float64).ravel()
    if int(cd.size) >= 1:
        m2 = np.isfinite(cd)
        if m2.any():
            lo, hi = float(np.min(cd[m2])), float(np.max(cd[m2]))
            if lo < -1e-3 or hi > 2.0 + 1e-3:
                out.append(f"cos_dist_next: finite вне [0,2] (min={lo}, max={hi})")

    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            tfi, pfi = int(tf), int(pf)
            if tfi >= 0 and pfi > tfi:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass
    if pf is not None and n >= 1:
        try:
            if int(pf) != n:
                out.append(
                    f"meta.processed_frames={pf} != N={n} (ожидается len(frame_indices))"
                )
        except (TypeError, ValueError):
            pass

    stm = meta.get("stage_timings_ms")
    if isinstance(stm, dict):
        for k, v in stm.items():
            if isinstance(v, (int, float)) and v < -1e-3:
                out.append(f"meta.stage_timings_ms[{k!r}]: отрицательное значение")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    c = 0
    for npz in sorted(root.rglob(f"uniqueness/{_ARTIFACT}")):
        c += 1
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
    print(f"Проверено файлов: {c}", flush=True)
    return ex if c else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate uniqueness/{_ARTIFACT} (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta + tabular (view_csv_feature_qa).")
    p.add_argument(
        "--struct", action="store_true", help="N, max_sim, cos_dist_next, _FEATURE_NAMES_V1."
    )
    p.add_argument(
        "--ranges",
        action="store_true",
        help="max_sim [0,1], cos_dist [0,2], meta frames, тайминги (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/uniqueness/{_ARTIFACT}",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
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
            n = int(np.asarray(d0["frame_indices"], dtype=np.int64).ravel().size)
            f = len(_FEATURE_NAMES_V1)
            print(
                f"✅ Structure OK (N={n}, F={f} tabular, {_SCHEMA}, uniqueness/{_ARTIFACT})"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print(
                "✅ Ranges OK (max_sim [0,1], cos_dist [0,2] finite, times, meta frames, timings)"
            )
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
