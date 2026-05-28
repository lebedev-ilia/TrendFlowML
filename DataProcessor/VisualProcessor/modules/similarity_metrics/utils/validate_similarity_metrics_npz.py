#!/usr/bin/env python3
"""Валидатор similarity_metrics/results.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "similarity_metrics_npz_v3"
_ARTIFACT = "results.npz"
_REQUIRED = (
    "frame_indices",
    "times_s",
    "centroid_sims",
    "temporal_sim_next",
    "reference_present",
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
        return "similarity_metrics_npz_v3" in sv or (
            "similarity_metrics" in sv and "npz" in sv
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
    comp = "similarity_metrics"
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

    fi = np.asarray(d["frame_indices"], dtype=np.int64).reshape(-1)
    n = int(fi.size)
    if n < 1:
        out.append("контракт: N >= 1")
    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if n != int(ts.size):
        out.append(f"len(times_s)={ts.size} != N={n}")

    cs = np.asarray(d["centroid_sims"], dtype=np.float64).reshape(-1)
    if int(cs.size) != n:
        out.append(f"centroid_sims: len={cs.size} != N={n}")

    tn = np.asarray(d["temporal_sim_next"], dtype=np.float64).reshape(-1)
    expect_tm = max(0, n - 1)
    if int(tn.size) != expect_tm:
        out.append(f"temporal_sim_next: len={tn.size} != N-1={expect_tm}")

    rp = d["reference_present"]
    if isinstance(rp, np.ndarray):
        if rp.shape != () and rp.size != 1:
            out.append("reference_present: ожидается скаляр bool")
    elif not isinstance(rp, (bool, np.bool_)):
        out.append("reference_present: неверный тип")

    fn = np.asarray(d["feature_names"], dtype=object).reshape(-1)
    f = int(fn.size)
    fv = np.asarray(d["feature_values"], dtype=np.float64).reshape(-1)
    if int(fv.size) != f:
        out.append("feature_values: len != len(feature_names)")

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
    """См. docs/FEATURE_DESCRIPTION.md; cos-sim в [-1,1], тайминги ≥0, processed≤total."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int64).ravel().size)
    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if n > 1 and int(ts.size) == n and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд")

    cs = np.asarray(d["centroid_sims"], dtype=np.float64).reshape(-1)
    if n >= 1 and int(cs.size) == n:
        m = np.isfinite(cs)
        if m.any():
            cmin, cmax = float(np.min(cs[m])), float(np.max(cs[m]))
            if cmin < -1.0001 or cmax > 1.0001:
                out.append(f"centroid_sims: finite вне [-1,1] (min={cmin}, max={cmax})")

    tn = np.asarray(d["temporal_sim_next"], dtype=np.float64).reshape(-1)
    if int(tn.size) >= 1:
        m2 = np.isfinite(tn)
        if m2.any():
            tmin, tmax = float(np.min(tn[m2])), float(np.max(tn[m2]))
            if tmin < -1.0001 or tmax > 1.0001:
                out.append(f"temporal_sim_next: finite вне [-1,1] (min={tmin}, max={tmax})")

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
                out.append(f"meta.processed_frames={pf} != N={n} (ожидается len(frame_indices))")
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
    for npz in sorted(root.rglob(f"similarity_metrics/{_ARTIFACT}")):
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
        description=f"validate similarity_metrics/{_ARTIFACT} (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta + feature_values (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, согласованность массивов/фич.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="cos-sim ∈ [-1,1], times_s, processed/total, stage_timings (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/similarity_metrics/{_ARTIFACT}",
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
            fna = np.asarray(d0["feature_names"], dtype=object).ravel()
            fcount = int(fna.size)
            print(
                f"✅ Structure OK (N={n}, F={fcount}, {_SCHEMA}, similarity_metrics/{_ARTIFACT})"
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
                "✅ Ranges OK (cos-sim [-1,1] finite, times, processed/total, stage_timings, N=processed)"
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
