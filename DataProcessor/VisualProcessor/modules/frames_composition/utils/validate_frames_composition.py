#!/usr/bin/env python3
"""Single-file валидатор frames_composition/frames_composition.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "frames_composition_npz_v1"
_ARTIFACT = "frames_composition.npz"
_REQUIRED = (
    "frame_indices",
    "times_s",
    "frame_feature_names",
    "frame_feature_values",
    "frame_feature_present_ratio",
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
        return _SCHEMA in sv or "frames_composition" in sv
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
    comp = "frames_composition"
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

    st = meta.get("status", "")
    if st not in ("ok", "empty", "error", ""):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int64).reshape(-1)
    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    n = int(fi.size)
    if n != int(ts.size):
        out.append(f"len(frame_indices)={fi.size} != len(times_s)={ts.size}")

    if n > 1 and not np.all(fi[1:] > fi[:-1]):
        out.append("frame_indices: ожидается строго возрастающая последовательность")

    if n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд")

    fn = d["feature_names"]
    fv = d["feature_values"]
    if isinstance(fn, np.ndarray) and isinstance(fv, np.ndarray):
        if int(fn.size) != int(fv.size):
            out.append(f"feature_names ({fn.size}) != feature_values ({fv.size})")

    ffn = d["frame_feature_names"]
    ffv = np.asarray(d["frame_feature_values"])
    ffpr = np.asarray(d["frame_feature_present_ratio"], dtype=np.float64).reshape(-1)
    if ffv.ndim != 2:
        out.append(f"frame_feature_values: ожидается 2D, факт {getattr(ffv, 'shape', None)}")
    elif int(ffn.size) != int(ffv.shape[1]):
        out.append(f"len(frame_feature_names)={ffn.size} != D={ffv.shape[1]}")
    elif n > 0 and int(ffv.shape[0]) != n:
        out.append(f"frame_feature_values rows={ffv.shape[0]} != N={n}")
    if int(ffpr.size) != int(ffn.size):
        out.append(
            f"frame_feature_present_ratio len={ffpr.size} != len(frame_feature_names)={ffn.size}"
        )

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
    """Диапазоны и согласованность (см. docs/FEATURE_DESCRIPTION.md)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    ffpr = np.asarray(d["frame_feature_present_ratio"], dtype=np.float64).reshape(-1)
    if ffpr.size and (np.any(ffpr < -1e-4) or np.any(ffpr > 1.0 + 1e-4)):
        out.append("frame_feature_present_ratio: ожидается [0, 1]")

    fs = meta.get("feature_set")
    if isinstance(fs, str) and fs.strip():
        low = fs.strip().lower()
        allowed = {"default", "ml", "model", "all", "full"}
        if low not in allowed:
            out.append(f"meta.feature_set: неизвестное значение {fs!r} (ожид. {allowed})")

    nw = meta.get("num_workers")
    if nw is not None:
        try:
            nwi = int(nw)
            if nwi < 1 or nwi > 128:
                out.append(f"meta.num_workers: вне [1, 128], факт {nwi}")
        except (TypeError, ValueError):
            out.append("meta.num_workers: не int")

    stm = meta.get("stage_timings_ms")
    if isinstance(stm, dict):
        for k, v in stm.items():
            if isinstance(v, (int, float)) and v < -1e-3:
                out.append(f"meta.stage_timings_ms[{k!r}]: отрицательное значение")

    er = meta.get("empty_reason")
    st = str(meta.get("status", ""))
    if st == "empty":
        ers = (str(er).strip() if er is not None else "")
        if ers and ers != "no_faces_in_video":
            out.append(
                f"meta.empty_reason при status=empty: ожидается no_faces_in_video, факт {er!r}"
            )

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


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob(f"frames_composition/{_ARTIFACT}")):
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
    p = argparse.ArgumentParser(
        description=f"validate frames_composition NPZ (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument(
        "--results-base",
        help="[батч] корень result_store; обход **/frames_composition/frames_composition.npz",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
    p.add_argument("--qa", action="store_true", help="Плоский meta + feature_values (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи NPZ, согласованность N/D/F.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Диапазоны (frame_feature_present_ratio, feature_set, num_workers, stage_timings, empty).",
    )
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
            ffv = np.asarray(d0["frame_feature_values"])
            dcol = int(ffv.shape[1]) if ffv.ndim == 2 else 0
            fna = np.asarray(d0["feature_names"], dtype=object).ravel()
            fcount = int(fna.size)
            print(f"✅ Structure OK (N={n}, D={dcol}, F={fcount}, {_SCHEMA})")
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (present_ratio, feature_set, workers, timings, empty, processed≤total)")
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
