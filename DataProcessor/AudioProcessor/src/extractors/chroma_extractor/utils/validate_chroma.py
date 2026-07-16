#!/usr/bin/env python3
"""Валидатор chroma_extractor: схема NPZ, --struct (формы/согласованность), --qa (плоский meta, view_csv_feature_qa.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {
        k: (data[k].item() if data[k].dtype == object and data[k].size == 1 else data[k])
        for k in data.files
    }


def extract_meta(d: Dict) -> Dict:
    m = d.get("meta")
    return m.item() if hasattr(m, "item") else (m or {})


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        if "meta" not in d:
            return False
        meta = extract_meta(d)
        sv = str(meta.get("schema_version", ""))
        return "chroma_extractor_npz_v1" in sv or "chroma_extractor" in sv
    except Exception:
        return False


def _dataprocessor_on_path() -> Path:
    return Path(__file__).resolve().parents[5]


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
    flat = flatten_meta(meta, prefix="meta_")
    warnings: List[str] = []
    comp = "chroma_extractor"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    is_empty = meta.get("status") == "empty"  # при empty: chroma_mean=[nan], dominant_class=-1 by design

    cm = d.get("chroma_mean")
    if cm is not None:
        a = np.asarray(cm, dtype=np.float64).reshape(-1)
        if is_empty:
            pass  # chroma_mean=[nan] by design при status=empty (npz_saver scalar→shape (1,))
        elif a.size != 12:
            out.append(f"chroma_mean: ожидается 12, получено {a.size}")
        elif np.any(~np.isfinite(a)) or np.any(a < 0):
            out.append("chroma_mean: ожидаются неотрицательные конечные значения")
    dc = d.get("chroma_dominant_class")
    if dc is not None:
        v = int(np.asarray(dc).reshape(-1)[0])
        if is_empty:
            pass  # dominant_class=-1 by design при status=empty (sentinel)
        elif v < 0 or v > 11:
            out.append(f"chroma_dominant_class: ожидается 0..11, получено {v}")
    seg_block = ("segment_centers_sec", "segment_durations_sec", "segment_mask", "chroma_mean_by_segment")
    has_any = any(k in d for k in seg_block)
    if has_any:
        missing = [k for k in seg_block if k not in d]
        if missing:
            out.append(f"сегментный блок неполный, нет: {missing}")
        elif "segment_centers_sec" in d:
            n1 = int(np.asarray(d["segment_centers_sec"]).size)
            n2 = int(np.asarray(d["segment_durations_sec"]).size)
            n3 = int(np.asarray(d["segment_mask"]).size)
            if not (n1 == n2 == n3):
                out.append(f"segment_*: несовпадение длин {n1}, {n2}, {n3}")
            m = np.asarray(d["chroma_mean_by_segment"], dtype=np.float64)
            if m.ndim == 2:
                if m.shape[1] != 12:
                    out.append(f"chroma_mean_by_segment: ожидается (N,12), shape={m.shape}")
                elif m.shape[0] != n1:
                    out.append("chroma_mean_by_segment: первая ось != числу сегментов")
    ch = d.get("chroma")
    if ch is not None:
        ch = np.asarray(ch, dtype=np.float64)
        if ch.ndim == 2 and ch.shape[0] != 12:
            out.append(f"chroma: ожидается первая размерность 12, got {ch.shape}")
    fn, fv = d.get("feature_names"), d.get("feature_values")
    if fn is not None and fv is not None:
        try:
            names = [str(x) for x in np.asarray(fn, dtype=object).ravel()]
            vals = np.asarray(fv, dtype=np.float64).ravel()
            if len(names) != len(vals):
                out.append("feature_names/feature_values: несовпадение длины")
        except Exception as e:  # pragma: no cover
            out.append(f"tabular: {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="validate chroma_extractor NPZ")
    p.add_argument("npz_path")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Проверить плоский meta по view_csv_feature_qa.json (components.chroma_extractor).",
    )
    p.add_argument(
        "--struct",
        action="store_true",
        help="Проверить размерности chroma_mean, сегментов, опционального chroma [12,T].",
    )
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
