#!/usr/bin/env python3
"""Валидатор key_extractor: схема, --struct (N, key_scores), --qa (плоский meta)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_AXIS = (
    "segment_start_sec",
    "segment_end_sec",
    "segment_center_sec",
    "segment_mask",
    "key_id_by_segment",
    "key_confidence_by_segment",
)


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
        return "key_extractor_npz_v1" in sv or "key_extractor" in sv
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
    comp = "key_extractor"
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
    miss = [k for k in _AXIS if k not in d]
    if miss:
        out.append(f"нет обязательных ключей: {miss}")
        return out
    sizes: Dict[str, int] = {}
    for k in _AXIS:
        sizes[k] = int(np.asarray(d[k]).size)
    n0 = next(iter(sizes.values()))
    for k, n in sizes.items():
        if n != n0:
            out.append(f"{k}: длина {n} != {n0} (строка сегмента)")

    ks = d.get("key_scores")
    if ks is not None:
        a = np.asarray(ks, dtype=np.float64).ravel()
        if a.size != 24:
            out.append(f"key_scores: ожидается 24 скаляра, size={a.size}")

    ts = d.get("key_confidences_sequence")
    if ts is not None and np.size(ts) > 0:
        if int(np.asarray(ts).size) != n0:
            out.append(f"key_confidences_sequence: длина {np.asarray(ts).size} != N={n0}")
    for name in ("key_names_sequence", "key_modes_sequence"):
        seq = d.get(name)
        if seq is not None and np.size(seq) > 0:
            ln = len(np.asarray(seq, dtype=object).ravel())
            if ln != n0:
                out.append(f"{name}: длина {ln} != N={n0}")

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
    p = argparse.ArgumentParser(description="validate key_extractor NPZ")
    p.add_argument("npz_path")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Проверить плоский meta (view_csv_feature_qa.json → key_extractor).",
    )
    p.add_argument(
        "--struct",
        action="store_true",
        help="Согласованность N, форма key_scores(24), опц. time_series.",
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
