#!/usr/bin/env python3
"""Валидатор speech_analysis_extractor: схема NPZ, --struct (tabular + 1D массивы), --qa (meta + tabular)."""
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
        return "speech_analysis_extractor_npz_v1" in sv or "speech_analysis_extractor" in sv
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
    comp = "speech_analysis_extractor"
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
    for req in ("feature_names", "feature_values", "meta"):
        if req not in d:
            out.append(f"нет обязательного ключа {req}")
    if out:
        return out
    fn, fv = d.get("feature_names"), d.get("feature_values")
    try:
        names = [str(x) for x in np.asarray(fn, dtype=object).ravel()]
        vals = np.asarray(fv, dtype=np.float64).ravel()
        if len(names) != len(vals):
            out.append("feature_names/feature_values: несовпадение длины")
    except Exception as e:  # pragma: no cover
        out.append(f"tabular: {e}")
    al = d.get("asr_lang_id_by_segment")
    if al is not None:
        a = np.asarray(al).ravel()
        if a.size and a.dtype.kind not in ("i", "u"):
            out.append("asr_lang_id_by_segment: ожидаются целочисленные id")
    sp = d.get("speaker_ids")
    if sp is not None:
        a = np.asarray(sp).ravel()
        if a.size and a.dtype.kind not in ("i", "u"):
            out.append("speaker_ids: ожидаются целочисленные id")
    for key in ("asr_lang_distribution", "pitch_distribution"):
        x = d.get(key)
        if x is None:
            continue
        try:
            obj: Any = x
            if isinstance(obj, np.ndarray):
                obj = obj.item() if obj.size == 1 else obj
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
            if not isinstance(obj, dict):
                out.append(f"{key}: ожидается dict (object), got {type(obj).__name__}")
        except Exception as e:  # pragma: no cover
            out.append(f"{key}: {e}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="validate speech_analysis_extractor NPZ")
    p.add_argument("npz_path")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Проверить плоский meta + tabular (view_csv_feature_qa.json).",
    )
    p.add_argument(
        "--struct",
        action="store_true",
        help="Обязательные ключи, tabular, 1D массивы, object dicts.",
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
