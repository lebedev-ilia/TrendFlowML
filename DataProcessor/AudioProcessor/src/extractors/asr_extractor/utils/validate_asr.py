#!/usr/bin/env python3
"""Валидатор для asr_extractor: схема NPZ и опционально QA по диапазонам (view_csv_feature_qa.json)."""
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
        meta = extract_meta(d)
        if "meta" not in d:
            return False
        sv = str(meta.get("schema_version", ""))
        return "asr_extractor_npz_v2" in sv or "asr_extractor" in sv
    except Exception:
        return False


def _dataprocessor_on_path() -> Path:
    # .../extractors/asr_extractor/utils/this.py -> parents[5] = DataProcessor
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
    """Проверка плоского meta как в batch CSV (component=asr_extractor)."""
    from qa.component_feature_qa import flatten_meta

    d = load_npz(npz_path)
    meta = extract_meta(d)
    flat = flatten_meta(meta, prefix="meta_")
    warnings: List[str] = []
    comp = "asr_extractor"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    # merge-колонки duration_ms нет в NPZ-only — при необходимости сверяйте wide CSV
    return warnings


def main() -> int:
    p = argparse.ArgumentParser(description="validate asr_extractor NPZ")
    p.add_argument("npz_path")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Проверить значения meta по правилам storage/result_store/view_csv_feature_qa.json (components.asr_extractor)",
    )
    args = p.parse_args()
    ok = validate_schema(args.npz_path)
    print("✅ VALID schema" if ok else "❌ INVALID schema")
    if not ok:
        return 1
    if not args.qa:
        return 0
    try:
        qa, path = _load_qa_config()
    except Exception as e:
        print(f"QA: skip ({e})", flush=True)
        return 0
    warns = validate_qa_rows(args.npz_path, qa)
    if not warns:
        print(f"✅ QA OK (rules {path})")
        return 0
    print(f"⚠️  QA warnings ({path}):")
    for w in warns:
        print("  -", w)
    return 2


if __name__ == "__main__":
    sys.exit(main())
