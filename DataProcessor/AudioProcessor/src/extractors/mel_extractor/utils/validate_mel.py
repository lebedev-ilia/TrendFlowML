#!/usr/bin/env python3
"""Валидатор mel_extractor: схема, --struct (N, M, опц. time_series), --qa (плоский meta)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SEGMENT_AXIS = (
    "segment_start_sec",
    "segment_end_sec",
    "segment_center_sec",
    "segment_mask",
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


def _tabular_dict(d: Dict) -> Dict[str, float]:
    fn, fv = d.get("feature_names"), d.get("feature_values")
    if fn is None or fv is None:
        return {}
    names = [str(x) for x in np.asarray(fn, dtype=object).ravel()]
    vals = np.asarray(fv, dtype=np.float64).ravel()
    return {names[i]: float(vals[i]) for i in range(min(len(names), len(vals)))}


def _n_mels_from_tabular(td: Dict[str, float]) -> Optional[int]:
    if "n_mels" in td:
        try:
            v = float(td["n_mels"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):  # NaN/inf при status=empty — пропускаем структурные проверки
            return None
        v_int = int(round(v))
        if v_int > 0:
            return v_int
    return None


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        if "meta" not in d:
            return False
        meta = extract_meta(d)
        sv = str(meta.get("schema_version", ""))
        return "mel_extractor_npz_v2" in sv or "mel_extractor" in sv
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
    comp = "mel_extractor"
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
    miss = [k for k in _SEGMENT_AXIS if k not in d]
    if miss:
        out.append(f"нет сегментных ключей: {miss}")
        return out
    sizes = {k: int(np.asarray(d[k]).size) for k in _SEGMENT_AXIS}
    n0 = next(iter(sizes.values()))
    for k, n in sizes.items():
        if n != n0:
            out.append(f"{k}: длина {n} != {n0} (N)")

    td = _tabular_dict(d)
    m_expected = _n_mels_from_tabular(td)

    for key in ("mel_mean", "mel_std", "mel_min", "mel_max"):
        a = d.get(key)
        if a is not None and np.size(a) > 0:
            ln = int(np.asarray(a).ravel().size)
            if m_expected is not None and ln != m_expected:
                out.append(f"{key}: длина {ln} != n_mels={m_expected} (tabular)")

    sv = d.get("mel_stats_vector")
    if sv is not None and np.size(sv) > 0 and m_expected is not None:
        lsv = int(np.asarray(sv).ravel().size)
        if lsv != 4 * m_expected:
            out.append(f"mel_stats_vector: ожидается 4*n_mels={4 * m_expected}, size={lsv}")

    mmb = d.get("mel_mean_by_segment")
    if mmb is not None and np.size(mmb) > 0:
        mm = np.asarray(mmb, dtype=np.float64)
        if mm.ndim != 2:
            out.append(f"mel_mean_by_segment: ожидается 2D, ndim={mm.ndim}")
        else:
            n_r, m_r = int(mm.shape[0]), int(mm.shape[1])
            if n_r != n0:
                out.append(f"mel_mean_by_segment: ось0 {n_r} != N={n0}")
            if m_expected is not None and m_r != m_expected:
                out.append(f"mel_mean_by_segment: ось1 {m_r} != n_mels={m_expected}")

    for skey in ("mel_energy_by_segment", "mel_centroid_mean_by_segment", "mel_bandwidth_mean_by_segment"):
        a = d.get(skey)
        if a is not None and np.size(a) > 0:
            if int(np.asarray(a).size) != n0:
                out.append(f"{skey}: длина {int(np.asarray(a).size)} != N={n0}")

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
    p = argparse.ArgumentParser(description="validate mel_extractor NPZ")
    p.add_argument("npz_path")
    p.add_argument(
        "--qa",
        action="store_true",
        help="Проверить плоский meta (view_csv_feature_qa.json → mel_extractor).",
    )
    p.add_argument(
        "--struct",
        action="store_true",
        help="Согласованность N, M, опциональные mel_* массивы.",
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
