#!/usr/bin/env python3
"""Single-file валидатор place_semantics.npz: схема, --struct, --qa (view_csv_feature_qa)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "frame_indices",
    "times_s",
    "semantic_label_names",
    "semantic_object_ids",
    "threshold_per_label_arr",
    "track_ids",
    "track_present_mask",
    "track_topk_ids",
    "track_topk_scores",
    "track_is_confident_top1",
    "track_topk_evidence_frame_indices",
    "frame_topk_ids",
    "frame_topk_scores",
    "frame_is_confident_top1",
    "meta",
    "meta_json",
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
        return "place_semantics_npz_v2" in sv or "place_semantics" in sv
    except Exception:
        return False


def _dataprocessor_on_path() -> Path:
    return Path(__file__).resolve().parents[6]


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
    comp = "place_semantics"
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
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    fi = np.asarray(d["frame_indices"], dtype=np.int32).reshape(-1)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    n = int(fi.size)
    if n != int(ts.size):
        out.append(f"len(frame_indices)={fi.size} != len(times_s)={ts.size}")

    f_ids = np.asarray(d["frame_topk_ids"])
    f_sc = np.asarray(d["frame_topk_scores"])
    if f_ids.ndim == 2 and f_sc.ndim == 2:
        if int(f_ids.shape[0]) != n or int(f_sc.shape[0]) != n:
            out.append("frame_topk_*: первая размерность != N")
        if f_ids.shape != f_sc.shape:
            out.append("frame_topk_ids / frame_topk_scores: shape mismatch")
    else:
        out.append("frame_topk_ids / frame_topk_scores: ожидается 2D (N,K)")

    tid = np.asarray(d["track_ids"], dtype=np.int32).reshape(-1)
    t = int(tid.size)
    tk = np.asarray(d["track_topk_ids"])
    tks = np.asarray(d["track_topk_scores"])
    if tk.ndim == 2 and tks.ndim == 2:
        if int(tk.shape[0]) != t or int(tks.shape[0]) != t:
            out.append("track_topk_*: первая размерность != len(track_ids)")
        if tk.shape != tks.shape:
            out.append("track_topk_ids / track_topk_scores: shape mismatch")
    else:
        out.append("track_topk_ids / track_topk_scores: ожидается 2D (T,K)")

    a = int(np.asarray(d["semantic_label_names"]).size)
    th = np.asarray(d["threshold_per_label_arr"], dtype=np.float32).reshape(-1)
    if int(th.size) != a:
        out.append("threshold_per_label_arr: длина != len(semantic_label_names)")

    mj = d.get("meta_json")
    if mj is not None:
        if isinstance(mj, np.ndarray):
            s = mj.item() if mj.shape == () else np.asarray(mj).reshape(-1)[0]
        else:
            s = mj
        if not isinstance(s, str) or not str(s).strip():
            out.append("meta_json: пусто или неверный тип")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="validate place_semantics/place_semantics.npz")
    p.add_argument("npz_path")
    p.add_argument("--qa", action="store_true", help="Плоский meta + tabular (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Обязательные ключи и согласованность N/T/K.")
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
