#!/usr/bin/env python3
"""Single-file валидатор core_face_identity/face_identity.npz: схема, --struct, --qa, --ranges."""
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
    "face_ids",
    "face_names",
    "face_similarities",
    "face_bbox_xyxy",
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
        return "core_face_identity_npz_v2" in sv or "core_face_identity" in sv
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
    comp = "core_face_identity"
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
    n = int(fi.size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if n != int(ts.size):
        out.append(f"len(times_s)={ts.size} != N={n}")

    sln = np.asarray(d["semantic_label_names"], dtype=object).reshape(-1)
    soi = np.asarray(d["semantic_object_ids"], dtype=object).reshape(-1)
    if int(sln.size) != int(soi.size):
        out.append("semantic_label_names / semantic_object_ids: разная длина A")

    fids = np.asarray(d["face_ids"], dtype=np.int32)
    fnm = np.asarray(d["face_names"])
    fsim = np.asarray(d["face_similarities"], dtype=np.float32)
    bbox = np.asarray(d["face_bbox_xyxy"], dtype=np.float32)

    if fids.ndim != 2:
        out.append("face_ids: ожидается (N, K)")
    else:
        n1, k = int(fids.shape[0]), int(fids.shape[1])
        if n1 != n:
            out.append(f"face_ids: N={n1} != len(frame_indices)={n}")
        if int(fnm.shape[0]) != n or int(fsim.shape[0]) != n:
            out.append("face_names / face_similarities: первая ось != N")
        if int(fnm.shape[1]) != k or int(fsim.shape[1]) != k:
            out.append("face_names / face_similarities: K не совпадает с face_ids")
    if bbox.ndim != 2 or int(bbox.shape[0]) != n or int(bbox.shape[1]) != 4:
        out.append("face_bbox_xyxy: ожидается (N, 4)")

    mj = d.get("meta_json")
    if isinstance(mj, np.ndarray) and getattr(mj, "shape", None) == ():
        _ = str(mj.item())
    else:
        out.append("meta_json: ожидается 0-d array/str")

    return out


def validate_ranges(npz_path: str) -> List[str]:
    """Типичные диапазоны: сходства [0,1], face_ids, K vs meta.top_k, bbox, times_s (см. docs/FEATURE_DESCRIPTION.md)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    try:
        tk = int(meta.get("top_k", -1))
    except (TypeError, ValueError):
        tk = -1

    A = int(np.asarray(d["semantic_label_names"], dtype=object).size)
    fsim = np.asarray(d["face_similarities"], dtype=np.float64)
    if fsim.size and np.isfinite(fsim).any():
        t = fsim[np.isfinite(fsim)]
        if np.any(t < -1e-3) or np.any(t > 1.0 + 1e-3):
            out.append("face_similarities: вне [0, 1] (finite)")

    fids = np.asarray(d["face_ids"], dtype=np.int64)
    if fids.size and A > 0:
        ok = (fids == -1) | ((fids >= 0) & (fids < A))
        if not np.all(ok):
            out.append("face_ids: ожидается -1 или индекс в [0, A) для label-space")
    elif fids.size and A == 0:
        if np.any(fids != -1):
            out.append("face_ids: при пустом label-space ожидается только -1")

    if fids.ndim == 2 and tk >= 0 and int(fids.shape[1]) != tk:
        out.append("meta.top_k != K (вторая ось face_*)")
    if fsim.ndim == 2 and tk >= 0 and int(fsim.shape[1]) != tk:
        out.append("face_similarities: K != meta.top_k")

    bbox = np.asarray(d["face_bbox_xyxy"], dtype=np.float64)
    if bbox.ndim == 2 and int(bbox.shape[1]) == 4 and int(bbox.shape[0]) > 0:
        bad_row = None
        for i in range(int(bbox.shape[0])):
            row = bbox[i]
            all_nan = bool(np.isnan(row).all())
            all_fin = bool(np.isfinite(row).all())
            if not (all_nan or all_fin):
                bad_row = "face_bbox_xyxy: строка должна быть полностью NaN или полностью конечной"
                break
            if all_fin:
                x1, y1, x2, y2 = map(float, row)
                if x2 < x1 - 1e-3 or y2 < y1 - 1e-3:
                    bad_row = "face_bbox_xyxy: x2>=x1, y2>=y1 (finite rows)"
                    break
        if bad_row:
            out.append(bad_row)

    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    n_ts = int(ts.size)
    if n_ts > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="validate core_face_identity/face_identity.npz")
    p.add_argument("npz_path")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, (N,K), label-space, meta_json.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="face_similarities [0,1], face_ids / bbox, K vs meta.top_k, times_s (см. docs/FEATURE_DESCRIPTION.md).",
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
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (sim, ids, K, bbox, times_s)")
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
