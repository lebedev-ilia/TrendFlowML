#!/usr/bin/env python3
"""Валидатор car_semantics NPZ: схема, --struct, --qa (view_csv_feature_qa.json)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_K = 5
_SCHEMA = "car_semantics_npz_v2"

_REQUIRED = (
    "frame_indices",
    "times_s",
    "semantic_label_names",
    "semantic_object_ids",
    "semantic_label_make",
    "semantic_label_model",
    "threshold_per_label_arr",
    "track_ids",
    "track_present_mask",
    "track_topk_ids",
    "track_topk_scores",
    "track_is_confident_top1",
    "track_best_frame_pos",
    "track_best_det_idx",
    "track_best_bbox_xyxy",
    "track_best_det_score",
    "track_best_class_id",
    "det_present_mask",
    "det_topk_ids",
    "det_topk_scores",
    "det_is_confident_top1",
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
        return _SCHEMA in sv
    except Exception:
        return False


def _label_ids(sem: np.ndarray) -> Set[int]:
    s: Set[int] = set()
    for x in np.asarray(sem, dtype=object).ravel():
        ss = str(x)
        if ":" not in ss:
            continue
        try:
            s.add(int(ss.split(":", 1)[0].strip()))
        except Exception:
            continue
    return s


def _check_scores_01(name: str, arr: np.ndarray, out: List[str]) -> None:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return
    if float(v.min()) < -0.0001 or float(v.max()) > 1.0001:
        out.append(f"{name}: значения вне [0,1] (ES similarity)")


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    fi = np.asarray(d["frame_indices"], dtype=np.int32).ravel()
    ts = np.asarray(d["times_s"], dtype=np.float32).ravel()
    if fi.size != ts.size:
        out.append(f"len(frame_indices)={fi.size} != len(times_s)={ts.size}")
    n = int(fi.size)

    sem = np.asarray(d["semantic_label_names"], dtype=object)
    oid = np.asarray(d["semantic_object_ids"], dtype=object)
    smk = np.asarray(d["semantic_label_make"], dtype=object)
    smd = np.asarray(d["semantic_label_model"], dtype=object)
    thr = np.asarray(d["threshold_per_label_arr"], dtype=np.float32).ravel()
    a = int(sem.size)
    if a > 0 and int(oid.size) != a:
        out.append(f"len(semantic_object_ids)={oid.size} != A={a}")
    if a > 0 and int(smk.size) != a:
        out.append(f"len(semantic_label_make)={smk.size} != A={a}")
    if a > 0 and int(smd.size) != a:
        out.append(f"len(semantic_label_model)={smd.size} != A={a}")
    if a > 0 and thr.size != a:
        out.append(f"len(threshold_per_label_arr)={thr.size} != A={a}")

    lids = _label_ids(sem)

    tr = np.asarray(d["track_ids"], dtype=np.int32).ravel()
    t = int(tr.size)
    tm = np.asarray(d["track_present_mask"], dtype=bool).ravel()
    if tm.size != t:
        out.append(f"track_present_mask len={tm.size} != T={t}")

    tids = np.asarray(d["track_topk_ids"], dtype=np.int32)
    tsc = np.asarray(d["track_topk_scores"], dtype=np.float32)
    if tids.shape != (t, _K) or tsc.shape != (t, _K):
        out.append(f"track_topk: ож. (T={t}, {_K}), ids={tids.shape}, sc={tsc.shape}")
    _check_scores_01("track_topk_scores", tsc, out)

    det_pr = np.asarray(d["det_present_mask"], dtype=bool)
    did = np.asarray(d["det_topk_ids"], dtype=np.int32)
    dsc = np.asarray(d["det_topk_scores"], dtype=np.float32)
    m = int(det_pr.shape[1]) if det_pr.ndim == 2 else 0
    if det_pr.shape[0] != n:
        out.append(f"det_present: N={det_pr.shape[0]} != {n}")
    if did.shape != (n, m, _K) or dsc.shape != (n, m, _K):
        out.append(f"det_topk: ож. ({n},{m},{_K}), ids={did.shape}, sc={dsc.shape}")
    _check_scores_01("det_topk_scores", dsc, out)

    fid = np.asarray(d["frame_topk_ids"], dtype=np.int32)
    fsc = np.asarray(d["frame_topk_scores"], dtype=np.float32)
    if fid.shape != (n, _K) or fsc.shape != (n, _K):
        out.append(f"frame_topk: ож. ({n},{_K}), ids={fid.shape}, sc={fsc.shape}")
    _check_scores_01("frame_topk_scores", fsc, out)

    for name, arr in (
        ("frame_topk_ids", fid),
        ("track_topk_ids", tids),
        ("det_topk_ids", did),
    ):
        for x in np.asarray(arr, dtype=np.int32).ravel():
            xi = int(x)
            if xi < -1:
                out.append(f"{name}: id {xi} < -1")
            if xi >= 0 and lids and xi not in lids:
                out.append(f"{name}: id {xi} вне label space")

    mj = d.get("meta_json")
    s = mj.item() if isinstance(mj, np.ndarray) and getattr(mj, "shape", None) == () else mj
    if not isinstance(s, str) or not str(s).strip():
        out.append("meta_json: пусто или неверный тип")
    else:
        try:
            o = json.loads(s)
            if not isinstance(o, dict):
                out.append("meta_json: JSON не object")
        except Exception as e:
            out.append(f"meta_json: не JSON: {e}")

    return out


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
    comp = "car_semantics"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def main() -> int:
    p = argparse.ArgumentParser(description="validate car_semantics NPZ (VisualProcessor)")
    p.add_argument("npz_path")
    p.add_argument("--qa", action="store_true")
    p.add_argument("--struct", action="store_true")
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
