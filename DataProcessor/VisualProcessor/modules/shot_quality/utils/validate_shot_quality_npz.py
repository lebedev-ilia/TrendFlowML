#!/usr/bin/env python3
"""Валидатор shot_quality/shot_quality.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "frame_indices",
    "times_s",
    "feature_names",
    "frame_features",
    "frame_feature_present_ratio",
    "quality_probs",
    "shot_ids",
    "shot_start_frame",
    "shot_end_frame",
    "shot_frame_count",
    "shot_features_mean",
    "shot_features_std",
    "shot_features_min",
    "shot_features_max",
    "shot_frame_feature_present_ratio",
    "shot_quality_topk_ids",
    "shot_quality_topk_probs",
    "shot_quality_conf_mean",
    "shot_quality_entropy_mean",
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
        return "shot_quality_npz_v3" in sv or (
            "shot_quality" in sv and "npz" in sv
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
    warnings: List[str] = []
    comp = "shot_quality"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _validate_data_dict(
    d: Dict[str, Any], meta: Dict[str, Any]
) -> List[str]:
    out: List[str] = []
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out
    st = str(meta.get("status", "") or "")
    if st and st not in ("ok", "empty", "error"):
        out.append(f"meta.status неожидан: {st!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int32).reshape(-1)
    n = int(fi.size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if n != int(ts.size):
        out.append(f"len(times_s)={ts.size} != N={n}")
    if n < 1:
        out.append("контракт: N >= 1 (семпл не пустой)")

    fn = np.asarray(d["feature_names"], dtype=object).reshape(-1)
    f = int(fn.size)
    ff = np.asarray(d["frame_features"], dtype=np.float32)
    if ff.ndim != 2 or int(ff.shape[0]) != n or int(ff.shape[1]) != f:
        out.append(
            f"frame_features: ожидается (N={n}, F={f}), факт {getattr(ff, 'shape', None)}"
        )

    fpr = np.asarray(d["frame_feature_present_ratio"], dtype=np.float32).reshape(-1)
    if int(fpr.size) != f:
        out.append("frame_feature_present_ratio: len != F")

    qp = np.asarray(d["quality_probs"], dtype=np.float16)
    if qp.ndim != 2 or int(qp.shape[0]) != n:
        out.append("quality_probs: ожидается (N, P)")
    p = int(qp.shape[1]) if qp.ndim == 2 else 0
    if p >= 1 and n >= 1:
        row_sums = np.sum(qp.astype(np.float64, copy=False), axis=1)
        if np.any(~np.isfinite(row_sums)):
            out.append("quality_probs: не все строки конечны")
        else:
            dev = float(np.max(np.abs(row_sums - 1.0)))
            if dev > 0.02:
                out.append(
                    f"quality_probs: max|row_sum-1|={dev:.4f} (ожидается softmax, допуск 0.02)"
                )

    sid = np.asarray(d["shot_ids"], dtype=np.int32).reshape(-1)
    if int(sid.size) != n:
        out.append("shot_ids: len != N")
    s_st = np.asarray(d["shot_start_frame"], dtype=np.int32).reshape(-1)
    s_en = np.asarray(d["shot_end_frame"], dtype=np.int32).reshape(-1)
    s_fc = np.asarray(d["shot_frame_count"], dtype=np.int32).reshape(-1)
    s = int(s_st.size)
    if s != int(s_en.size) or s != int(s_fc.size):
        out.append("shot_start_frame / shot_end_frame / shot_frame_count: разные длины")
    if s >= 1 and int(sid.size) >= 1:
        mx = int(np.max(sid))
        mn = int(np.min(sid))
        if mn < 0 or mx > s - 1:
            out.append(f"shot_ids: вне [0, S-1] при S={s} (min={mn}, max={mx})")

    for name in (
        "shot_features_mean",
        "shot_features_std",
        "shot_features_min",
        "shot_features_max",
        "shot_frame_feature_present_ratio",
    ):
        a = np.asarray(d[name], dtype=np.float32)
        if a.ndim != 2 or int(a.shape[0]) != s or int(a.shape[1]) != f:
            out.append(f"{name}: ожидается (S={s}, F={f})")

    tk = np.asarray(d["shot_quality_topk_ids"], dtype=np.int32)
    tkp = np.asarray(d["shot_quality_topk_probs"], dtype=np.float32)
    if tk.ndim != 2 or int(tk.shape[0]) != s:
        out.append("shot_quality_topk_ids: ожидается (S, K)")
    if tkp.shape != tk.shape:
        out.append("shot_quality_topk_probs: shape != shot_quality_topk_ids")

    for name in ("shot_quality_conf_mean", "shot_quality_entropy_mean"):
        a = np.asarray(d[name], dtype=np.float32).reshape(-1)
        if int(a.size) != s:
            out.append(f"{name}: len != S={s}")

    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    st = m.get("status", "")
    if st == "error":
        return [
            f"meta.status=error: {m.get('empty_reason')!r} (struct не валидирует payload)"
        ]
    return _validate_data_dict(d, m)


def validate_ranges(npz_path: str) -> List[str]:
    """См. docs/FEATURE_DESCRIPTION.md: time axis, meta processed/total, мягкие bound по prob/conf."""
    out: List[str] = []
    d = load_npz(npz_path)
    m = extract_meta(d)
    if m.get("status") == "error":
        return out

    fi = np.asarray(d.get("frame_indices"), dtype=np.int32).ravel()
    ts = np.asarray(d.get("times_s"), dtype=np.float32).ravel()
    n = int(fi.size)
    if n > 1 and int(ts.size) == n and np.any(np.diff(ts.astype(np.float64)) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    tf = m.get("total_frames")
    pf = m.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            if int(pf) > int(tf) >= 0:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass

    tkp = np.asarray(d.get("shot_quality_topk_probs"), dtype=np.float32)
    if tkp.size and np.isfinite(tkp).any():
        t = tkp[np.isfinite(tkp)]
        if np.any(t < -0.01) or np.any(t > 1.01):
            out.append(
                f"shot_quality_topk_probs: вне [0,1] (finite), min={float(np.min(t)):.4f} max={float(np.max(t)):.4f}"
            )
    cfm = np.asarray(d.get("shot_quality_conf_mean"), dtype=np.float32).ravel()
    if cfm.size and np.isfinite(cfm).any():
        t = cfm[np.isfinite(cfm)]
        if np.any(t < -0.01) or np.any(t > 1.01):
            out.append(
                f"shot_quality_conf_mean: вне [0,1] (finite), min={float(np.min(t)):.4f} max={float(np.max(t)):.4f}"
            )
    ent = np.asarray(d.get("shot_quality_entropy_mean"), dtype=np.float32).ravel()
    if ent.size and np.isfinite(ent).any():
        t = ent[np.isfinite(ent)]
        if np.any(t < -0.01):
            out.append(
                f"shot_quality_entropy_mean: отрицательные значения, min={float(np.min(t)):.4f}"
            )
    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("shot_quality/shot_quality.npz")):
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
        description="validate shot_quality/shot_quality.npz (shot_quality_npz_v3)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к shot_quality.npz (если не задан --results-base)",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/shot_quality/shot_quality.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube")
    p.add_argument(
        "--struct",
        action="store_true",
        help="Ключи, N/S/F/P/K, согласованность, softmax по строкам quality_probs",
    )
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Ось times_s, meta processed≤total, topk/conf∈[0,1], entropy≥0 (softmax в --struct).",
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
            n = int(np.asarray(d0["frame_indices"]).ravel().size)
            s = int(np.asarray(d0["shot_start_frame"]).ravel().size)
            f = int(np.asarray(d0["feature_names"], dtype=object).ravel().size)
            qp = np.asarray(d0["quality_probs"])
            p_ = int(qp.shape[1]) if qp.ndim == 2 else 0
            tki = np.asarray(d0.get("shot_quality_topk_ids"), dtype=np.int32)
            k = int(tki.shape[1]) if tki.ndim == 2 and s else 0
            print(
                f"✅ Structure OK (N={n}, S={s}, F={f}, P={p_}, K={k}, shot_quality_npz_v3)"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (time axis, meta, topk/conf/entropy)")

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
