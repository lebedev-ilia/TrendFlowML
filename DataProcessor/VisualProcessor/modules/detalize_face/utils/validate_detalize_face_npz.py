#!/usr/bin/env python3
"""Валидатор detalize_face/detalize_face.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_COMPACT_DIM = 40
_SCHEMA = "detalize_face_npz_v3"
_ARTIFACT = "detalize_face.npz"

_REQUIRED = (
    "summary",
    "frame_indices",
    "times_s",
    "face_present",
    "processed_mask",
    "primary_valid",
    "face_count",
    "primary_tracking_id",
    "primary_compact_features",
    "aggregated",
    "faces_agg",
    "meta",
)

_OPTIONAL_PRIMARY_CURVES = (
    "primary_gaze_at_camera_prob",
    "primary_blink_rate",
    "primary_attention_score",
    "primary_quality_proxy_score",
    "primary_face_sharpness",
    "primary_occlusion_proxy",
    "primary_speech_activity_prob",
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
        return "detalize_face_npz_v3" in sv or "detalize_face" in sv
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
    comp = "detalize_face"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _unbox(a: Any) -> Any:
    if isinstance(a, np.ndarray) and a.dtype == object and getattr(a, "shape", None) == ():
        try:
            return a.item()
        except Exception:
            return a
    return a


def _validate_data_dict(d: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k in _REQUIRED:
        if k not in d:
            out.append(f"отсутствует ключ: {k}")
    if out:
        return out

    stv = meta.get("status")
    if stv is not None:
        s = str(stv)
        if s not in ("ok", "empty", "error"):
            out.append(f"meta.status неожидан: {s!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int32).reshape(-1)
    n = int(fi.size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if n != int(ts.size):
        out.append(f"len(times_s)={ts.size} != N={n}")

    for k in ("face_present", "processed_mask", "primary_valid", "face_count", "primary_tracking_id"):
        a = np.asarray(d[k])
        if int(a.shape[0]) != n:
            out.append(f"{k}: len по оси 0 != N={n}")

    pcf = np.asarray(d["primary_compact_features"], dtype=np.float32)
    if pcf.ndim != 2 or int(pcf.shape[0]) != n or int(pcf.shape[1]) != _COMPACT_DIM:
        out.append(
            f"primary_compact_features: ожидается (N, {_COMPACT_DIM}), факт {getattr(pcf, 'shape', None)}"
        )

    agg = _unbox(d["aggregated"])
    if not isinstance(agg, dict):
        out.append("aggregated: ожидается dict (boxed)")

    fagg = _unbox(d["faces_agg"])
    if not isinstance(fagg, dict):
        out.append("faces_agg: ожидается dict (boxed)")

    sm = _unbox(d["summary"])
    if not isinstance(sm, dict):
        out.append("summary: ожидается dict (boxed)")

    for k in _OPTIONAL_PRIMARY_CURVES:
        if k not in d:
            continue
        a = np.asarray(d[k], dtype=np.float32).reshape(-1)
        if int(a.size) != n:
            out.append(f"{k}: len != N={n}")

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
    """См. docs/FEATURE_DESCRIPTION.md."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).ravel().size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if n > 1 and int(ts.size) == n and np.any(np.diff(np.asarray(ts, dtype=np.float64)) < -1e-4):
        out.append("times_s: не неубывающий ряд")

    fc = np.asarray(d["face_count"], dtype=np.float64).reshape(-1)
    if int(fc.size) == n and n:
        m = np.isfinite(fc)
        if m.any() and np.min(fc[m]) < -1e-3:
            out.append("face_count: отрицательные finite (не ожидается)")

    er = meta.get("empty_reason")
    st = str(meta.get("status", ""))
    if st == "empty":
        ers = str(er).strip() if er is not None else ""
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
    # meta.processed_frames = sum(processed_mask), не len(frame_indices) (см. detalize_face_refactored.run)
    pm = np.asarray(d["processed_mask"])
    try:
        exp_pf = int(np.sum(pm.astype(bool, copy=False).reshape(-1)))
    except Exception:
        exp_pf = -1
    if pf is not None and exp_pf >= 0:
        try:
            if int(pf) != exp_pf:
                out.append(
                    f"meta.processed_frames={pf} != sum(processed_mask)={exp_pf}"
                )
        except (TypeError, ValueError):
            pass
    if n >= 1 and exp_pf > n:
        out.append("sum(processed_mask) > N=len(frame_indices)")

    stm = meta.get("stage_timings_ms")
    if isinstance(stm, dict):
        for k, v in stm.items():
            if isinstance(v, (int, float)) and v < -1e-3:
                out.append(f"meta.stage_timings_ms[{k!r}]: отрицательное значение")

    sm = _unbox(d.get("summary"))
    if isinstance(sm, dict):
        st2 = sm.get("stage_timings_ms")
        if isinstance(st2, dict):
            for k, v in st2.items():
                if isinstance(v, (int, float)) and v < -1e-3:
                    out.append(f"summary.stage_timings_ms[{k!r}]: отрицательное значение")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    c = 0
    for npz in sorted(root.rglob(f"detalize_face/{_ARTIFACT}")):
        c += 1
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
    print(f"Проверено файлов: {c}", flush=True)
    return ex if c else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=f"validate detalize_face/{_ARTIFACT} (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, (N,40), boxed dicts, опц. кривые.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="times_s, face_count, empty_reason, кадры meta, тайминги (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/detalize_face/{_ARTIFACT}",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
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
            n = int(np.asarray(d0["frame_indices"], dtype=np.int32).ravel().size)
            print(
                f"✅ Structure OK (N={n}, C={_COMPACT_DIM}, {_SCHEMA}, detalize_face/{_ARTIFACT})"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print(
                "✅ Ranges OK (times, face_count≥0, empty, sum(mask)=meta.processed, stage_timings, total)"
            )
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
