#!/usr/bin/env python3
"""Валидатор scene_classification/scene_classification_features.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_TOPK = 5
_SCHEMA = "scene_classification_npz_v2"
_ARTIFACT = "scene_classification_features.npz"

_REQUIRED = (
    "frame_indices",
    "times_s",
    "label_fusion",
    "min_scene_seconds",
    "frame_topk_ids",
    "frame_topk_probs",
    "frame_entropy",
    "frame_top1_prob",
    "frame_top1_top2_gap",
    "frame_scene_id",
    "scene_ids",
    "scene_label",
    "fusion_mode",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "length_frames",
    "length_seconds",
    "mean_score",
    "class_entropy_mean",
    "top1_prob_mean",
    "top1_vs_top2_gap_mean",
    "fraction_high_confidence_frames",
    "mean_aesthetic_score",
    "aesthetic_std",
    "aesthetic_frac_high",
    "mean_luxury_score",
    "mean_cozy",
    "mean_scary",
    "mean_epic",
    "mean_neutral",
    "atmosphere_entropy",
    "scene_change_score",
    "label_stability",
    "scenes",
    "scenes_raw",
    "scene_aesthetic_prompts",
    "scene_luxury_prompts",
    "scene_atmosphere_prompts",
    "places365_prompts",
    "indices",
    "dominant_places_topk_ids",
    "dominant_places_topk_probs",
    "summary",
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
        return "scene_classification_npz_v2" in sv or "scene_classification" in sv
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
    comp = "scene_classification"
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

    ftk = np.asarray(d["frame_topk_ids"], dtype=np.int32)
    ftp = np.asarray(d["frame_topk_probs"], dtype=np.float32)
    if ftk.ndim != 2 or int(ftk.shape[0]) != n or int(ftk.shape[1]) != _TOPK:
        out.append(f"frame_topk_ids: ожидается (N, {_TOPK})")
    if ftp.shape != ftk.shape:
        out.append("frame_topk_probs: shape != frame_topk_ids")

    for k in ("frame_entropy", "frame_top1_prob", "frame_top1_top2_gap", "frame_scene_id"):
        a = np.asarray(d[k]).reshape(-1)
        if int(a.size) != n:
            out.append(f"{k}: len != N={n}")
    fsi = np.asarray(d["frame_scene_id"], dtype=np.int32).ravel()
    if fsi.size and int(np.min(fsi)) < 0:
        out.append("frame_scene_id: отрицательный индекс сцены (контракт: >=0)")

    sid = np.asarray(d["scene_ids"], dtype=object).reshape(-1)
    s = int(sid.size)

    def _s1(name: str) -> None:
        a = np.asarray(d[name]).reshape(-1)
        if int(a.size) != s:
            out.append(f"{name}: len={a.size} != S={s}")

    for name in (
        "scene_label",
        "fusion_mode",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "length_frames",
        "length_seconds",
        "mean_score",
        "class_entropy_mean",
        "top1_prob_mean",
        "top1_vs_top2_gap_mean",
        "fraction_high_confidence_frames",
        "mean_aesthetic_score",
        "aesthetic_std",
        "aesthetic_frac_high",
        "mean_luxury_score",
        "mean_cozy",
        "mean_scary",
        "mean_epic",
        "mean_neutral",
        "atmosphere_entropy",
        "scene_change_score",
        "label_stability",
    ):
        if s > 0:
            _s1(name)

    for name in ("indices", "dominant_places_topk_ids", "dominant_places_topk_probs"):
        a = d[name]
        if not isinstance(a, np.ndarray) or int(a.shape[0]) != s:
            out.append(f"{name}: object-массив длины S={s} ожидается")

    if not isinstance(_unbox(d["scenes"]), dict):
        out.append("scenes: ожидается dict (boxed)")
    if not isinstance(_unbox(d["scenes_raw"]), dict):
        out.append("scenes_raw: ожидается dict (boxed)")

    su = _unbox(d["summary"])
    if not isinstance(su, dict):
        out.append("summary: ожидается dict (boxed)")

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
    """См. docs/FEATURE_DESCRIPTION.md: вероятности, время, кадры, тайминги."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).ravel().size)
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    if n > 1 and int(ts.size) == n and np.any(np.diff(np.asarray(ts, dtype=np.float64)) < -1e-4):
        out.append("times_s: не неубывающий ряд")

    msc = d.get("min_scene_seconds")
    if msc is not None:
        try:
            msci = float(msc)
            if np.isfinite(msci) and msci < -1e-4:
                out.append("min_scene_seconds: отрицательное (не ожидается)")
        except (TypeError, ValueError):
            out.append("min_scene_seconds: не число")

    ftp = np.asarray(d["frame_topk_probs"], dtype=np.float32).reshape(-1)
    m = np.isfinite(ftp)
    if m.any() and (np.min(ftp[m]) < -1e-3 or np.max(ftp[m]) > 1.0 + 1e-3):
        out.append("frame_topk_probs: finite вне [0, 1]")

    t1 = np.asarray(d["frame_top1_prob"], dtype=np.float64).reshape(-1)
    m1 = np.isfinite(t1)
    if m1.any() and (np.min(t1[m1]) < -1e-3 or np.max(t1[m1]) > 1.0 + 1e-3):
        out.append("frame_top1_prob: finite вне [0, 1]")

    fe = np.asarray(d["frame_entropy"], dtype=np.float64).reshape(-1)
    mf = np.isfinite(fe)
    if mf.any() and np.min(fe[mf]) < -1e-3:
        out.append("frame_entropy: отрицательные finite (не ожидается)")

    fhf = np.asarray(d["fraction_high_confidence_frames"], dtype=np.float64).reshape(-1)
    mh = np.isfinite(fhf)
    if mh.any() and (np.min(fhf[mh]) < -1e-3 or np.max(fhf[mh]) > 1.0 + 1e-3):
        out.append("fraction_high_confidence_frames: finite вне [0, 1]")

    afrac = np.asarray(d["aesthetic_frac_high"], dtype=np.float64).reshape(-1)
    ma = np.isfinite(afrac)
    if ma.any() and (np.min(afrac[ma]) < -1e-3 or np.max(afrac[ma]) > 1.0 + 1e-3):
        out.append("aesthetic_frac_high: finite вне [0, 1]")

    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            tfi, pfi = int(tf), int(pf)
            if tfi >= 0 and pfi > tfi:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass
    if pf is not None and n >= 1:
        try:
            if int(pf) != n:
                out.append(
                    f"meta.processed_frames={pf} != N={n} (ожидается len(frame_indices))"
                )
        except (TypeError, ValueError):
            pass

    stm = meta.get("stage_timings_ms")
    if isinstance(stm, dict):
        for k, v in stm.items():
            if isinstance(v, (int, float)) and v < -1e-3:
                out.append(f"meta.stage_timings_ms[{k!r}]: отрицательное значение")

    stms = _unbox(d.get("summary"))
    if isinstance(stms, dict):
        st2 = stms.get("stage_timings_ms")
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
    for npz in sorted(root.rglob(f"scene_classification/{_ARTIFACT}")):
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
        description=f"validate scene_classification/{_ARTIFACT} (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, S, (N,5) topk, scenes/summary.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Вероятности [0,1], times_s, min_scene_seconds, кадры meta, тайминги (см. docs).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/scene_classification/{_ARTIFACT}",
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
            sid = np.asarray(d0["scene_ids"], dtype=object).reshape(-1)
            s = int(sid.size)
            print(
                f"✅ Structure OK (N={n}, S={s}, K={_TOPK}, {_SCHEMA}, scene_classification/{_ARTIFACT})"
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
                "✅ Ranges OK (probs/доли [0,1], entropy≥0, times, meta/summary timings, N=processed)"
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
