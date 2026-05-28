#!/usr/bin/env python3
"""Валидатор high_level_semantic/high_level_semantic.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCHEMA = "high_level_semantic_npz_v2"
_ARTIFACT = "high_level_semantic.npz"

_REQUIRED = (
    "frame_indices",
    "times_s",
    "scene_id",
    "scene_embeddings",
    "scene_start_frame_idx",
    "scene_end_frame_idx",
    "scene_start_time_s",
    "scene_end_time_s",
    "scene_duration_s",
    "scene_representative_frame_idx",
    "scene_embedding_mean_norm",
    "frame_feature_names",
    "frame_features",
    "frame_feature_present_ratio",
    "event_times_s",
    "event_type_id",
    "event_strength",
    "event_frame_pos",
    "text_feature_names",
    "text_feature_values",
    "features",
    "ui",
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
        return _SCHEMA in sv or (
            "high_level_semantic" in sv.lower() and "npz" in sv.lower()
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
    tfn, tfv = d.get("text_feature_names"), d.get("text_feature_values")
    if tfn is not None and tfv is not None:
        try:
            tnames = [str(x) for x in np.asarray(tfn, dtype=object).ravel()]
            tvals = np.asarray(tfv, dtype=np.float64).ravel()
            for n, v in zip(tnames, tvals):
                flat[str(n)] = v
        except Exception:
            pass
    feat = d.get("features")
    if isinstance(feat, dict):
        for k, v in feat.items():
            if isinstance(v, (bool, int, float, np.floating, np.integer)):
                if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
                    continue
                flat[str(k)] = v
    warnings: List[str] = []
    comp = "high_level_semantic"
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

    if not meta:
        out.append("meta: пусто или неверный тип")
        return out

    stv = meta.get("status")
    if stv is not None:
        s = str(stv)
        if s not in ("ok", "empty", "error"):
            out.append(f"meta.status неожидан: {s!r}")

    fi = np.asarray(d["frame_indices"], dtype=np.int64).ravel()
    ts = np.asarray(d["times_s"], dtype=np.float64).ravel()
    if fi.size != ts.size:
        out.append(f"len(frame_indices)={fi.size} != len(times_s)={ts.size}")
    n = int(fi.size)
    if n > 1 and not np.all(np.diff(fi) > 0):
        out.append("frame_indices должен быть строго возрастать (unique sorted)")
    if n > 1 and np.any(np.diff(ts) < 0):
        out.append("times_s должен быть неубывающим")

    sid = np.asarray(d["scene_id"], dtype=np.int32).ravel()
    if sid.size != n:
        out.append(f"len(scene_id)={sid.size} != N={n}")

    se = d["scene_embeddings"]
    if not isinstance(se, np.ndarray) or se.ndim != 2:
        out.append(f"scene_embeddings: ож. 2D, shape={getattr(se, 'shape', None)}")
    else:
        nuniq = int(len(np.unique(sid)))
        if int(se.shape[0]) != nuniq:
            out.append(f"scene_embeddings rows S={se.shape[0]} != unique scenes={nuniq}")

    ff = np.asarray(d["frame_features"], dtype=np.float64)
    ffn = np.asarray(d["frame_feature_names"], dtype=object).ravel()
    if ff.ndim != 2:
        out.append(f"frame_features: ож. 2D, got shape {getattr(ff, 'shape', None)}")
    elif int(ff.shape[0]) != n:
        out.append(f"frame_features rows {ff.shape[0]} != N={n}")
    elif int(ff.shape[1]) != int(ffn.size):
        out.append(f"frame_feature_names F={ffn.size} != columns {ff.shape[1]}")

    ffpr = np.asarray(d["frame_feature_present_ratio"], dtype=np.float64).ravel()
    if int(ffpr.size) != int(ffn.size):
        out.append("frame_feature_present_ratio len != F")

    ets = np.asarray(d["event_times_s"], dtype=np.float64).ravel()
    eti = np.asarray(d["event_type_id"], dtype=np.int64).ravel()
    es = np.asarray(d["event_strength"], dtype=np.float64).ravel()
    efp = np.asarray(d["event_frame_pos"], dtype=np.int64).ravel()
    e = int(ets.size)
    if e != int(eti.size) or e != int(es.size) or e != int(efp.size):
        out.append("event_*: несовпадение длин (times_s, type_id, strength, frame_pos)")
    if e > 1 and np.any(np.diff(ets) < 0):
        out.append("event_times_s: ож. неубывающий порядок")
    for i in efp:
        if int(i) < 0 or (n > 0 and int(i) >= n):
            out.append("event_frame_pos: индекс вне 0..N-1")
            break

    tfn = np.asarray(d["text_feature_names"], dtype=object).ravel()
    tfv = np.asarray(d["text_feature_values"], dtype=np.float64).ravel()
    if tfn.size != tfv.size:
        out.append("text_feature_names/values: разная длина")

    feat = d.get("features")
    if not isinstance(feat, dict):
        out.append("features: ож. dict (scalar object в NPZ)")

    u = d.get("ui")
    if u is not None:
        u0 = _unbox(u) if isinstance(u, np.ndarray) else u
        if not isinstance(u0, dict):
            out.append("ui: ож. dict")

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
    """См. docs/FEATURE_DESCRIPTION.md: доли, кадры meta, силы событий, сцены."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    ffn = np.asarray(d["frame_feature_names"], dtype=object).ravel()
    ffpr = np.asarray(d["frame_feature_present_ratio"], dtype=np.float64).ravel()
    if int(ffpr.size) == int(ffn.size) and ffpr.size and (
        np.any(np.isfinite(ffpr) & (ffpr < -1e-4)) or np.any(np.isfinite(ffpr) & (ffpr > 1.0 + 1e-4))
    ):
        out.append("frame_feature_present_ratio: finite вне [0,1] (допуск)")

    n = int(np.asarray(d["frame_indices"], dtype=np.int64).ravel().size)
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

    sdu = np.asarray(d["scene_duration_s"], dtype=np.float64).ravel()
    if sdu.size and np.isfinite(sdu).any() and float(np.min(sdu[np.isfinite(sdu)])) < -1e-6:
        out.append("scene_duration_s: отрицательные finite (не ожидается)")

    es = np.asarray(d["event_strength"], dtype=np.float64).ravel()
    if es.size:
        m = np.isfinite(es)
        if m.any() and float(np.min(es[m])) < -1e-3:
            out.append("event_strength: отрицательные finite (не ожидается)")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    c = 0
    for npz in sorted(root.rglob(f"high_level_semantic/{_ARTIFACT}")):
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
        description=f"validate high_level_semantic/{_ARTIFACT} (schema {_SCHEMA}, VisualProcessor)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta + text/features (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, сцены, dense, event_*, features/ui.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="frame_feature_present_ratio, кадры meta, тайминги, scene_duration, event_strength (см. docs).",
    )
    p.add_argument(
        "--results-base",
        help=f"[батч] корень result_store; обход **/high_level_semantic/{_ARTIFACT}",
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
            n = int(np.asarray(d0["frame_indices"], dtype=np.int64).ravel().size)
            ffn = int(np.asarray(d0["frame_feature_names"], dtype=object).ravel().size)
            e = int(np.asarray(d0["event_times_s"], dtype=np.float64).ravel().size)
            se = d0.get("scene_embeddings")
            srows = int(np.asarray(se).shape[0]) if isinstance(se, np.ndarray) and se.ndim == 2 else 0
            print(
                f"✅ Structure OK (N={n}, F={ffn} dense, S={srows} scenes, E={e} events, {_SCHEMA}, high_level_semantic/{_ARTIFACT})"
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
                "✅ Ranges OK (present_ratio, meta frames, timings, scene_duration≥0, event_strength≥0)"
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
