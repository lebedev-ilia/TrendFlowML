#!/usr/bin/env python3
"""Валидатор cut_detection/cut_detection_features_*.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "frame_indices",
    "times_s",
    "features",
    "detections",
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
        return "cut_detection_npz_v1" in sv or "cut_detection" in sv
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


def _unbox_object(val: Any) -> Any:
    if isinstance(val, np.ndarray) and val.dtype == object and val.shape == ():
        try:
            return val.item()
        except Exception:
            return val
    return val


def validate_qa_rows(npz_path: str, qa: Any) -> List[str]:
    from qa.component_feature_qa import flatten_meta

    d = load_npz(npz_path)
    meta = extract_meta(d)
    flat: Dict[str, Any] = dict(flatten_meta(meta, prefix="meta_"))
    warnings: List[str] = []
    comp = "cut_detection"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _validate_data_dict(d: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
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
    if n < 2:
        out.append("контракт: N >= 2 (пара кадров для переходов)")

    for key in ("features", "detections"):
        v = _unbox_object(d[key])
        if not isinstance(v, dict):
            out.append(f"{key}: ожидается dict (boxed в NPZ)")

    if st == "ok" and isinstance(_unbox_object(d.get("detections")), dict):
        det = _unbox_object(d["detections"])
        if "shot_boundaries_frame_indices" not in det:
            out.append("detections: нет ключа shot_boundaries_frame_indices (нужен downstream, напр. shot_quality)")

    mfp = d.get("model_facing_npz_path")
    if mfp is not None:
        s = mfp
        if isinstance(mfp, np.ndarray):
            s = mfp.item() if getattr(mfp, "shape", None) in ((), (1,)) else str(mfp)
        if s is not None and str(s).strip() and not isinstance(s, (str, bytes)):
            out.append("model_facing_npz_path: неверный тип (ожидается str)")

    if not meta:
        out.append("meta пустой")
    return out


def validate_structure(npz_path: str) -> List[str]:
    d = load_npz(npz_path)
    m = extract_meta(d)
    if m.get("status") == "error":
        return [
            f"meta.status=error: {m.get('empty_reason')!r} (struct не валидирует payload)"
        ]
    return _validate_data_dict(d, m)


def validate_ranges(npz_path: str) -> List[str]:
    """См. docs/FEATURE_DESCRIPTION.md: ось времени, meta, мягкие неотрицательные счётчики в features."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).ravel().size)
    ts = np.asarray(d["times_s"], dtype=np.float64).ravel()
    if int(ts.size) == n and n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    tf = meta.get("total_frames")
    pf = meta.get("processed_frames")
    if tf is not None and pf is not None:
        try:
            if int(pf) > int(tf) >= 0:
                out.append("meta.processed_frames > meta.total_frames")
        except (TypeError, ValueError):
            pass
    if tf is not None and n:
        try:
            if n > int(tf) >= 0:
                out.append("len(frame_indices) > meta.total_frames")
        except (TypeError, ValueError):
            pass

    fe = _unbox_object(d.get("features"))
    if isinstance(fe, dict):
        for k, v in fe.items():
            if not str(k).endswith("_count"):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                if np.isfinite(v) and float(v) < -1e-3:
                    out.append(f"features.{k}: отрицательный scalar")

    return out


def _iter_feature_npz_paths(root: Path) -> List[Path]:
    """Основной артефакт: cut_detection/cut_detection_features_*.npz (не model_facing)."""
    out: List[Path] = []
    for p in root.rglob("*.npz"):
        if p.parent.name != "cut_detection":
            continue
        if not p.name.startswith("cut_detection_features_"):
            continue
        out.append(p)
    return sorted(out)


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in _iter_feature_npz_paths(root):
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
        description="validate cut_detection/cut_detection_features_*.npz (cut_detection_npz_v1)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к cut_detection_features_*.npz (если не задан --results-base)",
    )
    p.add_argument("--struct", action="store_true", help="Ключи, N, features/detections, shot_boundaries")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="times_s, meta N/processed, неотрицательные *count в features (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; **/cut_detection/cut_detection_features_*.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube")
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
            has_sb = False
            det = _unbox_object(d0.get("detections"))
            if isinstance(det, dict):
                has_sb = "shot_boundaries_frame_indices" in det
            print(
                f"✅ Structure OK (N={n}, features+detections, shot_boundaries={has_sb}, cut_detection_npz_v1)"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (times_s, meta, *count)")

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
