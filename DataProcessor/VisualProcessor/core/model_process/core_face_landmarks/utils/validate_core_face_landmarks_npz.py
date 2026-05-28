#!/usr/bin/env python3
"""Валидатор core_face_landmarks/landmarks.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "version",
    "created_at",
    "model_name",
    "total_frames",
    "frame_indices",
    "times_s",
    "face_landmarks",
    "face_landmarks_raw",
    "face_present",
    "person_present",
    "face_mesh_ran",
    "has_any_face",
    "has_any_pose",
    "has_any_hands",
    "empty_reason",
    "face_empty_reason",
    "pose_empty_reason",
    "hands_empty_reason",
    "meta",
)

_N_AXIS1 = ("times_s", "person_present", "face_mesh_ran")


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
        return "core_face_landmarks_npz_v2" in sv or "core_face_landmarks" in sv
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
    comp = "core_face_landmarks"
    rmap = qa.rules_for_column(comp)
    for col in sorted(rmap.keys()):
        if col not in flat:
            continue
        raw = str(flat[col])
        w = qa.warning_for(comp, col, raw)
        if w:
            warnings.append(f"{col}: {w}")
    return warnings


def _check_axis_n(name: str, a: np.ndarray, n: int) -> bool:
    a = np.asarray(a)
    if a.ndim < 1:
        return False
    return int(a.shape[0]) == n


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

    for k in _N_AXIS1:
        a = np.asarray(d[k])
        if a.ndim < 1 or int(a.shape[0]) != n:
            out.append(f"{k}: ожидается длина N={n}, shape={getattr(a, 'shape', None)}")

    fl = np.asarray(d["face_landmarks"], dtype=np.float32)
    fr = np.asarray(d["face_landmarks_raw"], dtype=np.float32)
    if fl.shape != fr.shape:
        out.append("face_landmarks / face_landmarks_raw: разный shape")
    if fl.ndim != 4 or int(fl.shape[0]) != n or int(fl.shape[2]) != 468 or int(fl.shape[3]) != 3:
        out.append(
            f"face_landmarks: ожидается (N, F, 468, 3), N={n}, факт {getattr(fl, 'shape', None)}"
        )
    f = int(fl.shape[1]) if fl.ndim == 4 else 0
    fp = np.asarray(d["face_present"])
    if fp.ndim != 2 or int(fp.shape[0]) != n or int(fp.shape[1]) != f:
        out.append("face_present: ожидается (N, F) с тем же F, что и face_landmarks")

    for key in ("pose_landmarks", "pose_landmarks_raw"):
        if key not in d:
            continue
        if not _check_axis_n(key, d[key], n):
            out.append(f"{key}: первая ось != N={n}")
    if "pose_present" in d:
        pp = np.asarray(d["pose_present"], dtype=bool).reshape(-1)
        if int(pp.size) != n:
            out.append("pose_present: len != N")

    for key in ("hands_landmarks", "hands_landmarks_raw"):
        if key not in d:
            continue
        if not _check_axis_n(key, d[key], n):
            out.append(f"{key}: первая ось != N={n}")
    if "hands_present" in d:
        hp = np.asarray(d["hands_present"], dtype=bool)
        if hp.ndim != 2 or int(hp.shape[0]) != n:
            out.append("hands_present: ожидается (N, H) с N как у frame_indices")

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


def _as_scalar_bool(x: Any) -> bool:
    a = np.asarray(x)
    if a.shape == ():
        return bool(a.item())
    if a.size == 1:
        return bool(np.asarray(a, dtype=bool).reshape(-1)[0])
    return bool(a)


def validate_ranges(npz_path: str) -> List[str]:
    """
    Типичные инварианты (см. docs/FEATURE_DESCRIPTION.md): has_any_face vs face_present,
    NaN-политика для face_landmarks, согласованность pose/hands при наличии массивов.
    """
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).size)
    fl = np.asarray(d["face_landmarks"], dtype=np.float64)
    fp = np.asarray(d["face_present"], dtype=bool)
    if n == 0 and _as_scalar_bool(d.get("has_any_face")):
        out.append("has_any_face: при N=0 ожидается false")
    if n > 0 and fl.ndim == 4 and fp.ndim == 2 and fp.shape == fl.shape[:2]:
        if np.any(fp) != _as_scalar_bool(d.get("has_any_face")):
            out.append("has_any_face != np.any(face_present)")
        absent = ~fp
        if absent.any() and np.isfinite(fl[absent]).any():
            out.append("face_landmarks: при face_present=0 ожидаются NaN (filtered)")
        present = fp
        if present.any() and not np.isfinite(fl[present]).all():
            out.append("face_landmarks: при face_present=1 ожидаются конечные координаты")

    for flag_key, present_key in (
        ("has_any_pose", "pose_present"),
        ("has_any_hands", "hands_present"),
    ):
        if present_key not in d:
            continue
        h = d.get(flag_key)
        pr = np.asarray(d[present_key], dtype=bool)
        any_pr = bool(np.any(pr))
        if any_pr != _as_scalar_bool(h):
            out.append(f"{flag_key} != np.any({present_key})")

    fmc = meta.get("face_mesh_frames_count")
    if fmc is not None and n >= 0:
        try:
            fmc_i = int(fmc)
        except (TypeError, ValueError):
            fmc_i = -1
        if fmc_i < 0 or fmc_i > n:
            out.append("meta.face_mesh_frames_count вне [0, N] кадров выборки")

    tf = meta.get("total_frames")
    if tf is not None and n:
        try:
            if n > int(tf) >= 0:
                out.append("len(frame_indices) > meta.total_frames")
        except (TypeError, ValueError):
            pass

    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if int(ts.size) == n and n > 1:
        if np.any(np.diff(ts) < -1e-4):
            out.append("times_s: не неубывающий ряд (подсортировка union)")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("core_face_landmarks/landmarks.npz")):
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
        description="validate core_face_landmarks/landmarks.npz (core_face_landmarks_npz_v2)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к landmarks.npz (если не задан --results-base)",
    )
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, формы face/pose/hands.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="has_any_* vs present*, NaN-политика face_landmarks, times_s, face_mesh_frames_count, N vs total_frames (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/core_face_landmarks/landmarks.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube", help="Субкаталог платформы")
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
            fl = np.asarray(d0["face_landmarks"], dtype=np.float32)
            f = int(fl.shape[1]) if fl.ndim == 4 else 0
            print(
                f"✅ Structure OK (N={n}, F={f}, 468, core_face_landmarks_npz_v2; pose/hands опц.)"
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
                "✅ Ranges OK (flags, NaN-политика, times_s, face_mesh_frames_count, total_frames)"
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
