#!/usr/bin/env python3
"""Валидатор core_object_detections/detections.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_MAX_DETECTIONS = 100
_N_CLASS_NAMES = 41  # таксономия v1: class_id 0..40, 41 строка id:name
_REQUIRED = (
    "meta",
    "meta_json",
    "frame_indices",
    "times_s",
    "boxes",
    "boxes_norm",
    "centers_norm",
    "areas_frac",
    "scores",
    "class_ids",
    "valid_mask",
    "class_names",
    "det_count",
    "person_count",
    "text_region_count",
    "logo_region_count",
    "sum_person_area_frac",
    "max_person_area_frac",
    "sum_text_area_frac",
    "max_text_area_frac",
    "sum_logo_area_frac",
    "max_logo_area_frac",
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
        return "core_object_detections_npz_v2" in sv or "core_object_detections" in sv
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
    comp = "core_object_detections"
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
    ts = np.asarray(d["times_s"], dtype=np.float32).reshape(-1)
    n = int(fi.size)
    if n != int(ts.size):
        out.append(f"len(frame_indices)={fi.size} != len(times_s)={ts.size}")

    box = np.asarray(d["boxes"], dtype=np.float32)
    if (
        box.ndim != 3
        or int(box.shape[0]) != n
        or int(box.shape[1]) != _MAX_DETECTIONS
        or int(box.shape[2]) != 4
    ):
        out.append(
            f"boxes: ожидается (N={n},{_MAX_DETECTIONS},4), факт {getattr(box, 'shape', None)}"
        )

    for name, need_ndim in [
        ("scores", 2),
        ("class_ids", 2),
        ("valid_mask", 2),
        ("boxes_norm", 3),
        ("centers_norm", 3),
        ("areas_frac", 2),
    ]:
        a = np.asarray(d[name])
        if a.ndim != need_ndim or int(a.shape[0]) != n or int(a.shape[1]) != _MAX_DETECTIONS:
            out.append(f"{name}: неверная форма относительно N/M")

    cn = np.asarray(d["class_names"], dtype=object)
    if cn.size != _N_CLASS_NAMES:
        out.append(
            f"class_names: ожидается {_N_CLASS_NAMES} записей (0..40), факт {cn.size}"
        )

    for one_d in (
        "det_count",
        "person_count",
        "text_region_count",
        "logo_region_count",
        "sum_person_area_frac",
        "max_person_area_frac",
        "sum_text_area_frac",
        "max_text_area_frac",
        "sum_logo_area_frac",
        "max_logo_area_frac",
    ):
        a = np.asarray(d[one_d]).reshape(-1)
        if int(a.size) != n:
            out.append(f"{one_d}: len={a.size} != N={n}")

    mj = d.get("meta_json")
    if mj is not None:
        if isinstance(mj, np.ndarray):
            s = (
                mj.item()
                if getattr(mj, "shape", ()) == ()
                else np.asarray(mj).reshape(-1)[0]
            )
        else:
            s = mj
        if not isinstance(s, str) or not str(s).strip():
            out.append("meta_json: пусто или неверный тип")

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
    """Согласованность счётчиков и диапазоны (см. docs/FEATURE_DESCRIPTION.md)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    vm = np.asarray(d["valid_mask"], dtype=bool)
    if vm.ndim != 2:
        return [f"valid_mask: ожидается 2D, факт {getattr(vm, 'ndim', None)}"]
    n, m = int(vm.shape[0]), int(vm.shape[1])
    if m != _MAX_DETECTIONS:
        out.append(f"valid_mask: M={m} != {_MAX_DETECTIONS}")

    dc = np.asarray(d["det_count"], dtype=np.int32).ravel()
    srow = np.sum(vm, axis=1).astype(np.int32)
    if int(dc.size) != n or int(srow.size) != n:
        out.append("det_count: длина != N")
    elif n > 0 and not np.array_equal(dc, srow):
        out.append("det_count != sum(valid_mask, axis=1)")

    scores = np.asarray(d["scores"], dtype=np.float64)
    cid = np.asarray(d["class_ids"], dtype=np.int64)
    if vm.any():
        if np.any(scores[vm] < -1e-3) or np.any(scores[vm] > 1.0 + 1e-3):
            out.append("scores: вне [0, 1] на valid_mask")
        c_v = cid[vm]
        if c_v.size and (int(np.min(c_v)) < 0 or int(np.max(c_v)) > 40):
            out.append("class_ids: вне 0..40 на valid_mask")

    bn = np.asarray(d["boxes_norm"], dtype=np.float64)
    if np.any(bn[vm] < -1e-3) or np.any(bn[vm] > 1.0 + 1e-3):
        out.append("boxes_norm: компонент вне [0, 1] на valid_mask")

    cnorm = np.asarray(d["centers_norm"], dtype=np.float64)
    if np.any(cnorm[vm, :2] < -1e-3) or np.any(cnorm[vm, :2] > 1.0 + 1e-3):
        out.append("centers_norm: вне [0, 1] на valid_mask")

    af = np.asarray(d["areas_frac"], dtype=np.float64)
    if np.any(af[vm] < -1e-3) or np.any(af[vm] > 1.0 + 1e-3):
        out.append("areas_frac: вне [0, 1] на valid_mask")

    for name in (
        "person_count",
        "text_region_count",
        "logo_region_count",
    ):
        a = np.asarray(d[name], dtype=np.int32).ravel()
        if int(a.size) == n and n > 0:
            if np.any(a > dc):
                out.append(f"{name}: > det_count")
            if np.any(a < 0):
                out.append(f"{name}: отрицательное значение")

    for name in (
        "sum_person_area_frac",
        "max_person_area_frac",
        "sum_text_area_frac",
        "max_text_area_frac",
        "sum_logo_area_frac",
        "max_logo_area_frac",
    ):
        a = np.asarray(d[name], dtype=np.float64).ravel()
        if int(a.size) == n and n > 0:
            if np.any(a < -1e-3) or np.any(a > 1.0 + 1e-3):
                out.append(f"{name}: вне [0, 1]")

    bt = meta.get("box_threshold")
    if isinstance(bt, (int, float)) and (bt < -1e-3 or bt > 1.0 + 1e-3):
        out.append("meta.box_threshold: вне [0, 1]")

    tf = meta.get("total_frames")
    if tf is not None and n:
        try:
            if n > int(tf) >= 0:
                out.append("len(frame_indices) > meta.total_frames")
        except (TypeError, ValueError):
            pass

    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if int(ts.size) == n and n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    td = meta.get("total_detections")
    if td is not None and n >= 0:
        try:
            tdi = int(td)
        except (TypeError, ValueError):
            tdi = -1
        if tdi >= 0:
            sum_mask = int(np.sum(vm))
            if tdi != sum_mask:
                out.append("meta.total_detections != sum(valid_mask)")

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("core_object_detections/detections.npz")):
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
        description="validate core_object_detections/detections.npz (core_object_detections_npz_v2)"
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        help="Путь к detections.npz (если не задан --results-base)",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/core_object_detections/detections.npz",
    )
    p.add_argument("--platform-id", type=str, default="youtube")
    p.add_argument(
        "--struct",
        action="store_true",
        help="Ключи NPZ, N, M=100, class_names=41, meta_json",
    )
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Счётчики, [0,1] на valid_mask, class 0..40, times_s, total_detections (см. docs).",
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
            print(
                f"✅ Structure OK (N={n}, M={_MAX_DETECTIONS}, class_names={_N_CLASS_NAMES}, core_object_detections_npz_v2)"
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
                "✅ Ranges OK (det_count, scores/geom, class 0..40, times_s, total_detections, total_frames)"
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
