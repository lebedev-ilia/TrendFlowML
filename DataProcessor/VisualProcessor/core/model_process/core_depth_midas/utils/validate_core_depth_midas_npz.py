#!/usr/bin/env python3
"""Валидатор core_depth_midas/depth.npz: схема, --struct, --qa, --ranges; батч --results-base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REQUIRED = (
    "frame_indices",
    "times_s",
    "depth_maps",
    "depth_maps_norm",
    "depth_mean",
    "depth_std",
    "depth_p05",
    "depth_p95",
    "depth_range_robust",
    "depth_complexity_score",
    "foreground_background_separation_proxy",
    "preview_frame_indices",
    "preview_times_s",
    "preview_depth_maps",
    "preview_depth_maps_norm",
    "meta",
)

_N_PER_FRAME_1D = (
    "depth_mean",
    "depth_std",
    "depth_p05",
    "depth_p95",
    "depth_range_robust",
    "depth_complexity_score",
    "foreground_background_separation_proxy",
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
        return "core_depth_midas_npz_v3" in sv or "core_depth_midas" in sv
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
    comp = "core_depth_midas"
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
    if int(ts.size) != n:
        out.append(f"len(times_s)={ts.size} != N={n}")

    dm = np.asarray(d["depth_maps"], dtype=np.float32)
    dn = np.asarray(d["depth_maps_norm"], dtype=np.float32)
    if dm.ndim != 3 or int(dm.shape[0]) != n:
        out.append(f"depth_maps: ожидается (N,H,W), N={n}, факт {getattr(dm, 'shape', None)}")
    if dn.shape != dm.shape:
        out.append("depth_maps_norm: shape != depth_maps")

    for k in _N_PER_FRAME_1D:
        a = np.asarray(d[k], dtype=np.float32).reshape(-1)
        if int(a.size) != n:
            out.append(f"{k}: len != N={n}")

    pfi = np.asarray(d["preview_frame_indices"], dtype=np.int32).reshape(-1)
    pts = np.asarray(d["preview_times_s"], dtype=np.float32).reshape(-1)
    kprev = int(pfi.size)
    if int(pts.size) != kprev:
        out.append("preview_frame_indices / preview_times_s: разная длина")
    pdm = np.asarray(d["preview_depth_maps"], dtype=np.float32)
    pdn = np.asarray(d["preview_depth_maps_norm"], dtype=np.float32)
    if pdm.ndim != 3 or int(pdm.shape[0]) != kprev:
        out.append("preview_depth_maps: первая ось != K preview")
    if pdn.shape != pdm.shape:
        out.append("preview_depth_maps_norm: shape != preview_depth_maps")
    if kprev > 0 and dm.ndim == 3:
        h, w = int(dm.shape[1]), int(dm.shape[2])
        if int(pdm.shape[1]) != h or int(pdm.shape[2]) != w:
            out.append("preview maps H,W != depth_maps H,W")

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
    """Проверка типичных диапазонов (см. docs/FEATURE_DESCRIPTION.md)."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    def _check01(a: np.ndarray, name: str) -> None:
        if a.size and np.isfinite(a).any():
            t = a[np.isfinite(a)]
            if np.any(t < -1e-3) or np.any(t > 1.0 + 1e-3):
                out.append(f"{name}: вне [0, 1] (finite)")

    dn = np.asarray(d["depth_maps_norm"], dtype=np.float64)
    _check01(dn, "depth_maps_norm")
    pdn = np.asarray(d["preview_depth_maps_norm"], dtype=np.float64)
    _check01(pdn, "preview_depth_maps_norm")

    for name in ("depth_std", "depth_range_robust"):
        a = np.asarray(d[name], dtype=np.float64).reshape(-1)
        if a.size and np.isfinite(a).any():
            t = a[np.isfinite(a)]
            if np.any(t < -1e-3):
                out.append(f"{name}: отрицательные finite")

    p0 = np.asarray(d["depth_p05"], dtype=np.float64).reshape(-1)
    p9 = np.asarray(d["depth_p95"], dtype=np.float64).reshape(-1)
    if p0.size == p9.size and p0.size:
        m = np.isfinite(p0) & np.isfinite(p9)
        if m.any() and np.any(p0[m] > p9[m] + 1e-4):
            out.append("depth_p05: есть кадры с p05 > p95")

    dc = np.asarray(d["depth_complexity_score"], dtype=np.float64).reshape(-1)
    if dc.size and np.isfinite(dc).any():
        t = dc[np.isfinite(dc)]
        if np.any(t < -1e-3) or np.any(t > 1.0 + 1e-2):
            out.append("depth_complexity_score: вне [0, 1] (finite)")

    pk = meta.get("preview_k")
    pfi = np.asarray(d["preview_frame_indices"], dtype=np.int32).reshape(-1)
    try:
        pki = int(pk) if pk is not None else -1
    except (TypeError, ValueError):
        pki = -1
    if pki >= 0 and int(pfi.size) != pki:
        out.append("meta.preview_k != len(preview_frame_indices)")

    n = int(np.asarray(d["frame_indices"], dtype=np.int32).size)
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

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("core_depth_midas/depth.npz")):
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
    p = argparse.ArgumentParser(description="validate core_depth_midas/depth.npz")
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи, N, формы карт/preview.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Норм. карты [0,1], p05/p95, std≥0, preview_k, times_s (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/core_depth_midas/depth.npz",
    )
    p.add_argument(
        "--platform-id",
        type=str,
        default="youtube",
        help="Субкаталог платформы (батч)",
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
            n = int(np.asarray(d0["frame_indices"], dtype=np.int32).ravel().size)
            dm = np.asarray(d0["depth_maps"], dtype=np.float32)
            h = w = kpv = 0
            if dm.ndim == 3:
                h, w = int(dm.shape[1]), int(dm.shape[2])
            pfi = np.asarray(d0["preview_frame_indices"], dtype=np.int32).ravel()
            kpv = int(pfi.size)
            print(
                f"✅ Structure OK (N={n}, H={h}, W={w}, K_preview={kpv}, core_depth_midas_npz_v3)"
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
                "✅ Ranges OK (norm [0,1], p-order, preview_k, times_s, total_frames)"
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
