#!/usr/bin/env python3
"""Single-file валидатор core_optical_flow/flow.npz: схема, --struct, --qa, --ranges."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_PER_FRAME = (
    "frame_indices",
    "times_s",
    "motion_norm_per_sec_mean",
    "dt_seconds",
    "flow_mag_std_per_sec_norm",
    "flow_mag_p95_per_sec_norm",
    "flow_dx_mean_per_sec_norm",
    "flow_dy_mean_per_sec_norm",
    "flow_dir_sin_mean",
    "flow_dir_cos_mean",
    "flow_dir_dispersion",
    "flow_div_abs_mean",
    "flow_consistency",
    "cam_affine_scale",
    "cam_affine_rotation",
    "cam_tx_per_sec_norm",
    "cam_ty_per_sec_norm",
    "cam_shake_std_norm",
    "bg_ratio",
)

_PREVIEW_1D = (
    "preview_pair_pos",
    "preview_prev_frame_indices",
    "preview_cur_frame_indices",
    "preview_prev_times_s",
    "preview_cur_times_s",
)

_REQUIRED = _PER_FRAME + _PREVIEW_1D + ("preview_flow_mag_map_norm", "meta")


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
        return "core_optical_flow_npz_v3" in sv or "core_optical_flow" in sv
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
    comp = "core_optical_flow"
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
    for k in _PER_FRAME:
        a = np.asarray(d[k]).reshape(-1)
        if int(a.size) != n:
            out.append(f"{k}: len={a.size} != N={n}")

    p0 = np.asarray(d["preview_pair_pos"]).reshape(-1)
    pk = int(p0.size)
    for k in _PREVIEW_1D:
        a = np.asarray(d[k]).reshape(-1)
        if int(a.size) != pk and pk > 0:
            out.append(f"{k}: len preview-оси != preview_pair_pos ({pk})")

    pm = np.asarray(d["preview_flow_mag_map_norm"], dtype=np.float32)
    if pm.ndim == 3:
        if int(pm.shape[0]) != pk:
            out.append("preview_flow_mag_map_norm: dim0 != K preview")
    else:
        out.append("preview_flow_mag_map_norm: ожидается 3D (K,H,W)")

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
    """Проверка типичных диапазонов (см. docs/FEATURE_DESCRIPTION.md), только для finite."""
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    if meta.get("status") == "error":
        return out

    n = int(np.asarray(d["frame_indices"]).size)

    def _check01(name: str) -> None:
        a = np.asarray(d[name], dtype=np.float64).reshape(-1)
        if int(a.size) != n:
            return
        m = np.isfinite(a)
        if not m.any():
            return
        t = a[m]
        if np.any(t < -1e-6) or np.any(t > 1.0 + 1e-6):
            out.append(f"{name}: вне [0, 1] (finite)")

    def _checkm11(name: str) -> None:
        a = np.asarray(d[name], dtype=np.float64).reshape(-1)
        if int(a.size) != n:
            return
        m = np.isfinite(a)
        if not m.any():
            return
        t = a[m]
        if np.any(t < -1.0 - 1e-3) or np.any(t > 1.0 + 1e-3):
            out.append(f"{name}: вне [-1, 1] (finite)")

    _check01("bg_ratio")
    _check01("flow_dir_dispersion")
    _check01("flow_consistency")
    _checkm11("flow_dir_sin_mean")
    _checkm11("flow_dir_cos_mean")

    mnm = np.asarray(d["motion_norm_per_sec_mean"], dtype=np.float64).reshape(-1)
    if int(mnm.size) == n:
        m = np.isfinite(mnm)
        if m.any() and np.any(mnm[m] < -1e-6):
            out.append("motion_norm_per_sec_mean: отрицательные finite")

    for name in (
        "flow_mag_std_per_sec_norm",
        "flow_mag_p95_per_sec_norm",
    ):
        a = np.asarray(d[name], dtype=np.float64).reshape(-1)
        if int(a.size) == n:
            m = np.isfinite(a)
            if m.any() and np.any(a[m] < -1e-6):
                out.append(f"{name}: отрицательные finite (ожидается ≥0)")

    fdiv = np.asarray(d["flow_div_abs_mean"], dtype=np.float64).reshape(-1)
    if int(fdiv.size) == n:
        m = np.isfinite(fdiv)
        if m.any() and np.any(fdiv[m] < -1e-6):
            out.append("flow_div_abs_mean: отрицательные finite (ожидается ≥0)")

    fcon = np.asarray(d["flow_consistency"], dtype=np.float64).reshape(-1)
    if int(fcon.size) == n and int(fdiv.size) == n:
        both = np.isfinite(fdiv) & np.isfinite(fcon)
        if both.any():
            exp = 1.0 / (1.0 + fdiv[both])
            got = fcon[both]
            if np.max(np.abs(got - exp)) > 5e-3:
                out.append("flow_consistency: не согласован с 1/(1+flow_div_abs_mean)")

    csc = np.asarray(d["cam_affine_scale"], dtype=np.float64).reshape(-1)
    if int(csc.size) == n:
        m = np.isfinite(csc)
        if m.any() and np.any(csc[m] < -1e-6):
            out.append("cam_affine_scale: отрицательные finite (ожидается ≥0)")

    ts = np.asarray(d["times_s"], dtype=np.float64).reshape(-1)
    if int(ts.size) == n and n > 1 and np.any(np.diff(ts) < -1e-4):
        out.append("times_s: не неубывающий ряд (union)")

    dt = np.asarray(d["dt_seconds"], dtype=np.float64).reshape(-1)
    if int(dt.size) == n and n >= 2:
        if np.isfinite(float(dt[0])):
            out.append("dt_seconds[0]: ожидается NaN (нет пары с предыдущим кадром)")
        tail = np.asarray(dt[1:], dtype=np.float64).reshape(-1)
        m2 = np.isfinite(tail)
        if m2.any() and np.any(tail[m2] <= 0):
            out.append("dt_seconds[1:]: ожидаются > 0, где finite")

    pm = np.asarray(d["preview_flow_mag_map_norm"], dtype=np.float64)
    if pm.size and np.isfinite(pm).any():
        t = pm[np.isfinite(pm)]
        if np.any(t < -1e-3) or np.any(t > 1.0 + 1e-3):
            out.append("preview_flow_mag_map_norm: вне [0, 1] (finite)")

    pk = int(meta.get("preview_k", -1) or -1) if meta else -1
    if isinstance(pk, (int, float)) and pk >= 0:
        p0 = np.asarray(d["preview_pair_pos"]).reshape(-1)
        if int(p0.size) != int(pk):
            out.append("meta.preview_k != len(preview_pair_pos)")

    tf = meta.get("total_frames")
    if tf is not None and n:
        try:
            if n > int(tf) >= 0:
                out.append("len(frame_indices) > meta.total_frames")
        except (TypeError, ValueError):
            pass

    return out


def _run_batch_rglob(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    n = 0
    for npz in sorted(root.rglob("core_optical_flow/flow.npz")):
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
    p = argparse.ArgumentParser(description="validate core_optical_flow/flow.npz")
    p.add_argument("npz_path", nargs="?", help="Путь к NPZ (если не задан --results-base)")
    p.add_argument("--qa", action="store_true", help="Плоский meta (view_csv_feature_qa).")
    p.add_argument("--struct", action="store_true", help="Ключи и согласованность N / preview K.")
    p.add_argument(
        "--ranges",
        action="store_true",
        help="bg/flow/preview, dt/times, div↔consistency, cam_scale (см. docs/FEATURE_DESCRIPTION.md).",
    )
    p.add_argument(
        "--results-base",
        type=str,
        help="Корень result_store; обход **/core_optical_flow/flow.npz",
    )
    p.add_argument(
        "--platform-id", type=str, default="youtube", help="Субкаталог платформы (батч)"
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
            p0 = np.asarray(d0["preview_pair_pos"]).reshape(-1)
            kpv = int(p0.size)
            phw: Tuple[int, int] = (0, 0)
            pm = np.asarray(d0["preview_flow_mag_map_norm"], dtype=np.float32)
            if pm.ndim == 3:
                phw = (int(pm.shape[1]), int(pm.shape[2]))
            print(
                f"✅ Structure OK (N={n}, K_preview={kpv}, preview_HW={phw[0]}×{phw[1]}, core_optical_flow_npz_v3)"
            )
    if args.ranges:
        rg = validate_ranges(args.npz_path)
        if rg:
            print("⚠️  ranges:")
            for s in rg:
                print("  -", s)
            ex = max(ex, 2)
        else:
            print("✅ Ranges OK (dt/times, div/consistency, [0,1] bands, total_frames)")
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
