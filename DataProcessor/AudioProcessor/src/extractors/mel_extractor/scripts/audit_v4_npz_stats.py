#!/usr/bin/env python3
"""
Статистика по NPZ `mel_extractor` для Audit v4 / 4.2 (план §4).

Сводка по файлам, агрегат по прогонам, корреляции tabular, опционально PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[6]


def _tabular_dict(data: np.lib.npyio.NpzFile) -> Tuple[List[str], np.ndarray]:
    names = data["feature_names"]
    vals = data["feature_values"]
    if names.dtype != object:
        names = names.astype(object)
    name_list = [str(x) for x in names.tolist()]
    v = np.asarray(vals, dtype=np.float64)
    return name_list, v


def _numeric_array_stats(arr: np.ndarray, name: str) -> Dict[str, Any]:
    a = np.asarray(arr)
    if a.dtype == object:
        return {"key": name, "dtype": "object", "shape": list(a.shape), "skipped": True}
    flat = a.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(flat)
    n = int(flat.size)
    n_fin = int(finite.sum())
    out: Dict[str, Any] = {
        "key": name,
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "n_el": n,
        "nan_frac": float(np.isnan(flat).sum() / max(n, 1)),
        "inf_frac": float(np.isinf(flat).sum() / max(n, 1)),
        "zero_frac": float((flat == 0.0).sum() / max(n, 1)) if n else 0.0,
    }
    if n_fin == 0:
        out["finite_stats"] = None
        return out
    x = flat[finite]
    out["finite_stats"] = {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }
    return out


def _parse_meta(meta_obj: Any) -> Dict[str, Any]:
    if not isinstance(meta_obj, dict):
        return {}
    flat: Dict[str, Any] = {}
    for k in (
        "schema_version",
        "status",
        "producer_version",
        "empty_reason",
        "device_used",
        "mel_contract_version",
        "features_enabled",
    ):
        if k in meta_obj:
            flat[k] = meta_obj[k]
    return flat


def summarize_npz(path: Path) -> Dict[str, Any]:
    path = Path(path)
    rel = path.parts
    video_id, run_id = "", ""
    try:
        i = rel.index("result_store")
        if i + 3 < len(rel):
            video_id = rel[i + 2]
            run_id = rel[i + 3]
    except ValueError:
        pass

    out: Dict[str, Any] = {
        "path": str(path),
        "video_id": video_id,
        "run_id": run_id,
        "keys": [],
        "tabular": {},
        "arrays": [],
        "derived": {},
    }
    with np.load(path, allow_pickle=True) as data:
        out["keys"] = sorted(list(data.files))
        if "feature_names" in data.files and "feature_values" in data.files:
            names, vals = _tabular_dict(data)
            out["tabular"] = {
                "names": names,
                "values": vals.tolist(),
                "pairwise": dict(zip(names, vals.tolist())),
            }

        for key in sorted(data.files):
            if key in ("feature_names", "feature_values", "meta"):
                continue
            out["arrays"].append(_numeric_array_stats(np.asarray(data[key]), key))

        if "meta" in data.files:
            mmeta = data["meta"]
            mo = mmeta.item() if hasattr(mmeta, "item") and mmeta.dtype == object and mmeta.shape == () else {}
            out["meta_flat"] = _parse_meta(mo if isinstance(mo, dict) else {})

        if "segment_mask" in data.files:
            m = np.asarray(data["segment_mask"]).astype(bool).ravel()
            out["derived"]["segments_total"] = int(m.size)
            out["derived"]["segments_valid"] = int(m.sum())

        for k in ("mel_energy_by_segment", "mel_centroid_mean_by_segment", "mel_bandwidth_mean_by_segment"):
            if k in data.files:
                x = np.asarray(data[k], dtype=np.float64).ravel()
                fin = x[np.isfinite(x)]
                out["derived"][k] = {
                    "nan_frac": float(np.isnan(x).sum() / max(x.size, 1)),
                    "min": float(fin.min()) if fin.size else None,
                    "max": float(fin.max()) if fin.size else None,
                    "mean": float(fin.mean()) if fin.size else None,
                }

    return out


def _stack_tabular(summaries: Sequence[Mapping[str, Any]]) -> Tuple[List[str], np.ndarray]:
    all_names: List[str] = []
    seen = set()
    for s in summaries:
        for n in s.get("tabular", {}).get("names", []):
            if n not in seen:
                seen.add(n)
                all_names.append(n)
    M = len(summaries)
    F = len(all_names)
    mat = np.full((M, F), np.nan, dtype=np.float64)
    name_to_i = {n: i for i, n in enumerate(all_names)}
    for r, s in enumerate(summaries):
        names = s.get("tabular", {}).get("names", [])
        vals = s.get("tabular", {}).get("values", [])
        for n, v in zip(names, vals):
            j = name_to_i.get(n)
            if j is not None:
                try:
                    mat[r, j] = float(v)
                except (TypeError, ValueError):
                    mat[r, j] = np.nan
    return all_names, mat


def _corr_matrix(mat: np.ndarray) -> np.ndarray:
    M, F = mat.shape
    if M < 2:
        return np.full((F, F), np.nan, dtype=np.float64)
    corr = np.eye(F, dtype=np.float64)
    for i in range(F):
        for j in range(i + 1, F):
            xi = mat[:, i]
            xj = mat[:, j]
            ok = np.isfinite(xi) & np.isfinite(xj)
            if ok.sum() < 2:
                c = np.nan
            else:
                a = xi[ok]
                b = xj[ok]
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    c = np.nan
                else:
                    c = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = c
            corr[j, i] = c
    return corr


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def _try_plot_histograms(names: List[str], mat: np.ndarray, out_dir: Path, *, max_features: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for j, name in enumerate(names):
        if k >= max_features:
            break
        col = mat[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        plt.figure(figsize=(6, 3))
        plt.hist(col, bins=min(24, max(6, col.size)), color="#0ea5e9", edgecolor="white")
        plt.title(name[:80])
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:60]
        plt.savefig(out_dir / f"hist_tabular_{j:03d}_{safe}.png", dpi=120)
        plt.close()
        k += 1


def _try_plot_corr_heatmap(names: List[str], corr: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = corr.shape[0]
    if F == 0:
        return
    fig, ax = plt.subplots(figsize=(max(7, F * 0.45), max(6, F * 0.45)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(F))
    ax.set_yticks(range(F))
    ax.set_xticklabels([n[:12] for n in names], rotation=90, fontsize=7)
    ax.set_yticklabels([n[:12] for n in names], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pearson corr (tabular across runs)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit v4 NPZ stats for mel_extractor")
    p.add_argument("--result-store", type=Path, default=None)
    p.add_argument("--platform", default="youtube")
    p.add_argument("--component", default="mel_extractor")
    p.add_argument("--npz-name", default="mel_extractor_features.npz")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--npz", type=Path, nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--max-plot-features", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    np.random.seed(args.seed)

    repo = _repo_root_from_script()
    rs = args.result_store or (repo / "storage" / "result_store")

    if args.npz:
        paths = [Path(x) for x in args.npz]
    else:
        base = rs / args.platform
        paths = sorted(base.glob(f"*/*/{args.component}/{args.npz_name}")) if base.is_dir() else []
        if args.limit is not None:
            paths = paths[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file = [summarize_npz(pth) for pth in paths]
    names, mat = _stack_tabular(per_file)
    corr = _corr_matrix(mat)

    aggregate: Dict[str, Any] = {"n_runs": len(per_file), "tabular_feature_names": names, "tabular_per_feature": {}}
    for j, name in enumerate(names):
        col = mat[:, j]
        fin = col[np.isfinite(col)]
        aggregate["tabular_per_feature"][name] = {
            "n_finite": int(fin.size),
            "nan_frac_across_runs": float(np.isnan(col).sum() / max(len(col), 1)),
        }
        if fin.size:
            aggregate["tabular_per_feature"][name].update(
                {
                    "min": float(fin.min()),
                    "max": float(fin.max()),
                    "mean": float(fin.mean()),
                    "std": float(fin.std()),
                    "p01": float(np.percentile(fin, 1)),
                    "p50": float(np.percentile(fin, 50)),
                    "p99": float(np.percentile(fin, 99)),
                }
            )

    report = {
        "audit": "v4.2",
        "component": args.component,
        "result_store": str(rs.resolve()),
        "paths": [str(x.resolve()) for x in paths],
        "per_file": per_file,
        "aggregate": aggregate,
        "correlation_tabular": {"features": names, "matrix": corr.tolist() if corr.size else []},
    }
    json_path = args.out_dir / "mel_extractor_audit_v4_stats.json"
    json_path.write_text(json.dumps(_sanitize_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", file=sys.stderr)

    if not args.no_plots and len(names) and mat.shape[0] >= 1:
        try:
            _try_plot_histograms(names, mat, args.out_dir / "figures", max_features=args.max_plot_features)
            if mat.shape[0] >= 2:
                _try_plot_corr_heatmap(names, corr, args.out_dir / "figures" / "tabular_corr_heatmap.png")
        except ImportError:
            print("matplotlib not installed; skipped plots", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

