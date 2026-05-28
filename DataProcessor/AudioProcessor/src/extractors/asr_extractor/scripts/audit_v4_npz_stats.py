#!/usr/bin/env python3
"""
Статистика по NPZ `asr_extractor` для Audit v4 / 4.2 (план §4).

Сканирует ``result_store`` (или явный список файлов), строит:
  - сводку по каждому файлу (ключи, shapes, NaN/Inf/нули, табличные перцентили);
  - агрегат по нескольким прогонам: перцентили фич по «видео», матрица корреляций tabular;
  - опционально PNG: гистограммы фич, heatmap корреляций, длины токенов, доли языков.

Пример:

  cd DataProcessor/AudioProcessor
  ../.data_venv/bin/python src/extractors/asr_extractor/scripts/audit_v4_npz_stats.py \\
    --result-store ../../../../storage/result_store \\
    --out-dir ../../../../storage/audit_v4/asr_extractor_stats

Зависимости: numpy; matplotlib для `--no-plots` не нужен (графики пропускаются).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _repo_root_from_script() -> Path:
    # .../TrendFlowML/DataProcessor/AudioProcessor/src/extractors/asr_extractor/scripts/this.py
    return Path(__file__).resolve().parents[6]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return None


def _iter_asr_npz_files(
    result_store: Path,
    *,
    platform: str,
    component: str,
    npz_name: str,
    limit: Optional[int] = None,
) -> List[Path]:
    base = result_store / platform
    if not base.is_dir():
        return []
    paths = sorted(base.glob(f"*/*/{component}/{npz_name}"))
    if limit is not None:
        paths = paths[:limit]
    return [p for p in paths if p.is_file()]


def _parse_npz_meta(meta_obj: Any) -> Dict[str, Any]:
    if not isinstance(meta_obj, dict):
        return {}
    flat: Dict[str, Any] = {}
    for k in (
        "schema_version",
        "status",
        "producer_version",
        "empty_reason",
        "device_used",
    ):
        if k in meta_obj:
            flat[k] = meta_obj[k]
    return flat


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
    if a.dtype == object or a.ndim == 0 and a.dtype.type == np.str_:
        return {"key": name, "kind": "object_or_scalar", "skipped": True}
    flat = a.astype(np.float64).ravel()
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


def summarize_npz(path: Path) -> Dict[str, Any]:
    path = Path(path)
    rel = path.parts
    video_id, run_id = "", ""
    try:
        i = rel.index("result_store")
        # .../result_store/youtube/{video_id}/{run_id}/asr_extractor/...
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
            arr = data[key]
            if key in ("feature_names", "feature_values", "meta"):
                continue
            entry = _numeric_array_stats(arr, key)
            out["arrays"].append(entry)

        if "meta" in data.files:
            m = data["meta"]
            mo = m.item() if m.dtype == object and m.shape == () else {}
            out["meta_flat"] = _parse_npz_meta(mo)

        # object-derived metrics
        token_lens: List[int] = []
        if "token_ids_by_segment" in data.files:
            obj = data["token_ids_by_segment"]
            for row in obj.tolist():
                if row is None:
                    continue
                a = np.asarray(row)
                token_lens.append(int(a.size))
        out["derived"]["token_segments_count"] = len(token_lens)
        out["derived"]["token_lens"] = token_lens
        if token_lens:
            tl = np.asarray(token_lens, dtype=np.float64)
            out["derived"]["token_len_stats"] = {
                "sum": int(tl.sum()),
                "min": int(tl.min()),
                "max": int(tl.max()),
                "mean": float(tl.mean()),
            }

        lang_codes: List[str] = []
        if "lang_code_by_segment" in data.files:
            for x in data["lang_code_by_segment"].tolist():
                lang_codes.append(str(x) if x is not None else "")
        out["derived"]["lang_codes"] = lang_codes

        sq_vals: Dict[str, List[float]] = {}
        if "segment_quality_by_segment" in data.files:
            for item in data["segment_quality_by_segment"].tolist():
                if not isinstance(item, dict):
                    continue
                for k, v in item.items():
                    fv = _safe_float(v)
                    if fv is None:
                        continue
                    sq_vals.setdefault(str(k), []).append(fv)
        out["derived"]["segment_quality_keys"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)) if len(v) > 1 else 0.0}
            for k, v in sq_vals.items()
        }

    return out


def _stack_tabular(summaries: Sequence[Mapping[str, Any]]) -> Tuple[List[str], np.ndarray]:
    """Rows = runs, cols = features (union of names)."""
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


def _corr_matrix(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Pearson correlation across runs; columns with <2 finite values dropped from pairs."""
    M, F = mat.shape
    if M < 2:
        return np.full((F, F), np.nan), []
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
    return corr, list(range(F))


def _try_plot_histograms(
    names: List[str],
    mat: np.ndarray,
    out_dir: Path,
    *,
    max_features: int,
) -> None:
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
        plt.hist(col, bins=min(20, max(5, col.size)), color="#2c5282", edgecolor="white")
        plt.title(name[:80])
        plt.xlabel("value")
        plt.ylabel("count (runs)")
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
    fig, ax = plt.subplots(figsize=(max(6, F * 0.35), max(5, F * 0.35)))
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


def _sanitize_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for strict JSON (RFC 8259)."""
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit v4 NPZ stats for asr_extractor")
    p.add_argument(
        "--result-store",
        type=Path,
        default=None,
        help="Корень result_store (по умолчанию: <repo>/storage/result_store)",
    )
    p.add_argument("--platform", default="youtube")
    p.add_argument("--component", default="asr_extractor")
    p.add_argument("--npz-name", default="asr_extractor_features.npz")
    p.add_argument("--limit", type=int, default=None, help="Максимум файлов (отладка)")
    p.add_argument(
        "--npz",
        type=Path,
        nargs="*",
        default=None,
        help="Явные пути к NPZ (вместо сканирования)",
    )
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
        paths = _iter_asr_npz_files(
            rs,
            platform=args.platform,
            component=args.component,
            npz_name=args.npz_name,
            limit=args.limit,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file = [summarize_npz(path) for path in paths]

    names, mat = _stack_tabular(per_file)
    corr, _ = _corr_matrix(mat)

    aggregate = {
        "n_runs": len(per_file),
        "tabular_feature_names": names,
        "tabular_per_feature": {},
    }
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
        "correlation_tabular": {
            "features": names,
            "matrix": corr.tolist() if corr.size else [],
        },
    }
    json_path = args.out_dir / "asr_extractor_audit_v4_stats.json"
    json_path.write_text(
        json.dumps(_sanitize_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}", file=sys.stderr)

    if not args.no_plots and len(names) and mat.shape[0] >= 1:
        try:
            _try_plot_histograms(names, mat, args.out_dir / "figures", max_features=args.max_plot_features)
            if mat.shape[0] >= 2:
                _try_plot_corr_heatmap(
                    names,
                    corr,
                    args.out_dir / "figures" / "tabular_corr_heatmap.png",
                )
            # token total histogram across runs
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            totals = []
            for s in per_file:
                pdict = s.get("tabular", {}).get("pairwise", {})
                if "token_total" in pdict:
                    totals.append(float(pdict["token_total"]))
            if len(totals) >= 2:
                plt.figure(figsize=(5, 3))
                plt.hist(totals, bins=min(15, len(totals)), color="#276749", edgecolor="white")
                plt.title("token_total across runs")
                plt.savefig(args.out_dir / "figures" / "hist_token_total_runs.png", dpi=120)
                plt.close()
        except ImportError:
            print("matplotlib not installed; skipped plots", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
