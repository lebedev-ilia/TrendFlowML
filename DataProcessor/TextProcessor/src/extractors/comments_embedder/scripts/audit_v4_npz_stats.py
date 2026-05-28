#!/usr/bin/env python3
"""
Audit v4 / 4.2: статистика по срезу `tp_commentsemb_*` в `text_processor/text_features.npz`.

Особенность TextProcessor: экстракторы пишут **общий** табличный слой в `text_features.npz`,
а эмбеддинги комментариев — отдельным артефактом `text_processor/_artifacts/comments_embeddings.npy`.

Скрипт:
  - читает `feature_names`/`feature_values`;
  - выделяет только `tp_commentsemb_*` (контракт comments_embedder_output_v1);
  - фиксирует качество набора (meta.status ok/error);
  - опционально проверяет артефакт `.npy` по форме и конечности.

Пример:

  cd DataProcessor/TextProcessor
  ../.data_venv/bin/python \
    src/extractors/comments_embedder/scripts/audit_v4_npz_stats.py \
    --out-dir ../../storage/audit_v4/comments_embedder_l2 \
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PREFIX = "tp_commentsemb_"
EXPECTED_KEYS = 18

# По умолчанию — тот же набор A+B run_id, что используется в RUN_LOG для VisualProcessor L2.
_DEFAULT_NPZ_PATHS: Tuple[str, ...] = (
    "storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz",
    "storage/result_store/youtube/-15jH8mtfJw/30e1183d-0068-46ee-9b04-ae4f693a9bb2/text_processor/text_features.npz",
    "storage/result_store/youtube/-5EYUqIlyJU/b9761f4a-2227-4b0e-b780-2d2725234c53/text_processor/text_features.npz",
    "storage/result_store/youtube/-7Ei8e05x30/45c451ad-4b6a-4845-a71b-4245ec579f45/text_processor/text_features.npz",
    "storage/result_store/youtube/-Ga4edhrfog/e2dc8851-6c51-43c0-9757-3c0fed803348/text_processor/text_features.npz",
)


def _repo_root_from_script() -> Path:
    # .../TrendFlowML/DataProcessor/TextProcessor/src/extractors/.../scripts/this.py
    return Path(__file__).resolve().parents[6]


def _parse_meta(meta_obj: Any) -> Dict[str, Any]:
    if meta_obj is None:
        return {}
    if isinstance(meta_obj, np.ndarray) and meta_obj.dtype == object and meta_obj.shape == ():
        meta_obj = meta_obj.item()
    if not isinstance(meta_obj, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in (
        "schema_version",
        "producer",
        "producer_version",
        "status",
        "empty_reason",
    ):
        if k in meta_obj:
            out[k] = meta_obj[k]
    return out


def _filter_tabular(names: List[str], vals: np.ndarray) -> Tuple[List[str], np.ndarray]:
    idx = [i for i, n in enumerate(names) if n.startswith(PREFIX)]
    fn = [names[i] for i in idx]
    v = np.asarray(vals, dtype=np.float64).ravel()
    if v.size != len(names):
        raise ValueError("feature_values length mismatch vs feature_names")
    fv = v[idx]
    return fn, fv


def _artifact_path_from_npz(npz_path: Path) -> Path:
    # .../<video_id>/<run_id>/text_processor/text_features.npz -> .../text_processor/_artifacts/comments_embeddings.npy
    return npz_path.parent / "_artifacts" / "comments_embeddings.npy"


def _summarize_comments_embeddings(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "exists": bool(path.is_file()),
        "shape": None,
        "dtype": None,
        "finite": None,
        "row_l2_min": None,
        "row_l2_max": None,
    }
    if not path.is_file():
        return out
    arr = np.load(path)
    a = np.asarray(arr)
    out["shape"] = list(a.shape)
    out["dtype"] = str(a.dtype)
    if a.size == 0:
        out["finite"] = True
        return out
    fin = bool(np.isfinite(a).all())
    out["finite"] = fin
    if a.ndim == 2 and a.shape[0] >= 1:
        norms = np.linalg.norm(a.astype(np.float64), axis=1)
        out["row_l2_min"] = float(np.min(norms)) if norms.size else None
        out["row_l2_max"] = float(np.max(norms)) if norms.size else None
    return out


def summarize_text_npz(path: Path) -> Dict[str, Any]:
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
        "prefix": PREFIX,
        "expected_keys": EXPECTED_KEYS,
        "tabular_slice": {},
        "meta_flat": {},
        "text_processor_error": None,
        "artifact_comments_embeddings": None,
    }

    z = np.load(path, allow_pickle=True)
    try:
        if "meta" in z.files:
            mo = z["meta"]
            raw_meta = mo.item() if mo.dtype == object and mo.shape == () else {}
            out["meta_flat"] = _parse_meta(mo)
            if isinstance(raw_meta, dict) and raw_meta.get("error"):
                out["text_processor_error"] = str(raw_meta.get("error"))[:2000]

        if "feature_names" not in z.files or "feature_values" not in z.files:
            out["error"] = "missing feature_names or feature_values"
            return out

        names = [str(x) for x in z["feature_names"].astype(object).tolist()]
        vals = z["feature_values"]
        fn, fv = _filter_tabular(names, vals)

        out["tabular_slice"] = {
            "names": fn,
            "values": fv.tolist(),
            "pairwise": dict(zip(fn, fv.tolist())),
            "n_keys": len(fn),
        }

        nan_mask = np.isnan(fv) if fv.size else np.array([], dtype=bool)
        out["derived"] = {
            "nan_count": int(nan_mask.sum()) if fv.size else 0,
            "finite_count": int((~nan_mask).sum()) if fv.size else 0,
            "slice_ok": bool(fv.size == EXPECTED_KEYS),
            "tp_commentsemb_present": float(out["tabular_slice"]["pairwise"].get("tp_commentsemb_present"))
            if "tp_commentsemb_present" in out["tabular_slice"]["pairwise"]
            else None,
        }

        art = _artifact_path_from_npz(path)
        out["artifact_comments_embeddings"] = {
            "path": str(art),
            **_summarize_comments_embeddings(art),
        }
    finally:
        z.close()

    return out


def _stack_slice(summaries: Sequence[Mapping[str, Any]]) -> Tuple[List[str], np.ndarray]:
    all_names: List[str] = []
    seen = set()
    for s in summaries:
        for n in s.get("tabular_slice", {}).get("names", []):
            if n not in seen:
                seen.add(n)
                all_names.append(n)
    M = len(summaries)
    F = len(all_names)
    mat = np.full((M, F), np.nan, dtype=np.float64)
    name_to_i = {n: i for i, n in enumerate(all_names)}
    for r, s in enumerate(summaries):
        names = s.get("tabular_slice", {}).get("names", [])
        vals = s.get("tabular_slice", {}).get("values", [])
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
        return np.full((F, F), np.nan)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit v4 stats: tp_commentsemb_* slice in text_features.npz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--npz",
        type=Path,
        nargs="*",
        default=None,
        help="Явные пути к text_features.npz (по умолчанию — фиксированные 5 youtube run из RUN_LOG A+B)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    np.random.seed(args.seed)

    repo = _repo_root_from_script()
    paths = [Path(x) for x in args.npz] if args.npz else [repo / rel for rel in _DEFAULT_NPZ_PATHS]
    for path in paths:
        if not path.is_file():
            print(f"Missing NPZ: {path}", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_file = [summarize_text_npz(path) for path in paths]
    row_ok = np.array(
        [
            s.get("meta_flat", {}).get("status") == "ok" and s.get("derived", {}).get("slice_ok")
            for s in per_file
        ],
        dtype=bool,
    )
    n_ok = int(row_ok.sum())
    n_err = int(sum(1 for s in per_file if s.get("meta_flat", {}).get("status") == "error"))

    names, mat = _stack_slice(per_file)
    corr = _corr_matrix(mat)

    def _aggregate_for_rows(row_mask: np.ndarray) -> Dict[str, Any]:
        sub = mat[row_mask, :]
        agg: Dict[str, Any] = {
            "n_rows": int(row_mask.sum()),
            "tabular_feature_names": names,
            "tabular_per_feature": {},
        }
        for j, name in enumerate(names):
            col = sub[:, j]
            fin = col[np.isfinite(col)]
            agg["tabular_per_feature"][name] = {
                "n_finite": int(fin.size),
                "nan_frac_across_runs": float(np.isnan(col).sum() / max(len(col), 1)),
            }
            if fin.size:
                agg["tabular_per_feature"][name].update(
                    {
                        "min": float(fin.min()),
                        "max": float(fin.max()),
                        "mean": float(fin.mean()),
                        "std": float(fin.std()) if fin.size > 1 else 0.0,
                        "p01": float(np.percentile(fin, 1)),
                        "p50": float(np.percentile(fin, 50)),
                        "p99": float(np.percentile(fin, 99)),
                    }
                )
        return agg

    dataset_quality = {
        "n_paths": len(per_file),
        "n_text_processor_ok_with_slice": n_ok,
        "n_text_processor_error_or_empty_slice": int(len(per_file) - n_ok),
        "n_meta_error": n_err,
        "note": "L2 по смыслу плана требует 5 OK прогонов; при частичном text_processor перцентили/корреляции искажаются.",
    }

    report = {
        "audit": "v4.2",
        "component": "comments_embedder",
        "artifact": "text_processor/text_features.npz",
        "prefix": PREFIX,
        "expected_keys": EXPECTED_KEYS,
        "paths": [str(x.resolve()) for x in paths],
        "dataset_quality": dataset_quality,
        "per_file": per_file,
        "aggregate": _aggregate_for_rows(np.ones(len(per_file), dtype=bool)),
        "aggregate_ok_subset": (_aggregate_for_rows(row_ok) if row_ok.any() else None),
        "correlation_tabular": {
            "features": names,
            "matrix": corr.tolist() if corr.size else [],
            "low_sample_warning": bool(n_ok < 3),
        },
    }
    json_path = args.out_dir / "comments_embedder_audit_v4_stats.json"
    json_path.write_text(json.dumps(_sanitize_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", file=sys.stderr)
    if n_ok < 3:
        print("Skipped plots: fewer than 3 OK text_processor runs with tp_commentsemb slice", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

