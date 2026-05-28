#!/usr/bin/env python3
"""Валидатор среза `comments_aggregator` (39 полей) в `text_processor/text_features.npz`.

Схема: `comments_aggregator_output_v1` (семейства `tp_commentsagg_*`, `tp_comments_agg_*`, `tp_cagg_*`).

Пример:
  DataProcessor/.data_venv/bin/python \\
    TextProcessor/src/extractors/comments_aggregator/utils/validate_comments_aggregator_text_npz.py \\
    storage/.../text_processor/text_features.npz --struct --ranges
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/comments_aggregator_output_v1.json"


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Tuple[str, ...]:
    root = _repo_root_from_here()
    p = root / _SCHEMA_RELPATH
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return tuple(sorted(d["fields"].keys()))


EXPECTED_KEYS: Set[str] = set(_load_expected_keys())


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


def _build_slice(d: Dict[str, Any]) -> Tuple[Dict[str, float], Set[str]]:
    names = d.get("feature_names")
    vals = d.get("feature_values")
    if names is None or vals is None:
        return {}, set()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    names = [str(x) for x in names]
    v = np.asarray(vals, dtype=np.float64).ravel()
    if v.size != len(names):
        return {}, set()
    bag: Dict[str, float] = {}
    for i, n in enumerate(names):
        if n in EXPECTED_KEYS:
            bag[n] = float(v[i])
    return bag, set(bag.keys())


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        if "feature_names" not in d or "feature_values" not in d:
            return False
        bag, keys = _build_slice(d)
        meta = extract_meta(d)
        if not keys:
            return meta.get("status") != "ok"
        return keys == EXPECTED_KEYS
    except Exception:
        return False


def validate_structure(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    names = d.get("feature_names")
    vals = d.get("feature_values")
    if names is None or vals is None:
        return ["отсутствуют feature_names / feature_values"]
    if isinstance(names, np.ndarray):
        names = names.tolist()
    names = [str(x) for x in names]
    v = np.asarray(vals, dtype=np.float64).ravel()
    if v.size != len(names):
        out.append(
            f"len(feature_values)={v.size} != len(feature_names)={len(names)}"
        )
    bag, keys = _build_slice(d)
    meta = extract_meta(d)
    if not keys:
        if meta.get("status") == "ok":
            out.append(
                f"срез comments_aggregator пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
            )
        return out
    missing = sorted(EXPECTED_KEYS - keys)
    extra = [k for k in keys if k not in EXPECTED_KEYS]
    if missing:
        out.append(
            f"не хватает ключей схемы: {missing[:6]}{'...' if len(missing) > 6 else ''}"
        )
    if extra:
        out.append("лишние ключи в срезе (не ожидается)")
    if len(keys) != len(EXPECTED_KEYS):
        out.append(f"ожидается {len(EXPECTED_KEYS)} полей, получено {len(keys)}")
    return out


def _fin(x: float) -> bool:
    return bool(math.isfinite(x))


def validate_ranges(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    bag, keys = _build_slice(d)
    meta = extract_meta(d)
    if not keys or meta.get("status") != "ok":
        return out
    if keys != EXPECTED_KEYS:
        return out

    def f(name: str) -> float:
        return bag.get(name, float("nan"))

    binary = (
        "tp_commentsagg_present",
        "tp_comments_agg_present",
        "tp_cagg_present",
        "tp_commentsagg_compute_mean_enabled",
        "tp_commentsagg_compute_median_enabled",
        "tp_commentsagg_compute_std_enabled",
        "tp_commentsagg_write_artifacts_enabled",
        "tp_commentsagg_require_comment_embeddings_enabled",
        "tp_commentsagg_artifact_mean_written",
        "tp_commentsagg_artifact_median_written",
        "tp_commentsagg_weights_applied",
        "tp_commentsagg_weights_mask_likes",
        "tp_commentsagg_weights_mask_authority",
        "tp_commentsagg_weights_mask_recency",
        "tp_commentsagg_weights_align_present",
        "tp_commentsagg_weights_align_shape_ok",
        "tp_commentsagg_dim_mismatch_flag",
        "tp_commentsagg_unsafe_relpath_flag",
        "tp_comments_agg_weights_applied",
        "tp_comments_agg_weights_mask_likes",
        "tp_comments_agg_weights_mask_authority",
        "tp_comments_agg_weights_mask_recency",
        "tp_comments_agg_compute_std",
        "tp_comments_agg_compute_mean",
        "tp_comments_agg_compute_median",
    )
    for k in binary:
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    for k in (
        "tp_commentsagg_count",
        "tp_comments_agg_count",
        "tp_cagg_count",
    ):
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0")

    for k in (
        "tp_commentsagg_mean_std",
        "tp_commentsagg_median_std",
        "tp_comments_agg_mean_std",
        "tp_comments_agg_median_std",
        "tp_cagg_mean_std",
        "tp_cagg_median_std",
    ):
        x = f(k)
        if _fin(x) and x < 0:
            out.append(f"{k}: при finite ожидается >= 0")

    for k in ("tp_commentsagg_agg_mean_ms", "tp_commentsagg_agg_median_ms"):
        x = f(k)
        if _fin(x) and (x < 0 or x > 1.0e7):
            out.append(f"{k}: вне [0, 1e7] мс (подозрительно)")

    pres = f("tp_commentsagg_present")
    if _fin(pres) and pres > 0.5:
        for k in (
            "tp_commentsagg_dim",
            "tp_comments_agg_dim",
            "tp_cagg_dim",
        ):
            x = f(k)
            if not _fin(x) or x <= 0:
                out.append(
                    f"{k}: ожидается finite > 0 при tp_commentsagg_present=1 (got {x})"
                )

    if not _all_equal(
        f("tp_commentsagg_count"), f("tp_comments_agg_count"), f("tp_cagg_count")
    ):
        out.append("count: tp_commentsagg != tp_comments_agg != tp_cagg (зеркала)")
    if not _all_equal(
        f("tp_commentsagg_present"),
        f("tp_comments_agg_present"),
        f("tp_cagg_present"),
    ):
        out.append("present: зеркала не совпадают")
    for slot in (
        ("tp_commentsagg_dim", "tp_comments_agg_dim", "tp_cagg_dim"),
        ("tp_commentsagg_mean_std", "tp_comments_agg_mean_std", "tp_cagg_mean_std"),
        ("tp_commentsagg_median_std", "tp_comments_agg_median_std", "tp_cagg_median_std"),
    ):
        a, b, c = f(slot[0]), f(slot[1]), f(slot[2])
        if _fin(a) and _fin(b) and _fin(c):
            if abs(a - b) > 1e-4 or abs(a - c) > 1e-4:
                out.append(
                    f"зеркала {slot[0]}/{slot[1]}/{slot[2]}: расхождение (a={a}, b={b}, c={c})"
                )
        elif not (_nan_eq(a, b) and _nan_eq(a, c) and _nan_eq(b, c)):
            out.append(
                f"зеркала {slot[0]}/{slot[1]}/{slot[2]}: NaN не согласован (a={a}, b={b}, c={c})"
            )

    return out


def _nan_eq(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if _fin(a) and _fin(b):
        return abs(a - b) < 1e-4
    return False


def _all_equal(a: float, b: float, c: float) -> bool:
    return _nan_eq(a, b) and _nan_eq(a, c)


def _run_batch(*, results_base: str, platform_id: str) -> int:
    root = Path(results_base) / platform_id
    if not root.is_dir():
        print(f"❌ нет каталога: {root}", flush=True)
        return 1
    ex = 0
    c = 0
    for npz in sorted(root.rglob("text_processor/text_features.npz")):
        c += 1
        st = validate_structure(str(npz))
        rg: List[str] = []
        if not st and extract_meta(load_npz(str(npz))).get("status") == "ok":
            rg = validate_ranges(str(npz))
        if st or rg:
            ex = max(ex, 2)
        status = "OK" if not st and not rg else "ISSUES"
        print(f"[{status}] {npz}", flush=True)
        for line in st + rg:
            print(f"    - {line}", flush=True)
    print(f"Проверено файлов: {c}", flush=True)
    return ex if c else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Срез comments_aggregator (39 полей) в text_features.npz"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к text_features.npz")
    p.add_argument(
        "--struct",
        action="store_true",
        help="Согласованность feature_names/values и полного набора 39 ключей при status=ok",
    )
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Диапазоны и согласованность зеркал tp_* / tp_comments_agg_* / tp_cagg_*",
    )
    p.add_argument(
        "--results-base",
        help="[батч] корень result_store; обход **/text_processor/text_features.npz",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] платформа")
    args = p.parse_args()

    if args.results_base:
        return _run_batch(
            results_base=args.results_base, platform_id=args.platform_id or "youtube"
        )

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    ok = validate_schema(args.npz_path)
    print("✅ VALID (схема среза)" if ok else "❌ INVALID (схема среза)")
    if not ok and not (args.struct or args.ranges):
        return 1

    ex = 0
    stl = validate_structure(args.npz_path) if args.struct else []
    rgl = validate_ranges(args.npz_path) if args.ranges else []
    for line in stl:
        print(f"struct: {line}")
        ex = 2
    for line in rgl:
        print(f"ranges: {line}")
        ex = max(ex, 2)
    if args.struct and not stl:
        print("struct: OK")
    if args.ranges and not rgl and _build_slice(load_npz(args.npz_path))[0]:
        print("ranges: OK (срез непустой)")

    return ex


if __name__ == "__main__":
    raise SystemExit(main())
