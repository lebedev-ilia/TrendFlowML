#!/usr/bin/env python3
"""Валидатор среза `cosine_metrics_extractor` (39 × `tp_cos_*`) в `text_processor/text_features.npz`.

Схема: `cosine_metrics_extractor_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/cosine_metrics_extractor_output_v1.json"

_COS_KEYS: Tuple[str, ...] = (
    "tp_cos_title_desc",
    "tp_cos_title_transcript",
    "tp_cos_desc_transcript",
    "tp_cos_transcript_comments_mean",
    "tp_cos_transcript_comments_median",
)

_FLOAT_EXTRA: Tuple[str, ...] = (
    "tp_cos_load_ms",
    "tp_cos_compute_ms",
    "tp_cos_tc_n_comments_used",
    "tp_cos_tc_sims_std",
    "tp_cos_tc_sims_p95",
)

_TRANSCRIPT_ONEHOT: Tuple[str, ...] = (
    "tp_cos_transcript_agg_source_whisper",
    "tp_cos_transcript_agg_source_youtube_auto",
    "tp_cos_transcript_agg_source_combined",
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()


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
        _bag, keys = _build_slice(d)
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
                f"срез tp_cos_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
            )
        return out
    if keys != EXPECTED_KEYS:
        miss = sorted(EXPECTED_KEYS - keys)
        if miss:
            out.append(f"не хватает: {miss[:8]}{'...' if len(miss) > 8 else ''}")
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
    if not keys or meta.get("status") != "ok" or keys != EXPECTED_KEYS:
        return out

    def f(k: str) -> float:
        return bag.get(k, float("nan"))

    binary = EXPECTED_KEYS - set(_COS_KEYS) - set(_FLOAT_EXTRA)
    for k in binary:
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    for k in _COS_KEYS:
        x = f(k)
        if _fin(x) and (x < -1.0 - 1e-3 or x > 1.0 + 1e-3):
            out.append(f"{k}: косинус вне [-1,1] ({x})")

    for k in _FLOAT_EXTRA:
        if k in ("tp_cos_load_ms", "tp_cos_compute_ms"):
            x = f(k)
            if _fin(x) and (x < -1e-6 or x > 1.0e7):
                out.append(f"{k}: вне [0, 1e7] мс")
            continue
        if k == "tp_cos_tc_n_comments_used":
            x = f(k)
            if _fin(x) and x < -1e-6:
                out.append("tp_cos_tc_n_comments_used: ожидается >= 0")
            continue
        if k == "tp_cos_tc_sims_std":
            x = f(k)
            if _fin(x) and x < 0:
                out.append("tp_cos_tc_sims_std: ожидается >= 0")
            continue
        if k == "tp_cos_tc_sims_p95":
            x = f(k)
            if _fin(x) and (x < -1.0 - 1e-3 or x > 1.0 + 1e-3):
                out.append("tp_cos_tc_sims_p95: ожидается в [-1,1] для косинусов")

    ohx = [f(k) for k in _TRANSCRIPT_ONEHOT]
    if all(_fin(x) for x in ohx):
        tsum = float(sum(ohx))
        if abs(tsum) > 1e-6 and abs(tsum - 1.0) > 1e-3:
            out.append(
                f"one-hot transcript source: сумма {tsum} (ожидается 0 или 1)"
            )

    ma = f("tp_cos_comments_mode_aggregates")
    mm = f("tp_cos_comments_mode_matrix")
    if _fin(ma) and _fin(mm) and ma > 0.5 and mm > 0.5:
        out.append("comments_mode: оба mode_aggregates и mode_matrix = 1 (взаимоисключающе?)")

    return out


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
        description="Срез tp_cos_* (39) в text_features.npz (cosine_metrics_extractor_output_v1)"
    )
    p.add_argument("npz_path", nargs="?", help="text_features.npz")
    p.add_argument("--struct", action="store_true")
    p.add_argument("--ranges", action="store_true")
    p.add_argument("--results-base", help="[батч] result_store")
    p.add_argument("--platform-id", default="youtube")
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
