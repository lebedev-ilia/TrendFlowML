#!/usr/bin/env python3
"""Валидатор среза `comments_embedder` (18 × `tp_commentsemb_*`) в `text_processor/text_features.npz`.

Схема: `comments_embedder_output_v1`.

Пример:
  DataProcessor/.data_venv/bin/python \\
    TextProcessor/src/extractors/comments_embedder/utils/validate_comments_embedder_text_npz.py \\
    storage/.../text_processor/text_features.npz --struct --ranges
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/comments_embedder_output_v1.json"
_EXTRA_KEYS: Set[str] = {
    "tp_commentsemb_cache_enabled",
    "tp_commentsemb_cache_hit",
    "tp_commentsemb_fp16",
    "tp_commentsemb_device_cuda",
    "tp_commentsemb_model_digest_u24",
    "tp_commentsemb_compute_enabled",
    "tp_commentsemb_write_artifact_enabled",
    "tp_commentsemb_artifact_written",
    "tp_commentsemb_select_ms",
    "tp_commentsemb_encode_ms",
}


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
                f"срез tp_commentsemb_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
            )
        return out
    if keys != EXPECTED_KEYS:
        miss = sorted(EXPECTED_KEYS - keys)
        if miss:
            out.append(f"не хватает: {miss}")
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

    for k in (
        "tp_commentsemb_present",
        "tp_commentsemb_truncated_by_total_chars_flag",
    ):
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    for k in (
        "tp_commentsemb_n_input",
        "tp_commentsemb_n_deduped",
        "tp_commentsemb_n_selected",
    ):
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0")

    tc = f("tp_commentsemb_total_chars_used")
    if _fin(tc) and tc < -1e-6:
        out.append("tp_commentsemb_total_chars_used: ожидается >= 0")

    cnt = f("tp_commentsemb_count")
    if _fin(cnt) and cnt < -1e-6:
        out.append("tp_commentsemb_count: ожидается >= 0")

    dim = f("tp_commentsemb_dim")
    pres = f("tp_commentsemb_present")
    if _fin(pres) and pres > 0.5:
        if not _fin(dim) or dim <= 0:
            out.append(
                f"tp_commentsemb_dim: ожидается finite > 0 при present=1 (got {dim})"
            )

    for k in _EXTRA_KEYS:
        x = f(k)
        if not _fin(x):
            continue
        if k == "tp_commentsemb_model_digest_u24":
            if x < -1e-6:
                out.append(f"{k}: ожидается >= 0")
            continue
        if k in ("tp_commentsemb_select_ms", "tp_commentsemb_encode_ms"):
            if x < -1e-6 or x > 1.0e7:
                out.append(f"{k}: вне [0, 1e7] мс (подозрительно)")
            continue
        if k == "tp_commentsemb_cache_hit":
            if x < -1e-6 or x > 1.0 + 1e-6:
                out.append(f"{k}: ожидается 0/1 при finite, got {x}")
            continue
        if x < -1e-6 or x > 1.0 + 1e-6:
            out.append(f"{k}: ожидается 0/1 при finite, got {x}")

    if _fin(pres) and pres > 0.5:
        ns = f("tp_commentsemb_n_selected")
        if _fin(cnt) and _fin(ns) and abs(cnt - ns) > 0.51:
            out.append(
                f"present=1: count={cnt} и n_selected={ns} ожидается согласованы"
            )

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
        description="Срез tp_commentsemb_* (18) в text_features.npz"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к text_features.npz")
    p.add_argument("--struct", action="store_true")
    p.add_argument("--ranges", action="store_true")
    p.add_argument("--results-base", help="[батч] result_store root")
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
