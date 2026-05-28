#!/usr/bin/env python3
"""Валидатор среза `title_embedder` (16 × `tp_titleemb_*`) в `text_processor/text_features.npz`.

Схема: `title_embedder_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/title_embedder_output_v1.json"

_BINARY: tuple[str, ...] = (
    "tp_titleemb_present",
    "tp_titleemb_title_present",
    "tp_titleemb_require_title_enabled",
    "tp_titleemb_compute_enabled",
    "tp_titleemb_write_artifact_enabled",
    "tp_titleemb_artifact_written",
    "tp_titleemb_cache_enabled",
    "tp_titleemb_fp16",
    "tp_titleemb_device_cuda",
    "tp_titleemb_compute_raw_norm",
)

_NON_BINARY: frozenset = frozenset(
    {
        "tp_titleemb_dim",
        "tp_titleemb_norm_raw",
        "tp_titleemb_l2_norm",
        "tp_titleemb_model_digest_u24",
        "tp_titleemb_encode_ms",
    }
) | frozenset({"tp_titleemb_cache_hit"})


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    with open(_repo_root_from_here() / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
assert len(EXPECTED_KEYS) == 16
assert not (set(_BINARY) & set(_NON_BINARY))
assert set(_BINARY) | set(_NON_BINARY) == EXPECTED_KEYS


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


def _build_slice(d: Dict[str, Any]) -> tuple[Dict[str, float], Set[str]]:
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
                f"срез tp_titleemb_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    for k in _BINARY:
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    ch = f("tp_titleemb_cache_hit")
    if _fin(ch) and (ch < -1e-6 or ch > 1.0 + 1e-6):
        out.append("tp_titleemb_cache_hit: ожидается 0/1 при finite")

    md = f("tp_titleemb_model_digest_u24")
    if _fin(md) and (md < 0.0 or md > 16777215.0 + 1.0):
        out.append("tp_titleemb_model_digest_u24: вне [0, 0xFFFFFF] (finite)")

    en = f("tp_titleemb_encode_ms")
    if _fin(en) and (en < -1e-3 or en > 1.0e7):
        out.append("tp_titleemb_encode_ms: вне [0, 1e7] мс")

    pres = f("tp_titleemb_present")
    if _fin(pres) and pres > 0.5:
        dim = f("tp_titleemb_dim")
        if (not _fin(dim)) or dim < 0.5:
            out.append("tp_titleemb_dim: при present=1 ожидается finite >= 1")
        ln = f("tp_titleemb_l2_norm")
        if _fin(ln) and (ln < 0.99 or ln > 1.01):
            out.append(
                f"tp_titleemb_l2_norm: при present=1 ожидается ~1 (got {ln})"
            )
        if f("tp_titleemb_compute_raw_norm") > 0.5:
            nr = f("tp_titleemb_norm_raw")
            if _fin(nr) and nr <= 0.0:
                out.append("tp_titleemb_norm_raw: при compute_raw_norm=1 ожидается >0 (finite)")

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
        description="Срез tp_titleemb_* (16) в text_features.npz (title_embedder_output_v1)"
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
    b0, _ = _build_slice(load_npz(args.npz_path))
    if args.ranges and not rgl and b0:
        print("ranges: OK (срез непустой)")

    return ex


if __name__ == "__main__":
    raise SystemExit(main())
