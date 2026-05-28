#!/usr/bin/env python3
"""Валидатор среза `embedding_stats_extractor` (39 × `tp_embstats_*`) в `text_processor/text_features.npz`.

Схема: `embedding_stats_extractor_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/embedding_stats_extractor_output_v1.json"

_TIME_KEYS: Tuple[str, ...] = (
    "tp_embstats_load_ms",
    "tp_embstats_compute_ms",
)

def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()

_NON_BINARY: Set[str] = {
    "tp_embstats_schema_topvar_slots_max",
    "tp_embstats_top_k_slots_requested",
    "tp_embstats_top_k_slots",
    "tp_embstats_min_chunks_required",
    "tp_embstats_topk",
    "tp_embstats_variance_ddof",
    "tp_embstats_n_chunks",
    "tp_embstats_dim",
    "tp_embstats_l2_variance",
    "tp_embstats_topic_entropy",
    "tp_embstats_topic_entropy_norm",
    "tp_embstats_topic_perplexity",
    "tp_embstats_load_ms",
    "tp_embstats_compute_ms",
} | {f"tp_embstats_topvar_{i}" for i in range(1, 9)}

_BINARY_KEYS: Set[str] = EXPECTED_KEYS - _NON_BINARY

_SCHEMA_MAX = 8.0


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
                f"срез tp_embstats_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    for k in _BINARY_KEYS:
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    xsm = f("tp_embstats_schema_topvar_slots_max")
    if _fin(xsm) and abs(xsm - _SCHEMA_MAX) > 1e-3:
        out.append(
            f"tp_embstats_schema_topvar_slots_max: ожидается {_SCHEMA_MAX}, got {xsm}"
        )

    for k in (f"tp_embstats_topvar_{i}" for i in range(1, 9)):
        t = f(k)
        if _fin(t) and t < -1e-6:
            out.append(f"{k}: topvar (finite) ожидается >= 0, got {t}")

    lv = f("tp_embstats_l2_variance")
    if _fin(lv) and lv < -1e-6:
        out.append("tp_embstats_l2_variance: ожидается >= 0")

    h = f("tp_embstats_topic_entropy")
    if _fin(h) and h < -1e-6:
        out.append("tp_embstats_topic_entropy: ожидается >= 0 (finite)")

    hn = f("tp_embstats_topic_entropy_norm")
    if _fin(hn) and (hn < -1e-3 or hn > 1.0 + 1e-3):
        out.append(
            f"tp_embstats_topic_entropy_norm: ожидается в [0,1], got {hn}"
        )

    pp = f("tp_embstats_topic_perplexity")
    if _fin(pp) and pp < 1.0 - 1e-3:
        out.append("tp_embstats_topic_perplexity: ожидается >= 1 (finite)")

    nch = f("tp_embstats_n_chunks")
    if _fin(nch) and nch < -1e-6:
        out.append("tp_embstats_n_chunks: ожидается >= 0")

    dm = f("tp_embstats_dim")
    if _fin(dm) and dm < 1.0 - 1e-3:
        out.append("tp_embstats_dim: ожидается >= 1 (finite)")

    w = f("tp_embstats_source_used_whisper")
    y = f("tp_embstats_source_used_youtube_auto")
    if _fin(w) and _fin(y) and w + y > 1.0 + 1e-3:
        out.append(
            f"source_used whisper+youtube: сумма {w+y} (ожидается <= 1)"
        )

    em = f("tp_embstats_emit_extra_metrics_enabled")
    if _fin(em) and em < 0.5:
        for k in _TIME_KEYS:
            if _fin(f(k)):
                out.append(
                    f"{k}: ожидается NaN при emit_extra_metrics_enabled=0, got {f(k)}"
                )
    if _fin(em) and em > 0.5:
        for k in _TIME_KEYS:
            t = f(k)
            if _fin(t) and (t < -1e-6 or t > 1.0e7):
                out.append(
                    f"{k}: вне [0, 1e7] мс ({t})"
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
        description="Срез tp_embstats_* (39) в text_features.npz (embedding_stats_extractor_output_v1)"
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
