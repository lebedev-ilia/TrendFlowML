#!/usr/bin/env python3
"""Валидатор среза `title_embedding_cluster_entropy` (24 × `tp_titleclent_*`) в `text_processor/text_features.npz`.

Схема: `title_embedding_cluster_entropy_extractor_output_v1`. Extra-поля — NaN при `emit_extra_metrics=False` или на empty-ветке.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/title_embedding_cluster_entropy_extractor_output_v1.json"

_EXTRA_BLOCK: tuple[str, ...] = (
    "tp_titleclent_n_clusters",
    "tp_titleclent_model_orig_dim",
    "tp_titleclent_model_reduced_dim",
    "tp_titleclent_margin_top2",
    "tp_titleclent_compute_ms",
)

_NON_BINARY: frozenset = frozenset(
    {
        "tp_titleclent_schema_top_k_slots_max",
        "tp_titleclent_top_k_slots_requested",
        "tp_titleclent_top_k_slots",
        "tp_titleclent_temperature",
        "tp_titleclent_entropy_raw",
        "tp_titleclent_entropy_norm",
        "tp_titleclent_perplexity",
        "tp_titleclent_top_k_used",
        "tp_titleclent_distinct_clusters_topk",
    }
) | frozenset(_EXTRA_BLOCK)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    with open(_repo_root_from_here() / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
_BINARY_KEYS: Set[str] = EXPECTED_KEYS - set(_NON_BINARY)

assert len(EXPECTED_KEYS) == 24
assert len(_NON_BINARY) == 14
assert len(_BINARY_KEYS) == 10


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
                f"срез tp_titleclent_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
            )
        return out
    if keys != EXPECTED_KEYS:
        miss = sorted(EXPECTED_KEYS - keys)
        if miss:
            out.append(
                f"не хватает: {miss[:8]}{'...' if len(miss) > 8 else ''}"
            )
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
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-3):
            out.append(f"{k}: ожидается 0/1, got {x}")

    sm = f("tp_titleclent_schema_top_k_slots_max")
    if _fin(sm) and abs(sm - 8.0) > 1e-3:
        out.append("tp_titleclent_schema_top_k_slots_max: ожидается 8.0")

    ts = f("tp_titleclent_top_k_slots")
    if _fin(ts) and (ts < 1.0 - 1e-3 or ts > 8.0 + 1e-3):
        out.append("tp_titleclent_top_k_slots: вне [1, 8] (finite)")

    tr = f("tp_titleclent_top_k_slots_requested")
    if _fin(tr) and tr < 1.0 - 1e-3:
        out.append("tp_titleclent_top_k_slots_requested: ожидается >= 1")

    te = f("tp_titleclent_temperature")
    if _fin(te) and te <= 0.0:
        out.append("tp_titleclent_temperature: ожидается > 0 (finite)")

    ent = f("tp_titleclent_entropy_raw")
    if _fin(ent) and ent < -1e-6:
        out.append("tp_titleclent_entropy_raw: ожидается >= 0")

    eno = f("tp_titleclent_entropy_norm")
    if _fin(eno) and (eno < -1e-3 or eno > 1.0 + 1e-3):
        out.append("tp_titleclent_entropy_norm: вне [0,1] (typical)")

    ppx = f("tp_titleclent_perplexity")
    if _fin(ppx) and ppx < 1.0 - 1e-3:
        out.append("tp_titleclent_perplexity: ожидается >= 1 (finite)")

    mar = f("tp_titleclent_margin_top2")
    if _fin(mar) and (mar < -2.0 - 1e-3 or mar > 2.0 + 1e-3):
        out.append("tp_titleclent_margin_top2: вне [-2,2] (typical)")

    for k in ("tp_titleclent_n_clusters", "tp_titleclent_model_orig_dim", "tp_titleclent_model_reduced_dim"):
        t = f(k)
        if _fin(t) and t < 1.0 - 1e-3:
            out.append(f"{k}: ожидается >= 1 (finite)")

    cm = f("tp_titleclent_compute_ms")
    if _fin(cm) and (cm < -1e-3 or cm > 1.0e7):
        out.append("tp_titleclent_compute_ms: вне [0, 1e7] мс")

    em = f("tp_titleclent_emit_extra_metrics_enabled")
    if _fin(em) and em < 0.5:
        for k in _EXTRA_BLOCK:
            if _fin(f(k)):
                out.append(
                    f"{k}: ожидается NaN при emit_extra_metrics_enabled=0, got {f(k)}"
                )
    if _fin(em) and em > 0.5:
        pr = f("tp_titleclent_present")
        if _fin(pr) and pr > 0.5:
            for k in _EXTRA_BLOCK:
                if k == "tp_titleclent_margin_top2":
                    continue
                if not _fin(f(k)):
                    out.append(
                        f"{k}: ожидается finite при present=1 и emit=1, got {f(k)}"
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
        description="Срез tp_titleclent_* (24) в text_features.npz (title_embedding_cluster_entropy_extractor_output_v1)"
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
