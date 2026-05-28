#!/usr/bin/env python3
"""Валидатор среза `speaker_turn_embeddings_aggregator` (17 × `tp_spkemb_*`) в `text_processor/text_features.npz`.

Схема: `speaker_turn_embeddings_aggregator_output_v1`. Extra-поля — NaN при `emit_extra_metrics=False`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/speaker_turn_embeddings_aggregator_output_v1.json"

_EXTRA: tuple[str, ...] = (
    "tp_spkemb_batch_size",
    "tp_spkemb_max_speakers",
    "tp_spkemb_max_turns_per_speaker",
    "tp_spkemb_min_chars_per_turn",
    "tp_spkemb_max_chars_per_turn",
)

_MODE_PAIR: tuple[str, str] = (
    "tp_spkemb_input_mode_diar_asr",
    "tp_spkemb_input_mode_legacy_doc_speakers",
)

_NON_BINARY: frozenset = frozenset(
    {
        "tp_spkemb_speakers_total",
        "tp_spkemb_speakers_embedded",
        "tp_spkemb_turns_total",
    }
)
_NON_BINARY |= frozenset(_EXTRA)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
_BINARY_KEYS: Set[str] = EXPECTED_KEYS - set(_NON_BINARY)

assert len(EXPECTED_KEYS) == 17
assert len(_NON_BINARY) == 8
assert len(_BINARY_KEYS) == 9
assert not (_NON_BINARY & _BINARY_KEYS)


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
                f"срез tp_spkemb_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    st = f("tp_spkemb_speakers_total")
    se = f("tp_spkemb_speakers_embedded")
    tt = f("tp_spkemb_turns_total")
    for k, v in (("tp_spkemb_speakers_total", st), ("tp_spkemb_speakers_embedded", se), ("tp_spkemb_turns_total", tt)):
        if _fin(v) and v < -1e-6:
            out.append(f"{k}: ожидается >= 0")
    if _fin(st) and _fin(se) and se - st > 1e-3:
        out.append("tp_spkemb_speakers_embedded: ожидается <= speakers_total")

    m0, m1 = f(_MODE_PAIR[0]), f(_MODE_PAIR[1])
    if _fin(m0) and _fin(m1) and m0 > 0.5 and m1 > 0.5:
        out.append("input_mode: diar_asr и legacy не могут быть 1 одновременно")

    pr = f("tp_spkemb_present")
    if _fin(pr) and pr > 0.5:
        if _fin(se) and se < 1.0 - 1e-3:
            out.append("tp_spkemb_present=1, но speakers_embedded < 1")

    ex_fin = [ _fin(f(k)) for k in _EXTRA ]
    if any(ex_fin) and not all(ex_fin):
        out.append("extra-поля: ожидаются либо все NaN, либо все finite")
    if all(ex_fin):
        bs = f("tp_spkemb_batch_size")
        if bs < 1.0 - 1e-3:
            out.append("tp_spkemb_batch_size: ожидается >= 1 (extra)")

        for k in ("tp_spkemb_max_speakers", "tp_spkemb_max_turns_per_speaker"):
            if f(k) < 1.0 - 1e-3:
                out.append(f"{k}: ожидается >= 1 (extra)")

        mn = f("tp_spkemb_min_chars_per_turn")
        mx = f("tp_spkemb_max_chars_per_turn")
        if mx < 1.0 - 1e-3:
            out.append("tp_spkemb_max_chars_per_turn: ожидается > 0 (extra)")
        if _fin(mn) and _fin(mx) and mn - mx > 1e-3:
            out.append("tp_spkemb_min_chars: ожидается <= max_chars (extra)")

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
        description="Срез tp_spkemb_* (17) в text_features.npz (speaker_turn_embeddings_aggregator_output_v1)"
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
