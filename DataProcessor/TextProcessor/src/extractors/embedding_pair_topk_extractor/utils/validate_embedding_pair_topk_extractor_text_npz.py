#!/usr/bin/env python3
"""Валидатор среза `embedding_pair_topk_extractor` (69 × `tp_embpair_*` + legacy `tp_pairtopk_*`) в `text_processor/text_features.npz`.

Схема: `embedding_pair_topk_extractor_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/embedding_pair_topk_extractor_output_v1.json"

_COSINE_KEYS: Tuple[str, ...] = (
    "tp_embpair_title_desc_cosine",
    "tp_pairtopk_title_desc_cosine",
) + tuple(
    f"tp_embpair_title_transcript_top{i}" for i in range(1, 9)
) + tuple(
    f"tp_pairtopk_title_transcript_top{i}" for i in range(1, 9)
) + (
    "tp_embpair_title_transcript_topk_max",
    "tp_embpair_title_transcript_topk_mean",
    "tp_pairtopk_title_transcript_topk_max",
    "tp_pairtopk_title_transcript_topk_mean",
)

_INT_GE1: Tuple[str, ...] = (
    "tp_embpair_top_k",
    "tp_embpair_top_k_slots",
    "tp_embpair_top_k_slots_requested",
    "tp_pairtopk_top_k",
)

_INT_GE0: Tuple[str, ...] = ("tp_embpair_min_corpus_for_faiss",)

_SCHEMA_SLOTS_KEY = "tp_embpair_schema_slots_max"
_SCHEMA_SLOTS_MAX = 8.0

_INDEX_KEYS: Tuple[str, ...] = tuple(
    f"tp_embpair_title_transcript_top{i}_idx" for i in range(1, 9)
)

_EXTRA_BLOCK: Tuple[str, ...] = (
    "tp_embpair_n_chunks",
    "tp_embpair_transcript_source_whisper",
    "tp_embpair_transcript_source_youtube_auto",
    "tp_embpair_transcript_source_combined",
    "tp_embpair_use_faiss_mode",
    "tp_embpair_require_faiss",
)

_FAISS_TRIPLET: Tuple[str, ...] = (
    "tp_embpair_use_faiss_mode_auto",
    "tp_embpair_use_faiss_mode_never",
    "tp_embpair_use_faiss_mode_always",
)

_TRANSCRIPT_ONEHOT: Tuple[str, ...] = (
    "tp_embpair_transcript_source_whisper",
    "tp_embpair_transcript_source_youtube_auto",
    "tp_embpair_transcript_source_combined",
)

_NON_BINARY: Set[str] = (
    set(_COSINE_KEYS)
    | set(_INT_GE1)
    | set(_INT_GE0)
    | {_SCHEMA_SLOTS_KEY}
    | set(_INDEX_KEYS)
    | set(_EXTRA_BLOCK)
)

def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
_BINARY_KEYS: Set[str] = EXPECTED_KEYS - _NON_BINARY


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
                f"срез embpair пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    trip = [f(k) for k in _FAISS_TRIPLET]
    if all(_fin(x) for x in trip) and abs(sum(trip) - 1.0) > 1e-3:
        out.append(
            f"FAISS triplet: ожидается one-hot, сумма={sum(trip)}"
        )

    xsm = f(_SCHEMA_SLOTS_KEY)
    if _fin(xsm) and abs(xsm - _SCHEMA_SLOTS_MAX) > 1e-3:
        out.append(
            f"{_SCHEMA_SLOTS_KEY}: ожидается {_SCHEMA_SLOTS_MAX}, got {xsm}"
        )

    for k in _INT_GE1:
        x = f(k)
        if _fin(x) and x < 1.0 - 1e-3:
            out.append(f"{k}: ожидается >= 1, got {x}")

    for k in _INT_GE0:
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0, got {x}")

    for k in _COSINE_KEYS:
        x = f(k)
        if _fin(x) and (x < -1.0 - 1e-3 or x > 1.0 + 1e-3):
            out.append(f"{k}: косинус/скор вне [-1,1] ({x})")

    for k in _INDEX_KEYS:
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0, got {x}")

    a = f("tp_embpair_title_desc_cosine")
    b = f("tp_pairtopk_title_desc_cosine")
    if _fin(a) and _fin(b) and abs(a - b) > 1e-4:
        out.append("tp_pairtopk_title_desc_cosine != tp_embpair_title_desc_cosine (legacy)")

    for i in range(1, 9):
        em = f("tp_embpair_title_transcript_top{i}")
        pr = f("tp_pairtopk_title_transcript_top{i}")
        if _fin(f(em)) and _fin(f(pr)) and abs(f(em) - f(pr)) > 1e-4:
            out.append(
                f"tp_pairtopk_title_transcript_top{i} != tp_embpair_title_transcript_top{i}"
            )
        if (math.isnan(f(em)) and _fin(f(pr))) or (_fin(f(em)) and math.isnan(f(pr))):
            out.append(
                f"top{i}: несоответствие NaN/числа между emb и legacy"
            )

    p_legacy = f("tp_pairtopk_present")
    p_topk = f("tp_embpair_title_transcript_topk_present")
    if _fin(p_legacy) and _fin(p_topk) and abs(p_legacy - p_topk) > 1e-4:
        out.append(
            "tp_pairtopk_present != tp_embpair_title_transcript_topk_present (ожидается зеркалирование)"
        )

    ex_all_nan = all(not _fin(f(k)) for k in _EXTRA_BLOCK)
    ex_all_fin = all(_fin(f(k)) for k in _EXTRA_BLOCK)
    if not ex_all_nan and not ex_all_fin:
        out.append(
            "блок extra: ожидается либо все NaN (emit_extra_metrics=False), либо все finite"
        )
    ohx = [f(k) for k in _TRANSCRIPT_ONEHOT]
    if ex_all_fin:
        tsum = float(sum(ohx))
        if tsum < -1e-6 or tsum > 1.0 + 1e-3:
            out.append(
                f"one-hot transcript source (extra): сумма {tsum} (ожидается [0,1])"
            )
        um = f("tp_embpair_use_faiss_mode")
        if _fin(um):
            dev = min(abs(um - 0.0), abs(um - 0.5), abs(um - 1.0))
            if dev > 1e-3:
                out.append(
                    f"tp_embpair_use_faiss_mode: ожидается 0 / 0.5 / 1, got {um}"
                )
        nc = f("tp_embpair_n_chunks")
        if _fin(nc) and nc < -1e-6:
            out.append("tp_embpair_n_chunks: ожидается >= 0")
        rf = f("tp_embpair_require_faiss")
        if _fin(rf) and (rf < -1e-6 or rf > 1.0 + 1e-6):
            out.append("tp_embpair_require_faiss: ожидается 0/1 (extra)")

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
        description="Срез tp_embpair_* + tp_pairtopk_* (69) в text_features.npz (embedding_pair_topk_extractor_output_v1)"
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
