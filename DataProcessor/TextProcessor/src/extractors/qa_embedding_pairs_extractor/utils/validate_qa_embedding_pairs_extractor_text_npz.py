#!/usr/bin/env python3
"""Валидатор среза `qa_embedding_pairs_extractor` (34 × `tp_qa_*`) в `text_processor/text_features.npz`.

Схема: `qa_embedding_pairs_extractor_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/qa_embedding_pairs_extractor_output_v1.json"

_POLICY_TRI: Tuple[str, ...] = (
    "tp_qa_transcript_source_policy_asr_only",
    "tp_qa_transcript_source_policy_asr_then_legacy",
    "tp_qa_transcript_source_policy_legacy_only",
)

_NON_BINARY: frozenset = frozenset(
    {
        "tp_qa_num_questions",
        "tp_qa_embedding_dim",
        "tp_qa_q_title",
        "tp_qa_q_description",
        "tp_qa_q_transcript",
        "tp_qa_q_comments",
        "tp_qa_require_min_questions",
        "tp_qa_max_questions_total",
        "tp_qa_max_questions_per_source",
        "tp_qa_max_comments",
        "tp_qa_max_chars_per_comment",
        "tp_qa_max_transcript_chars",
        "tp_qa_min_chars_per_question",
        "tp_qa_max_question_chars",
        "tp_qa_questions_per_min",
        "tp_qa_questions_per_1k_chars",
        "tp_qa_mean_cosine_to_centroid",
    }
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
_BINARY_KEYS: Set[str] = EXPECTED_KEYS - set(_NON_BINARY)

assert len(EXPECTED_KEYS) == 34
assert len(_NON_BINARY) == 17
assert len(_BINARY_KEYS) == 17


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
                f"срез tp_qa_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    pt = [f(k) for k in _POLICY_TRI]
    if all(_fin(x) for x in pt) and abs(sum(pt) - 1.0) > 1e-3:
        out.append(f"one-hot policy: сумма {sum(pt)} (ожидается 1)")

    nq = f("tp_qa_num_questions")
    ssum = sum(
        f(k)
        for k in (
            "tp_qa_q_title",
            "tp_qa_q_description",
            "tp_qa_q_transcript",
            "tp_qa_q_comments",
        )
    )
    if all(_fin(nq) and _fin(f(k)) for k in (
        "tp_qa_q_title",
        "tp_qa_q_description",
        "tp_qa_q_transcript",
        "tp_qa_q_comments",
    )):
        if abs(nq - ssum) > 0.5:
            out.append(
                f"сумма по источникам ({ssum}) != tp_qa_num_questions ({nq})"
            )

    mc = f("tp_qa_mean_cosine_to_centroid")
    if _fin(mc) and (mc < -1.0 - 1e-3 or mc > 1.0 + 1e-3):
        out.append(
            f"tp_qa_mean_cosine_to_centroid: вне [-1,1] ({mc})"
        )
    mcp = f("tp_qa_mean_cosine_to_centroid_present")
    if _fin(mcp) and mcp > 0.5 and (not _fin(mc)):
        out.append("mean_cosine_to_centroid_present=1 при NaN mean_cosine")

    pr = f("tp_qa_present")
    ed = f("tp_qa_embedding_dim")
    if _fin(pr) and pr > 0.5:
        if not _fin(ed) or ed < 1.0 - 1e-3:
            out.append(
                "tp_qa_present=1 ожидает tp_qa_embedding_dim >= 1 (finite)"
            )

    for k in set(_NON_BINARY) - {
        "tp_qa_mean_cosine_to_centroid",
        "tp_qa_questions_per_min",
        "tp_qa_questions_per_1k_chars",
    }:
        t = f(k)
        if _fin(t) and t < -1e-6:
            out.append(f"{k}: ожидается >= 0, got {t}")

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
        description="Срез tp_qa_* (34) в text_features.npz (qa_embedding_pairs_extractor_output_v1)"
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
