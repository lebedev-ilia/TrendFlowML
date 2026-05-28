#!/usr/bin/env python3
"""Валидатор среза `lexico_static_features` (67 × `tp_lex_*`) в `text_processor/text_features.npz`.

Схема: `lexico_static_features_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/lexico_static_features_output_v1.json"

_POLICY_TRI: Tuple[str, ...] = (
    "tp_lex_transcript_source_policy_asr_only",
    "tp_lex_transcript_source_policy_asr_then_legacy",
    "tp_lex_transcript_source_policy_legacy_only",
)

_USED_TRI: Tuple[str, ...] = (
    "tp_lex_transcript_source_used_asr",
    "tp_lex_transcript_source_used_legacy",
    "tp_lex_transcript_source_used_none",
)

# Метрики-доли / score в [0, 1] при finite (readability / upper / entropy — отдельно).
_RATIO_0_1: Tuple[str, ...] = (
    "tp_lex_title_stopword_ratio",
    "tp_lex_title_type_token_ratio",
    "tp_lex_title_punctuation_ratio",
    "tp_lex_title_capital_words_ratio",
    "tp_lex_title_clickbait_score",
    "tp_lex_transcript_question_ratio",
    "tp_lex_transcript_lexical_diversity",
    "tp_lex_transcript_rare_word_ratio",
    "tp_lex_transcript_stopword_ratio",
    "tp_lex_transcript_orthographic_error_rate",
    "tp_lex_transcript_avg_token_frequency_percentile",
    "tp_lex_emoji_diversity",
    "tp_lex_special_character_ratio",
)

_NON_BINARY: frozenset = frozenset(
    {
        "tp_lex_compute_ms",
        "tp_lex_description_chars_kept",
        "tp_lex_description_chars_used",
        "tp_lex_description_emoji_count",
        "tp_lex_description_len_words",
        "tp_lex_description_num_mentions",
        "tp_lex_description_num_urls",
        "tp_lex_emoji_diversity",
        "tp_lex_load_ms",
        "tp_lex_named_entity_density",
        "tp_lex_punctuation_entropy",
        "tp_lex_special_character_ratio",
        "tp_lex_title_avg_word_len",
        "tp_lex_title_capital_words_ratio",
        "tp_lex_title_chars_kept",
        "tp_lex_title_chars_used",
        "tp_lex_title_clickbait_score",
        "tp_lex_title_emoji_count",
        "tp_lex_title_exclamation_count",
        "tp_lex_title_len_chars",
        "tp_lex_title_len_words",
        "tp_lex_title_punctuation_ratio",
        "tp_lex_title_question_count",
        "tp_lex_title_stopword_ratio",
        "tp_lex_title_type_token_ratio",
        "tp_lex_transcript_avg_sentence_len",
        "tp_lex_transcript_avg_token_frequency_percentile",
        "tp_lex_transcript_chars_kept",
        "tp_lex_transcript_chars_used",
        "tp_lex_transcript_len_words",
        "tp_lex_transcript_lexical_diversity",
        "tp_lex_transcript_orthographic_error_rate",
        "tp_lex_transcript_question_ratio",
        "tp_lex_transcript_rare_word_ratio",
        "tp_lex_transcript_readability_score",
        "tp_lex_transcript_stopword_ratio",
        "tp_lex_upper_lower_ratio_title",
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

assert len(EXPECTED_KEYS) == 67
assert len(_NON_BINARY) == 37
assert len(_BINARY_KEYS) == 30
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
                f"срез tp_lex_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    ut = [f(k) for k in _USED_TRI]
    if all(_fin(x) for x in ut) and abs(sum(ut) - 1.0) > 1e-3:
        out.append(f"one-hot transcript used: сумма {sum(ut)} (ожидается 1)")

    for k in _RATIO_0_1:
        x = f(k)
        if _fin(x) and (x < -1e-3 or x > 1.0 + 1e-3):
            out.append(f"{k}: ожидается в [0,1] (finite), got {x}")

    rs = f("tp_lex_transcript_readability_score")
    if _fin(rs) and rs < -1e-6:
        out.append("tp_lex_transcript_readability_score: ожидается >= 0")

    ul = f("tp_lex_upper_lower_ratio_title")
    if _fin(ul) and ul < -1e-6:
        out.append("tp_lex_upper_lower_ratio_title: ожидается >= 0 (finite)")

    pe = f("tp_lex_punctuation_entropy")
    if _fin(pe) and pe < -1e-6:
        out.append("tp_lex_punctuation_entropy: ожидается >= 0 (finite)")

    seen = set(_RATIO_0_1) | {
        "tp_lex_transcript_readability_score",
        "tp_lex_upper_lower_ratio_title",
        "tp_lex_punctuation_entropy",
        "tp_lex_named_entity_density",
        "tp_lex_load_ms",
        "tp_lex_compute_ms",
    }
    for k in _NON_BINARY:
        if k in seen:
            continue
        t = f(k)
        if _fin(t) and t < -1e-6:
            out.append(f"{k}: ожидается >= 0 (finite), got {t}")

    for k in ("tp_lex_load_ms", "tp_lex_compute_ms"):
        t = f(k)
        if _fin(t) and (t < -1e-6 or t > 1.0e7):
            out.append(
                f"{k}: вне [0, 1e7] (finite), got {t}"
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
        description="Срез tp_lex_* (67) в text_features.npz (lexico_static_features_output_v1)"
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
