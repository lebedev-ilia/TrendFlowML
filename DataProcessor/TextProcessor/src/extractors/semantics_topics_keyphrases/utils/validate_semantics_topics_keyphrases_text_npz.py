#!/usr/bin/env python3
"""Валидатор среза `semantics_topics_keyphrases` (116 × `tp_topics_*`) в `text_processor/text_features.npz`.

Схема: `semantics_topics_keyphrases_output_v1`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/semantics_topics_keyphrases_output_v1.json"

_TRANSCRIPT_ONEHOT: Tuple[str, ...] = (
    "tp_topics_transcript_source_policy_asr_only",
    "tp_topics_transcript_source_policy_asr_then_legacy",
    "tp_topics_transcript_source_policy_legacy_only",
)

_EXPORT_ONEHOT: Tuple[str, ...] = (
    "tp_topics_export_keyphrases_mode_raw",
    "tp_topics_export_keyphrases_mode_hashed",
    "tp_topics_export_keyphrases_mode_none",
)

_EXTRA_BLOCK: Tuple[str, ...] = (
    "tp_topics_extra_model_load_ms",
    "tp_topics_extra_topics_pipeline_ms",
    "tp_topics_extra_keyphrases_encode_ms",
    "tp_topics_extra_topics_db_digest_u24",
    "tp_topics_extra_model_digest_u24",
)


def _build_non_binary() -> Set[str]:
    s: Set[str] = {
        "tp_topics_text_chars",
        "tp_topics_schema_topic_slots_max",
        "tp_topics_top_k_slots_requested",
        "tp_topics_top_k_slots",
        "tp_topics_schema_kp_slots_max",
        "tp_topics_keyphrase_slots_requested",
        "tp_topics_keyphrase_slots",
        "tp_topics_top_k_topics",
        "tp_topics_temperature",
        "tp_topics_entropy_topk",
        "tp_topics_entropy_topk_norm",
        "tp_topics_perplexity_topk",
        "tp_topics_keyphrases_count",
        "tp_topics_keyphrases_dim",
        "tp_topics_style_faq_qmarks",
        "tp_topics_keyphrase_score_top1",
        "tp_topics_keyphrase_score_mean",
    }
    s.update(_EXTRA_BLOCK)
    for i in range(1, 9):
        s.add(f"tp_topics_topic_top{i}_id")
        s.add(f"tp_topics_topic_top{i}_score")
        s.add(f"tp_topics_topic_top{i}_prob")
    for i in range(1, 17):
        s.add(f"tp_topics_kp_top{i}_hash01")
        s.add(f"tp_topics_kp_top{i}_len")
    return s


_NON_BINARY: frozenset = frozenset(_build_non_binary())


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
_BINARY_KEYS: Set[str] = EXPECTED_KEYS - set(_NON_BINARY)

assert len(EXPECTED_KEYS) == 116
assert len(_NON_BINARY) == 78
assert len(_BINARY_KEYS) == 38
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
                f"срез tp_topics_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    tr = [f(k) for k in _TRANSCRIPT_ONEHOT]
    if all(_fin(x) for x in tr) and abs(sum(tr) - 1.0) > 1e-2:
        out.append(
            f"transcript one-hot: сумма {sum(tr)} (ожидается 1)"
        )

    exm = [f(k) for k in _EXPORT_ONEHOT]
    if all(_fin(x) for x in exm) and abs(sum(exm) - 1.0) > 1e-2:
        out.append(
            f"export_keyphrases one-hot: сумма {sum(exm)} (ожидается 1)"
        )

    tc = f("tp_topics_text_chars")
    if _fin(tc) and tc < -1e-6:
        out.append("tp_topics_text_chars: ожидается >= 0")

    t8 = f("tp_topics_schema_topic_slots_max")
    if _fin(t8) and abs(t8 - 8.0) > 1e-3:
        out.append("tp_topics_schema_topic_slots_max: ожидается 8.0")
    t16 = f("tp_topics_schema_kp_slots_max")
    if _fin(t16) and abs(t16 - 16.0) > 1e-3:
        out.append("tp_topics_schema_kp_slots_max: ожидается 16.0")

    tpos = f("tp_topics_top_k_slots")
    if _fin(tpos) and (tpos < 1.0 - 1e-3 or tpos > 8.0 + 1e-3):
        out.append("tp_topics_top_k_slots: вне [1, 8]")

    kps = f("tp_topics_keyphrase_slots")
    if _fin(kps) and (kps < 0.0 - 1e-3 or kps > 16.0 + 1e-3):
        out.append("tp_topics_keyphrase_slots: вне [0, 16]")

    te = f("tp_topics_temperature")
    if _fin(te) and te <= 0.0:
        out.append("tp_topics_temperature: ожидается > 0 (finite)")

    ent = f("tp_topics_entropy_topk")
    if _fin(ent) and ent < -1e-6:
        out.append("tp_topics_entropy_topk: ожидается >= 0")

    eno = f("tp_topics_entropy_topk_norm")
    if _fin(eno) and (eno < -1e-3 or eno > 1.0 + 1e-3):
        out.append("tp_topics_entropy_topk_norm: вне [0,1] (typical)")

    ppx = f("tp_topics_perplexity_topk")
    if _fin(ppx) and ppx < 1.0 - 1e-3:
        out.append("tp_topics_perplexity_topk: ожидается >= 1 (finite)")

    nkp = f("tp_topics_keyphrases_count")
    if _fin(nkp) and nkp < 0.0 - 1e-6:
        out.append("tp_topics_keyphrases_count: ожидается >= 0")

    sfq = f("tp_topics_style_faq_qmarks")
    if _fin(sfq) and sfq < 0.0 - 1e-6:
        out.append("tp_topics_style_faq_qmarks: ожидается >= 0")

    for i in range(1, 9):
        tid, sc, pr = f(f"tp_topics_topic_top{i}_id"), f(
            f"tp_topics_topic_top{i}_score"
        ), f(f"tp_topics_topic_top{i}_prob")
        any_f = _fin(tid) or _fin(sc) or _fin(pr)
        all_f = _fin(tid) and _fin(sc) and _fin(pr)
        if any_f and not all_f:
            out.append(
                f"topic top{i}: id/score/prob должны быть все finite или все NaN"
            )
        if all_f:
            if tid < -0.5:
                out.append(
                    f"tp_topics_topic_top{i}_id: ожидается >= 0, got {tid}"
                )
            if sc < -1.0 - 1e-2 or sc > 1.0 + 1e-2:
                out.append(
                    f"tp_topics_topic_top{i}_score: вне [-1,1] ({sc})"
                )
            if pr < -1e-3 or pr > 1.0 + 1e-3:
                out.append(
                    f"tp_topics_topic_top{i}_prob: вне [0,1] ({pr})"
                )

    for i in range(1, 17):
        pre = f(f"tp_topics_kp_top{i}_present")
        h0 = f(f"tp_topics_kp_top{i}_hash01")
        ln = f(f"tp_topics_kp_top{i}_len")
        if not _fin(pre):
            continue
        if pre < 0.5:
            if _fin(h0) or _fin(ln):
                out.append(
                    f"tp_topics_kp_top{i}_hash/len: ожидается NaN при present=0"
                )
        else:
            if not _fin(h0) or h0 < -1e-3 or h0 > 255.0 + 1e-3:
                out.append(
                    f"tp_topics_kp_top{i}_hash01: вне [0,255] при present=1"
                )
            if not _fin(ln) or ln < 0.0 - 1e-3:
                out.append(
                    f"tp_topics_kp_top{i}_len: ожидается >=0 при present=1"
                )

    for k in ("tp_topics_keyphrase_score_top1", "tp_topics_keyphrase_score_mean"):
        x = f(k)
        if _fin(x) and x < 0.0 - 1e-6:
            out.append(f"{k}: ожидается >=0 (typical)")

    em = f("tp_topics_emit_extra_metrics_enabled")
    if _fin(em) and em < 0.5:
        for k in _EXTRA_BLOCK:
            if _fin(f(k)):
                out.append(
                    f"{k}: ожидается NaN при emit_extra=0, got {f(k)}"
                )
    if _fin(em) and em > 0.5:
        for k in _EXTRA_BLOCK:
            if k.endswith("_ms"):
                v = f(k)
                if _fin(v) and (v < -1e-3 or v > 1.0e7):
                    out.append(
                        f"{k}: вне [0, 1e7] мс (finite extra)"
                    )
            elif k.endswith("_u24"):
                v = f(k)
                if _fin(v) and (v < -1.0 or v > 16777215.0 + 1.0):
                    out.append(
                        f"{k}: ожидается u24 digest [0, 0xFFFFFF], got {v}"
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
        description="Срез tp_topics_* (116) в text_features.npz (semantics_topics_keyphrases_output_v1)"
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
