#!/usr/bin/env python3
"""Валидатор среза `tags_extractor` (`tp_tags_*`) в `text_processor/text_features.npz`.

Схема: `tags_extractor_output_v1` (база) + динамические `tp_tags_top{i}_*` для i=1..K (K = `tp_tags_topk_slots`).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/tags_extractor_output_v1.json"

_EXPORT_CLEAN: Tuple[str, str] = (
    "tp_tags_export_cleaned_texts_mode_none",
    "tp_tags_export_cleaned_texts_mode_raw",
)

_EXPORT_HT: Tuple[str, str, str] = (
    "tp_tags_export_hashtags_mode_none",
    "tp_tags_export_hashtags_mode_raw",
    "tp_tags_export_hashtags_mode_hashed",
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_base_keys() -> Set[str]:
    root = _repo_root_from_here()
    with open(root / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


BASE_KEYS: Set[str] = _load_base_keys()
assert len(BASE_KEYS) == 28

# Для 0/1: все base кроме явно непрерывных счётчиков/плотностей/avg/topk_slots
_CATEGORICAL_OR_BINARY: Set[str] = set(BASE_KEYS) - {
    "tp_tags_json_hashtag_merged_count",
    "tp_tags_title_hashtag_found_count",
    "tp_tags_description_hashtag_found_count",
    "tp_tags_hashtag_total_found_count",
    "tp_tags_hashtag_unique_count",
    "tp_tags_hashtag_avg_len",
    "tp_tags_hashtag_max_len",
    "tp_tags_title_hashtag_density_per_char",
    "tp_tags_description_hashtag_density_per_char",
    "tp_tags_topk_slots",
}


def _expected_tag_keys(k_slots: int) -> Set[str]:
    k_slots = int(k_slots)
    ex = set(BASE_KEYS)
    for i in range(1, k_slots + 1):
        ex.add(f"tp_tags_top{i}_present")
        ex.add(f"tp_tags_top{i}_hash01")
        ex.add(f"tp_tags_top{i}_len")
    return ex


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


def _build_full_slice(d: Dict[str, Any]) -> Tuple[Dict[str, float], Set[str]]:
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
        bag[n] = float(v[i])
    return bag, set(bag.keys())


def _tags_only(bag: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in bag.items() if k.startswith("tp_tags_")}


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        bag, keys = _build_full_slice(d)
        if not keys:
            return extract_meta(d).get("status") != "ok"
        tbag = _tags_only(bag)
        if not tbag:
            return extract_meta(d).get("status") != "ok"
        k0 = int(round(tbag.get("tp_tags_topk_slots", 0.0)))
        if k0 < 1:
            return False
        return set(tbag.keys()) == _expected_tag_keys(k0)
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
    bag, _ = _build_full_slice(d)
    tbag = _tags_only(bag)
    meta = extract_meta(d)
    if not tbag:
        if meta.get("status") == "ok":
            out.append("срез tp_tags_* пуст при meta.status=ok")
        return out

    ks = tbag.get("tp_tags_topk_slots")
    if not math.isfinite(ks) or int(round(ks)) < 1:
        out.append("tp_tags_topk_slots: ожидается finite >= 1")
        return out
    k0 = int(round(ks))
    exp = _expected_tag_keys(k0)
    got = set(tbag.keys())
    if got != exp:
        miss = sorted(exp - got)
        extra = sorted(got - exp)
        if miss:
            out.append(
                f"не хватает tp_tags_*: {miss[:10]}{'...' if len(miss) > 10 else ''}"
            )
        if extra:
            out.append(
                f"лишние tp_tags_*: {extra[:10]}{'...' if len(extra) > 10 else ''}"
            )
    return out


def _fin(x: float) -> bool:
    return bool(math.isfinite(x))


def validate_ranges(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    bag, _ = _build_full_slice(d)
    tbag = _tags_only(bag)
    meta = extract_meta(d)
    if not tbag or meta.get("status") != "ok":
        return out
    k0 = int(round(tbag.get("tp_tags_topk_slots", 0.0)))
    if k0 < 1 or set(tbag.keys()) != _expected_tag_keys(k0):
        return out

    def f(k: str) -> float:
        return tbag.get(k, float("nan"))

    for k in _CATEGORICAL_OR_BINARY:
        if k in ("tp_tags_export_cleaned_texts_mode_none", "tp_tags_export_cleaned_texts_mode_raw"):
            continue
        if k in _EXPORT_HT:
            continue
        x = f(k)
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    c0, c1 = f(_EXPORT_CLEAN[0]), f(_EXPORT_CLEAN[1])
    if all(_fin(x) for x in (c0, c1)) and abs(c0 + c1 - 1.0) > 1e-2:
        out.append("export_cleaned one-hot: сумма должна быть 1")

    h0, h1, h2 = f(_EXPORT_HT[0]), f(_EXPORT_HT[1]), f(_EXPORT_HT[2])
    if all(_fin(x) for x in (h0, h1, h2)) and abs(h0 + h1 + h2 - 1.0) > 1e-2:
        out.append("export_hashtags one-hot: сумма должна быть 1")

    title_c = f("tp_tags_title_hashtag_found_count")
    desc_c = f("tp_tags_description_hashtag_found_count")
    tot = f("tp_tags_hashtag_total_found_count")
    if all(_fin(x) for x in (title_c, desc_c, tot)) and abs(tot - (title_c + desc_c)) > 1e-2:
        out.append("hashtag_total_found_count: ожидается title+description")

    for k in (
        "tp_tags_json_hashtag_merged_count",
        "tp_tags_title_hashtag_found_count",
        "tp_tags_description_hashtag_found_count",
        "tp_tags_hashtag_total_found_count",
        "tp_tags_hashtag_unique_count",
    ):
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0")

    uq = f("tp_tags_hashtag_unique_count")
    if _fin(uq) and uq < 0.5:
        av, mx = f("tp_tags_hashtag_avg_len"), f("tp_tags_hashtag_max_len")
        if _fin(av) or _fin(mx):
            out.append("avg_len/max_len: ожидается NaN при unique=0")

    tfc = f("tp_tags_title_hashtag_found_count")
    dfc = f("tp_tags_description_hashtag_found_count")
    td = f("tp_tags_title_hashtag_density_per_char")
    dd = f("tp_tags_description_hashtag_density_per_char")
    if _fin(tfc) and tfc == 0.0 and _fin(td) and abs(td) > 1e-6:
        out.append("title density: при count=0 ожидается 0.0 или NaN, не >0")
    if _fin(dfc) and dfc == 0.0 and _fin(dd) and abs(dd) > 1e-6:
        out.append("description density: при count=0 ожидается 0.0 или NaN, не >0")

    seen_empty = False
    for i in range(1, k0 + 1):
        pr = f(f"tp_tags_top{i}_present")
        h0 = f(f"tp_tags_top{i}_hash01")
        ln = f(f"tp_tags_top{i}_len")
        if not _fin(pr):
            continue
        if pr < 0.5:
            seen_empty = True
            if _fin(h0) or _fin(ln):
                out.append(
                    f"tp_tags_top{i}_*: hash/len ожидаются NaN при present=0"
                )
        else:
            if seen_empty:
                out.append("tp_tags_top: present=1 после пустого слота (нужен префикс)")
            if not _fin(h0) or h0 < -1e-6 or h0 > 1.0 + 1e-3:
                out.append(
                    f"tp_tags_top{i}_hash01: вне [0,1] (finite)"
                )
            if not _fin(ln) or ln < 0.5:
                out.append(
                    f"tp_tags_top{i}_len: ожидается >=1 при present=1"
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
        description="Срез tp_tags_* в text_features.npz (tags_extractor_output_v1 + top-K динамика)"
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
    b0, _ = _build_full_slice(load_npz(args.npz_path))
    if args.ranges and not rgl and _tags_only(b0):
        print("ranges: OK (срез tp_tags_* непустой)")

    return ex


if __name__ == "__main__":
    raise SystemExit(main())
