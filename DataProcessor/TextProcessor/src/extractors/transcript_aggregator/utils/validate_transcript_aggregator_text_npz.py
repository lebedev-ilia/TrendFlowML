#!/usr/bin/env python3
"""Валидатор среза `transcript_aggregator` (19 × `tp_tragg_*`) в `text_processor/text_features.npz`.

Схема: `transcript_aggregator_output_v1`.

Опционально: `--timings` — `payload.timings_by_extractor["TranscriptAggregatorExtractor"]` (load/aggregate/total, сек).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/transcript_aggregator_output_v1.json"
_EXTRACTOR_CLASS = "TranscriptAggregatorExtractor"

_BINARY: frozenset = frozenset(
    {
        "tp_tragg_present",
        "tp_tragg_present_whisper",
        "tp_tragg_present_youtube",
        "tp_tragg_present_combined",
        "tp_tragg_compute_std",
        "tp_tragg_compute_mean",
        "tp_tragg_compute_max",
        "tp_tragg_compute_combined",
        "tp_tragg_write_artifacts",
    }
)

_N_CHUNK: frozenset = frozenset(
    {
        "tp_tragg_whisper_n_chunks",
        "tp_tragg_youtube_auto_n_chunks",
        "tp_tragg_combined_n_chunks",
    }
)

_STD_KEYS: frozenset = frozenset(
    {
        "tp_tragg_whisper_mean_std",
        "tp_tragg_whisper_max_std",
        "tp_tragg_youtube_auto_mean_std",
        "tp_tragg_youtube_auto_max_std",
        "tp_tragg_combined_mean_std",
        "tp_tragg_combined_max_std",
    }
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    with open(_repo_root_from_here() / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
assert len(EXPECTED_KEYS) == 19


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


def extract_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    p = d.get("payload")
    if p is None:
        return {}
    if isinstance(p, np.ndarray) and p.dtype == object and p.shape == ():
        p = p.item()
    return p if isinstance(p, dict) else {}


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
                f"срез tp_tragg_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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
        if _fin(x) and (x < -1e-6 or x > 1.0 + 1e-3):
            out.append(f"{k}: ожидается 0/1, got {x}")

    dr = f("tp_tragg_decay_rate")
    if _fin(dr) and dr < -1e-6:
        out.append("tp_tragg_decay_rate: ожидается >=0 (finite)")

    for k in _N_CHUNK:
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >=0 (finite)")

    for k in _STD_KEYS:
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >=0 (finite)")

    if f("tp_tragg_compute_std") < 0.5:
        for k in _STD_KEYS:
            x = f(k)
            if _fin(x):
                out.append(
                    f"{k}: ожидается NaN при tp_tragg_compute_std=0, got {x}"
                )

    emit_on = any(_fin(f(k)) for k in (_N_CHUNK | _STD_KEYS))
    if emit_on:
        for pk, nk in (
            ("tp_tragg_present_whisper", "tp_tragg_whisper_n_chunks"),
            ("tp_tragg_present_youtube", "tp_tragg_youtube_auto_n_chunks"),
            ("tp_tragg_present_combined", "tp_tragg_combined_n_chunks"),
        ):
            if f(pk) > 0.5 and not _fin(f(nk)):
                out.append(
                    f"{nk}: ожидается finite при emit_extra_metrics и {pk}=1"
                )

    return out


def validate_timings(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    payload = extract_payload(d)
    tbe = payload.get("timings_by_extractor")
    if not isinstance(tbe, dict):
        return []
    td = tbe.get(_EXTRACTOR_CLASS)
    if not isinstance(td, dict):
        return []
    for key in ("load", "aggregate", "total"):
        v = td.get(key)
        if v is None:
            out.append(f"timings.{key}: отсутствует")
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            out.append(f"timings.{key}: не число: {v!r}")
            continue
        if not _fin(x):
            out.append(f"timings.{key}: ожидается finite, got {v!r}")
        elif x < -1e-9:
            out.append(f"timings.{key}: ожидается >=0, got {x}")
    load = td.get("load")
    agg = td.get("aggregate")
    tot = td.get("total")
    try:
        ls = float(load) if load is not None else float("nan")
        ag = float(agg) if agg is not None else float("nan")
        ts = float(tot) if tot is not None else float("nan")
    except (TypeError, ValueError):
        return out
    if all(_fin(x) for x in (ls, ag, ts)):
        if ts + 1e-4 < ls + ag:
            out.append(
                f"timings.total={ts} < load+aggregate={ls + ag} (ожидается total >= load+aggregate)"
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
        meta = extract_meta(load_npz(str(npz)))
        rg: List[str] = []
        if not st and meta.get("status") == "ok":
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
        description="Срез tp_tragg_* (19) + опционально тайминги (transcript_aggregator_output_v1)"
    )
    p.add_argument("npz_path", nargs="?", help="text_features.npz")
    p.add_argument("--struct", action="store_true")
    p.add_argument("--ranges", action="store_true")
    p.add_argument("--timings", action="store_true")
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
    if not ok and not (args.struct or args.ranges or args.timings):
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
    if args.timings:
        pld = extract_payload(load_npz(args.npz_path))
        tbe = pld.get("timings_by_extractor") if isinstance(pld, dict) else None
        has_tragg = isinstance(tbe, dict) and _EXTRACTOR_CLASS in tbe
        if not has_tragg:
            print("timings: (нет в payload) — пропуск")
        else:
            tml = validate_timings(args.npz_path)
            for line in tml:
                print(f"timings: {line}")
                ex = max(ex, 2)
            if not tml:
                print("timings: OK")
    if args.struct and not stl:
        print("struct: OK")
    b0, _ = _build_slice(load_npz(args.npz_path))
    if args.ranges and not rgl and b0:
        print("ranges: OK (срез непустой)")

    return ex


if __name__ == "__main__":
    raise SystemExit(main())
