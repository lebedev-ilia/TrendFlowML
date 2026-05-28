#!/usr/bin/env python3
"""Валидатор среза `transcript_chunk_embedder` (16 × `tp_tchunk_*`) в `text_processor/text_features.npz`.

Схема: `transcript_chunk_embedder_output_v1`.

Опционально: `--timings` — `payload.timings_by_extractor["TranscriptChunkEmbedder"]` (ожидается `total`, сек).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_SCHEMA_RELPATH = "DataProcessor/TextProcessor/schemas/transcript_chunk_embedder_output_v1.json"
_EXTRACTOR_CLASS = "TranscriptChunkEmbedder"

_BINARY: frozenset = frozenset(
    {
        "tp_tchunk_present",
        "tp_tchunk_whisper_present",
        "tp_tchunk_youtube_auto_present",
        "tp_tchunk_conf_present",
    }
)

_EXTRA_KEYS: frozenset = frozenset(
    {
        "tp_tchunk_batch_size",
        "tp_tchunk_max_chunk_tokens_model",
        "tp_tchunk_overlap_ratio",
        "tp_tchunk_max_chunks_total",
        "tp_tchunk_cache_enabled",
    }
)

_CONF_STATS: frozenset = frozenset(
    {
        "tp_tchunk_conf_mean",
        "tp_tchunk_conf_min",
        "tp_tchunk_conf_max",
    }
)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Set[str]:
    with open(_repo_root_from_here() / _SCHEMA_RELPATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d["fields"].keys())


EXPECTED_KEYS: Set[str] = _load_expected_keys()
assert len(EXPECTED_KEYS) == 16


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
                f"срез tp_tchunk_* пуст при meta.status=ok (ожидается {len(EXPECTED_KEYS)} ключей)"
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

    sc = f("tp_tchunk_sources_count")
    if _fin(sc) and sc < -1e-6:
        out.append("tp_tchunk_sources_count: ожидается >=0 (finite)")

    for k in ("tp_tchunk_whisper_chunks", "tp_tchunk_youtube_chunks"):
        x = f(k)
        if _fin(x) and x < -1e-6:
            out.append(f"{k}: ожидается >=0 (finite)")

    ed = f("tp_tchunk_embedding_dim")
    if _fin(ed) and ed < 1.0 - 1e-6:
        out.append("tp_tchunk_embedding_dim: ожидается >=1 (finite) или NaN (empty)")

    for k in _CONF_STATS:
        x = f(k)
        if _fin(x) and (x < -1e-4 or x > 1.0 + 1e-4):
            out.append(
                f"{k}: типично [0,1] для ASR confidence (finite), got {x}"
            )

    if f("tp_tchunk_conf_present") > 0.5:
        for k in _CONF_STATS:
            if not _fin(f(k)):
                out.append(
                    f"{k}: ожидается finite при tp_tchunk_conf_present=1"
                )
                break

    emit_on = any(_fin(f(k)) for k in _EXTRA_KEYS)
    if emit_on:
        ce = f("tp_tchunk_cache_enabled")
        if _fin(ce) and (ce < -1e-6 or ce > 1.0 + 1e-3):
            out.append(
                f"tp_tchunk_cache_enabled: ожидается 0/1 при emit_extra_metrics, got {ce}"
            )
        ov = f("tp_tchunk_overlap_ratio")
        if _fin(ov) and (ov < -1e-6 or ov >= 1.0):
            out.append(
                "tp_tchunk_overlap_ratio: ожидается [0, 1) при emit_extra_metrics"
            )
        for k in (
            "tp_tchunk_batch_size",
            "tp_tchunk_max_chunk_tokens_model",
            "tp_tchunk_max_chunks_total",
        ):
            x = f(k)
            if _fin(x) and x < 1.0 - 1e-6:
                out.append(f"{k}: ожидается >=1 при emit_extra_metrics, got {x}")

    if f("tp_tchunk_whisper_present") > 0.5 and f("tp_tchunk_whisper_chunks") < 1.0 - 1e-6:
        out.append("tp_tchunk_whisper_chunks: ожидается >=1 при whisper_present=1")

    if f("tp_tchunk_youtube_auto_present") > 0.5 and f("tp_tchunk_youtube_chunks") < 1.0 - 1e-6:
        out.append("tp_tchunk_youtube_chunks: ожидается >=1 при youtube_auto_present=1")

    if f("tp_tchunk_present") > 0.5:
        if f("tp_tchunk_sources_count") < 1.0 - 1e-6:
            out.append("tp_tchunk_sources_count: ожидается >=1 при present=1")
        if not _fin(f("tp_tchunk_embedding_dim")):
            out.append("tp_tchunk_embedding_dim: ожидается finite при present=1")

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
    tot = td.get("total")
    if tot is None:
        out.append("timings.total: отсутствует")
        return out
    try:
        t = float(tot)
    except (TypeError, ValueError):
        out.append(f"timings.total: не число: {tot!r}")
        return out
    if not _fin(t):
        out.append("timings.total: ожидается finite")
    elif t < -1e-9:
        out.append("timings.total: ожидается >=0")
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
        description="Срез tp_tchunk_* (16) + опционально тайминги (transcript_chunk_embedder_output_v1)"
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
        has_ext = isinstance(tbe, dict) and _EXTRACTOR_CLASS in tbe
        if not has_ext:
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
