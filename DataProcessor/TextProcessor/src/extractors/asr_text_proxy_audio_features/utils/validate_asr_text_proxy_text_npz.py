#!/usr/bin/env python3
"""Валидатор среза `tp_asrproxy_*` в `text_processor/text_features.npz` (агрегат TextProcessor).

Проверяет соответствие набору `asr_text_proxy_audio_features_output_v1`, длину векторов, опционально
диапазоны значений (`--ranges`). Тайминги экстрактора в агрегированном NPZ нет — см. docs/FEATURE_DESCRIPTION.md.

Пример:
  DataProcessor/.data_venv/bin/python \\
    TextProcessor/src/extractors/asr_text_proxy_audio_features/utils/validate_asr_text_proxy_text_npz.py \\
    storage/.../text_processor/text_features.npz --struct --ranges
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

PREFIX = "tp_asrproxy_"
_SCHEMA_RELPATH = (
    "DataProcessor/TextProcessor/schemas/asr_text_proxy_audio_features_output_v1.json"
)


def _repo_root_from_here() -> Path:
    # .../TrendFlowML/DataProcessor/TextProcessor/src/extractors/.../utils/this.py
    return Path(__file__).resolve().parents[6]


def _load_expected_keys() -> Tuple[str, ...]:
    root = _repo_root_from_here()
    p = root / _SCHEMA_RELPATH
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    keys = sorted(d["fields"].keys())
    return tuple(keys)


EXPECTED_KEYS: Tuple[str, ...] = _load_expected_keys()


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


def _slice_asr(
    d: Dict[str, Any],
) -> Tuple[List[str], np.ndarray, Set[str]]:
    names = d.get("feature_names")
    vals = d.get("feature_values")
    if names is None or vals is None:
        return [], np.array([]), set()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    names = [str(x) for x in names]
    v = np.asarray(vals, dtype=np.float64).ravel()
    idx = [i for i, n in enumerate(names) if n.startswith(PREFIX)]
    fn = [names[i] for i in idx]
    fv = v[idx] if len(idx) else np.array([])
    return fn, fv, set(fn)


def validate_schema(npz_path: str) -> bool:
    try:
        d = load_npz(npz_path)
        if "feature_names" not in d or "feature_values" not in d:
            return False
        names, _vals, asr = _slice_asr(d)
        if not asr and extract_meta(d).get("status") == "ok":
            # пустой срез при status=ok — не соответствует контракту компонента
            return False
        exp = set(EXPECTED_KEYS)
        if asr and asr != exp:
            return False
        return True
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
    fn, fv, asr = _slice_asr(d)
    if not asr:
        m = extract_meta(d)
        if m.get("status") == "ok":
            out.append("срез tp_asrproxy_* пуст при meta.status=ok (ожидается 37 ключей)")
        # при status=error пустой срез — ожидаемо (пайплайн не дошёл до заполнения)
        return out
    exp = set(EXPECTED_KEYS)
    missing = sorted(exp - asr)
    extra = sorted(asr - exp)
    if missing:
        out.append(f"не хватает ключей: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if extra:
        out.append(f"лишние tp_asrproxy_*: {extra}")
    if len(fn) != len(EXPECTED_KEYS):
        out.append(f"ожидается 37 tp_asrproxy_*; получено {len(fn)}")
    return out


def _finite(x: float) -> bool:
    return bool(math.isfinite(x))


def validate_ranges(npz_path: str) -> List[str]:
    out: List[str] = []
    d = load_npz(npz_path)
    meta = extract_meta(d)
    fn, fv, asr = _slice_asr(d)
    if not asr or meta.get("status") != "ok":
        return out
    if len(fn) != len(fv):
        return out
    bag: Dict[str, float] = {fn[i]: float(fv[i]) for i in range(len(fn))}

    def f(name: str) -> Optional[float]:
        v0 = bag.get(name)
        if v0 is None:
            return None
        return float(v0)

    binary = (
        "tp_asrproxy_present",
        "tp_asrproxy_has_confidence",
        "tp_asrproxy_enabled",
        "tp_asrproxy_basic_enabled",
        "tp_asrproxy_noise_enabled",
        "tp_asrproxy_rhythm_enabled",
        "tp_asrproxy_intonation_enabled",
        "tp_asrproxy_require_asr_text_enabled",
        "tp_asrproxy_strict_document_duration_enabled",
        "tp_asrproxy_text_truncated_flag",
        "tp_asrproxy_asr_schema_invalid_flag",
        "tp_asrproxy_conf_invalid_flag",
        "tp_asrproxy_token_decode_failed_flag",
        "tp_asrproxy_duration_from_payload_flag",
        "tp_asrproxy_duration_invalid_flag",
        "tp_asrproxy_noise_proxy_present",
    )
    for k in binary:
        x = f(k)
        if x is not None and _finite(x) and (x < -1e-6 or x > 1.0 + 1e-6):
            out.append(f"{k}: ожидается 0/1, got {x}")

    thr = f("tp_asrproxy_low_conf_threshold")
    if thr is not None and _finite(thr) and (thr < -1e-6 or thr > 1.0 + 1e-6):
        out.append(f"tp_asrproxy_low_conf_threshold: вне [0,1] ({thr})")

    wpm0 = f("tp_asrproxy_words_per_minute_baseline")
    if wpm0 is not None and _finite(wpm0) and wpm0 <= 0:
        out.append("tp_asrproxy_words_per_minute_baseline: ожидается > 0")

    mchars = f("tp_asrproxy_max_text_chars")
    if mchars is not None and _finite(mchars) and mchars < 0:
        out.append("tp_asrproxy_max_text_chars: ожидается >= 0")

    ad = f("tp_asrproxy_audio_duration_sec")
    if ad is not None and _finite(ad) and ad <= 0:
        out.append("tp_asrproxy_audio_duration_sec: ожидается > 0 при finite")

    for k in (
        "tp_asrproxy_segments_count",
        "tp_asrproxy_text_chars",
        "tp_asrproxy_word_count",
    ):
        x = f(k)
        if x is not None and _finite(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0")

    rate01 = (
        "tp_asrproxy_confidence_present_rate",
        "tp_asrproxy_confidence_mean",
        "tp_asrproxy_confidence_chunked_min",
        "tp_asrproxy_low_conf_rate",
        "tp_asrproxy_text_noise_rare_ratio",
        "tp_asrproxy_text_noise_oov_ratio",
        "tp_asrproxy_noise_proxy",
        "tp_asrproxy_filler_ratio",
        "tp_asrproxy_sentence_intonation",
    )
    for k in rate01:
        x = f(k)
        if x is not None and _finite(x) and (x < -1e-3 or x > 1.0 + 1e-3):
            out.append(f"{k}: при finite ожидается [0,1] ({x})")

    cstd = f("tp_asrproxy_confidence_std")
    if cstd is not None and _finite(cstd) and cstd < 0:
        out.append("tp_asrproxy_confidence_std: ожидается >= 0")

    for k in (
        "tp_asrproxy_speech_rate_wpm",
        "tp_asrproxy_speech_char_density",
        "tp_asrproxy_pause_density",
    ):
        x = f(k)
        if x is not None and _finite(x) and x < -1e-6:
            out.append(f"{k}: ожидается >= 0")

    ratio = f("tp_asrproxy_speech_rate_wpm_ratio_to_baseline")
    if ratio is not None and _finite(ratio) and ratio < -1e-6:
        out.append("tp_asrproxy_speech_rate_wpm_ratio_to_baseline: ожидается >= 0")

    p = f("tp_asrproxy_present")
    rhy = f("tp_asrproxy_rhythm_enabled")
    if p is not None and p > 0.5 and rhy is not None and rhy > 0.5:
        wpm = f("tp_asrproxy_speech_rate_wpm")
        if wpm is not None and not _finite(wpm):
            out.append("tp_asrproxy_present=1, rhythm on: ожидается finite tp_asrproxy_speech_rate_wpm")
        sden = f("tp_asrproxy_speech_char_density")
        if sden is not None and not _finite(sden):
            out.append("tp_asrproxy_present=1, rhythm on: ожидается finite tp_asrproxy_speech_char_density")
        rrat = f("tp_asrproxy_speech_rate_wpm_ratio_to_baseline")
        if rrat is not None and not _finite(rrat):
            out.append("tp_asrproxy_present=1, rhythm on: ожидается finite ratio WPM/baseline")

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
        description="Срез tp_asrproxy_* в text_processor/text_features.npz (asr_text_proxy_audio_features_output_v1)"
    )
    p.add_argument("npz_path", nargs="?", help="Путь к text_features.npz (если не задан --results-base)")
    p.add_argument(
        "--struct",
        action="store_true",
        help="Проверка длины feature_values и полного набора 37 tp_asrproxy_* при status=ok",
    )
    p.add_argument(
        "--ranges",
        action="store_true",
        help="Проверка типичных диапазонов (0/1, [0,1], длительность > 0) — см. docs/FEATURE_DESCRIPTION.md",
    )
    p.add_argument(
        "--results-base",
        help="[батч] корень result_store; обход **/text_processor/text_features.npz",
    )
    p.add_argument("--platform-id", default="youtube", help="[батч] субкаталог платформы")
    args = p.parse_args()

    if args.results_base:
        return _run_batch(results_base=args.results_base, platform_id=args.platform_id or "youtube")

    if not args.npz_path:
        p.error("нужен npz_path или --results-base")
        return 1

    ok = validate_schema(args.npz_path)
    print("✅ VALID (схема среза)" if ok else "❌ INVALID (схема среза)")
    if not ok and not (args.struct or args.ranges):
        return 1

    ex = 0
    stl: List[str] = []
    rgl: List[str] = []
    if args.struct:
        stl = validate_structure(args.npz_path)
    if args.ranges:
        rgl = validate_ranges(args.npz_path)
    for line in stl:
        print(f"struct: {line}")
        ex = 2
    for line in rgl:
        print(f"ranges: {line}")
        ex = max(ex, 2)

    if args.struct and not stl and args.npz_path:
        print("struct: OK")
    if args.ranges and not rgl and load_npz(args.npz_path).get("feature_names") is not None:
        d = load_npz(args.npz_path)
        if _slice_asr(d)[2]:
            print("ranges: OK (срез непустой)")

    return ex


if __name__ == "__main__":
    raise SystemExit(main())
