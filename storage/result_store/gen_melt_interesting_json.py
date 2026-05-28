#!/usr/bin/env python3
"""
Собрать view_csv_melt_interesting.json: для каждого component — колонки,
где в batch CSV реально есть значения, с отсечением «шума» (пути, контракты, версии).

  cd storage/result_store
  python3 gen_melt_interesting_json.py --csv batch_features_report_2videos.csv -o view_csv_melt_interesting.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# align with view_csv.MELT_ID_COLS and typical tech noise
MERGE_INTO_EACH = [
    "duration_ms",
    "manifest_status",
    "manifest_empty_reason",
    "device_used",
    "npz_error",
]

# substrings: если встречается в имени колонки (lower) — не в include (тайминги отдельно)
_NOISE = (
    "contract_version",
    "sampling_policy_version",
    "asr_text_contract",
    "config_hash",
    "dataprocessor_version",
    "meta_dataprocessor",
    "meta_sampling_policy",
    "docker",
    "db_name",
    "db_path",
    "db_digest",
    "db_version",
    "git_",
    "model_signature",
    "core_clip_model",
    "clip_text",
    "tokenizer",
    "triton",
    "backend_proxy",
    "whisper",
    "yolo",
    "emonet",
    "embedding_service",
    "domain_db",
    "npz_artifact",
    "result_store",
    "frames_dir",
    "produced_at",
    "created_at",
    "mediapipe",  # обычно id модели, не сигнал
    "meta_platform_id",
    "meta_video_id",
    "meta_run_id",
    "meta_producer",
    "meta_schema",
    "meta_created",
    "meta_config_hash",
    "diarization_contract",
    "meta_backend_proxy",
)

_SKIP_SUFFIX = ("_path", "_url", "_dir", "_file")
_TIMING_PREFIX = "meta_timing_"
_ASR_TIMING_PREFIX = "meta_asr_timing_"


def _is_noise_key(name: str) -> bool:
    n = name.lower()
    if n.startswith(_TIMING_PREFIX):
        return True
    if n.startswith(_ASR_TIMING_PREFIX):
        return True
    for s in _NOISE:
        if s in n:
            return True
    for suf in _SKIP_SUFFIX:
        if n.endswith(suf):
            return True
    return False


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def collect(
    path: Path,
) -> tuple[list[str], dict[str, set[str]]]:
    """Заголовок CSV и для каждого component — множество колонок с непустым значением в строке."""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return [], {}
        base = [c for c in r.fieldnames if c]
        # как в _melt_feature_columns (без id, без run/npz/meta-suppress)
        idc = {
            "platform_id",
            "video_id",
            "run_id",
            "component",
            "component_type",
        }
        suppress = {
            "run_path",
            "npz",
            "meta_video_id",
            "meta_schema_version",
            "meta_sampling_policy_version",
            "meta_run_id",
            "meta_producer_version",
            "meta_producer",
            "meta_platform_id",
        }
        feature_like = [c for c in base if c not in idc and c not in suppress]

        by_comp: dict[str, set[str]] = defaultdict(set)
        for row in r:
            comp = (row.get("component") or "").strip() or "—"
            for c in feature_like:
                v = row.get(c, "")
                if v is None:
                    continue
                t = str(v).strip()
                if not t or t.lower() in ("nan", "none", "null", "n/a", "na", "-", "—"):
                    continue
                by_comp[comp].add(c)
    return base, by_comp


def build_config(by_comp: dict[str, set[str]], *, all_columns: list[str]) -> dict[str, Any]:
    all_set = set(all_columns)
    components: dict[str, Any] = {}
    merge_in_defaults = [c for c in MERGE_INTO_EACH if c in all_set]
    for comp in sorted(by_comp.keys()):
        raw = by_comp[comp]
        # merge_into_each в defaults; здесь — только сигнальные meta_* / feature_*
        rest = [c for c in sorted(raw) if c not in MERGE_INTO_EACH and not _is_noise_key(c)]
        include: list[str] = list(rest)
        block: dict[str, Any] = {"include": include, "add_all_meta_timing": True}
        if comp == "asr_extractor":
            block["add_all_meta_asr_timing"] = True
        components[comp] = block
    return {
        "comment": "Автосбор: непустые колонки; meta_timing_* — add_all_meta_timing; "
        "asr: meta_asr_timing_* (стадии ASR) — add_all_meta_asr_timing. "
        "Пересобрать: python3 gen_melt_interesting_json.py --csv …",
        "defaults": {
            "meta_timing_prefix": "meta_timing_",
            "meta_asr_timing_prefix": "meta_asr_timing_",
            "merge_into_each": merge_in_defaults,
        },
        "components": components,
        "fallback_unlisted": {"include": [], "add_all_meta_timing": True},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_script_dir() / "batch_features_report_2videos.csv",
        help="Wide batch CSV",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_script_dir() / "view_csv_melt_interesting.json",
    )
    args = ap.parse_args()
    if not args.csv.is_file():
        print("Нет CSV:", args.csv, flush=True)
        return 1
    all_columns, by_comp = collect(args.csv)
    if not by_comp:
        print("Пусто или нет component", flush=True)
        return 1
    cfg = build_config(by_comp, all_columns=all_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as w:
        json.dump(cfg, w, ensure_ascii=False, indent=2)
        w.write("\n")
    print("OK", args.output, "components:", len(cfg["components"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
