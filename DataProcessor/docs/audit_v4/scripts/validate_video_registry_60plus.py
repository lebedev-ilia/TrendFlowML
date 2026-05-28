#!/usr/bin/env python3
"""
Проверка структуры VIDEO_REGISTRY_60PLUS.yaml (батч 60+, чеклист п. 3.x).

Запуск (нужен PyYAML), из корня репо:

  python DataProcessor/docs/audit_v4/scripts/validate_video_registry_60plus.py
  python DataProcessor/docs/audit_v4/scripts/validate_video_registry_60plus.py --registry /path/to/VIDEO_REGISTRY_60PLUS.yaml
  python .../validate_video_registry_60plus.py --strict-count   # ошибка, если len(videos) < target_video_count

Код выхода: 0 — нет ошибок схемы, 1 — ошибки, 2 — файл / YAML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Нужен пакет PyYAML (pip install pyyaml).", file=sys.stderr)
    sys.exit(2)


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "VIDEO_REGISTRY_60PLUS.yaml"


def validate(data: dict[str, Any], *, strict_count: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required_top = (
        "schema_version",
        "registry_status",
        "target_video_count",
        "videos",
    )
    for key in required_top:
        if key not in data:
            errors.append(f"Отсутствует обязательное поле верхнего уровня: {key!r}")

    videos = data.get("videos")
    if not isinstance(videos, list):
        errors.append("Поле 'videos' должно быть списком.")
        return errors, warnings

    target = data.get("target_video_count")
    if isinstance(target, int) and len(videos) < target:
        msg = (
            f"Записей videos: {len(videos)}, target_video_count: {target} — не хватает строк "
            f"(при --strict-count это ошибка)."
        )
        if strict_count:
            errors.append(msg)
        else:
            warnings.append(msg)

    seen_ids: set[str] = set()
    for i, item in enumerate(videos):
        if not isinstance(item, dict):
            errors.append(f"videos[{i}]: ожидается объект, получено {type(item).__name__}.")
            continue
        vid = item.get("video_id")
        if not vid or not isinstance(vid, str):
            errors.append(f"videos[{i}]: нужен непустой строковый video_id.")
        elif vid in seen_ids:
            errors.append(f"Дубликат video_id: {vid!r}")
        else:
            seen_ids.add(vid)
        rid = item.get("run_id")
        if rid is not None and not isinstance(rid, str):
            errors.append(f"videos[{i}]: run_id должен быть строкой или null.")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--registry",
        type=Path,
        default=_default_registry_path(),
        help="Путь к YAML реестра",
    )
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Считать ошибкой, если число записей меньше target_video_count",
    )
    args = parser.parse_args()
    path: Path = args.registry
    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 2
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except Exception as e:
        print(f"Ошибка чтения YAML: {e}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        print("Корень YAML должен быть объектом (mapping).", file=sys.stderr)
        return 2

    errors, warnings = validate(data, strict_count=args.strict_count)
    if warnings:
        print(f"Реестр: {path} — предупреждения ({len(warnings)}):", file=sys.stderr)
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)
    if errors:
        print(f"Реестр: {path}\nОшибки ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return 1

    n = len(data.get("videos", []))
    print(f"OK: {path} — {n} записей, структура в порядке.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
