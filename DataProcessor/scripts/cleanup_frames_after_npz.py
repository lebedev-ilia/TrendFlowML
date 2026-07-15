#!/usr/bin/env python3
"""
Очистка кэша кадров (frames) после того, как NPZ-артефакты записаны (ASSESSMENT §2, для 200k).

Плотные окна action_recognition + union-выборка раздувают кэш frames (~10 GB/видео). Кадры нужны
только на время обработки компонентов; после появления NPZ их можно удалять. Скрипт безопасен:
удаляет ТОЛЬКО директории кадров и только если найдены ожидаемые NPZ (иначе пропускает).

Использование:
  python cleanup_frames_after_npz.py <rs_path> [--frames-subdir frames] [--require action_recognition,core_object_detections] [--apply]
Без --apply — dry-run (только печатает, что удалил бы).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _has_required_npz(rs_path: str, required: list[str]) -> bool:
    for comp in required:
        comp_dir = os.path.join(rs_path, comp)
        if not os.path.isdir(comp_dir):
            return False
        if not any(f.endswith(".npz") for f in os.listdir(comp_dir)):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Удаление кэша frames после записи NPZ (dry-run по умолчанию)")
    ap.add_argument("rs_path", help="Каталог результатов run (где лежат <component>/*.npz и frames)")
    ap.add_argument("--frames-subdir", default="frames", help="Имя поддиректории кадров")
    ap.add_argument("--require", default="core_object_detections",
                    help="Список компонентов через запятую, чьи NPZ обязаны существовать перед удалением")
    ap.add_argument("--apply", action="store_true", help="Реально удалить (без флага — dry-run)")
    args = ap.parse_args()

    required = [c.strip() for c in str(args.require).split(",") if c.strip()]
    freed = 0
    removed = 0
    for frames_dir in Path(args.rs_path).rglob(args.frames_subdir):
        if not frames_dir.is_dir():
            continue
        rs = str(frames_dir.parent)
        if not _has_required_npz(rs, required):
            print(f"SKIP (нет обязательных NPZ {required}): {frames_dir}")
            continue
        size = sum(f.stat().st_size for f in frames_dir.rglob("*") if f.is_file())
        freed += size
        removed += 1
        if args.apply:
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"REMOVED {size/1e9:.2f} GB: {frames_dir}")
        else:
            print(f"WOULD REMOVE {size/1e9:.2f} GB: {frames_dir}")

    verb = "освобождено" if args.apply else "будет освобождено"
    print(f"\nИтог: {removed} директорий, {verb} {freed/1e9:.2f} GB"
          + ("" if args.apply else "  (запусти с --apply чтобы удалить)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
