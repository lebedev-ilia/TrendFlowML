#!/usr/bin/env python3
"""
Добавить по нескольку демо-объектов `place` и `franchise` в Embedding Service (Triton + Postgres).

Без `sync_*.py` в runbook — это offline-аналог: manager.add() с картинками.

Требования:
- Postgres (как в e2e_env.sh) и Triton: place по умолчанию clip_448; в models_t_1 E2E обычно
  **нет** clip_image_448 — тогда: export EMBEDDING_DEV_MAP_PLACE_TO_CLIP336=1
- franchise: clip_224 → clip_image_224 в Triton
- Запуск из каталога DataProcessor, PYTHONPATH=., venv .data_venv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Корень DataProcessor
_DP = Path(__file__).resolve().parent.parent.parent
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from embedding_service.config.settings import EmbeddingServiceConfig  # noqa: E402
from embedding_service.core.embedding_manager import EmbeddingManager  # noqa: E402

# 6+ разных файлов (runbook: 5–8)
_DEFAULT_PHOTO_DIR = _DP.parent / "example" / "example_photo"
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _list_images(photo_dir: Path, limit: int) -> list[Path]:
    if not photo_dir.is_dir():
        return []
    files: list[Path] = []
    for p in sorted(photo_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            files.append(p)
    return files[:limit]


def main() -> None:
    photo_dir = Path(
        os.environ.get("E2E_SEED_PLACE_FRANCHISE_PHOTO_DIR", str(_DEFAULT_PHOTO_DIR))
    )
    n_each = int(os.environ.get("E2E_SEED_PLACE_FRANCHISE_COUNT", "6"))
    if n_each < 1:
        raise SystemExit("E2E_SEED_PLACE_FRANCHISE_COUNT must be >= 1")

    images = _list_images(photo_dir, n_each * 2)
    if len(images) < n_each * 2:
        raise SystemExit(
            f"Нужно минимум {n_each * 2} картинок в {photo_dir}, нашли {len(images)}. "
            "Положите PNG/JPG в каталог или задайте E2E_SEED_PLACE_FRANCHISE_PHOTO_DIR."
        )

    place_paths = images[:n_each]
    franchise_paths = images[n_each : n_each * 2]

    if os.environ.get("EMBEDDING_DEV_MAP_PLACE_TO_CLIP336", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        print(
            "[info] EMBEDDING_DEV_MAP_PLACE_TO_CLIP336 не задан: для place ожидается Triton model "
            "clip_image_448. Если её нет в репо — export EMBEDDING_DEV_MAP_PLACE_TO_CLIP336=1",
            file=sys.stderr,
        )

    print("Инициализация EmbeddingManager...")
    config = EmbeddingServiceConfig()
    manager = EmbeddingManager(config)

    added = 0
    for i, p in enumerate(place_paths, start=1):
        name = f"e2e_place_{i:02d}"
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] пропуск (не читается): {p}", file=sys.stderr)
            continue
        mid = manager.add(
            category="place",
            image=img,
            name=name,
            metadata={"source": "seed_e2e_place_and_franchise", "file": p.name},
        )
        print(f"[ok] place: {name} id={mid} file={p.name}")
        added += 1

    for i, p in enumerate(franchise_paths, start=1):
        name = f"e2e_franchise_{i:02d}"
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] пропуск (не читается): {p}", file=sys.stderr)
            continue
        mid = manager.add(
            category="franchise",
            image=img,
            name=name,
            metadata={"source": "seed_e2e_place_and_franchise", "file": p.name},
        )
        print(f"[ok] franchise: {name} id={mid} file={p.name}")
        added += 1

    manager.close()
    print(f"=== готово, добавлено: {added} (ожидалось {n_each * 2}) ===")


if __name__ == "__main__":
    main()
