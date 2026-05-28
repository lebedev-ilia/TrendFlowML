#!/usr/bin/env python3
"""
Синхронизация локальной базы `known_people/` с Embedding Service (категория `face`).

Для каждого человека в `known_people/<person_name>/`:
- Собирает эмбеддинги по всем фото (InsightFace ArcFace, buffalo_l)
- Усредняет эмбеддинги и L2-нормализует
- Добавляет один объект в Embedding Service через add_from_embedding:
  - category = "face"
  - name = <person_name>
  - embedding = avg_emb
  - metadata = {"source": "known_people", "num_images": N}

Требования:
- Запущен Embedding Service (см. embedding_service/README.md)
- Настроены переменные окружения для подключения к PostgreSQL (или .env)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Добавить корень DataProcessor в путь для импорта embedding_service
# Ищем DataProcessor, поднимаясь вверх по дереву директорий
current = Path(__file__).resolve().parent
dp_root = None
while current != current.parent:
    if (current / "embedding_service").exists():
        dp_root = current
        break
    current = current.parent

# Fallback: предполагаем стандартную структуру
if dp_root is None:
    dp_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent

if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))

from embedding_service.config.settings import EmbeddingServiceConfig
from embedding_service.core.embedding_manager import EmbeddingManager


# Путь к known_people относительно корня DataProcessor
KNOWN_ROOT = dp_root / "VisualProcessor" / "core" / "model_process" / "core_identity" / "face_identity" / "known_people"
CATEGORY = "face"


def collect_person_embeddings(app: FaceAnalysis, person_dir: Path) -> List[np.ndarray]:
    """
    Собрать L2-нормализованные эмбеддинги для всех фото человека.

    Args:
        app: Инициализированный InsightFace FaceAnalysis
        person_dir: Путь к директории человека с фото

    Returns:
        Список эмбеддингов (np.ndarray shape (512,))
    """
    embeddings: List[np.ndarray] = []

    for img_path in sorted(person_dir.iterdir()):
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] Не удалось прочитать изображение: {img_path}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"[warn] Лицо не найдено на {img_path}")
            continue

        # Берём первое лицо (как в add_person_in_base.py)
        face = faces[0]
        emb = face.embedding.astype(np.float32)
        # L2-нормализация
        norm = np.linalg.norm(emb) + 1e-9
        emb /= norm
        embeddings.append(emb)

    return embeddings


def main() -> None:
    # 1. Проверка директории known_people
    if not KNOWN_ROOT.is_dir():
        raise SystemExit(f"Директория с базой лиц не найдена: {KNOWN_ROOT}")

    print(f"Синхронизация known_people -> Embedding Service (category='{CATEGORY}')")
    print(f"Корень базы лиц: {KNOWN_ROOT}")

    # 2. Инициализация InsightFace (ArcFace)
    print("Инициализация InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 3. Инициализация Embedding Service manager
    print("Инициализация Embedding Service manager...")
    config = EmbeddingServiceConfig()
    manager = EmbeddingManager(config)

    total_people = 0
    total_added = 0

    # 4. Проход по всем людям в known_people
    for person_dir in sorted(KNOWN_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        total_people += 1
        print(f"\n=== Обработка человека: {person_name} ===")
        embs = collect_person_embeddings(app, person_dir)

        if not embs:
            print(f"[warn] Нет валидных эмбеддингов для {person_name}, пропуск.")
            continue

        embs_arr = np.stack(embs, axis=0)  # (N, D)
        avg_emb = embs_arr.mean(axis=0)
        # Финальная L2-нормализация
        avg_emb /= np.linalg.norm(avg_emb) + 1e-9

        metadata: Dict[str, object] = {
            "source": "known_people",
            "num_images": len(embs),
        }

        try:
            obj_id = manager.add_from_embedding(
                category=CATEGORY,
                embedding=avg_emb,
                name=person_name,
                metadata=metadata,
            )
            total_added += 1
            print(
                f"[ok] Добавлен в Embedding Service: {person_name} "
                f"(images={len(embs)}, id={obj_id})"
            )
        except Exception as e:
            print(f"[error] Не удалось добавить {person_name} в Embedding Service: {e}")

    # 5. Финализация
    manager.close()
    print("\n=== Готово ===")
    print(f"Людей в known_people: {total_people}")
    print(f"Добавлено в Embedding Service: {total_added}")


if __name__ == "__main__":
    main()


