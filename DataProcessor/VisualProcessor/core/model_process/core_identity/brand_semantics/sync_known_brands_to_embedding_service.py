#!/usr/bin/env python3
"""
Синхронизация локальной базы `known_brands/` с Embedding Service (категория `brand`).

Для каждого бренда в `known_brands/<brand_name>/`:
- Собирает эмбеддинги по всем фото (CLIP clip_336 через Triton)
- Усредняет эмбеддинги и L2-нормализует
- Добавляет один объект в Embedding Service через add_from_embedding:
  - category = "brand"
  - name = <brand_name>
  - embedding = avg_emb
  - metadata = {"source": "known_brands", "num_images": N}

Требования:
- Запущен Embedding Service (см. embedding_service/README.md)
- Запущен Triton Inference Server с моделью clip_image_336
- Настроены переменные окружения для подключения к PostgreSQL (или .env)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

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


# Путь к known_brands относительно корня DataProcessor
KNOWN_ROOT = (
    dp_root
    / "VisualProcessor"
    / "core"
    / "model_process"
    / "core_identity"
    / "brand_semantics"
    / "known_brands"
)
CATEGORY = "brand"


def collect_brand_embeddings(
    manager: EmbeddingManager, brand_dir: Path
) -> List[np.ndarray]:
    """
    Собрать L2-нормализованные эмбеддинги для всех фото бренда через CLIP.

    Args:
        manager: EmbeddingManager с инициализированным CLIP extractor
        brand_dir: Путь к директории бренда с фото

    Returns:
        Список эмбеддингов (np.ndarray shape (512,))
    """
    embeddings: List[np.ndarray] = []

    # Получаем manager для категории brand (использует CLIP clip_336)
    brand_manager = manager._get_manager(CATEGORY)

    for img_path in sorted(brand_dir.iterdir()):
        if not img_path.is_file():
            continue

        # Пропускаем не-изображения
        if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] Не удалось прочитать изображение: {img_path}")
            continue

        try:
            # Извлекаем эмбеддинг через CLIP extractor
            emb = brand_manager.extract_embedding(img)
            # Эмбеддинг уже L2-нормализован в CLIPExtractor
            embeddings.append(emb)
        except Exception as e:
            print(f"[warn] Не удалось извлечь эмбеддинг для {img_path}: {e}")
            continue

    return embeddings


def main() -> None:
    # 1. Проверка директории known_brands
    if not KNOWN_ROOT.is_dir():
        raise SystemExit(f"Директория с базой брендов не найдена: {KNOWN_ROOT}")

    print(f"Синхронизация known_brands -> Embedding Service (category='{CATEGORY}')")
    print(f"Корень базы брендов: {KNOWN_ROOT}")

    # 2. Инициализация Embedding Service manager
    print("Инициализация Embedding Service manager...")
    config = EmbeddingServiceConfig()
    manager = EmbeddingManager(config)

    total_brands = 0
    total_added = 0

    # 3. Проход по всем брендам в known_brands
    for brand_dir in sorted(KNOWN_ROOT.iterdir()):
        if not brand_dir.is_dir():
            continue

        brand_name = brand_dir.name
        total_brands += 1
        print(f"\n=== Обработка бренда: {brand_name} ===")
        embs = collect_brand_embeddings(manager, brand_dir)

        if not embs:
            print(f"[warn] Нет валидных эмбеддингов для {brand_name}, пропуск.")
            continue

        embs_arr = np.stack(embs, axis=0)  # (N, D)
        avg_emb = embs_arr.mean(axis=0)
        # Финальная L2-нормализация (на всякий случай, хотя эмбеддинги уже нормализованы)
        avg_emb /= np.linalg.norm(avg_emb) + 1e-9

        metadata: Dict[str, object] = {
            "source": "known_brands",
            "num_images": len(embs),
        }

        try:
            obj_id = manager.add_from_embedding(
                category=CATEGORY,
                embedding=avg_emb,
                name=brand_name,
                metadata=metadata,
            )
            total_added += 1
            print(
                f"[ok] Добавлен в Embedding Service: {brand_name} "
                f"(images={len(embs)}, id={obj_id})"
            )
        except Exception as e:
            print(f"[error] Не удалось добавить {brand_name} в Embedding Service: {e}")

    # 4. Финализация
    manager.close()
    print("\n=== Готово ===")
    print(f"Брендов в known_brands: {total_brands}")
    print(f"Добавлено в Embedding Service: {total_added}")


if __name__ == "__main__":
    main()

