#!/usr/bin/env python3
"""
Удаление маленьких изображений из базы known_brands.

Удаляет все изображения, у которых хотя бы одна сторона меньше 100 пикселей.
Это необходимо для обеспечения качества эмбеддингов CLIP.

Использование:
    python cleanup_small_images.py [--dry-run] [--min-size SIZE]
    
    --dry-run: только показать, что будет удалено, без реального удаления
    --min-size: минимальный размер (по умолчанию 100)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
HAS_PIL = False
HAS_CV2 = True

# Добавить корень DataProcessor в путь
current = Path(__file__).resolve().parent
dp_root = None
while current != current.parent:
    if (current / "embedding_service").exists() or (current.parent / "embedding_service").exists():
        dp_root = current.parent if (current.parent / "embedding_service").exists() else current
        break
    current = current.parent

# Fallback: предполагаем стандартную структуру
if dp_root is None:
    dp_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent

if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))

# Путь к known_brands
KNOWN_ROOT = (
    dp_root
    / "VisualProcessor"
    / "core"
    / "model_process"
    / "core_identity"
    / "brand_semantics"
    / "known_brands"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_size(image_path: Path) -> Tuple[int, int] | None:
    """
    Получить размер изображения.
    
    Returns:
        (width, height) или None если не удалось прочитать
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception as e:
        print(f"[warn] Не удалось прочитать {image_path}: {e}")
        return None


def find_small_images(
    root: Path, min_size: int = 100
) -> List[Tuple[Path, int, int]]:
    """
    Найти все изображения меньше min_size пикселей.
    
    Args:
        root: Корневая директория для поиска
        min_size: Минимальный размер (по меньшей стороне)
    
    Returns:
        Список кортежей (путь, width, height)
    """
    small_images: List[Tuple[Path, int, int]] = []
    
    if not root.exists():
        print(f"[error] Директория не найдена: {root}")
        return small_images
    
    print(f"Поиск маленьких изображений в: {root}")
    print(f"Минимальный размер: {min_size}x{min_size} пикселей\n")
    
    total_images = 0
    for img_path in root.rglob("*"):
        if not img_path.is_file():
            continue
        
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        
        total_images += 1
        size = get_image_size(img_path)
        
        if size is None:
            continue
        
        w, h = size
        if min(w, h) < min_size:
            small_images.append((img_path, w, h))
    
    print(f"Всего проверено изображений: {total_images}")
    print(f"Найдено маленьких изображений (< {min_size}x{min_size}): {len(small_images)}\n")
    
    return small_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Удаление маленьких изображений из базы known_brands"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет удалено, без реального удаления",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Минимальный размер изображения (по умолчанию 100)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Путь к known_brands (по умолчанию используется стандартный путь)",
    )
    
    args = parser.parse_args()
    
    # Определяем корневую директорию
    root = Path(args.root) if args.root else KNOWN_ROOT
    
    if not root.exists():
        print(f"[error] Директория не найдена: {root}")
        sys.exit(1)
    
    # Находим маленькие изображения
    small_images = find_small_images(root, min_size=args.min_size)
    
    if not small_images:
        print("Маленьких изображений не найдено. Всё в порядке!")
        return
    
    # Показываем список
    print("=" * 80)
    print("Маленькие изображения для удаления:")
    print("=" * 80)
    
    for img_path, w, h in small_images:
        rel_path = img_path.relative_to(root)
        print(f"  {rel_path}: {w}x{h} пикселей")
    
    print("=" * 80)
    
    if args.dry_run:
        print(f"\n[DRY-RUN] Будет удалено {len(small_images)} изображений")
        print("Запустите без --dry-run для реального удаления")
        return
    
    # Подтверждение
    print(f"\nБудет удалено {len(small_images)} изображений.")
    response = input("Продолжить? (yes/no): ").strip().lower()
    
    if response not in ("yes", "y", "да", "д"):
        print("Отменено.")
        return
    
    # Удаление
    deleted = 0
    errors = 0
    
    for img_path, w, h in small_images:
        try:
            img_path.unlink()
            deleted += 1
            rel_path = img_path.relative_to(root)
            print(f"[ok] Удалено: {rel_path} ({w}x{h})")
        except Exception as e:
            errors += 1
            print(f"[error] Не удалось удалить {img_path}: {e}")
    
    # Итоги
    print("\n" + "=" * 80)
    print("Итоги:")
    print(f"  Удалено: {deleted}")
    if errors > 0:
        print(f"  Ошибок: {errors}")
    print("=" * 80)


if __name__ == "__main__":
    main()

