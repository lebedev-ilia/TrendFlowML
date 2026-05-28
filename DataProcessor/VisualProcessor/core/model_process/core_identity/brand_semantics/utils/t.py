#!/usr/bin/env python3
"""
Статистика по базе брендов с валидацией размеров изображений.
"""

import os
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    try:
        from PIL import Image
        HAS_PIL = True
        HAS_CV2 = False
    except ImportError:
        print("[error] Необходим либо OpenCV (cv2), либо PIL (Pillow)")
        exit(1)
else:
    HAS_PIL = False

p = Path("DataProcessor/VisualProcessor/core/model_process/core_identity/brand_semantics/known_brands")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = 100  # Минимальный размер изображения


def get_image_size(image_path: Path):
    """Получить размер изображения."""
    try:
        if HAS_CV2:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            h, w = img.shape[:2]
            return (w, h)
        else:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
    except Exception:
        return None


def validate_images(brand_dir: Path):
    """Проверить размеры всех изображений в директории бренда."""
    invalid = []
    for img_file in brand_dir.iterdir():
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in IMG_EXTS:
            continue
        
        size = get_image_size(img_file)
        if size is None:
            invalid.append(img_file.name)
            continue
        
        w, h = size
        if min(w, h) < MIN_SIZE:
            invalid.append(f"{img_file.name} ({w}x{h})")
    
    return invalid


l = []

for category_dir in sorted(p.iterdir()):
    if not category_dir.is_dir():
        continue
    
    for brand_dir in sorted(category_dir.iterdir()):
        if not brand_dir.is_dir():
            continue
        
        # Подсчет файлов изображений
        image_files = [f for f in brand_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in IMG_EXTS]
        files_count = len(image_files)
        
        # Валидация размеров
        invalid_images = validate_images(brand_dir)
        
        label = f"{category_dir.name}/{brand_dir.name}"
        status = "complete" if files_count >= 20 else str(files_count)
        
        l.append((label, status, invalid_images))

# Сортировка: сначала complete, затем по убыванию количества
def sort_key(x):
    label, status, invalid = x
    if status == "complete":
        return (0, 0)  # complete в начале
    try:
        count = int(status)
        return (1, -count)  # затем по убыванию количества
    except:
        return (2, 0)

l = sorted(l, key=sort_key)

# Вывод таблицы
print("=" * 100)
print(f"{'Бренд':<50} {'Фото':<10} {'Проблемные изображения':<40}")
print("=" * 100)

for label, status, invalid in l:
    status_str = status
    invalid_str = ", ".join(invalid) if invalid else "-"
    
    # Обрезаем длинные строки
    if len(invalid_str) > 38:
        invalid_str = invalid_str[:35] + "..."
    
    print(f"{label:<50} {status_str:<10} {invalid_str:<40}")

print("=" * 100)
print(f"\nВсего брендов: {len(l)}")
complete_count = sum(1 for _, status, _ in l if status == "complete")
print(f"Завершено (>=20 фото): {complete_count}")
print(f"Минимальный размер изображения: {MIN_SIZE}x{MIN_SIZE} пикселей")