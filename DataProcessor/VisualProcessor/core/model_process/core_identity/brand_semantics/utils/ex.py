from ultralytics import YOLO # type: ignore
import numpy as np # type: ignore
from typing import Tuple, Optional
import cv2
import os

# Путь к изображению
img_path = "/media/ilya/Новый том/TrendFlowML/DataProcessor/VisualProcessor/core/model_process/core_identity/brand_semantics/known_brands_auto/car/bmw/2.jpg"

# Загружаем изображение
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Не удалось загрузить изображение: {img_path}")

model_path = '/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt'
pad_ratio = 0.15
min_size = 16
logo_class_ids = [33]  # ID класса logo_region
conf_thr = 0.1  # Порог confidence

model = YOLO(model_path)

# Выводим информацию о классах модели
if hasattr(model, 'names'):
    print("Классы модели YOLO:")
    for cls_id, cls_name in model.names.items():
        marker = " ← ожидаемый" if cls_id in logo_class_ids else ""
        print(f"  Класс {cls_id}: {cls_name}{marker}")
    print()

# Детекция через YOLO (можно передать как путь, так и numpy array)
results = model(img, verbose=False)

def show_brand_and_get_choice(
    crop: np.ndarray,
    known_root: str,
    preview_name: str = "preview_brand.jpg",
    *,
    default_brand: Optional[str] = None,
    no_gui: bool = False,
    auto_accept: bool = False,
) -> Optional[str]:
    """
    Показать кроп логотипа и запросить у пользователя label (или использовать default_brand).

    Поведение:
    - Пытается показать кроп через OpenCV (GUI-режим).
    - После показа спрашивает в консоли строковый label.
    - Специальные значения:
        - Для default_brand:
            - Enter -> принять default_brand
            - 's' -> пропустить (вернёт None)
            - 'q' -> выйти (вернёт 'q')
            - любой другой ввод -> использовать как brand_name
        - Без default_brand:
            - Enter -> пропустить (вернёт None)
            - 'q' -> выйти (вернёт 'q')

    Если GUI недоступен (ошибка OpenCV), сохраняет превью в KNOWN_ROOT/preview_name
    и всё равно спрашивает label в консоли.
    """
    if auto_accept and default_brand:
        return default_brand

    if not no_gui:
        # Сначала пробуем GUI-режим
        try:
            cv2.imshow("brand", crop)
            # Просто ждём любое нажатие клавиши, а сам label спрашиваем в консоли
            cv2.waitKey(0)
            cv2.destroyWindow("brand")
        except cv2.error:
            # Fallback: пробуем показать через matplotlib (если доступен),
            # если и он недоступен — сохраняем превью в файл.
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt.figure("brand")
                # matplotlib ожидает RGB
                img_rgb = (
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    if len(crop.shape) == 3
                    else crop
                )
                plt.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
                plt.axis("off")
                plt.show()
            except Exception:
                preview_path = os.path.join(known_root, preview_name)
                os.makedirs(known_root, exist_ok=True)
                cv2.imwrite(preview_path, crop)
                print(
                    "Не удалось отобразить превью логотипа (OpenCV/matplotlib недоступны). "
                    f"Превью сохранено в: {preview_path}"
                )

    # В любом случае спрашиваем label в консоли
    if default_brand:
        print(f"Default brand: {default_brand}")
        print("Enter — принять default, 's' — пропустить, 'q' — выход, либо введите другое имя бренда.")
        choice = input("Brand name: ").strip()
        if choice.lower() == "q":
            return "q"
        if choice.lower() == "s":
            return None
        if not choice:
            return default_brand
        return choice

    print("Введите название бренда для этого логотипа (например: coca_cola, nike, apple).")
    print("Специальные значения: 'q' — выход, Enter — пропустить.")
    choice = input("Brand name: ").strip()
    if choice.lower() == "q":
        return "q"
    if not choice:
        return None
    return choice


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.15,
    min_size: int = 32,
) -> np.ndarray:
    """
    Crop image from bounding box with padding.

    This function extracts a region from an image based on a bounding box,
    with additional padding on all sides. The padding helps preserve context
    that may be important for recognition (e.g., logo surroundings, car details).

    The function ensures:
    - Padding stays within image bounds
    - Minimum crop size is maintained
    - Coordinates are properly clipped

    Args:
        image: Input image as numpy array, shape (H, W, C) for color or (H, W) for grayscale
        bbox: Bounding box coordinates as (x1, y1, x2, y2) in image pixel coordinates
        pad_ratio: Padding ratio applied to bounding box dimensions (default: 0.15 = 15%)
            - A value of 0.15 means 15% padding on each side (total 30% added width/height)
        min_size: Minimum size (width or height) of the cropped image in pixels (default: 32)

    Returns:
        Cropped image as numpy array with same number of channels as input

    Example:
        ```python
        bbox = (100, 200, 150, 250)  # x1, y1, x2, y2
        crop = crop_with_padding(image, bbox, pad_ratio=0.20)
        # Crop will include ~20% padding around the original bbox
        ```
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * pad_ratio
    pad_h = box_h * pad_ratio

    # Apply padding
    x1n = max(0, int(x1 - pad_w))
    y1n = max(0, int(y1 - pad_h))
    x2n = min(w, int(x2 + pad_w))
    y2n = min(h, int(y2 + pad_h))

    # Ensure minimum size
    if x2n - x1n < min_size:
        center_x = (x1n + x2n) // 2
        x1n = max(0, center_x - min_size // 2)
        x2n = min(w, x1n + min_size)

    if y2n - y1n < min_size:
        center_y = (y1n + y2n) // 2
        y1n = max(0, center_y - min_size // 2)
        y2n = min(h, y1n + min_size)

    # Crop
    crop = image[y1n:y2n, x1n:x2n]
    return crop


print(f"Обработка изображения: {img_path}")
print(f"Размер изображения: {img.shape}")
print(f"Ожидаемые классы логотипов: {logo_class_ids}")
print(f"Порог confidence: {conf_thr}\n")

processed_logos = 0
all_detections = []
for res in results:
    if res.boxes is None or len(res.boxes) == 0:
        print("Логотипы не найдены")
        continue

    print(f"Всего детекций YOLO: {len(res.boxes)}")
    print("=" * 60)
    
    # Сначала выводим ВСЕ детекции для отладки
    for i, box in enumerate(res.boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy
        
        status = ""
        if cls_id not in logo_class_ids:
            status = " [ПРОПУЩЕН: не в списке классов логотипов]"
        elif conf < conf_thr:
            status = f" [ПРОПУЩЕН: confidence {conf:.2f} < {conf_thr}]"
        else:
            status = " [✓ ПРИНЯТ]"
        
        print(f"Детекция #{i+1}: Класс={cls_id}, Confidence={conf:.3f}, BBox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}){status}")
        all_detections.append((cls_id, conf, (x1, y1, x2, y2)))
    
    print("=" * 60)
    print(f"\nОбработка принятых детекций:\n")

    for i, box in enumerate(res.boxes):
        # Проверяем класс (только logo_region/text_region)
        cls_id = int(box.cls.item())
        if cls_id not in logo_class_ids:
            continue

        conf = float(box.conf.item())
        if conf < conf_thr:
            continue

        # Получаем bbox
        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy

        print(f"\nНайден логотип #{i+1}:")
        print(f"  Класс: {cls_id}, Confidence: {conf:.2f}")
        print(f"  BBox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        # Кроп с padding
        crop = crop_with_padding(
            img,
            (x1, y1, x2, y2),
            pad_ratio=pad_ratio,
            min_size=min_size,
        )

        if crop.size == 0:
            print("  ⊗ Кроп пустой, пропускаем")
            continue

        processed_logos += 1

        # Показать кроп и запросить label
        choice = show_brand_and_get_choice(
            crop,
            known_root="/media/ilya/Новый том/TrendFlowML/DataProcessor/VisualProcessor/core/model_process/core_identity/brand_semantics",
            preview_name=f"preview_{os.path.basename(img_path)}_{i}.jpg",
            default_brand="car/bmw",
            no_gui=False,
            auto_accept=False,
        )
        
        if choice == "q":
            print("Выход по 'q'")
            break
        elif choice is None:
            print("Пропущен")
        else:
            print(f"Выбран бренд: {choice}")

print(f"\nИтого обработано логотипов: {processed_logos}")