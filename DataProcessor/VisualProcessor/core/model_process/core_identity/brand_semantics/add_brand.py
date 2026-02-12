#!/usr/bin/env python3
"""
Интерактивное добавление брендов в локальную базу `known_brands/`.

Скрипт проходит по всем видео/фото в указанных директориях, находит логотипы через YOLO,
показывает кроп логотипа и спрашивает у пользователя label (название бренда).

Для каждого бренда создаётся папка `known_brands/<brand_name>/`, в которую
сохраняются изображения логотипов с уникальными числовыми именами (`1.jpg`, `2.jpg`, ...).

После заполнения базы используйте `sync_known_brands_to_embedding_service.py`
для синхронизации с Embedding Service.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Добавить корень DataProcessor в путь
_vp_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_vp_root) not in sys.path:
    sys.path.insert(0, str(_vp_root))

# Добавить текущую директорию для импорта crop_utils
_brand_semantics_dir = Path(__file__).parent
if str(_brand_semantics_dir) not in sys.path:
    sys.path.insert(0, str(_brand_semantics_dir))

from ultralytics import YOLO  # type: ignore

from crop_utils import crop_with_padding

# Директория с видео, из которых будем вытаскивать логотипы
videos = ""  # "/home/ilya/Рабочий стол/TrendFlowML/example/example_videos"
photos = ""  # "/home/ilya/Рабочий стол/TrendFlowML/example/example_photos"

# Корень локальной базы брендов
KNOWN_ROOT = "DataProcessor/VisualProcessor/core/model_process/core_identity/brand_semantics/known_brands"

# Путь к YOLO модели
DP_MODELS_ROOT = os.environ.get("DP_MODELS_ROOT", "")
if DP_MODELS_ROOT:
    YOLO_MODEL_PATH = os.path.join(DP_MODELS_ROOT, "visual/yolo/yolo11x_41_best.pt")
else:
    # Относительный путь от корня проекта
    YOLO_MODEL_PATH = "DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt"

# ID класса "logo_region" в таксономии YOLO
# Проверьте актуальный ID в вашей модели (обычно это один из классов: logo_region, text_region, brand)
LOGO_CLASS_IDS = [2, 3]  # Может быть несколько классов (logo_region, text_region)


def ensure_root_dir() -> None:
    """Убедиться, что корневая директория для базы брендов существует."""
    os.makedirs(KNOWN_ROOT, exist_ok=True)


def get_next_index_for_brand(brand_dir: str) -> int:
    """
    Найти следующий свободный числовой индекс для файлов в папке бренда.

    Имена файлов ожидаются в формате `<number>.<ext>` (например, `1.jpg`, `2.png`).
    """
    if not os.path.isdir(brand_dir):
        return 1

    max_id = 0
    for fname in os.listdir(brand_dir):
        name, _ = os.path.splitext(fname)
        try:
            num = int(name)
        except ValueError:
            continue
        max_id = max(max_id, num)
    return max_id + 1


def show_brand_and_get_choice(
    crop: np.ndarray, preview_name: str = "preview_brand.jpg"
) -> Optional[str]:
    """
    Показать кроп логотипа и запросить у пользователя label.

    Поведение:
    - Пытается показать кроп через OpenCV (GUI-режим).
    - После показа спрашивает в консоли строковый label.
    - Специальные значения:
        - пустой ввод -> пропустить этот логотип (вернёт None)
        - 'q' -> выйти из скрипта (вернёт 'q')

    Если GUI недоступен (ошибка OpenCV), сохраняет превью в KNOWN_ROOT/preview_name
    и всё равно спрашивает label в консоли.
    """
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
            # Если даже matplotlib недоступен — сохраняем превью в файл,
            # чтобы можно было открыть его вручную.
            preview_path = os.path.join(KNOWN_ROOT, preview_name)
            os.makedirs(KNOWN_ROOT, exist_ok=True)
            cv2.imwrite(preview_path, crop)
            print(
                "Не удалось отобразить превью логотипа (OpenCV/matplotlib недоступны). "
                f"Превью сохранено в: {preview_path}"
            )

    # В любом случае спрашиваем label в консоли
    print("Введите название бренда для этого логотипа (например: coca_cola, nike, apple).")
    print("Специальные значения: 'q' — выход, пустая строка — пропустить.")
    choice = input("Brand name: ").strip()

    if not choice:
        # Пропустить этот логотип
        return None
    if choice.lower() == "q":
        return "q"
    return choice


def main() -> None:
    """Основной цикл: проход по видео/фото, детекция логотипов и интерактивная разметка."""
    ensure_root_dir()

    # Проверка наличия модели YOLO
    if not os.path.exists(YOLO_MODEL_PATH):
        raise SystemExit(
            f"Модель YOLO не найдена: {YOLO_MODEL_PATH}\n"
            "Установите переменную окружения DP_MODELS_ROOT или убедитесь, что модель находится по указанному пути."
        )

    # Инициализация модели YOLO
    print(f"Загрузка модели YOLO: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # Ограничение по количеству обработанных кадров на видео
    max_frames_per_video = 20

    # --- 1. Проходим по всем видео в директории ---
    if videos and os.path.isdir(videos):
        for video in sorted(os.listdir(videos)):
            video_path = os.path.join(videos, video)
            if not os.path.isfile(video_path):
                continue

            print(f"\n=== Обработка видео: {video_path} ===")
            cap = cv2.VideoCapture(video_path)

            # --- Равномерная выборка кадров по всему видео ---
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                # fallback: просто берём до max_frames_per_video первых успешных кадров
                frame_indices = list(range(max_frames_per_video))
            else:
                num_samples = min(max_frames_per_video, frame_count)
                frame_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)

            processed_frames = 0
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok:
                    continue

                # Детекция через YOLO
                results = model(frame, verbose=False)

                # Обработка результатов
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue

                    for i, box in enumerate(res.boxes):
                        # Проверяем класс (только logo_region/text_region)
                        cls_id = int(box.cls.item())
                        if cls_id not in LOGO_CLASS_IDS:
                            continue

                        conf = float(box.conf.item())
                        if conf < 0.5:  # Минимальный порог уверенности
                            continue

                        # Получаем bbox
                        xyxy = box.xyxy[0].cpu().numpy().astype(float)
                        x1, y1, x2, y2 = xyxy

                        # Кроп с padding
                        crop = crop_with_padding(
                            frame,
                            (x1, y1, x2, y2),
                            pad_ratio=0.15,
                            min_size=32,
                        )

                        if crop.size == 0:
                            continue

                        processed_frames += 1

                        # Показать кроп и запросить label
                        choice = show_brand_and_get_choice(
                            crop, preview_name=f"preview_{video}_{processed_frames}_{i}.jpg"
                        )

                        if choice == "q":
                            print("Выход по 'q'. Останавливаем обработку.")
                            cap.release()
                            return

                        if choice is None:
                            # Пользователь решил пропустить этот логотип
                            continue

                        # Папка для конкретного бренда
                        brand_dir = os.path.join(KNOWN_ROOT, choice)
                        os.makedirs(brand_dir, exist_ok=True)

                        # Следующий свободный индекс
                        idx = get_next_index_for_brand(brand_dir)
                        filename = os.path.join(brand_dir, f"{idx}.jpg")
                        cv2.imwrite(filename, crop)
                        print(f"Сохранен логотип в {filename}")

                if processed_frames >= max_frames_per_video:
                    print(
                        f"Достигнут лимит {max_frames_per_video} кадров для видео, переходим к следующему."
                    )
                    break

            cap.release()

    # --- 2. Обработка отдельных фото (если директория существует) ---
    if photos and os.path.isdir(photos):
        print(f"\n=== Обработка фото из: {photos} ===")
        for img_name in sorted(os.listdir(photos)):
            img_path = os.path.join(photos, img_name)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось прочитать изображение: {img_path}")
                continue

            # Детекция через YOLO
            results = model(img, verbose=False)

            for res in results:
                if res.boxes is None or len(res.boxes) == 0:
                    continue

                for i, box in enumerate(res.boxes):
                    # Проверяем класс (только logo_region/text_region)
                    cls_id = int(box.cls.item())
                    if cls_id not in LOGO_CLASS_IDS:
                        continue

                    conf = float(box.conf.item())
                    if conf < 0.5:
                        continue

                    # Получаем bbox
                    xyxy = box.xyxy[0].cpu().numpy().astype(float)
                    x1, y1, x2, y2 = xyxy

                    # Кроп с padding
                    crop = crop_with_padding(
                        img,
                        (x1, y1, x2, y2),
                        pad_ratio=0.15,
                        min_size=32,
                    )

                    if crop.size == 0:
                        continue

                    choice = show_brand_and_get_choice(
                        crop, preview_name=f"preview_photo_{img_name}_{i}.jpg"
                    )

                    if choice == "q":
                        print("Выход по 'q'. Останавливаем обработку.")
                        return

                    if choice is None:
                        continue

                    brand_dir = os.path.join(KNOWN_ROOT, choice)
                    os.makedirs(brand_dir, exist_ok=True)

                    idx = get_next_index_for_brand(brand_dir)
                    filename = os.path.join(brand_dir, f"{idx}.jpg")
                    cv2.imwrite(filename, crop)
                    print(f"Сохранен логотип из фото в {filename}")

    print("\n=== Готово ===")
    print(f"База брендов сохранена в: {KNOWN_ROOT}")
    print(
        "Для синхронизации с Embedding Service запустите: "
        "sync_known_brands_to_embedding_service.py"
    )


if __name__ == "__main__":
    main()

