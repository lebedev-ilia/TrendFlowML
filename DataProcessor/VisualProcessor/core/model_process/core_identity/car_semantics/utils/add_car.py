"""
Интерактивное добавление машин в локальную базу `known_cars/`.

Скрипт проходит по всем видео в директории `videos`, находит машины через YOLO,
показывает кроп машины и спрашивает у пользователя label (марка/модель/идентификатор).

Для каждого label создаётся папка `known_cars/<label>/`, в которую
сохраняются изображения машин с уникальными числовыми именами (`1.jpg`, `2.jpg`, ...).
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
_car_semantics_dir = Path(__file__).parent
if str(_car_semantics_dir) not in sys.path:
    sys.path.insert(0, str(_car_semantics_dir))

from ultralytics import YOLO  # type: ignore

from crop_utils import crop_with_padding

# Директория с видео, из которых будем вытаскивать машины
videos = "" # "/home/ilya/Рабочий стол/TrendFlowML/example/example_videos"
photos = "/home/ilya/Изображения/Снимки экрана/cars"

# Корень локальной базы машин
KNOWN_ROOT = "DataProcessor/VisualProcessor/core/model_process/core_identity/car_semantics/known_cars"

# Путь к YOLO модели
# Пробуем найти через DP_MODELS_ROOT или относительный путь
DP_MODELS_ROOT = os.environ.get("DP_MODELS_ROOT", "")
if DP_MODELS_ROOT:
    YOLO_MODEL_PATH = os.path.join(DP_MODELS_ROOT, "visual/yolo/yolo11x_41_best.pt")
else:
    # Относительный путь от корня проекта
    YOLO_MODEL_PATH = "DataProcessor/dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt"

# ID класса "car" в таксономии YOLO (из _FINAL_TAXONOMY_V1_CLASSES)
CAR_CLASS_ID = 2  # "car" - второй класс в списке (индекс 2)


def ensure_root_dir() -> None:
    """Убедиться, что корневая директория для базы машин существует."""
    os.makedirs(KNOWN_ROOT, exist_ok=True)


def get_next_index_for_car(car_dir: str) -> int:
    """
    Найти следующий свободный числовой индекс для файлов в папке машины.

    Имена файлов ожидаются в формате `<number>.<ext>` (например, `1.jpg`, `2.png`).
    """
    if not os.path.isdir(car_dir):
        return 1

    max_id = 0
    for fname in os.listdir(car_dir):
        name, _ = os.path.splitext(fname)
        try:
            num = int(name)
        except ValueError:
            continue
        max_id = max(max_id, num)
    return max_id + 1


def show_car_and_get_choice(crop: np.ndarray, preview_name: str = "preview_car.jpg") -> Optional[str]:
    """
    Показать кроп машины и запросить у пользователя label.

    Поведение:
    - Пытается показать кроп через OpenCV (GUI-режим).
    - После показа спрашивает в консоли строковый label.
    - Специальные значения:
        - пустой ввод -> пропустить эту машину (вернёт None)
        - 'q' -> выйти из скрипта (вернёт 'q')

    Если GUI недоступен (ошибка OpenCV), сохраняет превью в KNOWN_ROOT/preview_name
    и всё равно спрашивает label в консоли.
    """
    # Сначала пробуем GUI-режим
    try:
        cv2.imshow("car", crop)
        # Просто ждём любое нажатие клавиши, а сам label спрашиваем в консоли
        cv2.waitKey(0)
        cv2.destroyWindow("car")
    except cv2.error:
        # Fallback: пробуем показать через matplotlib (если доступен),
        # если и он недоступен — сохраняем превью в файл.
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure("car")
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
                "Не удалось отобразить превью машины (OpenCV/matplotlib недоступны). "
                f"Превью сохранено в: {preview_path}"
            )

    # В любом случае спрашиваем label в консоли
    print("Введите label для этой машины (например: tesla_model_3, bmw_x5, toyota_camry).")
    print("Специальные значения: 'q' — выход, пустая строка — пропустить.")
    choice = input("Label: ").strip()

    if not choice:
        # Пропустить эту машину
        return None
    if choice.lower() == "q":
        return "q"
    return choice


def main() -> None:
    """Основной цикл: проход по видео, детекция машин и интерактивная разметка."""
    ensure_root_dir()

    # # Проверка наличия модели YOLO
    # if not os.path.exists(YOLO_MODEL_PATH):
    #     raise SystemExit(
    #         f"Модель YOLO не найдена: {YOLO_MODEL_PATH}\n"
    #         "Установите переменную окружения DP_MODELS_ROOT или убедитесь, что модель находится по указанному пути."
    #     )

    # Инициализация модели YOLO
    print(f"Загрузка модели YOLO: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # Ограничение по количеству обработанных кадров на видео
    max_frames_per_video = 20

    # --- 1. Проходим по всем видео в директории ---
    if os.path.isdir(videos):
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
                        # Проверяем класс (только "car")
                        cls_id = int(box.cls.item())
                        if cls_id != CAR_CLASS_ID:
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
                        choice = show_car_and_get_choice(
                            crop, preview_name=f"preview_{video}_{processed_frames}_{i}.jpg"
                        )

                        if choice == "q":
                            print("Выход по 'q'. Останавливаем обработку.")
                            cap.release()
                            return

                        if choice is None:
                            # Пользователь решил пропустить эту машину
                            continue

                        # Папка для конкретной машины
                        car_dir = os.path.join(KNOWN_ROOT, choice)
                        os.makedirs(car_dir, exist_ok=True)

                        # Следующий свободный индекс
                        idx = get_next_index_for_car(car_dir)
                        filename = os.path.join(car_dir, f"{idx}.jpg")
                        cv2.imwrite(filename, crop)
                        print(f"Сохранена машина в {filename}")

                if processed_frames >= max_frames_per_video:
                    print(f"Достигнут лимит {max_frames_per_video} кадров для видео, переходим к следующему.")
                    break

            cap.release()

    # --- 2. Обработка отдельных фото (если директория существует) ---
    if os.path.isdir(photos):
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
                    # Проверяем класс (только "car")
                    cls_id = int(box.cls.item())
                    if cls_id != CAR_CLASS_ID:
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

                    choice = show_car_and_get_choice(
                        crop, preview_name=f"preview_photo_{img_name}_{i}.jpg"
                    )

                    if choice == "q":
                        print("Выход по 'q'. Останавливаем обработку.")
                        return

                    if choice is None:
                        continue

                    car_dir = os.path.join(KNOWN_ROOT, choice)
                    os.makedirs(car_dir, exist_ok=True)

                    idx = get_next_index_for_car(car_dir)
                    filename = os.path.join(car_dir, f"{idx}.jpg")
                    cv2.imwrite(filename, crop)
                    print(f"Сохранена машина из фото в {filename}")


if __name__ == "__main__":
    main()

