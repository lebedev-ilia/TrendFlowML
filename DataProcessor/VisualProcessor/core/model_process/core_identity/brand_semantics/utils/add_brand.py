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
import argparse
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
LOGO_CLASS_IDS = [33, 34]  # Класс logo_region (можно добавить 34 для text_region, если нужно)


def _parse_int_list(s: str) -> list[int]:
    items: list[int] = []
    for part in s.replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Интерактивное/полуавтоматическое пополнение `known_brands/`.\n"
            "- videos: выборка кадров из видео -> детекция логотипов -> ручной label\n"
            "- photos: плоская папка с фото -> детекция -> ручной label\n"
            "- photos-by-brand: папка с поддиректориями-брендами -> детекция -> auto label из имени папки\n"
            "- known-brands-auto: папка со структурой <category>/<brand_name>/<images> -> детекция -> auto label"
        )
    )
    p.add_argument("--videos", default="", help="Директория с видео (опционально).")
    p.add_argument("--photos", default="", help="Директория с фото (плоская структура, опционально).")
    p.add_argument(
        "--photos-by-brand",
        default="",
        help="Директория, где каждая поддиректория = brand_name и содержит фото этого бренда.",
    )
    p.add_argument(
        "--known-brands-auto",
        default="",
        help=(
            "Директория со структурой <category>/<brand_name>/<images> "
            "(например, known_brands_auto/car/audi/1.jpg). "
            "Автоматически использует путь category/brand_name как default_brand."
        ),
    )
    p.add_argument(
        "--known-root",
        default=KNOWN_ROOT,
        help="Куда сохранять `known_brands/<brand>/N.jpg` (по умолчанию в репозиторий).",
    )
    p.add_argument(
        "--yolo-model-path",
        default=YOLO_MODEL_PATH,
        help="Путь к YOLO модели. Можно также задать через DP_MODELS_ROOT.",
    )
    p.add_argument(
        "--logo-class-ids",
        default=",".join(map(str, LOGO_CLASS_IDS)),
        help="ID классов логотипов в YOLO (пример: '2,3').",
    )
    p.add_argument("--conf", type=float, default=0.5, help="Порог confidence для детекций.")
    p.add_argument("--pad-ratio", type=float, default=0.15, help="Padding вокруг bbox (доля).")
    p.add_argument("--min-size", type=int, default=32, help="Минимальный размер кропа (px).")
    p.add_argument("--max-frames-per-video", type=int, default=20, help="Сколько кадров сэмплить из каждого видео.")
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Не пытаться открывать окна OpenCV/matplotlib (подходит для headless).",
    )
    p.add_argument(
        "--auto-accept",
        action="store_true",
        help=(
            "Автоматически сохранять кропы без вопросов. "
            "Полезно вместе с --photos-by-brand (label берётся из имени подпапки)."
        ),
    )
    return p.parse_args()


def ensure_root_dir(known_root: str) -> None:
    """Убедиться, что корневая директория для базы брендов существует."""
    os.makedirs(known_root, exist_ok=True)


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


def main() -> None:
    """Основной цикл: проход по видео/фото, детекция логотипов и интерактивная разметка."""
    args = parse_args()
    logo_class_ids = _parse_int_list(str(args.logo_class_ids))
    known_root = str(args.known_root)
    ensure_root_dir(known_root)

    # Проверка наличия модели YOLO
    yolo_model_path = str(args.yolo_model_path)
    if not os.path.exists(yolo_model_path):
        raise SystemExit(
            f"Модель YOLO не найдена: {yolo_model_path}\n"
            "Установите переменную окружения DP_MODELS_ROOT или убедитесь, что модель находится по указанному пути."
        )

    # Инициализация модели YOLO
    print(f"Загрузка модели YOLO: {yolo_model_path}")
    model = YOLO(yolo_model_path)

    # Ограничение по количеству обработанных кадров на видео
    max_frames_per_video = int(args.max_frames_per_video)
    conf_thr = float(args.conf)
    pad_ratio = float(args.pad_ratio)
    min_size = int(args.min_size)
    no_gui = bool(args.no_gui)
    auto_accept = bool(args.auto_accept)

    # --- 1. Проходим по всем видео в директории ---
    if args.videos and os.path.isdir(args.videos):
        for video in sorted(os.listdir(args.videos)):
            video_path = os.path.join(args.videos, video)
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
                        if cls_id not in logo_class_ids:
                            continue

                        conf = float(box.conf.item())
                        if conf < conf_thr:  # Минимальный порог уверенности
                            continue

                        # Получаем bbox
                        xyxy = box.xyxy[0].cpu().numpy().astype(float)
                        x1, y1, x2, y2 = xyxy

                        # Кроп с padding
                        crop = crop_with_padding(
                            frame,
                            (x1, y1, x2, y2),
                            pad_ratio=pad_ratio,
                            min_size=min_size,
                        )

                        if crop.size == 0:
                            continue

                        processed_frames += 1

                        # Показать кроп и запросить label
                        choice = show_brand_and_get_choice(
                            crop,
                            known_root=known_root,
                            preview_name=f"preview_{video}_{processed_frames}_{i}.jpg",
                            no_gui=no_gui,
                            auto_accept=False,  # видео режим всегда требует явного label
                        )

                        if choice == "q":
                            print("Выход по 'q'. Останавливаем обработку.")
                            cap.release()
                            return

                        if choice is None:
                            # Пользователь решил пропустить этот логотип
                            continue

                        # Папка для конкретного бренда
                        brand_dir = os.path.join(known_root, choice)
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

    # --- 2. Обработка фото из поддиректорий-брендов (auto label) ---
    if args.photos_by_brand and os.path.isdir(args.photos_by_brand):
        print(f"\n=== Обработка фото по брендам из: {args.photos_by_brand} ===")
        root = Path(args.photos_by_brand)
        for brand_dir_src in sorted([p for p in root.iterdir() if p.is_dir()]):
            brand_name = brand_dir_src.name
            print(f"\n--- Бренд: {brand_name} ---")
            for img_path in sorted([p for p in brand_dir_src.iterdir() if p.is_file()]):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Не удалось прочитать изображение: {img_path}")
                    continue
                results = model(img, verbose=False)
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    for i, box in enumerate(res.boxes):
                        cls_id = int(box.cls.item())
                        if cls_id not in logo_class_ids:
                            continue
                        conf = float(box.conf.item())
                        if conf < conf_thr:
                            continue
                        xyxy = box.xyxy[0].cpu().numpy().astype(float)
                        x1, y1, x2, y2 = xyxy
                        crop = crop_with_padding(
                            img,
                            (x1, y1, x2, y2),
                            pad_ratio=pad_ratio,
                            min_size=min_size,
                        )
                        if crop.size == 0:
                            continue
                        choice = show_brand_and_get_choice(
                            crop,
                            known_root=known_root,
                            preview_name=f"preview_{brand_name}_{img_path.name}_{i}.jpg",
                            default_brand=brand_name,
                            no_gui=no_gui,
                            auto_accept=auto_accept,
                        )
                        if choice == "q":
                            print("Выход по 'q'. Останавливаем обработку.")
                            return
                        if choice is None:
                            continue
                        out_dir = os.path.join(known_root, choice)
                        os.makedirs(out_dir, exist_ok=True)
                        idx = get_next_index_for_brand(out_dir)
                        out_path = os.path.join(out_dir, f"{idx}.jpg")
                        cv2.imwrite(out_path, crop)
                        print(f"Сохранен логотип в {out_path}")

    # --- 3. Обработка отдельных фото (если директория существует) ---
    if args.photos and os.path.isdir(args.photos):
        print(f"\n=== Обработка фото из: {args.photos} ===")
        for img_name in sorted(os.listdir(args.photos)):
            img_path = os.path.join(args.photos, img_name)
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
                    if cls_id not in logo_class_ids:
                        continue

                    conf = float(box.conf.item())
                    if conf < conf_thr:
                        continue

                    # Получаем bbox
                    xyxy = box.xyxy[0].cpu().numpy().astype(float)
                    x1, y1, x2, y2 = xyxy

                    # Кроп с padding
                    crop = crop_with_padding(
                        img,
                        (x1, y1, x2, y2),
                        pad_ratio=pad_ratio,
                        min_size=min_size,
                    )

                    if crop.size == 0:
                        continue

                    choice = show_brand_and_get_choice(
                        crop,
                        known_root=known_root,
                        preview_name=f"preview_photo_{img_name}_{i}.jpg",
                        no_gui=no_gui,
                        auto_accept=False,
                    )

                    if choice == "q":
                        print("Выход по 'q'. Останавливаем обработку.")
                        return

                    if choice is None:
                        continue

                    brand_dir = os.path.join(known_root, choice)
                    os.makedirs(brand_dir, exist_ok=True)

                    idx = get_next_index_for_brand(brand_dir)
                    filename = os.path.join(brand_dir, f"{idx}.jpg")
                    cv2.imwrite(filename, crop)
                    print(f"Сохранен логотип из фото в {filename}")

    # --- 4. Обработка known_brands_auto (структура category/brand_name/images) ---
    if args.known_brands_auto and os.path.isdir(args.known_brands_auto):
        print(f"\n=== Обработка known_brands_auto из: {args.known_brands_auto} ===")
        root = Path(args.known_brands_auto)
        
        # Проходим по категориям (car, sport_wear, wear и т.д.)
        for category_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            category_name = category_dir.name
            print(f"\n--- Категория: {category_name} ---")
            
            # Проходим по брендам внутри категории
            for brand_dir_src in sorted([p for p in category_dir.iterdir() if p.is_dir()]):
                brand_name = brand_dir_src.name
                # Используем путь category/brand_name как default_brand
                default_brand_path = f"{category_name}/{brand_name}"
                print(f"\n--- Бренд: {default_brand_path} ---")
                
                # Проходим по изображениям в папке бренда
                image_files = sorted([p for p in brand_dir_src.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
                total_images = len(image_files)
                found_logos_count = 0
                
                for img_idx, img_path in enumerate(image_files, 1):
                    print(f"  Обработка изображения {img_idx}/{total_images}: {img_path.name}")
                    
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"    ⚠ Не удалось прочитать изображение: {img_path}")
                        continue
                    
                    # Детекция через YOLO
                    results = model(img, verbose=False)
                    
                    logos_found_in_image = 0
                    for res in results:
                        if res.boxes is None or len(res.boxes) == 0:
                            continue
                        
                        for i, box in enumerate(res.boxes):
                            cls_id = int(box.cls.item())
                            if cls_id not in logo_class_ids:
                                continue
                            
                            conf = float(box.conf.item())
                            if conf < conf_thr:
                                continue
                            
                            xyxy = box.xyxy[0].cpu().numpy().astype(float)
                            x1, y1, x2, y2 = xyxy
                            
                            crop = crop_with_padding(
                                img,
                                (x1, y1, x2, y2),
                                pad_ratio=pad_ratio,
                                min_size=min_size,
                            )
                            
                            if crop.size == 0:
                                continue
                            
                            logos_found_in_image += 1
                            found_logos_count += 1
                            
                            print(f"    ✓ Найден логотип #{logos_found_in_image} (confidence: {conf:.2f})")
                            
                            # Показать кроп и запросить label (с default_brand из пути)
                            if not auto_accept:
                                print(f"    → Default brand: {default_brand_path}")
                                print(f"    → Enter — принять default, 's' — пропустить, 'q' — выход, либо введите другое имя бренда")
                            
                            choice = show_brand_and_get_choice(
                                crop,
                                known_root=known_root,
                                preview_name=f"preview_{category_name}_{brand_name}_{img_path.name}_{i}.jpg",
                                default_brand=default_brand_path,
                                no_gui=no_gui,
                                auto_accept=auto_accept,
                            )
                            
                            if choice == "q":
                                print("Выход по 'q'. Останавливаем обработку.")
                                return
                            
                            if choice is None:
                                print("    ⊗ Пропущен")
                                continue
                            
                            # Сохраняем в known_root с сохранением структуры пути (если указан путь с /)
                            # или просто как brand_name (если указано только имя)
                            out_dir = os.path.join(known_root, choice)
                            os.makedirs(out_dir, exist_ok=True)
                            
                            idx = get_next_index_for_brand(out_dir)
                            out_path = os.path.join(out_dir, f"{idx}.jpg")
                            cv2.imwrite(out_path, crop)
                            print(f"    ✓ Сохранен логотип в {out_path}")
                    
                    if logos_found_in_image == 0:
                        print(f"    ⊗ Логотипы не найдены на изображении")
                
                print(f"  Итого найдено логотипов для {default_brand_path}: {found_logos_count}")

    print("\n=== Готово ===")
    print(f"База брендов сохранена в: {known_root}")
    print(
        "Для синхронизации с Embedding Service запустите: "
        "sync_known_brands_to_embedding_service.py"
    )


if __name__ == "__main__":
    main()

