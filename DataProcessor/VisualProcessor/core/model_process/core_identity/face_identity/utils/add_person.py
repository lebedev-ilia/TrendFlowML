"""
Интерактивное добавление лиц в локальную базу `known_people/`.

Скрипт проходит по всем видео в директории `videos`, находит лица,
показывает выровненное лицо и спрашивает у пользователя label (имя/идентификатор).

Для каждого label создаётся папка `known_people/<label>/`, в которую
сохраняются изображения лиц с уникальными числовыми именами (`1.jpg`, `2.jpg`, ...).
"""

import os
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_aligment import align_face

# Директория с видео, из которых будем вытаскивать лица
videos = "/home/ilya/Рабочий стол/TrendFlowML/example/example_videos"
photos = "/home/ilya/Рабочий стол/TrendFlowML/example/example_photo"

# Корень локальной базы лиц
KNOWN_ROOT = "DataProcessor/VisualProcessor/core/model_process/core_identity/face_identity/known_people"


def ensure_root_dir() -> None:
    """Убедиться, что корневая директория для базы лиц существует."""
    os.makedirs(KNOWN_ROOT, exist_ok=True)


def get_next_index_for_person(person_dir: str) -> int:
    """
    Найти следующий свободный числовой индекс для файлов в папке человека.

    Имена файлов ожидаются в формате `<number>.<ext>` (например, `1.jpg`, `2.png`).
    """
    if not os.path.isdir(person_dir):
        return 1

    max_id = 0
    for fname in os.listdir(person_dir):
        name, _ = os.path.splitext(fname)
        try:
            num = int(name)
        except ValueError:
            continue
        max_id = max(max_id, num)
    return max_id + 1


def show_face_and_get_choice(aligned: np.ndarray, preview_name: str = "preview_face.jpg") -> Optional[str]:
    """
    Показать выровненное лицо и запросить у пользователя label.

    Поведение:
    - Пытается показать лицо через OpenCV (GUI-режим).
    - После показа спрашивает в консоли строковый label.
    - Специальные значения:
        - пустой ввод -> пропустить это лицо (вернёт None)
        - 'q' -> выйти из скрипта (вернёт 'q')

    Если GUI недоступен (ошибка OpenCV), сохраняет превью в KNOWN_ROOT/preview_name
    и всё равно спрашивает label в консоли.
    """
    # Сначала пробуем GUI-режим
    try:
        cv2.imshow("face", aligned)
        # Просто ждём любое нажатие клавиши, а сам label спрашиваем в консоли
        cv2.waitKey(0)
        cv2.destroyWindow("face")
    except cv2.error:
        # Fallback: пробуем показать через matplotlib (если доступен),
        # если и он недоступен — сохраняем превью в файл.
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure("face")
            # matplotlib ожидает RGB
            img_rgb = (
                cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                if len(aligned.shape) == 3
                else aligned
            )
            plt.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
            plt.axis("off")
            plt.show()
        except Exception:
            # Если даже matplotlib недоступен — сохраняем превью в файл,
            # чтобы можно было открыть его вручную.
            preview_path = os.path.join(KNOWN_ROOT, preview_name)
            os.makedirs(KNOWN_ROOT, exist_ok=True)
            cv2.imwrite(preview_path, aligned)
            print(
                "Не удалось отобразить превью лица (OpenCV/matplotlib недоступны). "
                f"Превью сохранено в: {preview_path}"
            )

    # В любом случае спрашиваем label в консоли
    print("Введите label для этого лица .")
    print("Специальные значения: 'q' — выход, пустая строка — пропустить.")
    choice = input("Label: ").strip()

    if not choice:
        # Пропустить это лицо
        return None
    if choice.lower() == "q":
        return "q"
    return choice


def main() -> None:
    """Основной цикл: проход по видео, детекция лиц и интерактивная разметка."""
    ensure_root_dir()

    # Инициализация модели лиц (InsightFace)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # --- 1. Проходим по всем видео в директории ---
    for video in sorted(os.listdir(videos)):
        video_path = os.path.join(videos, video)
        if not os.path.isfile(video_path):
            continue

        print(f"\n=== Обработка видео: {video_path} ===")
        cap = cv2.VideoCapture(video_path)

        # Ограничение по количеству обработанных кадров на видео (можно настроить)
        max_frames_per_video = 20

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

            faces = app.get(frame)
            if not faces:
                continue

            processed_frames += 1

            for i, f in enumerate(faces):
                bbox = f.bbox.astype(int)  # x1, y1, x2, y2
                kps = f.kps  # 5 ключевых точек

                x1, y1, x2, y2 = bbox
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Переводим ключевые точки в координаты внутри кропа
                kps_local = kps.copy()
                kps_local[:, 0] -= x1
                kps_local[:, 1] -= y1

                # Выровненное лицо
                aligned = align_face(face_crop, kps_local, output_size=(224, 224))

                # Показать лицо и запросить label
                choice = show_face_and_get_choice(
                    aligned, preview_name=f"preview_{video}_{processed_frames}_{i}.jpg"
                )

                if choice == "q":
                    print("Выход по 'q'. Останавливаем обработку.")
                    cap.release()
                    return

                if choice is None:
                    # Пользователь решил пропустить это лицо
                    continue

                # Папка для конкретного человека
                person_dir = os.path.join(KNOWN_ROOT, choice)
                os.makedirs(person_dir, exist_ok=True)

                # Следующий свободный индекс
                idx = get_next_index_for_person(person_dir)
                filename = os.path.join(person_dir, f"{idx}.jpg")
                cv2.imwrite(filename, aligned)
                print(f"Сохранено лицо в {filename}")

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

            faces = app.get(img)
            if not faces:
                continue

            for i, f in enumerate(faces):
                bbox = f.bbox.astype(int)
                kps = f.kps

                x1, y1, x2, y2 = bbox
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                kps_local = kps.copy()
                kps_local[:, 0] -= x1
                kps_local[:, 1] -= y1

                aligned = align_face(face_crop, kps_local, output_size=(224, 224))

                choice = show_face_and_get_choice(
                    aligned, preview_name=f"preview_photo_{img_name}_{i}.jpg"
                )

                if choice == "q":
                    print("Выход по 'q'. Останавливаем обработку.")
                    return

                if choice is None:
                    continue

                person_dir = os.path.join(KNOWN_ROOT, choice)
                os.makedirs(person_dir, exist_ok=True)

                idx = get_next_index_for_person(person_dir)
                filename = os.path.join(person_dir, f"{idx}.jpg")
                cv2.imwrite(filename, aligned)
                print(f"Сохранено лицо из фото в {filename}")


if __name__ == "__main__":
    main()