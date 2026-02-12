import numpy as np
import cv2

# Canonical 5-point ArcFace landmarks
ARC_FACE_5PTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)


def align_face(img, landmarks, output_size=(224, 224)):
    """
    Выравнивание лица под формат ArcFace.

    :param img: BGR‑изображение лица (обрезанное по bbox), np.ndarray (H, W, 3)
    :param landmarks: np.ndarray формы (5, 2) — 5 ключевых точек лица
    :param output_size: (width, height) выходного изображения, по умолчанию 224x224
    :return: выровненное BGR‑изображение лица заданного размера
    """
    src = np.array(landmarks, dtype=np.float32)

    # Получаем аффинное преобразование (аналог ArcFace align)
    M, _ = cv2.estimateAffinePartial2D(src, ARC_FACE_5PTS, method=cv2.LMEDS)

    if M is None:
        # Если по какой‑то причине матрица не нашлась, возвращаем исходный кроп
        return cv2.resize(img, output_size)

    # Применяем трансформацию
    aligned = cv2.warpAffine(
        img, M, output_size,
        borderValue=0.0
    )
    return aligned


