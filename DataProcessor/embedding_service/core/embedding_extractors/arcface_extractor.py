"""ArcFace embedding extractor"""

import os
from typing import Any, List

import cv2
import numpy as np

from ..errors import EmbeddingExtractionError, InvalidModelError
from .base import EmbeddingExtractor


def _insightface_onnxruntime_providers() -> List[str]:
    """
    Предпочтить GPU, если в onnxruntime зарегистрирован CUDAExecutionProvider
    (нужен пакет onnxruntime-gpu, совместимый с драйвером/CUDA).
    INSIGHTFACE_DISABLE_CUDA=1 — принудительно CPU.
    """
    if os.environ.get("INSIGHTFACE_DISABLE_CUDA", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


class ArcFaceExtractor(EmbeddingExtractor):
    """ArcFace embedding extractor using InsightFace"""

    def __init__(self):
        """Initialize ArcFace extractor"""
        try:
            from insightface.app import FaceAnalysis

            _providers = _insightface_onnxruntime_providers()
            self.app = FaceAnalysis(name="buffalo_l", providers=_providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self._embedding_dim = 512  # ArcFace produces 512-d embeddings
            self._model_name = "arcface"
        except ImportError as e:
            raise EmbeddingExtractionError(f"InsightFace not available: {e}") from e
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to initialize ArcFace: {e}") from e

    def extract(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Extract ArcFace embedding from image.

        Args:
            image: Input image (BGR format from OpenCV)
            **kwargs: Additional arguments
                     - bbox: Optional bounding box [x1, y1, x2, y2]
                     - kps: Optional keypoints

        Returns:
            Normalized embedding vector (512-d)
        """
        try:
            # Detect faces (кропы из пайплайна могут быть малыми — ретраи без смены API)
            faces = self.app.get(image)

            if not faces and min(image.shape[0], image.shape[1]) < 160:
                scale = 160.0 / float(min(image.shape[0], image.shape[1]))
                new_w = max(1, int(round(image.shape[1] * scale)))
                new_h = max(1, int(round(image.shape[0] * scale)))
                up = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                faces = self.app.get(up)

            if not faces:
                h, w = int(image.shape[0]), int(image.shape[1])
                ph, pw = max(1, h // 8), max(1, w // 8)
                padded = cv2.copyMakeBorder(image, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
                faces = self.app.get(padded)

            if not faces:
                # Кропы из face_identity: уже «лицо», детектор на чипе часто пустой — recognition-only
                return self._extract_from_recognition_chip(image)

            # Use first face
            face = faces[0]
            embedding = face.embedding.astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            if isinstance(e, EmbeddingExtractionError):
                raise
            raise EmbeddingExtractionError(f"ArcFace extraction failed: {e}") from e

    def _extract_from_recognition_chip(self, image: np.ndarray) -> np.ndarray:
        """
        Когда `FaceAnalysis.get` не находит bbox на узком кропе — считаем, что
        весь кадр уже face chip (как в core_face_identity) и гоняем только ArcFace.
        """
        rec = self.app.models.get("recognition")
        if rec is None:
            raise EmbeddingExtractionError("No faces detected in image")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise EmbeddingExtractionError("Input must be BGR (H,W,3) for ArcFace")

        try:
            tw, th = rec.input_size
        except Exception:
            tw, th = 112, 112
        chip = cv2.resize(
            image, (int(tw), int(th)), interpolation=cv2.INTER_CUBIC
        )
        out = rec.get_feat(chip)
        embedding = np.asarray(out, dtype=np.float32).reshape(-1)
        if embedding.size == 0:
            raise EmbeddingExtractionError("No faces detected in image")
        n = float(np.linalg.norm(embedding))
        if n > 1e-10:
            embedding = embedding / n
        return embedding

    def extract_from_face_data(self, image: np.ndarray, bbox: list, kps: np.ndarray) -> np.ndarray:
        """Extract embedding from already detected face"""
        # For now, use the standard extract method
        # In future, could optimize to skip detection
        return self.extract(image)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self._embedding_dim

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name


