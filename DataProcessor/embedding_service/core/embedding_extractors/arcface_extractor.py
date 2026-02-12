"""ArcFace embedding extractor"""

from typing import Any, Optional

import cv2
import numpy as np

from ..errors import EmbeddingExtractionError, InvalidModelError
from .base import EmbeddingExtractor


class ArcFaceExtractor(EmbeddingExtractor):
    """ArcFace embedding extractor using InsightFace"""

    def __init__(self):
        """Initialize ArcFace extractor"""
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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
            # Detect faces
            faces = self.app.get(image)

            if not faces:
                raise EmbeddingExtractionError("No faces detected in image")

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


