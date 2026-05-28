"""Face-specific manager"""

from typing import Any, Dict, Optional

import numpy as np

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from ..errors import EmbeddingExtractionError
from embedding_service.config.settings import EmbeddingServiceConfig
from .base_manager import BaseManager


class FaceManager(BaseManager):
    """Manager for face embeddings"""

    def validate(self, image: np.ndarray) -> bool:
        """Validate face image for search/add.

        Клиенты (например face_identity) часто присылают **уже обрезанные** кропы лица.
        Повторная детекция в InsightFace на таком кропе часто даёт 0 лиц — это не «битая
        картинка». Ограничиваемся базовой проверкой; извлечение эмбеддинга само ретраит
        на upsample/pad в ArcFaceExtractor.
        """
        if not super().validate(image):
            return False
        h, w = int(image.shape[0]), int(image.shape[1])
        if h < 1 or w < 1:
            return False
        return True

    def extract_embedding(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Extract face embedding"""
        # ArcFace extractor handles face detection internally
        return super().extract_embedding(image, **kwargs)

    def normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize face metadata"""
        normalized = super().normalize_metadata(metadata)

        # Add face-specific fields
        normalized.setdefault("type", "face")

        return normalized


