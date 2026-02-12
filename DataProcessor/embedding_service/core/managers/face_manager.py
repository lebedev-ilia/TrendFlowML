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
        """Validate face image"""
        if not super().validate(image):
            return False

        # Check if face can be detected
        try:
            faces = self.extractor.app.get(image)
            return len(faces) > 0
        except Exception:
            return False

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


