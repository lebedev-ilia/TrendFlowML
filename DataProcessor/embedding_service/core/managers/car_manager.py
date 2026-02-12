"""Car-specific manager"""

from typing import Any, Dict, Optional

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from embedding_service.config.settings import EmbeddingServiceConfig
from .base_manager import BaseManager


class CarManager(BaseManager):
    """Manager for car embeddings"""

    def normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize car metadata"""
        normalized = super().normalize_metadata(metadata)

        # Add car-specific fields
        normalized.setdefault("type", "car")

        return normalized


