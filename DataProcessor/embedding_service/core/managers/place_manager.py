"""Place-specific manager"""

from typing import Any, Dict, Optional

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from embedding_service.config.settings import EmbeddingServiceConfig
from .base_manager import BaseManager


class PlaceManager(BaseManager):
    """Manager for place embeddings"""

    def normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize place metadata"""
        normalized = super().normalize_metadata(metadata)

        # Add place-specific fields
        normalized.setdefault("type", "place")

        return normalized


