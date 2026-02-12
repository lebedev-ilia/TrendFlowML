"""Brand-specific manager"""

from typing import Any, Dict, Optional

import numpy as np

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from embedding_service.config.settings import EmbeddingServiceConfig
from .base_manager import BaseManager


class BrandManager(BaseManager):
    """Manager for brand embeddings"""

    def normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize brand metadata"""
        normalized = super().normalize_metadata(metadata)

        # Add brand-specific fields
        normalized.setdefault("type", "brand")

        return normalized


