"""Factory for creating category managers"""

from typing import Dict

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from ..errors import InvalidCategoryError
from embedding_service.config.settings import EmbeddingServiceConfig
from .base_manager import BaseManager
from .brand_manager import BrandManager
from .car_manager import CarManager
from .face_manager import FaceManager
from .place_manager import PlaceManager


class ManagerFactory:
    """Factory for creating category-specific managers"""

    _manager_classes: Dict[str, type] = {
        "face": FaceManager,
        "face_semantic": FaceManager,
        "brand": BrandManager,
        "brand_semantic": BrandManager,
        "car": CarManager,
        "car_semantic": CarManager,
        "place": PlaceManager,
        "place_semantic": PlaceManager,
        # Generic fallback
        "person": BaseManager,
        "object": BaseManager,
        "logo": BaseManager,
        "franchise": BaseManager,  # Franchise recognition uses generic BaseManager
    }

    @classmethod
    def create(
        cls,
        category: str,
        config: EmbeddingServiceConfig,
        db_store: PostgresEmbeddingStore,
        faiss_manager: FaissIndexManager,
    ) -> BaseManager:
        """
        Create manager for a category.

        Args:
            category: Category name
            config: Service configuration
            db_store: PostgreSQL store
            faiss_manager: FAISS index manager

        Returns:
            BaseManager instance
        """
        manager_class = cls._manager_classes.get(category, BaseManager)

        return manager_class(category, config, db_store, faiss_manager)


def get_manager_for_category(
    category: str,
    config: EmbeddingServiceConfig,
    db_store: PostgresEmbeddingStore,
    faiss_manager: FaissIndexManager,
) -> BaseManager:
    """Get manager for a category"""
    if category not in config.category_model_mapping:
        raise InvalidCategoryError(category)

    return ManagerFactory.create(category, config, db_store, faiss_manager)

