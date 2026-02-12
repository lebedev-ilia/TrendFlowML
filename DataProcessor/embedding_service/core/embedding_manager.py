"""Main embedding manager"""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from embedding_service.config.settings import EmbeddingServiceConfig
from .database.faiss_index import FaissIndexManager
from .database.postgres import PostgresEmbeddingStore
from .errors import EmbeddingNotFoundError, InvalidCategoryError
from .managers.base_manager import BaseManager
from .managers.factory import ManagerFactory


class EmbeddingManager:
    """Main embedding manager - unified interface for all categories"""

    def __init__(self, config: EmbeddingServiceConfig):
        self.config = config

        # Initialize database store
        self.db_store = PostgresEmbeddingStore(
            host=config.postgres_host,
            port=config.postgres_port,
            database=config.postgres_db,
            user=config.postgres_user,
            password=config.postgres_password,
        )

        # Initialize FAISS manager
        self.faiss_manager = FaissIndexManager(config.faiss_index_path)

        # Cache of managers per category
        self._managers: Dict[str, BaseManager] = {}

    def _get_manager(self, category: str) -> BaseManager:
        """Get or create manager for category"""
        if category not in self._managers:
            self._managers[category] = ManagerFactory.create(
                category,
                self.config,
                self.db_store,
                self.faiss_manager,
            )
        return self._managers[category]

    def add(
        self,
        *,
        category: str,
        image: np.ndarray,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        object_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """Add object to embedding service"""
        manager = self._get_manager(category)
        return manager.add(
            image=image,
            name=name,
            metadata=metadata,
            image_url=image_url,
            object_id=object_id,
        )

    def add_from_embedding(
        self,
        *,
        category: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        object_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Add object using a precomputed embedding (без изображения).

        Используется для offline-сценариев (миграция баз, усреднение по нескольким фото),
        когда эмбеддинг уже рассчитан и не нужно гонять изображение через extractor.
        """
        manager = self._get_manager(category)
        return manager.add_from_embedding(
            embedding=embedding,
            name=name,
            metadata=metadata,
            image_url=image_url,
            object_id=object_id,
        )

    def get(self, object_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get object by ID"""
        return self.db_store.get(object_id)

    def delete(self, object_id: uuid.UUID) -> bool:
        """Delete object"""
        obj = self.db_store.get(object_id)
        if not obj:
            raise EmbeddingNotFoundError(str(object_id))

        category = obj["category"]
        manager = self._get_manager(category)
        return manager.delete(object_id)

    def search(
        self,
        *,
        category: str,
        embedding: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar objects"""
        manager = self._get_manager(category)
        return manager.search(
            embedding=embedding,
            image=image,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def update(
        self,
        *,
        object_id: uuid.UUID,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> bool:
        """Update object"""
        obj = self.db_store.get(object_id)
        if not obj:
            raise EmbeddingNotFoundError(str(object_id))

        category = obj["category"]
        manager = self._get_manager(category)
        return manager.update(
            object_id=object_id,
            name=name,
            metadata=metadata,
            image_url=image_url,
            image=image,
        )

    def list_categories(self) -> List[str]:
        """List all categories"""
        return self.db_store.list_categories()

    def count_by_category(self, category: Optional[str] = None) -> Dict[str, int]:
        """Count embeddings by category"""
        return self.db_store.count_by_category(category)

    def get_all_embeddings(
        self,
        category: str,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a category.
        
        This is useful for local similarity computation when you have frame embeddings
        and want to compare them with all franchise embeddings without making HTTP requests.
        
        Args:
            category: Category name (e.g., "franchise")
            embedding_model: Optional model name filter (e.g., "clip_224")
            
        Returns:
            List of embeddings with keys: id, category, name, embedding_model, embedding_dim,
            embedding (as numpy array), metadata, image_url, added_at
        """
        results = self.db_store.get_all_embeddings(category, embedding_model)
        
        # Convert embedding lists to numpy arrays
        for result in results:
            if isinstance(result.get("embedding"), list):
                result["embedding"] = np.array(result["embedding"], dtype=np.float32)
        
        return results

    def batch_add(
        self,
        *,
        category: str,
        images: List[np.ndarray],
        names: Optional[List[Optional[str]]] = None,
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        image_urls: Optional[List[Optional[str]]] = None,
    ) -> List[uuid.UUID]:
        """Batch add objects"""
        manager = self._get_manager(category)
        ids = []

        if names is None:
            names = [None] * len(images)
        if metadata_list is None:
            metadata_list = [None] * len(images)
        if image_urls is None:
            image_urls = [None] * len(images)

        for image, name, metadata, image_url in zip(images, names, metadata_list, image_urls):
            obj_id = manager.add(
                image=image,
                name=name,
                metadata=metadata,
                image_url=image_url,
            )
            ids.append(obj_id)

        return ids

    def save_faiss_indices(self) -> None:
        """Save all FAISS indices to disk"""
        self.faiss_manager.save_all()

    def close(self) -> None:
        """Close connections and save indices"""
        self.save_faiss_indices()
        self.db_store.close()

