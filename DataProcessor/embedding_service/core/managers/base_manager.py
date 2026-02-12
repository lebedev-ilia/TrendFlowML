"""Base manager for embedding operations"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..database.faiss_index import FaissIndexManager
from ..database.postgres import PostgresEmbeddingStore
from ..embedding_extractors.base import EmbeddingExtractor
from ..embedding_extractors.factory import EmbeddingExtractorFactory
from ..errors import EmbeddingExtractionError, EmbeddingNotFoundError, EmbeddingServiceError
from embedding_service.config.settings import EmbeddingServiceConfig


class BaseManager(ABC):
    """Base manager for category-specific operations"""

    def __init__(
        self,
        category: str,
        config: EmbeddingServiceConfig,
        db_store: PostgresEmbeddingStore,
        faiss_manager: FaissIndexManager,
    ):
        self.category = category
        self.config = config
        self.db_store = db_store
        self.faiss_manager = faiss_manager

        # Get model name for this category
        self.model_name = config.category_model_mapping.get(category)
        if not self.model_name:
            raise EmbeddingServiceError(
                f"No model mapping for category: {category}",
                error_code="invalid_category",
            )

        # Create extractor
        self.extractor = EmbeddingExtractorFactory.create(self.model_name, config)
        self.embedding_dim = self.extractor.get_embedding_dim()

    def validate(self, image: np.ndarray) -> bool:
        """
        Validate input image.

        Override in subclasses for category-specific validation.
        """
        if image is None or image.size == 0:
            return False
        if image.ndim < 2 or image.ndim > 3:
            return False
        return True

    def extract_embedding(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Extract embedding from image"""
        if not self.validate(image):
            raise EmbeddingServiceError("Invalid image", error_code="invalid_image")

        try:
            embedding = self.extractor.extract(image, **kwargs)
            return embedding
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to extract embedding: {e}") from e

    def normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize metadata.

        Override in subclasses for category-specific normalization.
        """
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            return {"raw": str(metadata)}
        return metadata

    def add(
        self,
        *,
        image: np.ndarray,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        object_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Add object to embedding service.

        Args:
            image: Input image
            name: Optional name/label
            metadata: Optional metadata dict
            image_url: Optional URL to original image
            object_id: Optional UUID (generated if not provided)

        Returns:
            UUID of created object
        """
        # Extract embedding
        embedding = self.extract_embedding(image)
        embedding_list = embedding.tolist()

        # Normalize metadata
        normalized_metadata = self.normalize_metadata(metadata)

        # Save to PostgreSQL
        db_id = self.db_store.add(
            object_id=object_id,
            category=self.category,
            name=name,
            embedding_model=self.model_name,
            embedding_dim=self.embedding_dim,
            embedding=embedding_list,
            metadata=normalized_metadata,
            image_url=image_url,
        )

        # Add to FAISS
        self.faiss_manager.add(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            object_id=str(db_id),
            embedding=embedding,
        )

        return db_id

    def add_from_embedding(
        self,
        *,
        embedding: np.ndarray,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        object_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """
        Add object using a precomputed embedding (без изображения).

        Используется для offline-процессов, где эмбеддинг уже рассчитан (например,
        усреднение по нескольким фото человека) и нужно напрямую записать его
        в PostgreSQL + FAISS.

        Args:
            embedding: Готовый эмбеддинг (1D numpy array длиной embedding_dim)
            name: Имя/label объекта
            metadata: Дополнительные метаданные (будут нормализованы)
            image_url: Опциональный URL исходного изображения (если есть)
            object_id: Опциональный UUID (если не указан — будет сгенерирован)

        Returns:
            UUID созданного объекта.
        """
        if embedding is None or embedding.size == 0:
            raise EmbeddingServiceError("Empty embedding", error_code="invalid_embedding")

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.embedding_dim:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb.shape[0]}",
                error_code="invalid_embedding_dim",
            )

        embedding_list = emb.tolist()
        normalized_metadata = self.normalize_metadata(metadata)

        # Save to PostgreSQL
        db_id = self.db_store.add(
            object_id=object_id,
            category=self.category,
            name=name,
            embedding_model=self.model_name,
            embedding_dim=self.embedding_dim,
            embedding=embedding_list,
            metadata=normalized_metadata,
            image_url=image_url,
        )

        # Add to FAISS
        self.faiss_manager.add(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            object_id=str(db_id),
            embedding=emb,
        )

        return db_id

    def get(self, object_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get object by ID"""
        return self.db_store.get(object_id)

    def delete(self, object_id: uuid.UUID) -> bool:
        """Delete object"""
        # Delete from PostgreSQL
        deleted = self.db_store.delete(object_id)

        if deleted:
            # Remove from FAISS (note: FAISS doesn't support direct removal efficiently)
            # We'll rely on periodic index rebuild or ignore FAISS inconsistency
            # For production, implement proper FAISS removal or use incremental index
            pass

        return deleted

    def search(
        self,
        *,
        embedding: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar objects.

        Args:
            embedding: Query embedding (optional if image provided)
            image: Query image (optional if embedding provided)
            top_k: Number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar objects with metadata
        """
        # Extract embedding if image provided
        if embedding is None:
            if image is None:
                raise EmbeddingServiceError("Either embedding or image must be provided", error_code="invalid_query")
            embedding = self.extract_embedding(image)

        # Use FAISS for fast search
        similarities, ids = self.faiss_manager.search(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            embedding=embedding,
            top_k=top_k,
        )

        # Filter by similarity threshold
        mask = similarities >= similarity_threshold
        filtered_ids = ids[mask]
        filtered_similarities = similarities[mask]

        # Get full metadata from PostgreSQL
        results = []
        for obj_id, similarity in zip(filtered_ids, filtered_similarities):
            obj = self.db_store.get(uuid.UUID(str(obj_id)))
            if obj:
                obj["similarity"] = float(similarity)
                results.append(obj)

        # Sort by similarity (descending)
        results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return results

    def update(
        self,
        *,
        object_id: uuid.UUID,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update object metadata or re-embed with new image.

        Args:
            object_id: Object ID
            name: New name (optional)
            metadata: New metadata (optional)
            image_url: New image URL (optional)
            image: New image (optional, will re-embed if provided)

        Returns:
            True if updated successfully
        """
        # Get existing object
        existing = self.db_store.get(object_id)
        if not existing:
            raise EmbeddingNotFoundError(str(object_id))

        # Re-extract embedding if image provided
        embedding = None
        if image is not None:
            embedding = self.extract_embedding(image)
            embedding_list = embedding.tolist()
        else:
            # Use existing embedding
            embedding_list = existing.get("embedding")

        # Merge metadata
        merged_metadata = existing.get("metadata", {})
        if metadata:
            merged_metadata.update(self.normalize_metadata(metadata))

        # Update name
        new_name = name if name is not None else existing.get("name")

        # Update in PostgreSQL
        self.db_store.add(
            object_id=object_id,
            category=self.category,
            name=new_name,
            embedding_model=self.model_name,
            embedding_dim=self.embedding_dim,
            embedding=embedding_list,
            metadata=merged_metadata,
            image_url=image_url if image_url is not None else existing.get("image_url"),
        )

        # Update FAISS if embedding changed
        if embedding is not None:
            # Remove old (if FAISS supported removal)
            # Add new
            self.faiss_manager.add(
                model_name=self.model_name,
                embedding_dim=self.embedding_dim,
                object_id=str(object_id),
                embedding=embedding,
            )

        return True


