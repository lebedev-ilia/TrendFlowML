"""FAISS index manager for fast similarity search"""

import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..errors import EmbeddingServiceError


class FaissIndexManager:
    """Manages FAISS indices per embedding_model"""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self._indices: Dict[str, faiss.Index] = {}
        self._id_mappings: Dict[str, List[str]] = {}  # model_name -> list of UUIDs

    def _get_index_path(self, model_name: str) -> str:
        """Get path for model-specific index"""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.index_dir, f"{safe_name}.faiss")

    def _get_ids_path(self, model_name: str) -> str:
        """Get path for model-specific ID mapping"""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.index_dir, f"{safe_name}_ids.npy")

    def _load_index(self, model_name: str, embedding_dim: int) -> faiss.Index:
        """Load or create index for a model"""
        if model_name in self._indices:
            return self._indices[model_name]

        index_path = self._get_index_path(model_name)
        ids_path = self._get_ids_path(model_name)

        if os.path.exists(index_path) and os.path.exists(ids_path):
            # Load existing index
            index = faiss.read_index(index_path)
            ids = np.load(ids_path, allow_pickle=True).tolist()
            self._indices[model_name] = index
            self._id_mappings[model_name] = ids
        else:
            # Create new index (IndexFlatIP for cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)
            self._indices[model_name] = index
            self._id_mappings[model_name] = []

        return index

    def add(
        self,
        *,
        model_name: str,
        embedding_dim: int,
        object_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Add embedding to FAISS index"""
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)
        if embedding.shape[0] != embedding_dim:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {embedding_dim}, got {embedding.shape[0]}",
                error_code="embedding_dim_mismatch",
            )

        # L2 normalize for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        embedding = embedding.reshape(1, -1)

        index = self._load_index(model_name, embedding_dim)
        index.add(embedding)
        self._id_mappings[model_name].append(object_id)

    def remove(self, *, model_name: str, embedding_dim: int, object_id: str) -> bool:
        """Remove embedding from FAISS index (by finding its position)"""
        if model_name not in self._id_mappings:
            return False

        ids = self._id_mappings[model_name]
        if object_id not in ids:
            return False

        idx = ids.index(object_id)
        index = self._indices[model_name]

        # FAISS doesn't have direct remove, so we need to rebuild
        # For now, mark as removed (we'll rebuild on save)
        # TODO: Implement proper removal with index reconstruction
        return False  # Placeholder

    def search(
        self,
        *,
        model_name: str,
        embedding_dim: int,
        embedding: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.

        Returns:
            similarities: np.ndarray of similarity scores
            ids: np.ndarray of object IDs (as strings in numpy array)
        """
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)
        if embedding.shape[0] != embedding_dim:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {embedding_dim}, got {embedding.shape[0]}",
                error_code="embedding_dim_mismatch",
            )

        # L2 normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        embedding = embedding.reshape(1, -1)

        index = self._load_index(model_name, embedding_dim)
        if index.ntotal == 0:
            return np.array([]), np.array([])

        similarities, indices = index.search(embedding, min(top_k, index.ntotal))

        # Map indices to object IDs
        ids = self._id_mappings[model_name]
        result_ids = [ids[i] for i in indices[0]]

        return similarities[0], np.array(result_ids, dtype=object)

    def save(self, model_name: str) -> None:
        """Save index to disk"""
        if model_name not in self._indices:
            return

        index = self._indices[model_name]
        ids = self._id_mappings[model_name]

        index_path = self._get_index_path(model_name)
        ids_path = self._get_ids_path(model_name)

        faiss.write_index(index, index_path)
        np.save(ids_path, np.array(ids, dtype=object))

    def save_all(self) -> None:
        """Save all indices to disk"""
        for model_name in self._indices.keys():
            self.save(model_name)

    def get_stats(self, model_name: str) -> Dict[str, int]:
        """Get statistics for a model's index"""
        if model_name not in self._indices:
            return {"count": 0}

        index = self._indices[model_name]
        return {
            "count": index.ntotal,
            "dimension": index.d,
        }

