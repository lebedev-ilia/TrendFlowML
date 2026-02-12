"""
Unified Embedding Service

Provides a single API for managing embeddings across different categories:
- face_semantic (ArcFace)
- brand_semantic (CLIP 336)
- car_semantic (CLIP 336)
- place_semantic (CLIP 448)
- generic_object (CLIP 224)
"""

from .config.settings import EmbeddingServiceConfig
from .core.embedding_manager import EmbeddingManager

__all__ = ["EmbeddingServiceConfig", "EmbeddingManager"]

