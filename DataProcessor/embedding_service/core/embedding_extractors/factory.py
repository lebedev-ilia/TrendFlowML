"""Factory for creating embedding extractors"""

from typing import Dict, Optional

from embedding_service.config.settings import EmbeddingServiceConfig
from ..errors import InvalidModelError
from .arcface_extractor import ArcFaceExtractor
from .base import EmbeddingExtractor
from .clip_extractor import CLIPExtractor


class EmbeddingExtractorFactory:
    """Factory for creating embedding extractors"""

    _cache: Dict[str, EmbeddingExtractor] = {}

    @classmethod
    def create(cls, model_name: str, config: EmbeddingServiceConfig) -> EmbeddingExtractor:
        """
        Create embedding extractor for a model.

        Args:
            model_name: Model name (e.g., "arcface", "clip_224", "clip_336", "clip_448")
            config: Service configuration

        Returns:
            EmbeddingExtractor instance
        """
        # Check cache
        cache_key = f"{model_name}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Create new extractor
        extractor: EmbeddingExtractor

        if model_name == "arcface":
            extractor = ArcFaceExtractor()
        elif model_name.startswith("clip_"):
            extractor = CLIPExtractor(model_name, config)
        else:
            raise InvalidModelError(f"Unknown model: {model_name}")

        # Cache it
        cls._cache[cache_key] = extractor

        return extractor

    @classmethod
    def clear_cache(cls) -> None:
        """Clear extractor cache"""
        cls._cache.clear()


def get_extractor_for_category(category: str, config: EmbeddingServiceConfig) -> EmbeddingExtractor:
    """Get extractor for a category based on model mapping"""
    model_name = config.category_model_mapping.get(category)
    if not model_name:
        raise InvalidModelError(f"No model mapping for category: {category}")

    return EmbeddingExtractorFactory.create(model_name, config)

