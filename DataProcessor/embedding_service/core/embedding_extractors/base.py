"""Base embedding extractor interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class EmbeddingExtractor(ABC):
    """Base class for embedding extractors"""

    @abstractmethod
    def extract(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Extract embedding from image.

        Args:
            image: Input image as numpy array (BGR or RGB)
            **kwargs: Additional arguments

        Returns:
            Embedding vector as 1D numpy array
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass

