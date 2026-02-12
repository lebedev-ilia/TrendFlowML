"""CLIP embedding extractor with Triton integration"""

from typing import Any

import cv2
import numpy as np
from PIL import Image

from embedding_service.config.settings import EmbeddingServiceConfig
from ..errors import EmbeddingExtractionError, EmbeddingServiceError, InvalidModelError
from .base import EmbeddingExtractor


class TritonClipClient:
    """Triton client for CLIP models"""

    def __init__(self, model_name: str, config: EmbeddingServiceConfig):
        self.model_name = model_name
        self.config = config

        # Map internal model names to Triton model names and settings
        # Internal names: clip_224, clip_336, clip_448
        # Triton model names: clip_image_224, clip_image_336, clip_image_448
        if model_name == "clip_224":
            self.triton_model_name = "clip_image_224"
            self.input_size = 224
            self.embedding_dim = 512
        elif model_name == "clip_336":
            self.triton_model_name = "clip_image_336"
            self.input_size = 336
            self.embedding_dim = 512
        elif model_name == "clip_448":
            self.triton_model_name = "clip_image_448"
            self.input_size = 448
            self.embedding_dim = 512
        else:
            raise InvalidModelError(f"Unknown CLIP model: {model_name}")

        # Triton parameters (from clip_image_336_triton.yaml spec)
        self.triton_model_version = "1"
        self.triton_input_name = "INPUT__0"
        self.triton_output_name = "OUTPUT__0"
        self.triton_datatype = "UINT8"  # Triton ensemble handles preprocessing

        # Initialize Triton client
        import sys
        from pathlib import Path
        # Add DataProcessor to path (embedding_service -> DataProcessor)
        dp_root = Path(__file__).parent.parent.parent.parent.parent
        if str(dp_root) not in sys.path:
            sys.path.insert(0, str(dp_root))
        from dp_triton.http_client import TritonHttpClient

        self.triton = TritonHttpClient(
            base_url=self.config.triton_base_url,
            timeout_sec=self.config.triton_timeout_sec,
        )

        # Check if Triton is ready
        if not self.triton.ready():
            raise EmbeddingServiceError(
                f"Triton server not ready at {self.config.triton_base_url}",
                error_code="triton_unavailable",
            )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CLIP Triton ensemble.
        
        For UINT8 datatype, we only resize (Triton ensemble handles normalization).
        Returns: (1, H, W, 3) uint8 array in RGB format.
        """
        # Convert BGR to RGB if needed
        if image.ndim == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize to target size (using PIL-style resampling for consistency with core_clip)
        from PIL import Image
        img_pil = Image.fromarray(image_rgb)
        img_resized = img_pil.resize((self.input_size, self.input_size), resample=Image.BICUBIC)
        image_resized = np.asarray(img_resized, dtype=np.uint8)

        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
        image_batch = image_resized[np.newaxis, :]

        return image_batch

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding via Triton"""
        # Preprocess (returns UINT8 for Triton ensemble)
        preprocessed = self._preprocess_image(image)

        try:
            # Infer via Triton
            result = self.triton.infer(
                model_name=self.triton_model_name,
                model_version=self.triton_model_version,
                input_name=self.triton_input_name,
                input_tensor=preprocessed,
                output_name=self.triton_output_name,
                datatype=self.triton_datatype,
            )

            embedding = result.output

            # Ensure 1D array
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            # L2 normalize (Triton may already normalize, but we ensure it)
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            return embedding.astype(np.float32)
        except Exception as e:
            raise EmbeddingExtractionError(f"Triton inference failed: {e}") from e


class CLIPExtractor(EmbeddingExtractor):
    """CLIP embedding extractor supporting multiple resolutions"""

    def __init__(self, model_name: str, config: EmbeddingServiceConfig):
        """
        Initialize CLIP extractor.

        Args:
            model_name: One of "clip_224", "clip_336", "clip_448"
            config: Service configuration
        """
        self.client = TritonClipClient(model_name, config)
        self._model_name = model_name

    def extract(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Extract CLIP embedding"""
        return self.client.embed(image)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.client.embedding_dim

    def get_model_name(self) -> str:
        """Get model name"""
        return self._model_name

