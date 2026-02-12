"""HTTP client for Embedding Service integration"""
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image


class EmbeddingServiceClient:
    """Client for interacting with Embedding Service API"""

    def __init__(
        self,
        base_url: str = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Embedding Service client.

        Args:
            base_url: Base URL of Embedding Service (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
        """
        if base_url is None:
            # Try to get from environment
            base_url = os.environ.get(
                "EMBEDDING_SERVICE_URL", "http://localhost:8001"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _ensure_url(self):
        """Ensure service is available"""
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Embedding Service unavailable at {self.base_url}: {e}"
            )

    def search(
        self,
        category: str,
        image: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar objects in Embedding Service with retry.

        Args:
            category: Category to search (e.g., "brand", "car")
            image: Image array (BGR or RGB, numpy array)
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            List of search results with keys: id, name, similarity, metadata
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG", quality=85)  # Reduced quality for better performance
        img_bytes.seek(0)

        # Prepare request
        url = f"{self.base_url}/search"
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        data = {
            "category": category,
            "top_k": str(top_k),
            "similarity_threshold": str(similarity_threshold),
        }

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                self._ensure_url()
                response = requests.post(
                    url, files=files, data=data, timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                results = result.get("results", [])
                if not results:
                    # Empty results - return empty list (not an error)
                    return []
                return results
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise RuntimeError(
                    f"Embedding Service search failed after {max_retries} attempts: {e}"
                ) from last_exception

        raise RuntimeError(
            f"Embedding Service search failed after {max_retries} attempts: {last_exception}"
        )

    def embed(
        self, category: str, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embedding from image.

        Args:
            category: Category (e.g., "brand", "car")
            image: Image array (BGR or RGB, numpy array)

        Returns:
            Tuple of (embedding array, metadata dict)
        """
        self._ensure_url()

        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)

        # Prepare request
        url = f"{self.base_url}/embed"
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"category": category}

        try:
            response = requests.post(
                url, files=files, data=data, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            embedding = np.array(result["embedding"], dtype=np.float32)
            metadata = result.get("metadata", {})
            return embedding, metadata
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding Service embed failed: {e}")

    def add_object(
        self,
        category: str,
        name: str,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add object to Embedding Service.

        Args:
            category: Category (e.g., "brand", "car")
            name: Object name
            image: Image array (BGR or RGB, numpy array)
            metadata: Optional metadata dict

        Returns:
            Response dict with object ID
        """
        self._ensure_url()

        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)

        # Prepare request
        url = f"{self.base_url}/objects/add"
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"category": category, "name": name}
        if metadata:
            import json

            data["metadata"] = json.dumps(metadata)

        try:
            response = requests.post(
                url, files=files, data=data, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding Service add_object failed: {e}")

    def search_batch(
        self,
        category: str,
        images: List[np.ndarray],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch search for similar objects in Embedding Service.

        NOTE: This method falls back to individual requests if batch API is not available.
        For better performance, implement batch API in Embedding Service.

        Args:
            category: Category to search (e.g., "brand", "car")
            images: List of image arrays (BGR or RGB, numpy arrays)
            top_k: Number of top results to return per image
            similarity_threshold: Minimum similarity threshold
            max_retries: Maximum number of retry attempts per request
            retry_delay: Delay between retries in seconds

        Returns:
            List of search results per image: [[result1, ...], [result2, ...], ...]
        """
        # Fallback to individual requests (TODO: implement batch API in Embedding Service)
        results = []
        for image in images:
            try:
                result = self.search(
                    category=category,
                    image=image,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                results.append(result)
            except Exception as e:
                # On error, return empty results for this image
                results.append([])
        return results

