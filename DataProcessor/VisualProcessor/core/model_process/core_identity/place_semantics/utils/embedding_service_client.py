"""
HTTP client for Embedding Service integration.

This module provides a client for interacting with the Embedding Service API,
including search, embedding extraction, and object addition operations.

The client supports:
- Automatic retry with exponential backoff
- Health check before operations
- Image format conversion (BGR/RGB to JPEG)
- Error handling and validation

Example:
    ```python
    from embedding_service_client import EmbeddingServiceClient

    client = EmbeddingServiceClient(base_url="http://localhost:8001")
    
    # Search for similar brands
    results = client.search(
        category="brand",
        image=crop_image,
        top_k=5,
        similarity_threshold=0.7
    )
    
    for result in results:
        print(f"Brand: {result['name']}, Similarity: {result['similarity']}")
    ```
"""
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from utils.embedding_service_errors import (
    EmbeddingServiceUnavailableError,
    brief_request_exception_message,
)


class EmbeddingServiceClient:
    """
    Client for interacting with Embedding Service API.

    This client provides methods to:
    - Search for similar objects in the embedding database
    - Extract embeddings from images
    - Add new objects to the database
    - Batch operations (with fallback to individual requests)

    The client automatically handles:
    - Image format conversion (BGR/RGB to JPEG)
    - Retry logic with exponential backoff
    - Health checks
    - Error handling

    Attributes:
        base_url (str): Base URL of the Embedding Service
        timeout (float): Request timeout in seconds

    Example:
        ```python
        client = EmbeddingServiceClient(base_url="http://localhost:8001")
        results = client.search(category="brand", image=img, top_k=5)
        ```
    """

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
        """
        Ensure Embedding Service is available by checking health endpoint.

        Raises:
            EmbeddingServiceUnavailableError: If the service cannot be reached or /health is not OK.
        """
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5.0)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            tip = brief_request_exception_message(e)
            raise EmbeddingServiceUnavailableError(self.base_url, tip) from None

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
        Extract embedding vector from image.

        This method sends an image to the Embedding Service and returns
        the extracted embedding vector along with metadata.

        Args:
            category: Category of the object (e.g., "brand", "car", "face")
            image: Image array in BGR or RGB format (numpy array)

        Returns:
            Tuple containing:
            - embedding (np.ndarray): Embedding vector (1D array, typically 512 dim)
            - metadata (Dict[str, Any]): Metadata about the embedding (model, dimension, etc.)

        Raises:
            RuntimeError: If Embedding Service is unavailable or request fails

        Example:
            ```python
            embedding, metadata = client.embed(category="brand", image=crop)
            print(f"Embedding dimension: {len(embedding)}")
            print(f"Model used: {metadata.get('model_name')}")
            ```
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
        Add a new object to the Embedding Service database.

        This method adds an object (brand, car, face, etc.) to the database
        along with its embedding and metadata. The object can then be searched
        using the search() method.

        Args:
            category: Category of the object (e.g., "brand", "car", "face")
            name: Name/label of the object (e.g., "Coca-Cola", "Toyota Camry")
            image: Image array in BGR or RGB format (numpy array)
            metadata: Optional metadata dictionary. Can include:
                - aliases_en: List of English aliases
                - aliases_ru: List of Russian aliases
                - prompts_en: List of English prompts
                - category: Subcategory or classification
                - make, model, segment: For cars
                - etc.

        Returns:
            Dictionary containing:
            - id (str): Unique identifier of the added object
            - status (str): Status of the operation
            - message (str): Optional message

        Raises:
            RuntimeError: If Embedding Service is unavailable or request fails

        Example:
            ```python
            result = client.add_object(
                category="brand",
                name="Nike",
                image=logo_image,
                metadata={
                    "aliases_en": ["NIKE", "nike"],
                    "category": "clothing"
                }
            )
            print(f"Added object with ID: {result['id']}")
            ```
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

    def get_all_embeddings(
        self,
        category: str,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a category from Embedding Service.
        
        This is useful for local similarity computation when you have frame embeddings
        and want to compare them with all category embeddings without making HTTP requests.
        
        Args:
            category: Category name (e.g., "place")
            embedding_model: Optional model name filter (e.g., "clip_224")
            
        Returns:
            List of embeddings with keys: id, category, name, embedding_model, embedding_dim,
            embedding (as numpy array), metadata, image_url, added_at
        """
        self._ensure_url()
        
        url = f"{self.base_url}/categories/{category}/embeddings"
        params = {}
        if embedding_model:
            params["embedding_model"] = embedding_model
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            # Convert embedding lists to numpy arrays
            for emb in embeddings:
                if isinstance(emb.get("embedding"), list):
                    emb["embedding"] = np.array(emb["embedding"], dtype=np.float32)
            
            return embeddings
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding Service get_all_embeddings failed: {e}")

    def get_labels(
        self,
        category: str,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get label-space metadata for a category (WITHOUT embeddings).

        This method retrieves the canonical list of labels (objects) for a category,
        which is used to build a stable label space and compute db_digest for reproducibility.

        Requires Embedding Service endpoint:
          GET /categories/{category}/labels

        Args:
            category: Category name (e.g., "place", "brand", "car")
            embedding_model: Optional embedding model filter

        Returns:
            List of labels with keys: id, name, embedding_model, embedding_dim, updated_at, etc.
        """
        self._ensure_url()
        url = f"{self.base_url}/categories/{category}/labels"
        params: Dict[str, str] = {}
        if embedding_model:
            params["embedding_model"] = str(embedding_model)

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json() or {}
            labels = data.get("labels", [])
            if not isinstance(labels, list):
                return []
            return labels
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Embedding Service get_labels failed: {e}") from e

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

