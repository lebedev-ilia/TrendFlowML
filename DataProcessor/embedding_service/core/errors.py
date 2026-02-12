"""Errors for Embedding Service"""


class EmbeddingServiceError(RuntimeError):
    """Base error for Embedding Service"""

    def __init__(self, message: str, *, error_code: str = "embedding_service_error") -> None:
        super().__init__(message)
        self.error_code = error_code


class EmbeddingNotFoundError(EmbeddingServiceError):
    """Embedding not found"""

    def __init__(self, object_id: str) -> None:
        super().__init__(f"Embedding not found: {object_id}", error_code="embedding_not_found")
        self.object_id = object_id


class InvalidCategoryError(EmbeddingServiceError):
    """Invalid category"""

    def __init__(self, category: str) -> None:
        super().__init__(f"Invalid category: {category}", error_code="invalid_category")
        self.category = category


class InvalidModelError(EmbeddingServiceError):
    """Invalid embedding model"""

    def __init__(self, model: str) -> None:
        super().__init__(f"Invalid embedding model: {model}", error_code="invalid_model")
        self.model = model


class EmbeddingExtractionError(EmbeddingServiceError):
    """Failed to extract embedding"""

    def __init__(self, message: str) -> None:
        super().__init__(f"Failed to extract embedding: {message}", error_code="extraction_failed")

