"""
Brand semantics component.

This module provides brand recognition functionality using Embedding Service.
It processes video frames to detect and recognize brand logos and text regions.

Main components:
- main: Main processing pipeline
- embedding_service_client: HTTP client for Embedding Service API
- crop_utils: Image cropping and preprocessing utilities

Usage:
    ```python
    from brand_semantics.main import main
    exit_code = main()
    ```
"""

__version__ = "0.1.0"
__all__ = ["main", "EmbeddingServiceClient", "crop_with_padding", "select_best_crop_for_track"]

