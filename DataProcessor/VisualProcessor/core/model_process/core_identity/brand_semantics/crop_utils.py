"""
Utilities for cropping and preprocessing images with padding.

This module provides functions for:
- Cropping images from bounding boxes with configurable padding
- Creating tight and loose crop variants
- Selecting the best crop from a track based on quality metrics

The padding helps preserve context around objects, which is important
for recognition tasks (brands, cars, etc.) as it may include important
visual features like emblems, text, or surroundings.

Example:
    ```python
    from crop_utils import crop_with_padding, select_best_crop_for_track

    # Crop with 15% padding
    crop = crop_with_padding(image, bbox, pad_ratio=0.15)

    # Select best crop from track
    best_idx, best_crop = select_best_crop_for_track(
        crops, scores, areas, use_sharpness=True
    )
    ```
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.15,
    min_size: int = 32,
) -> np.ndarray:
    """
    Crop image from bounding box with padding.

    This function extracts a region from an image based on a bounding box,
    with additional padding on all sides. The padding helps preserve context
    that may be important for recognition (e.g., logo surroundings, car details).

    The function ensures:
    - Padding stays within image bounds
    - Minimum crop size is maintained
    - Coordinates are properly clipped

    Args:
        image: Input image as numpy array, shape (H, W, C) for color or (H, W) for grayscale
        bbox: Bounding box coordinates as (x1, y1, x2, y2) in image pixel coordinates
        pad_ratio: Padding ratio applied to bounding box dimensions (default: 0.15 = 15%)
            - A value of 0.15 means 15% padding on each side (total 30% added width/height)
        min_size: Minimum size (width or height) of the cropped image in pixels (default: 32)

    Returns:
        Cropped image as numpy array with same number of channels as input

    Example:
        ```python
        bbox = (100, 200, 150, 250)  # x1, y1, x2, y2
        crop = crop_with_padding(image, bbox, pad_ratio=0.20)
        # Crop will include ~20% padding around the original bbox
        ```
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * pad_ratio
    pad_h = box_h * pad_ratio

    # Apply padding
    x1n = max(0, int(x1 - pad_w))
    y1n = max(0, int(y1 - pad_h))
    x2n = min(w, int(x2 + pad_w))
    y2n = min(h, int(y2 + pad_h))

    # Ensure minimum size
    if x2n - x1n < min_size:
        center_x = (x1n + x2n) // 2
        x1n = max(0, center_x - min_size // 2)
        x2n = min(w, x1n + min_size)

    if y2n - y1n < min_size:
        center_y = (y1n + y2n) // 2
        y1n = max(0, center_y - min_size // 2)
        y2n = min(h, y1n + min_size)

    # Crop
    crop = image[y1n:y2n, x1n:x2n]
    return crop


def create_tight_and_loose_crops(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    tight_pad: float = 0.05,
    loose_pad: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create both tight and loose crop variants of an object.

    This function creates two crop variants:
    - Tight crop: Minimal padding, focuses on the object itself
    - Loose crop: More padding, includes surrounding context

    This is useful for recognition tasks where you might want to try both
    approaches (e.g., for logos, tight crop focuses on the logo itself,
    loose crop includes surrounding text or context).

    Args:
        image: Input image as numpy array
        bbox: Bounding box coordinates as (x1, y1, x2, y2)
        tight_pad: Padding ratio for tight crop (default: 0.05 = 5%)
        loose_pad: Padding ratio for loose crop (default: 0.20 = 20%)

    Returns:
        Tuple of (tight_crop, loose_crop) as numpy arrays

    Example:
        ```python
        tight, loose = create_tight_and_loose_crops(image, bbox)
        # Use both crops for recognition and select best result
        ```
    """
    tight_crop = crop_with_padding(image, bbox, pad_ratio=tight_pad)
    loose_crop = crop_with_padding(image, bbox, pad_ratio=loose_pad)
    return tight_crop, loose_crop


def select_best_crop_for_track(
    crops: List[np.ndarray],
    scores: List[float],
    areas: List[float],
    use_sharpness: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    Select best crop from a track based on score, area, and optionally sharpness.

    Optimized: sharpness is computed once for all crops to avoid redundant calculations.

    Args:
        crops: List of crops from track
        scores: List of detection scores
        areas: List of crop areas
        use_sharpness: Whether to use sharpness as additional metric

    Returns:
        Tuple of (best_index, best_crop)
    """
    if not crops:
        raise ValueError("Empty crops list")

    if len(crops) == 1:
        return 0, crops[0]

    # Validate input lengths
    if len(crops) != len(scores) or len(crops) != len(areas):
        raise ValueError(
            f"Mismatched input lengths: crops={len(crops)}, scores={len(scores)}, areas={len(areas)}"
        )

    # Pre-compute sharpness values if needed (optimization)
    sharpness_values = None
    if use_sharpness:
        sharpness_values = []
        for crop in crops:
            gray = (
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if len(crop.shape) == 3
                else crop
            )
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_values.append(laplacian_var)
        max_sharpness = max(sharpness_values) + 1e-9

    # Calculate combined score
    max_area = max(areas) + 1e-9
    combined_scores = []

    for i, (score, area) in enumerate(zip(scores, areas)):
        # Normalize area (use log to reduce impact of very large areas)
        normalized_area = np.log1p(area) / np.log1p(max_area)
        combined = score * normalized_area

        if use_sharpness and sharpness_values is not None:
            # Use pre-computed sharpness
            normalized_sharpness = sharpness_values[i] / max_sharpness
            combined *= (1 + normalized_sharpness) / 2

        combined_scores.append(combined)

    # Select best
    best_idx = int(np.argmax(combined_scores))
    return best_idx, crops[best_idx]

