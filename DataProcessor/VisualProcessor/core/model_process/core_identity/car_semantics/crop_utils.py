"""Utilities for cropping and preprocessing images with padding"""
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

    Args:
        image: Input image (H, W, C) or (H, W)
        bbox: Bounding box (x1, y1, x2, y2) in image coordinates
        pad_ratio: Padding ratio (e.g., 0.15 = 15% on each side)
        min_size: Minimum size of cropped image

    Returns:
        Cropped image
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
    Create both tight and loose crops of an object.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        tight_pad: Padding for tight crop (smaller context)
        loose_pad: Padding for loose crop (more context)

    Returns:
        Tuple of (tight_crop, loose_crop)
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

