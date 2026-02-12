"""
Library-only similarity code (NOT used by baseline module).

Rationale:
- The baseline `SimilarityBaselineModuleV1` must be numpy-only and lightweight.
- The broader experimental/reference metrics rely on heavy deps (scipy/sklearn).

If you need these metrics, import:
  from modules.similarity_metrics.similarity_metrics_library import SimilarityMetrics
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

# Heavy deps: intentionally kept out of baseline module import graph.
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


class SimilarityMetrics:
    """
    Reference-based multi-aspect similarity between videos.
    (Library / future use; not invoked by VisualProcessor baseline pipeline.)
    """

    def __init__(self, top_n: int = 10, similarity_weights: Optional[Dict[str, float]] = None):
        self.top_n = int(top_n)
        self.similarity_weights = similarity_weights or {
            "semantic": 0.25,
            "topics": 0.15,
            "visual": 0.15,
            "text": 0.10,
            "audio": 0.15,
            "emotion": 0.10,
            "temporal": 0.10,
        }

    # NOTE: The full implementation historically lived in similarity_metrics.py.
    # We keep it here to avoid importing heavy deps in baseline.
    #
    # For now, the baseline module uses a different, deterministic implementation.

    def compute_high_level_scores(self, all_similarity_metrics: Dict[str, float], reference_videos_metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        weights = self.similarity_weights
        semantic_score = all_similarity_metrics.get("semantic_similarity_mean", 0.0)
        topics_score = all_similarity_metrics.get("topic_overlap_score", 0.0)
        visual_score = np.mean(
            [
                all_similarity_metrics.get("color_histogram_similarity", 0.0),
                all_similarity_metrics.get("lighting_pattern_similarity", 0.0),
                all_similarity_metrics.get("shot_type_distribution_similarity", 0.0),
            ]
        )
        text_score = all_similarity_metrics.get("ocr_text_semantic_similarity", 0.0)
        audio_score = all_similarity_metrics.get("audio_embedding_similarity", 0.0)
        emotion_score = all_similarity_metrics.get("emotion_curve_similarity", 0.0)
        temporal_score = all_similarity_metrics.get("pacing_curve_similarity", 0.0)

        overall_similarity = (
            weights["semantic"] * semantic_score
            + weights["topics"] * topics_score
            + weights["visual"] * float(visual_score)
            + weights["text"] * text_score
            + weights["audio"] * audio_score
            + weights["emotion"] * emotion_score
            + weights["temporal"] * temporal_score
        )
        overall_similarity = float(np.clip(overall_similarity, 0.0, 1.0))

        return {
            "overall_similarity_score": overall_similarity,
            "uniqueness_score": float(1.0 - overall_similarity),
            "trend_alignment_score": overall_similarity,
            "viral_pattern_score": overall_similarity,
        }


