"""
Группировка плоских фич TextProcessor (prefix tp_*) по имени экстрактора.

Используется в render_text_processor и в batch_runs_feature_report (строка на экстрактор).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np

# Совпадает с логикой в renderer.py (до вынесения)
_EXTRACTOR_PREFIX_MAP: Dict[str, str] = {
    "lex": "lexico_static_features",
    "tags": "tags_extractor",
    "asr": "asr_text_proxy_audio_features",
    "asrproxy": "asr_text_proxy_audio_features",
    "title": "title_embedder",
    "titleemb": "title_embedder",
    "desc": "description_embedder",
    "descemb": "description_embedder",
    "hashtag": "hashtag_embedder",
    "hashemb": "hashtag_embedder",
    "transcript": "transcript_chunk_embedder",
    "tchunk": "transcript_chunk_embedder",
    "comments": "comments_embedder",
    "commentsemb": "comments_embedder",
    "tragg": "transcript_aggregator",
    "commentsagg": "comments_aggregator",
    "cos": "cosine_metrics_extractor",
    "cosine": "cosine_metrics_extractor",
    "embpair": "embedding_pair_topk_extractor",
    "pairtopk": "embedding_pair_topk_extractor",
    "embstats": "embedding_stats_extractor",
    "embshift": "embedding_shift_indicator_extractor",
    "embid": "embedding_source_id_extractor",
    "spkemb": "speaker_turn_embeddings_aggregator",
    "embedding": "embedding_stats_extractor",
    "qa": "qa_embedding_pairs_extractor",
    "topics": "semantics_topics_keyphrases",
    "semantic": "semantics_topics_keyphrases",
    "semclust": "semantic_cluster_extractor",
    "titlehashcos": "title_to_hashtag_cosine_extractor",
    "topktitles": "topk_similar_titles_extractor",
    "titleclent": "title_embedding_cluster_entropy_extractor",
}


def build_feature_dict_from_npz_arrays(
    feature_names: Union[None, list, np.ndarray],
    feature_values: Union[None, list, np.ndarray],
) -> Dict[str, Any]:
    """Собрать словарь name -> value из NPZ; NaN/Inf -> None (как в render_text_processor)."""
    if feature_names is None:
        return {}
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if feature_values is None:
        feature_values = []
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    features: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            value = feature_values[i]
            if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
                features[str(name)] = None
            else:
                features[str(name)] = value
    return features


def build_feature_dict_from_npz_data(npz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Удобная обёртка для словаря после load_npz."""
    return build_feature_dict_from_npz_arrays(
        npz_data.get("feature_names"),
        npz_data.get("feature_values"),
    )


def group_text_features_by_extractor(features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Раскладка flat-фич по «экстракторам» (каталоги в src/extractors/…).

    Legacy: tp_title_hashtag_cosine_* -> title_to_hashtag_cosine_extractor
    (дубли с основным циклом по tp_* — поведение как в renderer, без изменения семантики.)
    """
    extractor_features: Dict[str, Dict[str, Any]] = {}

    for name, value in features.items():
        if str(name).startswith("tp_title_hashtag_cosine"):
            en = "title_to_hashtag_cosine_extractor"
            extractor_features.setdefault(en, {})[str(name)] = value

    for name, value in features.items():
        sname = str(name)
        if sname.startswith("tp_"):
            parts = sname.split("_", 2)
            if len(parts) >= 3:
                hint = parts[1]
                extractor_name = _EXTRACTOR_PREFIX_MAP.get(hint, hint)
                extractor_features.setdefault(extractor_name, {})[sname] = value
            else:
                extractor_features.setdefault("unclassified", {})[sname] = value
        else:
            extractor_features.setdefault("unclassified", {})[sname] = value
    return extractor_features


__all__ = [
    "build_feature_dict_from_npz_arrays",
    "build_feature_dict_from_npz_data",
    "group_text_features_by_extractor",
]
