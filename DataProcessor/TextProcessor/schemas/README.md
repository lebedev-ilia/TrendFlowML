# TextProcessor — machine schemas (NPZ)

Реестр JSON-схем для артефактов TextProcessor (`vp_schema_v1`), см. `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md`.

Валидатор (`VisualProcessor/utils/artifact_validator.py`) подхватывает этот каталог по `meta.schema_version`.

| schema_version | Артефакт | Описание |
|----------------|----------|----------|
| `text_npz_v1` | `text_processor/text_features.npz` | Агрегированный per-run NPZ из `run_cli.py` |
| `tags_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_tags_*` из `TagsExtractor` (`artifact_kind`: `extractor_features_flat`) |
| `lexico_static_features_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_lex_*` из `LexicalStatsExtractor` |
| `asr_text_proxy_audio_features_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_asrproxy_*` из `ASRTextProxyExtractor` |
| `title_embedder_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_titleemb_*` из `TitleEmbedder` (вектор — отдельный `.npy`) |
| `description_embedder_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_descemb_*` из `DescriptionEmbedder` (вектор — отдельный `.npy`) |
| `hashtag_embedder_output_v1` | вклад в `feature_names` / `feature_values` | Плоские скаляры `tp_hashemb_*` из `HashtagEmbedder` (вектор — отдельный `.npy`) |
| `transcript_chunk_embedder_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **16** ключей `tp_tchunk_*` из `TranscriptChunkEmbedder` (матрицы чанков — `transcript_*_chunk_embeddings.npy`) |
| `comments_embedder_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **18** ключей `tp_commentsemb_*` из `CommentsEmbedder` (матрица — `comments_embeddings.npy`) |
| `speaker_turn_embeddings_aggregator_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **17** ключей `tp_spkemb_*` из `SpeakerTurnEmbeddingsAggregatorExtractor` |
| `transcript_aggregator_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **19** ключей `tp_tragg_*` из `TranscriptAggregatorExtractor` |
| `comments_aggregator_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **39** ключей (`tp_commentsagg_*`, legacy `tp_comments_agg_*`, `tp_cagg_*`) из `CommentsAggregationExtractor` |
| `qa_embedding_pairs_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **34** ключа `tp_qa_*` из `QAEmbeddingPairsExtractor` |
| `embedding_pair_topk_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **69** ключей (`tp_embpair_*`, legacy `tp_pairtopk_*`) из `EmbeddingPairTopKExtractor` |
| `semantics_topics_keyphrases_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **116** ключей `tp_topics_*` из `SemanticTopicExtractor` (8×3 topic slots, 16×3 keyphrase slots) |
| `embedding_stats_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **39** ключей `tp_embstats_*` из `EmbeddingStatsExtractor` (8 topvar slots, 2 canonical transcript source flags) |
| `cosine_metrics_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **39** ключей `tp_cos_*` из `CosineMetricsExtractor` |
| `title_embedding_cluster_entropy_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **24** ключа `tp_titleclent_*` из `TitleEmbeddingClusterEntropyExtractor` |
| `title_to_hashtag_cosine_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **11** ключей `tp_titlehashcos_*` из `TitleToHashtagCosineExtractor` |
| `semantic_cluster_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **31** ключа `tp_semclust_*` из `SemanticClusterExtractor` (`allow_extra_keys: false`) |
| `topk_similar_titles_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **29** ключей `tp_topktitles_*` из `TopKSimilarCorpusTitlesExtractor` (`allow_extra_keys: false`) |
| `embedding_shift_indicator_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **27** ключей `tp_embshift_*` из `EmbeddingShiftIndicatorExtractor` (`allow_extra_keys: false`) |
| `embedding_source_id_extractor_output_v1` | вклад в `feature_names` / `feature_values` | Ровно **13** ключей `tp_embid_*` из `EmbeddingSourceIdExtractor` (`allow_extra_keys: false`) |

Реестр `artifact_validator` по `meta.schema_version` относится к **корневому** NPZ (`text_npz_v1`). Схемы экстракторов используются как **контракт вклада** и для будущих CI/линтеров; `allow_extra_keys=true` там, где есть динамический top‑K (`tp_tags_top{i}_*`).

Per-extractor NPZ (как у Audio) возможны позже для изолированных артефактов.
