ilya@ilya-B450M-DS3H:/media/ilya/Новый том1/TrendFlowML/DataProcessor$ python3 main.py   --video-path "/media/ilya/Новый том1/TrendFlowML/example/example_videos/video1.mp4"   --global-config configs/global_config.yaml   --run-audio   --platform-id youtube   --video-id test_video_1   --run-id test_run_1_no_optimizations --output-dir "/media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results"

================================================================================
  Segmenter
================================================================================
  [✓ OK] Запуск Segmenter
        .../TrendFlowML/example/example_videos/video1.mp4
  → Segmenter: построено 29 конфигов из /tmp/visual_cfg_tzl20ki3.yaml
  → Segmenter: обработка .../TrendFlowML/example/example_videos/video1.mp4
    primary sampling group budget: total_frames_source=863 fps=30.000 duration_s=28.8 requested_max=500 target_gap_sec=0.25 rate_fps=4.0 budget_n=115 chosen_n=115
    primary sampling group: set core_clip.frame_indices_source = N=115
    primary sampling group: set core_object_detections.frame_indices_source = N=115
    primary sampling group: set core_depth_midas.frame_indices_source = N=115
    primary sampling group: set core_face_landmarks.frame_indices_source = N=115
    primary sampling group: set core_optical_flow.frame_indices_source = N=115
    primary sampling group: set shot_quality.frame_indices_source = N=115
    primary sampling group: set frames_composition.frame_indices_source = N=115
    core_optical_flow reuse policy: set cut_detection.frame_indices_source = core_optical_flow (N=115)
    deps sampling align: ocr_extractor ⊆ core_object_detections | 250 -> 115 (parent=115)
    deps sampling align: content_domain ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: franchise_recognition ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: scene_classification ⊆ core_clip | 250 -> 115 (parent=115)
    deps sampling align: video_pacing ⊆ core_optical_flow | 120 -> 115 (parent=115)
    deps sampling align: uniqueness ⊆ core_clip | 120 -> 115 (parent=115)
    deps sampling align: story_structure ⊆ core_clip | 120 -> 115 (parent=115)
    deps sampling align: high_level_semantic ⊆ cut_detection | 250 -> 115 (parent=115)
    ✓ batch_00000.npy
    ✓ batch_00001.npy
    ✓ batch_00002.npy
    ✓ batch_00003.npy
    ✓ batch_00004.npy
    ✓ batch_00005.npy
    ✓ batch_00006.npy
    ✓ batch_00007.npy
    ✓ batch_00008.npy
    ✓ Segmenter/data/test_video_1/audio/audio.wav
    ✓ .../data/test_video_1/audio/audio.wav', 'duration_sec': 28.8, 'sample_rate': 22050, 'total_samples': 635040}
    ✓ Сохранено: Segmenter/data/test_video_1/audio/segments.json
    union mode done: union_frames=515 -> /media/ilya/Новый том1/TrendFlowML/DataProcessor/Segmenter/data/test_video_1/video/metadata.json
  [✓ OK] Segmenter завершен
        время: 4475ms

================================================================================
  TextProcessor
================================================================================
  [✓ OK] Запуск TextProcessor
        .../TrendFlowML/example/example_text_documents/video_document_1.json
  → TextProcessor: загружен документ из .../TrendFlowML/example/example_text_documents/video_document_1.json
  → TextProcessor: запуск 22 экстракторов
    → [1/22] LexicalStatsExtractor (cpu)
    ✓ [1/22] LexicalStatsExtractor 0.007s (66 feat)
    → [2/22] TagsExtractor (cpu)
    ✓ [2/22] TagsExtractor 0.001s (39 feat)
    → [3/22] ASRTextProxyExtractor (cpu)
    ✓ [3/22] ASRTextProxyExtractor 0.169s (33 feat)
    → [4/22] SemanticTopicExtractor (cpu)
    Batches:   0%|          | 0/1 [00:00<?, ?it/s]
    Batches: 100%|██████████| 1/1 [00:01<00:00,  1.74s/it]
    Batches:   0%|          | 0/1 [00:00<?, ?it/s]
    Batches: 100%|██████████| 1/1 [00:00<00:00,  2.36it/s]
    ✓ [4/22] SemanticTopicExtractor 99.337s (73 feat)
    → [5/22] TitleEmbedder (cuda)
    → [5/22] extractor (cpu)
    → [6/22] DescriptionEmbedder (cuda)
    → [6/22] extractor (cpu)
    → [7/22] HashtagEmbedder (cuda)
    → [7/22] extractor (cpu)
    → [8/22] TranscriptChunkEmbedder (cuda)
    → [8/22] extractor (cpu)
    → [9/22] CommentsEmbedder (cuda)
    → [9/22] extractor (cpu)
    → [10/22] SpeakerTurnEmbeddingsAggregatorExtractor (cuda)
    ✓ [10/22] SpeakerTurnEmbeddingsAggregatorExtractor 0.001s (12 feat)
    → [11/22] QAEmbeddingPairsExtractor (cuda)
    ✓ [11/22] QAEmbeddingPairsExtractor 0.089s (33 feat)
    → [12/22] TranscriptAggregatorExtractor (cpu)
    ✓ [12/22] TranscriptAggregatorExtractor 0.162s (10 feat)
    → [13/22] CommentsAggregationExtractor (cpu)
    ✓ [13/22] CommentsAggregationExtractor 0.025s (37 feat)
    → [14/22] CosineMetricsExtractor (cpu)
    ✓ [14/22] CosineMetricsExtractor 0.016s (23 feat)
    → [15/22] EmbeddingPairTopKExtractor (cpu)
    ✓ [15/22] EmbeddingPairTopKExtractor 0.007s (51 feat)
    → [16/22] EmbeddingStatsExtractor (cpu)
    ✓ [16/22] EmbeddingStatsExtractor 0.007s (35 feat)
    → [17/22] EmbeddingShiftIndicatorExtractor (cpu)
    ✓ [17/22] EmbeddingShiftIndicatorExtractor 0.004s (25 feat)
    → [18/22] EmbeddingSourceIdExtractor (cpu)
    ✓ [18/22] EmbeddingSourceIdExtractor 0.003s (9 feat)
    → [19/22] TitleToHashtagCosineExtractor (cpu)
    ✓ [19/22] TitleToHashtagCosineExtractor 0.004s (13 feat)
    → [20/22] TopKSimilarCorpusTitlesExtractor (cpu)
    ✓ [20/22] TopKSimilarCorpusTitlesExtractor 0.008s (28 feat)
    → [21/22] TitleEmbeddingClusterEntropyExtractor (cpu)
    ✓ [21/22] TitleEmbeddingClusterEntropyExtractor 0.003s (11 feat)
    → [22/22] SemanticClusterExtractor (cpu)
    ✓ [22/22] SemanticClusterExtractor 0.019s (13 feat)
  ✓ TextProcessor: завершено со статусом OK
    ✓ NPZ сохранен: .../test_run_1_no_optimizations/text_processor/text_features.npz
    ✓ render: отчет сохранен → .../text_processor/_render/render_context.json
    ✓ lexico static features: отчет сохранен → .../text_processor/_render/lexico_static_features_report.html
    ✓ embedding shift indicator: отчет сохранен → .../text_processor/_render/embedding_shift_indicator_extractor_report.html
    ✓ title embedding cluster entropy: отчет сохранен → .../text_processor/_render/title_embedding_cluster_entropy_extractor_report.html
    ✓ embedding stats: отчет сохранен → .../text_processor/_render/embedding_stats_extractor_report.html
    ✓ embedding pair topk: отчет сохранен → .../text_processor/_render/embedding_pair_topk_extractor_report.html
    ✓ embedding source id: отчет сохранен → .../text_processor/_render/embedding_source_id_extractor_report.html
    ✓ description: отчет сохранен → .../text_processor/_render/description_embedder_report.html
    ✓ transcript chunk: отчет сохранен → .../text_processor/_render/transcript_chunk_embedder_report.html
    ✓ semantics topics keyphrases: отчет сохранен → .../text_processor/_render/semantics_topics_keyphrases_report.html
    ✓ asr text proxy audio features: отчет сохранен → .../text_processor/_render/asr_text_proxy_audio_features_report.html
    ✓ qa embedding pairs: отчет сохранен → .../text_processor/_render/qa_embedding_pairs_extractor_report.html
    ✓ comments: отчет сохранен → .../text_processor/_render/comments_embedder_report.html
    ✓ transcript aggregator: отчет сохранен → .../text_processor/_render/transcript_aggregator_report.html
    ✓ title: отчет сохранен → .../text_processor/_render/title_embedder_report.html
    ✓ semantic cluster: отчет сохранен → .../text_processor/_render/semantic_cluster_extractor_report.html
    ✓ tags: отчет сохранен → .../text_processor/_render/tags_extractor_report.html
    ✓ cosine metrics: отчет сохранен → .../text_processor/_render/cosine_metrics_extractor_report.html
    ✓ comments aggregator: отчет сохранен → .../text_processor/_render/comments_aggregator_report.html
    ✓ topk similar titles: отчет сохранен → .../text_processor/_render/topk_similar_titles_extractor_report.html
    ✓ hashtag: отчет сохранен → .../text_processor/_render/hashtag_embedder_report.html
    ✓ speaker turn embeddings aggregator: отчет сохранен → .../text_processor/_render/speaker_turn_embeddings_aggregator_report.html
    ✓ title to hashtag cosine: отчет сохранен → .../test_run_1_no_optimizations/text_processor/_render/title_to_hashtag_cosine_extractor_report.html
  [✓ OK] TextProcessor завершен
        время: 132600ms
---

## Навигация

[TextProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
