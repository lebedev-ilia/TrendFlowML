# Audit v4.2 — доп. материалы (engineering bridge)

Документы, которые **связывают** эмпирические отчёты **L2** с последующими изменениями кода (профилирование, оптимизации, env), не заменяя сами отчёты в [`audio_processor/`](../audio_processor/), [`visual_processor/`](../visual_processor/) и [`text_processor/`](../text_processor/).

**Сквозной итог L2 и bridge 4.2 по трём процессорам:** [`../AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md`](../AUDIT_V4_2_L2_CROSS_PROCESSORS_SUMMARY.md).

## Раскладка папок

```text
audit_4_2/
├── README.md                 ← этот файл
├── audio_processor/          ← журналы по экстракторам AudioProcessor
├── text_processor/           ← журналы по TextProcessor (срезы text_features.npz)
└── visual_processor/
    ├── README.md             ← навигация по visual bridge
    ├── core/                 ← core_* + ocr_extractor
    └── modules/              ← продуктовые модули VisualProcessor
```

Сводка по статусам VisualProcessor: [`../../VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md`](../../VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md). Журнал прогонов: [`../../RUN_LOG.md`](../../RUN_LOG.md).

---

## AudioProcessor — engineering logs

| Документ | Описание |
|----------|----------|
| [`audio_processor/asr_extractor_engineering_log_v4_2.md`](audio_processor/asr_extractor_engineering_log_v4_2.md) | Журнал `asr_extractor` + ссылка на статистики L2 |
| [`audio_processor/band_energy_extractor_engineering_log_v4_2.md`](audio_processor/band_energy_extractor_engineering_log_v4_2.md) | Профилирование/ускорение `band_energy_extractor` |
| [`audio_processor/chroma_extractor_engineering_log_v4_2.md`](audio_processor/chroma_extractor_engineering_log_v4_2.md) | Профилирование `chroma_extractor` |
| [`audio_processor/clap_extractor_engineering_log_v4_2.md`](audio_processor/clap_extractor_engineering_log_v4_2.md) | Profiling `clap_extractor` |
| [`audio_processor/emotion_diarization_extractor_engineering_log_v4_2.md`](audio_processor/emotion_diarization_extractor_engineering_log_v4_2.md) | Profiling `emotion_diarization_extractor` |
| [`audio_processor/key_extractor_engineering_log_v4_2.md`](audio_processor/key_extractor_engineering_log_v4_2.md) | Profiling `key_extractor` |
| [`audio_processor/loudness_extractor_engineering_log_v4_2.md`](audio_processor/loudness_extractor_engineering_log_v4_2.md) | Profiling `loudness_extractor` |
| [`audio_processor/hpss_extractor_engineering_log_v4_2.md`](audio_processor/hpss_extractor_engineering_log_v4_2.md) | Profiling `hpss_extractor` |
| [`audio_processor/mel_extractor_engineering_log_v4_2.md`](audio_processor/mel_extractor_engineering_log_v4_2.md) | Profiling `mel_extractor` |
| [`audio_processor/mfcc_extractor_engineering_log_v4_2.md`](audio_processor/mfcc_extractor_engineering_log_v4_2.md) | Profiling `mfcc_extractor` |
| [`audio_processor/onset_extractor_engineering_log_v4_2.md`](audio_processor/onset_extractor_engineering_log_v4_2.md) | Profiling `onset_extractor` |
| [`audio_processor/pitch_extractor_engineering_log_v4_2.md`](audio_processor/pitch_extractor_engineering_log_v4_2.md) | Profiling `pitch_extractor` |
| [`audio_processor/quality_extractor_engineering_log_v4_2.md`](audio_processor/quality_extractor_engineering_log_v4_2.md) | Profiling `quality_extractor` |
| [`audio_processor/rhythmic_extractor_engineering_log_v4_2.md`](audio_processor/rhythmic_extractor_engineering_log_v4_2.md) | Profiling `rhythmic_extractor` |
| [`audio_processor/source_separation_extractor_engineering_log_v4_2.md`](audio_processor/source_separation_extractor_engineering_log_v4_2.md) | Profiling `source_separation_extractor` |
| [`audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md`](audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md) | Profiling `speaker_diarization_extractor` |
| [`audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md`](audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md) | Profiling `spectral_entropy_extractor` |
| [`audio_processor/spectral_extractor_engineering_log_v4_2.md`](audio_processor/spectral_extractor_engineering_log_v4_2.md) | Profiling `spectral_extractor` |
| [`audio_processor/speech_analysis_extractor_engineering_log_v4_2.md`](audio_processor/speech_analysis_extractor_engineering_log_v4_2.md) | Profiling `speech_analysis_extractor` |
| [`audio_processor/tempo_extractor_engineering_log_v4_2.md`](audio_processor/tempo_extractor_engineering_log_v4_2.md) | Profiling `tempo_extractor` |
| [`audio_processor/voice_quality_extractor_engineering_log_v4_2.md`](audio_processor/voice_quality_extractor_engineering_log_v4_2.md) | Profiling `voice_quality_extractor` |

Сквозные отчёты L2: [`../audio_processor/asr_extractor_audit_v4.md`](../audio_processor/asr_extractor_audit_v4.md), … (остальные `*_audit_v4.md` в той же папке).

---

## TextProcessor — engineering logs

| Документ | Описание |
|----------|----------|
| [`text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md`](text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_asrproxy_*`, качество `result_store`, блокировка при error `text_processor` |
| [`text_processor/comments_embedder_engineering_log_v4_2.md`](text_processor/comments_embedder_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_commentsemb_*` + проверка `comments_embeddings.npy`, блокировка при error `text_processor` |
| [`text_processor/comments_aggregator_engineering_log_v4_2.md`](text_processor/comments_aggregator_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_commentsagg_*` + проверка mean/median/indices артефактов, блокировка при error `text_processor` |
| [`text_processor/description_embedder_engineering_log_v4_2.md`](text_processor/description_embedder_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_descemb_*` + проверка `description_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/title_embedder_engineering_log_v4_2.md`](text_processor/title_embedder_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_titleemb_*` + проверка `title_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/title_embedding_cluster_entropy_extractor_engineering_log_v4_2.md`](text_processor/title_embedding_cluster_entropy_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_titleclent_*` + проверка upstream `title_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/title_to_hashtag_cosine_extractor_engineering_log_v4_2.md`](text_processor/title_to_hashtag_cosine_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_titlehashcos_*` + сверка cosine с `title_embedding.npy`/`hashtag_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/topk_similar_titles_extractor_engineering_log_v4_2.md`](text_processor/topk_similar_titles_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_topktitles_*` + проверка upstream `title_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/transcript_chunk_embedder_engineering_log_v4_2.md`](text_processor/transcript_chunk_embedder_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_tchunk_*` + проверка `transcript_whisper_chunk_embeddings.npy`, блокировка при error `text_processor` |
| [`text_processor/transcript_aggregator_engineering_log_v4_2.md`](text_processor/transcript_aggregator_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_tragg_*` + проверка `transcript_*_agg_{mean,max}.npy` по флагам, блокировка при error `text_processor` |
| [`text_processor/cosine_metrics_extractor_engineering_log_v4_2.md`](text_processor/cosine_metrics_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_cos_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md`](text_processor/embedding_pair_topk_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_embpair_*` + legacy `tp_pairtopk_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md`](text_processor/embedding_shift_indicator_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_embshift_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/embedding_source_id_extractor_engineering_log_v4_2.md`](text_processor/embedding_source_id_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_embid_*` + nested `payload["embedding_source_id"]` + сверка `vector_id`, блокировка при error `text_processor` |
| [`text_processor/embedding_stats_extractor_engineering_log_v4_2.md`](text_processor/embedding_stats_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_embstats_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/hashtag_embedder_engineering_log_v4_2.md`](text_processor/hashtag_embedder_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_hashemb_*` + проверка `hashtag_embedding.npy`, блокировка при error `text_processor` |
| [`text_processor/lexico_static_features_engineering_log_v4_2.md`](text_processor/lexico_static_features_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_lex_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md`](text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_qa_*` + проверка `qa_question_embeddings.npy` (если `present=1`), блокировка при error `text_processor` |
| [`text_processor/semantic_cluster_extractor_engineering_log_v4_2.md`](text_processor/semantic_cluster_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_semclust_*` (без артефактов), блокировка при error `text_processor` |
| [`text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md`](text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_topics_*` + проверка `tp_topics_keyphrase_embeddings.npy`, блокировка при error `text_processor` |
| [`text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md`](text_processor/speaker_turn_embeddings_aggregator_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_spkemb_*` + проверка `speaker_spkXXX_{mean,max}.npy` (если `present=1`), блокировка при error `text_processor` |
| [`text_processor/tags_extractor_engineering_log_v4_2.md`](text_processor/tags_extractor_engineering_log_v4_2.md) | Скрипт L2 по срезу `tp_tags_*` (allow_extra_keys: true; top‑K слоты), блокировка при error `text_processor` |

Канонические отчёты: [`../text_processor/asr_text_proxy_audio_features_audit_v4.md`](../text_processor/asr_text_proxy_audio_features_audit_v4.md), [`../text_processor/comments_embedder_audit_v4.md`](../text_processor/comments_embedder_audit_v4.md), [`../text_processor/comments_aggregator_audit_v4.md`](../text_processor/comments_aggregator_audit_v4.md), [`../text_processor/description_embedder_audit_v4.md`](../text_processor/description_embedder_audit_v4.md), [`../text_processor/title_embedder_audit_v4.md`](../text_processor/title_embedder_audit_v4.md), [`../text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md`](../text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md), [`../text_processor/title_to_hashtag_cosine_extractor_audit_v4.md`](../text_processor/title_to_hashtag_cosine_extractor_audit_v4.md), [`../text_processor/topk_similar_titles_extractor_audit_v4.md`](../text_processor/topk_similar_titles_extractor_audit_v4.md), [`../text_processor/transcript_chunk_embedder_audit_v4.md`](../text_processor/transcript_chunk_embedder_audit_v4.md), [`../text_processor/transcript_aggregator_audit_v4.md`](../text_processor/transcript_aggregator_audit_v4.md), [`../text_processor/cosine_metrics_extractor_audit_v4.md`](../text_processor/cosine_metrics_extractor_audit_v4.md), [`../text_processor/embedding_pair_topk_extractor_audit_v4.md`](../text_processor/embedding_pair_topk_extractor_audit_v4.md), [`../text_processor/embedding_shift_indicator_extractor_audit_v4.md`](../text_processor/embedding_shift_indicator_extractor_audit_v4.md), [`../text_processor/embedding_source_id_extractor_audit_v4.md`](../text_processor/embedding_source_id_extractor_audit_v4.md), [`../text_processor/embedding_stats_extractor_audit_v4.md`](../text_processor/embedding_stats_extractor_audit_v4.md), [`../text_processor/hashtag_embedder_audit_v4.md`](../text_processor/hashtag_embedder_audit_v4.md), [`../text_processor/lexico_static_features_audit_v4.md`](../text_processor/lexico_static_features_audit_v4.md), [`../text_processor/qa_embedding_pairs_extractor_audit_v4.md`](../text_processor/qa_embedding_pairs_extractor_audit_v4.md), [`../text_processor/semantic_cluster_extractor_audit_v4.md`](../text_processor/semantic_cluster_extractor_audit_v4.md), [`../text_processor/semantics_topics_keyphrases_audit_v4.md`](../text_processor/semantics_topics_keyphrases_audit_v4.md), [`../text_processor/speaker_turn_embeddings_aggregator_audit_v4.md`](../text_processor/speaker_turn_embeddings_aggregator_audit_v4.md), [`../text_processor/tags_extractor_audit_v4.md`](../text_processor/tags_extractor_audit_v4.md).

---

## VisualProcessor — core (`visual_processor/core/`)

| Документ | Компонент |
|----------|-----------|
| [`visual_processor/core/core_clip_engineering_log_v4_2.md`](visual_processor/core/core_clip_engineering_log_v4_2.md) | `core_clip` |
| [`visual_processor/core/core_depth_midas_engineering_log_v4_2.md`](visual_processor/core/core_depth_midas_engineering_log_v4_2.md) | `core_depth_midas` |
| [`visual_processor/core/core_face_landmarks_engineering_log_v4_2.md`](visual_processor/core/core_face_landmarks_engineering_log_v4_2.md) | `core_face_landmarks` |
| [`visual_processor/core/core_object_detections_engineering_log_v4_2.md`](visual_processor/core/core_object_detections_engineering_log_v4_2.md) | `core_object_detections` |
| [`visual_processor/core/core_optical_flow_engineering_log_v4_2.md`](visual_processor/core/core_optical_flow_engineering_log_v4_2.md) | `core_optical_flow` |
| [`visual_processor/core/ocr_extractor_engineering_log_v4_2.md`](visual_processor/core/ocr_extractor_engineering_log_v4_2.md) | `ocr_extractor` |

---

## VisualProcessor — modules (`visual_processor/modules/`)

| Документ | Компонент |
|----------|-----------|
| [`visual_processor/modules/action_recognition_engineering_log_v4_2.md`](visual_processor/modules/action_recognition_engineering_log_v4_2.md) | `action_recognition` |
| [`visual_processor/modules/behavioral_engineering_log_v4_2.md`](visual_processor/modules/behavioral_engineering_log_v4_2.md) | `behavioral` |
| [`visual_processor/modules/color_light_engineering_log_v4_2.md`](visual_processor/modules/color_light_engineering_log_v4_2.md) | `color_light` |
| [`visual_processor/modules/cut_detection_engineering_log_v4_2.md`](visual_processor/modules/cut_detection_engineering_log_v4_2.md) | `cut_detection` |
| [`visual_processor/modules/detalize_face_engineering_log_v4_2.md`](visual_processor/modules/detalize_face_engineering_log_v4_2.md) | `detalize_face` |
| [`visual_processor/modules/emotion_face_engineering_log_v4_2.md`](visual_processor/modules/emotion_face_engineering_log_v4_2.md) | `emotion_face` |
| [`visual_processor/modules/frames_composition_engineering_log_v4_2.md`](visual_processor/modules/frames_composition_engineering_log_v4_2.md) | `frames_composition` |
| [`visual_processor/modules/high_level_semantic_engineering_log_v4_2.md`](visual_processor/modules/high_level_semantic_engineering_log_v4_2.md) | `high_level_semantic` |
| [`visual_processor/modules/micro_emotion_engineering_log_v4_2.md`](visual_processor/modules/micro_emotion_engineering_log_v4_2.md) | `micro_emotion` |
| [`visual_processor/modules/optical_flow_engineering_log_v4_2.md`](visual_processor/modules/optical_flow_engineering_log_v4_2.md) | `optical_flow` |
| [`visual_processor/modules/scene_classification_engineering_log_v4_2.md`](visual_processor/modules/scene_classification_engineering_log_v4_2.md) | `scene_classification` |
| [`visual_processor/modules/shot_quality_engineering_log_v4_2.md`](visual_processor/modules/shot_quality_engineering_log_v4_2.md) | `shot_quality` |
| [`visual_processor/modules/similarity_metrics_engineering_log_v4_2.md`](visual_processor/modules/similarity_metrics_engineering_log_v4_2.md) | `similarity_metrics` |
| [`visual_processor/modules/story_structure_engineering_log_v4_2.md`](visual_processor/modules/story_structure_engineering_log_v4_2.md) | `story_structure` |
| [`visual_processor/modules/text_scoring_engineering_log_v4_2.md`](visual_processor/modules/text_scoring_engineering_log_v4_2.md) | `text_scoring` |
| [`visual_processor/modules/uniqueness_engineering_log_v4_2.md`](visual_processor/modules/uniqueness_engineering_log_v4_2.md) | `uniqueness` |
| [`visual_processor/modules/video_pacing_engineering_log_v4_2.md`](visual_processor/modules/video_pacing_engineering_log_v4_2.md) | `video_pacing` |
