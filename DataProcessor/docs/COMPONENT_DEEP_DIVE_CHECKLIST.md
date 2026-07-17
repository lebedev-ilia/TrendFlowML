# Чеклист глубокого разбора компонентов (Final Report)

Процесс — [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](COMPONENT_DEEP_DIVE_PROTOCOL.md). Статусы: ⬜ не начат ·
🔄 в разборе · ✅ отчёт готов (`component_reports/<component>/FINAL_REPORT.md`).

Список компонентов — источник истины: `automation/runner/component_queue.py::stamped_components()`
(всё, что заштамповано ✅ в [`COMPONENT_VALIDATION_CHECKLIST.md`](COMPONENT_VALIDATION_CHECKLIST.md)).
Разбор имеет смысл начинать только после штампа валидации — иначе результатов тестов ещё нет.

Всего заштамповано: **75**. Разобрано глубоко: **0**.

---

## Visual (VisualProcessor)

| Компонент | Статус | Дата | Оценка (модели) | Оценка (аналитики) | Отчёт |
|---|---|---|---|---|---|
| `core_clip` | ⬜ | | | | |
| `core_depth_midas` | ⬜ | | | | |
| `core_optical_flow` | ⬜ | | | | |
| `optical_flow` | ⬜ | | | | |
| `core_object_detections` | ⬜ | | | | |
| `core_face_landmarks` | ⬜ | | | | |
| `clip_embeddings` | ⬜ | | | | |
| `clip_times_s/clip_frame_indices` | ⬜ | | | | |
| `clip_track_id` | ⬜ | | | | |
| `color_light` | ⬜ | | | | |
| `frames_composition` | ⬜ | | | | |
| `video_pacing` | ⬜ | | | | |
| `cut_detection` | ⬜ | | | | |
| `scene_classification` | ⬜ | | | | |
| `shot_quality` | ⬜ | | | | |
| `ocr_extractor` | ⬜ | | | | |
| `high_level_semantic` | ⬜ | | | | |
| `story_structure` | ⬜ | | | | |
| `emotion_face` | ⬜ | | | | |
| `micro_emotion` | ⬜ | | | | |
| `detalize_face` | ⬜ | | | | |
| `behavioral` | ⬜ | | | | |
| `action_recognition` | ⬜ | | | | |
| `text_scoring` | ⬜ | | | | |
| `core_identity/place_semantics` | ⬜ | | | | |
| `core_identity/brand_semantics` | ⬜ | | | | |
| `core_identity/car_semantics` | ⬜ | | | | |
| `core_identity/content_domain` | ⬜ | | | | |
| `core_identity/face_identity` | ⬜ | | | | |
| `core_identity/franchise_recognition` | ⬜ | | | | |
| `similarity_metrics` | ⬜ | | | | |
| `uniqueness` | ⬜ | | | | |

## Audio (AudioProcessor)

| Компонент | Статус | Дата | Оценка (модели) | Оценка (аналитики) | Отчёт |
|---|---|---|---|---|---|
| `clap_extractor` | ⬜ | | | | |
| `asr_extractor` | ⬜ | | | | |
| `speaker_diarization_extractor` | ⬜ | | | | |
| `loudness_extractor` | ⬜ | | | | |
| `spectral_extractor` | ⬜ | | | | |
| `mel_extractor` | ⬜ | | | | |
| `mfcc_extractor` | ⬜ | | | | |
| `chroma_extractor` | ⬜ | | | | |
| `tempo_extractor` | ⬜ | | | | |
| `onset_extractor` | ⬜ | | | | |
| `pitch_extractor` | ⬜ | | | | |
| `spectral_entropy_extractor` | ⬜ | | | | |
| `rhythmic_extractor` | ⬜ | | | | |
| `key_extractor` | ⬜ | | | | |
| `band_energy_extractor` | ⬜ | | | | |
| `quality_extractor` | ⬜ | | | | |
| `voice_quality_extractor` | ⬜ | | | | |
| `hpss_extractor` | ⬜ | | | | |
| `emotion_diarization_extractor` | ⬜ | | | | |
| `source_separation_extractor` | ⬜ | | | | |
| `speech_analysis_extractor` | ⬜ | | | | |

## Text (TextProcessor)

| Компонент | Статус | Дата | Оценка (модели) | Оценка (аналитики) | Отчёт |
|---|---|---|---|---|---|
| `title_embedder` | ⬜ | | | | |
| `description_embedder` | ⬜ | | | | |
| `hashtag_embedder` | ⬜ | | | | |
| `transcript_chunk_embedder` | ⬜ | | | | |
| `comments_embedder` | ⬜ | | | | |
| `semantic_cluster_extractor` | ⬜ | | | | |
| `tags_extractor` | ⬜ | | | | |
| `transcript_aggregator` | ⬜ | | | | |
| `comments_aggregator` | ⬜ | | | | |
| `speaker_turn_embeddings_aggregator` | ⬜ | | | | |
| `asr_text_proxy_audio_features` | ⬜ | | | | |
| `lexico_static_features` | ⬜ | | | | |
| `semantics_topics_keyphrases` | ⬜ | | | | |
| `cosine_metrics_extractor` | ⬜ | | | | |
| `embedding_stats_extractor` | ⬜ | | | | |
| `embedding_shift_indicator_extractor` | ⬜ | | | | |
| `embedding_source_id_extractor` | ⬜ | | | | |
| `embedding_pair_topk_extractor` | ⬜ | | | | |
| `qa_embedding_pairs_extractor` | ⬜ | | | | |
| `title_embedding_cluster_entropy_extractor` | ⬜ | | | | |
| `title_to_hashtag_cosine_extractor` | ⬜ | | | | |
| `topk_similar_titles_extractor` | ⬜ | | | | |

---

*Обновляй эту таблицу сразу после написания `FINAL_REPORT.md` для компонента (шаг 4 протокола). Если
появляются новые заштампованные компоненты, добавляй строки сверяясь с `stamped_components()`.*
