# Главный индекс документации TextProcessor

Этот документ служит единой точкой входа для навигации по всей документации TextProcessor. Каждый раздел содержит краткое описание документов и ссылки на полные версии.

---

## Документация

### Audit v3 (TextProcessor)
**Краткое описание**: Preflight Audit v3 — smoke с контролируемыми текстами, обязательный ASR, порядок **22** экстракторов, модель **`intfloat/multilingual-e5-large`**, corpus packs, отчёты в `docs/audit_v3/components/`.

**Полный документ (preflight)**: [`../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`](../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)

**Индекс каталога аудита**: [`docs/audit_v3/README.md`](audit_v3/README.md)

### BATCH_PROCESSING_PLAN.md
**Краткое описание**: План адаптации TextProcessor для батчевой обработки. Описывает двухуровневую параллельность (видео + сегменты), GPU batching для ML-моделей, CPU parallelism для signal processing extractors, изоляцию данных, валидацию, этапы реализации (Stage 0-5), примеры использования, производительность и оптимизации. Статус: все стадии завершены (Stage 0-5), batch processing полностью интегрирован в CLI и готов к production использованию.

**Полный документ**: [docs/BATCH_PROCESSING_PLAN.md](BATCH_PROCESSING_PLAN.md)

### LAST_FULL_RUN_LOG.md
**Краткое описание**: Лог последнего полного запуска DataProcessor с включенным TextProcessor. Содержит примеры вывода команд, статусы выполнения компонентов, тайминги и диагностическую информацию для отладки и валидации пайплайна.

**Полный документ**: [docs/LAST_FULL_RUN_LOG.md](LAST_FULL_RUN_LOG.md)

### semantics_topics_keyphrases/TOOLS.md
**Краткое описание**: Инструменты и инструкции по расширению базы данных тем (topics DB). Описывает расположение bundled asset (`dp_models/bundled_models/text/topics_v1/topics.jsonl`), использование кеша для prompt embeddings, инструкции по расширению до 200-500 тем, рекомендации по качеству и сложным обновлениям.

**Полный документ**: [src/extractors/semantics_topics_keyphrases/TOOLS.md](../src/extractors/semantics_topics_keyphrases/TOOLS.md)

### FAISS_AND_NUMPY_BACKEND.md
**Краткое описание**: Единая шпаргалка: флаги **`use_faiss` / `backend_faiss` / `faiss_available`**, fallback на **NumPy**, приближённый **HNSW** vs точный косинус в **`topk_similar_titles`**, рекомендации для сравнения прогонов (набор **B**).

**Полный документ**: [docs/FAISS_AND_NUMPY_BACKEND.md](FAISS_AND_NUMPY_BACKEND.md)

---

## Extractors

TextProcessor содержит 22 extractor'а для извлечения текстовых признаков из видео. Extractors организованы по уровням зависимостей и поддерживают batch processing, GPU ускорение и детерминированное кеширование.

### Tier-0: Baseline extractors (корень — tags)

**Audit v3 / runtime**: `TagsExtractor` — первый среди Tier-0, если включён общий документ с последующими экстракторами: он фиксирует хэштеги и (при включённой политике) **очищает** `title`/`description` до лексики, ASR proxy и эмбеддингов. Жёсткие зависимости заданы в `MainProcessor` (`LexicalStatsExtractor`, `ASRTextProxyExtractor`, `DescriptionEmbedder` → `TagsExtractor`); декларативно — стадия `text_processor_tier0` в `DataProcessor/docs/reference/component_graph.yaml`.

#### TagsExtractor (tags_extractor)
**Краткое описание**: Извлекает хэштеги из `doc.title`/`doc.description`, **мерджит** с `doc.hashtags` из JSON (dedupe по `casefold`), удаляет токены `#<tag>` из title/description и (опционально) пишет очищенные поля и итоговый список тегов в документ. Парсинг на окне `max_parse_chars`, хранение строк — до `max_text_chars`. Мутации и `tp_artifacts` — fail-fast с логом. Версия **1.2.0**, категория text, CPU-only.

**Контракт**: [src/extractors/tags_extractor/SCHEMA.md](../src/extractors/tags_extractor/SCHEMA.md) · machine: [schemas/tags_extractor_output_v1.json](../schemas/tags_extractor_output_v1.json)

**Полный документ**: [src/extractors/tags_extractor/README.md](../src/extractors/tags_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/tags_extractor_AUDIT_V3_REPORT.md](audit_v3/components/tags_extractor_AUDIT_V3_REPORT.md)

#### LexicalStatsExtractor (lexico_static_features)
**Краткое описание**: Детерминированные лексические признаки (title, description, ASR-транскрипт по умолчанию **`asr_only`**). Heavy NLP нет; baseline **`enable_emoji=true`**, **`emoji_policy=optional`**; опция **`require_transcript`** для строгого ASR. В стандартном ранне метрики title/description — **после** `TagsExtractor` (очищенные тексты). Версия **1.2.0**, CPU-only.

**Контракт**: [src/extractors/lexico_static_features/SCHEMA.md](../src/extractors/lexico_static_features/SCHEMA.md) · machine: [schemas/lexico_static_features_output_v1.json](../schemas/lexico_static_features_output_v1.json)

**Полный документ**: [src/extractors/lexico_static_features/README.md](../src/extractors/lexico_static_features/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/lexico_static_features_AUDIT_V3_REPORT.md](audit_v3/components/lexico_static_features_AUDIT_V3_REPORT.md)

#### ASRTextProxyExtractor (asr_text_proxy_audio_features)
**Краткое описание**: Извлекает audio-like proxy признаки из текста ASR (`doc.asr` / legacy `transcripts_meta`; **не** `doc.transcripts`) без анализа волны. Валидный empty по умолчанию; opt-in **`require_asr_text`** / **`strict_document_duration`**; duration из payload — деградация с флагом; token-id path Audit v3 с **`tp_asrproxy_token_decode_failed_flag`**; **`tp_asrproxy_speech_rate_wpm_ratio_to_baseline`**. Версия **1.2.0**, CPU-only. Зависит от `TagsExtractor`, если тот в прогоне.

**Контракт**: [src/extractors/asr_text_proxy_audio_features/SCHEMA.md](../src/extractors/asr_text_proxy_audio_features/SCHEMA.md) · machine: [schemas/asr_text_proxy_audio_features_output_v1.json](../schemas/asr_text_proxy_audio_features_output_v1.json)

**Полный документ**: [src/extractors/asr_text_proxy_audio_features/README.md](../src/extractors/asr_text_proxy_audio_features/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/asr_text_proxy_audio_features_AUDIT_V3_REPORT.md](audit_v3/components/asr_text_proxy_audio_features_AUDIT_V3_REPORT.md)

### Tier-1: Embedding Extractors (зависят от Tier-0)

#### TitleEmbedder (title_embedder)
**Краткое описание**: Извлекает L2-нормализованные эмбеддинги для заголовков видео с использованием моделей sentence transformers. Версия 1.2.0, категория text embeddings, опциональный GPU (CUDA с fp16). Поддерживает батчинг, кеширование на диск, GPU ускорение, возвращает нормализованные векторы и L2-нормы необработанных векторов. Модели загружаются через `dp_models` (no-network). Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/title_embedder/SCHEMA.md](../src/extractors/title_embedder/SCHEMA.md) · machine: [schemas/title_embedder_output_v1.json](../schemas/title_embedder_output_v1.json)

**Полный документ**: [src/extractors/title_embedder/README.md](../src/extractors/title_embedder/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/title_embedder_AUDIT_V3_REPORT.md](audit_v3/components/title_embedder_AUDIT_V3_REPORT.md)

#### DescriptionEmbedder (description_embedder)
**Краткое описание**: Извлекает L2-нормализованные эмбеддинги для описаний видео: token-aware chunking (`shared_tokenizer_v1`), pooling чанков, кеш, опциональный GPU. Версия 1.2.0. Зависит от `TagsExtractor` (описание после очистки хэштегов). Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/description_embedder/SCHEMA.md](../src/extractors/description_embedder/SCHEMA.md) · machine: [schemas/description_embedder_output_v1.json](../schemas/description_embedder_output_v1.json)

**Полный документ**: [src/extractors/description_embedder/README.md](../src/extractors/description_embedder/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/description_embedder_AUDIT_V3_REPORT.md](audit_v3/components/description_embedder_AUDIT_V3_REPORT.md)

#### HashtagEmbedder (hashtag_embedder)
**Краткое описание**: Агрегированный эмбеддинг по `doc.hashtags` (per-tag encode, mean/max/logsumexp, опционально частоты), артефакт `hashtag_embedding.npy`. Версия 1.2.0, опциональный GPU. Зависит от **TagsExtractor**. Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/hashtag_embedder/SCHEMA.md](../src/extractors/hashtag_embedder/SCHEMA.md) · machine: [schemas/hashtag_embedder_output_v1.json](../schemas/hashtag_embedder_output_v1.json)

**Полный документ**: [src/extractors/hashtag_embedder/README.md](../src/extractors/hashtag_embedder/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/hashtag_embedder_AUDIT_V3_REPORT.md](audit_v3/components/hashtag_embedder_AUDIT_V3_REPORT.md)

#### TranscriptChunkEmbedder (transcript_chunk_embedder)
**Краткое описание**: Извлекает эмбеддинги для чанков транскрипта (логический канал **whisper** + **youtube_auto**). Поддерживает батчинг переменной длины (собирает чанки всех документов → batch encode → распределяет обратно), GPU ускорение. Версия **1.3.0**, категория text embeddings, опциональный GPU. Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/transcript_chunk_embedder/SCHEMA.md](../src/extractors/transcript_chunk_embedder/SCHEMA.md) · machine: [schemas/transcript_chunk_embedder_output_v1.json](../schemas/transcript_chunk_embedder_output_v1.json)

**Полный документ**: [src/extractors/transcript_chunk_embedder/README.md](../src/extractors/transcript_chunk_embedder/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md](audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md)

#### CommentsEmbedder (comments_embedder)
**Краткое описание**: Извлекает L2-нормализованные эмбеддинги для комментариев видео. Поддерживает батчинг, детерминированный отбор/лимиты, optional cache, per-run sub-artifact. Версия **1.3.0**, категория text embedding, опциональный GPU. Строго через `dp_models` (offline/no-network). Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/comments_embedder/SCHEMA.md](../src/extractors/comments_embedder/SCHEMA.md) · machine: [schemas/comments_embedder_output_v1.json](../schemas/comments_embedder_output_v1.json)

**Полный документ**: [src/extractors/comments_embedder/README.md](../src/extractors/comments_embedder/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md](audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md)

#### SpeakerTurnEmbeddingsAggregatorExtractor (speaker_turn_embeddings_aggregator)
**Краткое описание**: Агрегирует эмбеддинги по спикерам (speaker turns): выравнивание **`speaker_diarization` + `doc.asr.segments`** по времени или legacy **`doc.speakers`**. Версия **1.3.0**, категория text embeddings aggregation, опциональный GPU. Preflight Audit v3: **`intfloat/multilingual-e5-large`**; для полного прохода рекомендуется **speaker diarization** в том же run, что и ASR.

**Контракт**: [src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md](../src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md) · machine: [schemas/speaker_turn_embeddings_aggregator_output_v1.json](../schemas/speaker_turn_embeddings_aggregator_output_v1.json)

**Полный документ**: [src/extractors/speaker_turn_embeddings_aggregator/README.md](../src/extractors/speaker_turn_embeddings_aggregator/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md](audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md)

### Tier-2: Aggregation Extractors (зависят от Tier-1)

#### TranscriptAggregatorExtractor (transcript_aggregator)
**Краткое описание**: Агрегирует эмбеддинги чанков транскрипта (**mean** / **max**, decay, optional std, **combined**) из **`doc.tp_artifacts`**. Версия **1.3.0**, категория aggregation, CPU-only. Зависит от **TranscriptChunkEmbedder**; модель не исполняется — метаданные **`model_name`** совпадают с чанковым эмбеддером (**Audit v3: `intfloat/multilingual-e5-large`**).

**Контракт**: [src/extractors/transcript_aggregator/SCHEMA.md](../src/extractors/transcript_aggregator/SCHEMA.md) · machine: [schemas/transcript_aggregator_output_v1.json](../schemas/transcript_aggregator_output_v1.json)

**Полный документ**: [src/extractors/transcript_aggregator/README.md](../src/extractors/transcript_aggregator/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md](audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md)

#### CommentsAggregationExtractor (comments_aggregator)
**Краткое описание**: Агрегирует эмбеддинги комментариев (**взвешенное среднее**, **покомпонентная медиана**). Версия **1.3.0**, категория aggregation, CPU-only. Зависит от **CommentsEmbedder**; модель не исполняется — **`model_name`** / **`weights_digest`** совпадают с эмбеддером (**Audit v3: `intfloat/multilingual-e5-large`**).

**Контракт**: [src/extractors/comments_aggregator/SCHEMA.md](../src/extractors/comments_aggregator/SCHEMA.md) · machine: [schemas/comments_aggregator_output_v1.json](../schemas/comments_aggregator_output_v1.json)

**Полный документ**: [src/extractors/comments_aggregator/README.md](../src/extractors/comments_aggregator/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/comments_aggregator_AUDIT_V3_REPORT.md](audit_v3/components/comments_aggregator_AUDIT_V3_REPORT.md)

#### QAEmbeddingPairsExtractor (qa_embedding_pairs_extractor)
**Краткое описание**: Извлекает **вопросоподобные** фразы из title/description/ASR/комментариев и считает **L2-нормализованные** эмбеддинги (**матрица N×D**, не пары Q–A). Версия **1.3.0**, опциональный GPU. В графе зависимостей указан **TranscriptChunkEmbedder** (операционный порядок конвейера). Preflight Audit v3: **`intfloat/multilingual-e5-large`**.

**Контракт**: [src/extractors/qa_embedding_pairs_extractor/SCHEMA.md](../src/extractors/qa_embedding_pairs_extractor/SCHEMA.md) · machine: [schemas/qa_embedding_pairs_extractor_output_v1.json](../schemas/qa_embedding_pairs_extractor_output_v1.json)

**Полный документ**: [src/extractors/qa_embedding_pairs_extractor/README.md](../src/extractors/qa_embedding_pairs_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md](audit_v3/components/qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md)

#### EmbeddingPairTopKExtractor (embedding_pair_topk_extractor)
**Краткое описание**: **Cosine(title, description)** и **top‑K** сходства **title** с **матрицей chunk embeddings** транскрипта (`tp_artifacts`), опционально **FAISS**. Версия **1.3.0**, CPU/GPU по **`device`** (по сути numpy/FAISS). Зависит от **TitleEmbedder**, **DescriptionEmbedder**, **TranscriptChunkEmbedder**. Контракт Audit v3: **8** фиксированных слотов экспорта; **`combined`** в приоритете — только если есть в `tp_artifacts`.

**Контракт**: [src/extractors/embedding_pair_topk_extractor/SCHEMA.md](../src/extractors/embedding_pair_topk_extractor/SCHEMA.md) · machine: [schemas/embedding_pair_topk_extractor_output_v1.json](../schemas/embedding_pair_topk_extractor_output_v1.json)

**Полный документ**: [src/extractors/embedding_pair_topk_extractor/README.md](../src/extractors/embedding_pair_topk_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/embedding_pair_topk_extractor_AUDIT_V3_REPORT.md](audit_v3/components/embedding_pair_topk_extractor_AUDIT_V3_REPORT.md)

#### SemanticTopicExtractor (semantics_topics_keyphrases)
**Краткое описание**: Извлекает глобальные (сопоставимые между видео) темы из текста через retrieval по фиксированной taxonomy (bundled `topics.jsonl` + embeddings через `dp_models`), а также ключевые фразы и стилистические proxy-флаги. Версия 2.1.0, категория topic modeling, keyphrase extraction, style analysis, опциональный GPU. Использует ASR от AudioProcessor как preferred source, bundled topics DB (offline).

**Контракт**: [src/extractors/semantics_topics_keyphrases/SCHEMA.md](../src/extractors/semantics_topics_keyphrases/SCHEMA.md) · machine: [schemas/semantics_topics_keyphrases_output_v1.json](../schemas/semantics_topics_keyphrases_output_v1.json)

**Полный документ**: [src/extractors/semantics_topics_keyphrases/README.md](../src/extractors/semantics_topics_keyphrases/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/semantics_topics_keyphrases_AUDIT_V3_REPORT.md](audit_v3/components/semantics_topics_keyphrases_AUDIT_V3_REPORT.md)

### Tier-3: Advanced Metrics Extractors (зависят от Tier-2)

#### EmbeddingStatsExtractor (embedding_stats_extractor)
**Краткое описание**: Дисперсия эмбеддингов между **чанками транскрипта** (матрица из `TranscriptChunkEmbedder`) + опциональная энтропия по `topic_probs` из `semantics_topics_keyphrases`. Версия 1.2.0, CPU-only. Жёсткая зависимость в DAG: **TranscriptChunkEmbedder**; topics — best-effort.

**Контракт**: [src/extractors/embedding_stats_extractor/SCHEMA.md](../src/extractors/embedding_stats_extractor/SCHEMA.md) · machine: [schemas/embedding_stats_extractor_output_v1.json](../schemas/embedding_stats_extractor_output_v1.json)

**Полный документ**: [src/extractors/embedding_stats_extractor/README.md](../src/extractors/embedding_stats_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/embedding_stats_extractor_AUDIT_V3_REPORT.md](audit_v3/components/embedding_stats_extractor_AUDIT_V3_REPORT.md)

#### CosineMetricsExtractor (cosine_metrics_extractor)
**Краткое описание**: Cosine similarity между эмбеддингами title, description, **mean-агрегатом транскрипта** и комментариями (aggregates/matrix), через **`tp_artifacts`**. Версия 1.3.0, CPU-only. Зависит от TitleEmbedder, DescriptionEmbedder, TranscriptAggregatorExtractor, CommentsEmbedder (и при matrix — CommentsEmbedder).

**Контракт**: [src/extractors/cosine_metrics_extractor/SCHEMA.md](../src/extractors/cosine_metrics_extractor/SCHEMA.md) · machine: [schemas/cosine_metrics_extractor_output_v1.json](../schemas/cosine_metrics_extractor_output_v1.json)

**Полный документ**: [src/extractors/cosine_metrics_extractor/README.md](../src/extractors/cosine_metrics_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/cosine_metrics_extractor_AUDIT_V3_REPORT.md](audit_v3/components/cosine_metrics_extractor_AUDIT_V3_REPORT.md)

#### TitleEmbeddingClusterEntropyExtractor (title_embedding_cluster_entropy_extractor)
**Краткое описание**: Энтропия распределения title embedding по таксономии **PCA+центроиды** (`semantic_clusters_v1` / `dp_models`). Версия **1.3.0**, CPU-only. Зависит от **TitleEmbedder**; опционально **FAISS** IndexFlatIP; **24** фиксированных ключа `features_flat`, кламп **`top_k_slots`** ≤ **8**.

**Контракт**: [src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md](../src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md) · machine: [schemas/title_embedding_cluster_entropy_extractor_output_v1.json](../schemas/title_embedding_cluster_entropy_extractor_output_v1.json)

**Полный документ**: [src/extractors/title_embedding_cluster_entropy_extractor/README.md](../src/extractors/title_embedding_cluster_entropy_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md](audit_v3/components/title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md)

#### TitleToHashtagCosineExtractor (title_to_hashtag_cosine_extractor)
**Краткое описание**: Cosine similarity между **title** и **hashtag** эмбеддингами из **`tp_artifacts`**. Версия **1.2.0**, CPU-only. Зависит от **TitleEmbedder** и **HashtagEmbedder**; **11** фиксированных ключей; раздельно **unsafe relpath** и **missing file**.

**Контракт**: [src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md](../src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md) · machine: [schemas/title_to_hashtag_cosine_extractor_output_v1.json](../schemas/title_to_hashtag_cosine_extractor_output_v1.json)

**Полный документ**: [src/extractors/title_to_hashtag_cosine_extractor/README.md](../src/extractors/title_to_hashtag_cosine_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md](audit_v3/components/title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md)

#### SemanticClusterExtractor (semantic_cluster_extractor)
**Краткое описание**: Ближайший центроид в **PCA**-пространстве по эмбеддингу **title** / **description** / **hashtag** (`semantic_clusters_v1` / `dp_models`). Версия **1.3.0**, CPU-only. Зависит от **TitleEmbedder**, **DescriptionEmbedder**, **HashtagEmbedder** (для primary/fallback **`hashtag`**). **31** фиксированный ключ `features_flat`; **`_*_present`** = успешная загрузка `.npy`; **`semantic_cluster_meta.backend`** на всех ветках; extra-блок — **NaN** при **`emit_extra_metrics=False`**.

**Контракт**: [src/extractors/semantic_cluster_extractor/SCHEMA.md](../src/extractors/semantic_cluster_extractor/SCHEMA.md) · machine: [schemas/semantic_cluster_extractor_output_v1.json](../schemas/semantic_cluster_extractor_output_v1.json)

**Полный документ**: [src/extractors/semantic_cluster_extractor/README.md](../src/extractors/semantic_cluster_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/semantic_cluster_extractor_AUDIT_V3_REPORT.md](audit_v3/components/semantic_cluster_extractor_AUDIT_V3_REPORT.md)

#### TopKSimilarCorpusTitlesExtractor (topk_similar_titles_extractor)
**Краткое описание**: Top-K похожих заголовков из **dp_models**-корпуса по title embedding (**inner product** на L2-нормах). Версия **1.3.0**, CPU-only. Зависит от **TitleEmbedder**. **29** фиксированных ключей `features_flat`; **`corpus`** в payload на всех ветках; **`tp_topktitles_title_embed_missing_flag`**; **HNSW** — приближённый поиск относительно numpy. Tier **analytics** до фиксации corpus pack.

**Контракт**: [src/extractors/topk_similar_titles_extractor/SCHEMA.md](../src/extractors/topk_similar_titles_extractor/SCHEMA.md) · machine: [schemas/topk_similar_titles_extractor_output_v1.json](../schemas/topk_similar_titles_extractor_output_v1.json)

**Полный документ**: [src/extractors/topk_similar_titles_extractor/README.md](../src/extractors/topk_similar_titles_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/topk_similar_titles_extractor_AUDIT_V3_REPORT.md](audit_v3/components/topk_similar_titles_extractor_AUDIT_V3_REPORT.md)

#### EmbeddingShiftIndicatorExtractor (embedding_shift_indicator_extractor)
**Краткое описание**: Косинус между усреднёнными чанками **начала** и **конца** транскрипта (semantic shift). Версия **1.3.0**, CPU-only. Зависит от **TranscriptChunkEmbedder**. **27** фиксированных ключей `features_flat`; **`load_ms`/`compute_ms`** — **NaN** при **`emit_extra_metrics=False`**; **`tp_embshift_chunk_embed_missing_flag`**; канон **`transcripts`[][].chunk_embeddings_relpath`** без обязательного **`transcript_chunks`**.

**Контракт**: [src/extractors/embedding_shift_indicator_extractor/SCHEMA.md](../src/extractors/embedding_shift_indicator_extractor/SCHEMA.md) · machine: [schemas/embedding_shift_indicator_extractor_output_v1.json](../schemas/embedding_shift_indicator_extractor_output_v1.json)

**Полный документ**: [src/extractors/embedding_shift_indicator_extractor/README.md](../src/extractors/embedding_shift_indicator_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md](audit_v3/components/embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md)

#### EmbeddingSourceIdExtractor (embedding_source_id_extractor)
**Краткое описание**: Primary embedding: **`vector_id`** (SHA256 по float32 байтам), privacy-safe **`embedding_source_id`**, фиксированные **13** ключа **`tp_embid_*`**. Версия **1.3.0**, metadata, CPU-only. Зависит от **TitleEmbedder**, **DescriptionEmbedder**, **TranscriptAggregatorExtractor**. Режим **`strict_missing_primary`** управляет fail-fast vs soft empty + **`error`**.

**Контракт**: [src/extractors/embedding_source_id_extractor/SCHEMA.md](../src/extractors/embedding_source_id_extractor/SCHEMA.md) · machine: [schemas/embedding_source_id_extractor_output_v1.json](../schemas/embedding_source_id_extractor_output_v1.json)

**Полный документ**: [src/extractors/embedding_source_id_extractor/README.md](../src/extractors/embedding_source_id_extractor/README.md)

**Отчёт Audit v3**: [docs/audit_v3/components/embedding_source_id_extractor_AUDIT_V3_REPORT.md](audit_v3/components/embedding_source_id_extractor_AUDIT_V3_REPORT.md)

---

## Архитектура и Core

### src/core/main_processor.py
**Краткое описание**: Главный оркестратор TextProcessor. Класс `MainProcessor` управляет списком extractors, последовательно применяет их к документу, поддерживает конфигурацию устройств (CPU/GPU), batch processing с параллелизмом по уровням зависимостей, топологическую сортировку extractors, обработку ошибок и прогресс-репортинг. Реализует registry extractors для ленивой загрузки, поддерживает required/optional extractors, artifacts_dir per-run.

**Расположение**: `TextProcessor/src/core/main_processor.py`

### src/core/base_extractor.py
**Краткое описание**: Базовый интерфейс для всех extractors. Абстрактный класс `BaseExtractor` определяет контракт `extract(doc)` и опциональный `extract_batch(docs)` для batch processing. Свойство `supports_batch` указывает на оптимизированную batch реализацию. Все extractors наследуются от этого класса.

**Расположение**: `TextProcessor/src/core/base_extractor.py`

### src/core/model_registry.py
**Краткое описание**: Реестр моделей для переиспользования между extractors. Обеспечивает единую точку загрузки моделей (sentence transformers, embeddings), кеширование в памяти, управление устройствами (CPU/GPU), интеграцию с `dp_models` для offline моделей. Используется всеми embedding extractors для оптимизации памяти и производительности.

**Расположение**: `TextProcessor/src/core/model_registry.py`

### src/core/metrics.py
**Краткое описание**: Утилиты для вычисления метрик и статистик. Содержит функции для вычисления cosine similarity, агрегации эмбеддингов (mean, max, min pooling), статистик по векторам, нормализации. Используется extractors для вычисления similarity метрик и агрегации.

**Расположение**: `TextProcessor/src/core/metrics.py`

### src/core/path_utils.py
**Краткое описание**: Утилиты для работы с путями и файловой системой. Содержит функции для нормализации путей, работы с artifacts_dir, генерации детерминированных путей для артефактов, атомарного сохранения файлов. Используется extractors для безопасной работы с файловой системой.

**Расположение**: `TextProcessor/src/core/path_utils.py`

### src/core/renderer.py
**Краткое описание**: Рендерер для генерации HTML/JSON визуализаций результатов TextProcessor. Создает human-readable представления extractor results, метрик, эмбеддингов (опционально), статистик. Используется для отладки и визуализации результатов обработки.

**Расположение**: `TextProcessor/src/core/renderer.py`

### src/core/text_utils.py
**Краткое описание**: Утилиты для работы с текстом. Содержит функции для нормализации текста (whitespace, unicode), токенизации, извлечения хэштегов, обработки специальных символов. Используется extractors для предобработки текстовых данных.

**Расположение**: `TextProcessor/src/core/text_utils.py`

---

## Схемы данных

### src/schemas/models.py
**Краткое описание**: Схемы данных для TextProcessor. Определяет структуру `VideoDocument` (входной документ с полями title, description, comments, asr, transcripts), `ExtractorResult` (результат extractor'а с features_flat, metadata, artifacts), типы для метаданных моделей, конфигурации extractors. Используется для валидации входных данных и сериализации результатов.

**Расположение**: `TextProcessor/src/schemas/models.py`

---

## Утилиты

### src/utils/meta_builder.py
**Краткое описание**: Утилиты для работы с метаданными моделей. Содержит функции для канонизации списка используемых моделей (`model_used()`), вычисления детерминированной подписи моделей (`compute_model_signature()`), применения метаданных моделей к мета-словарю (`apply_models_meta()`). Обеспечивает стабильную сортировку и детерминированное хеширование для reproducibility. Интегрирован с общим `DataProcessor/common/meta_builder.py`.

**Расположение**: `TextProcessor/src/utils/meta_builder.py`

### src/utils/validate_emb.py
**Краткое описание**: Утилиты для валидации эмбеддингов. Содержит функции для проверки размерности, нормализации, наличия NaN/Inf значений, соответствия ожидаемым форматам. Используется для отладки и тестирования embedding extractors.

**Расположение**: `TextProcessor/src/utils/validate_emb.py`

---

## Конфигурация

### config/config.py
**Краткое описание**: Парсер конфигурации TextProcessor. Читает настройки из global_config.yaml, генерирует конфигурацию для MainProcessor, извлекает параметры extractors, настройки устройств (CPU/GPU), batch processing параметры. Интегрирован с `DataProcessor/configs/config_parser.py` для единой системы конфигурации.

**Расположение**: `TextProcessor/config/config.py`

---

## Скрипты

### scripts/bench_batch_parallel.py
**Краткое описание**: Бенчмарк для измерения производительности batch processing с параллелизмом. Сравнивает последовательную обработку vs batch processing с CPU parallelism, измеряет ускорение, утилизацию ресурсов, тайминги по уровням зависимостей. Используется для оптимизации производительности.

**Расположение**: `TextProcessor/scripts/bench_batch_parallel.py`

### scripts/bench_titleembedder_batch.py
**Краткое описание**: Микро-бенчмарк для TitleEmbedder batch processing. Сравнивает loop `extract()` vs `extract_batch()` для измерения ускорения от GPU batching. Измеряет latency, throughput, утилизацию GPU/CPU.

**Расположение**: `TextProcessor/scripts/bench_titleembedder_batch.py`

### scripts/smoke_batch.py
**Краткое описание**: Smoke-тест для batch processing. Проверяет корректность batch обработки (эквивалентность результатов `run_batch([doc])` == `run(doc)`), изоляцию данных между документами, обработку ошибок, детерминизм. Используется для валидации batch processing реализации.

**Расположение**: `TextProcessor/scripts/smoke_batch.py`

### scripts/smoke_commentsembedder_batch.py
**Краткое описание**: Smoke-тест для CommentsEmbedder batch processing. Проверяет корректность батчинга комментариев, дедупликацию, политику отбора, сохранение артефактов per-doc.

**Расположение**: `TextProcessor/scripts/smoke_commentsembedder_batch.py`

### scripts/smoke_hashtagembedder_batch.py
**Краткое описание**: Smoke-тест для HashtagEmbedder batch processing. Проверяет корректность батчинга хэштегов переменной длины, агрегацию per-doc, сохранение артефактов.

**Расположение**: `TextProcessor/scripts/smoke_hashtagembedder_batch.py`

### scripts/smoke_transcriptchunkembedder_batch.py
**Краткое описание**: Smoke-тест для TranscriptChunkEmbedder batch processing. Проверяет корректность батчинга чанков транскрипта, маппинг между документами и чанками, сохранение артефактов per-doc.

**Расположение**: `TextProcessor/scripts/smoke_transcriptchunkembedder_batch.py`

### scripts/smoke_each_extractor_audit_v3.py
**Краткое описание**: Audit v3 — поочерёдный прогон **каждого** из **22** экстракторов на `inference.video_document` из `example/text_audit_v3_smoke/scenarios/audit_v3_20_scenarios.json`. Для цели запускается минимальная цепочка зависимостей (как в `MainProcessor`); эмбеддеры на **CPU**; временный union **`DP_MODELS_ROOT`** дополняет corpus-артефакты (`similar_titles`, `semantic_clusters`). Режим **`--all-scenarios`** — все сценарии × 22 экстрактора.

**Расположение**: `TextProcessor/scripts/smoke_each_extractor_audit_v3.py`  
**Команды и критерии**: [`example/text_audit_v3_smoke/scenarios/README.md`](../../../example/text_audit_v3_smoke/scenarios/README.md)

---

## CLI и Entry Points

### run_cli.py
**Краткое описание**: CLI entry point для TextProcessor. Парсит аргументы командной строки, загружает конфигурацию из global_config.yaml, инициализирует MainProcessor, обрабатывает один или несколько документов (single-file и batch mode), сохраняет результаты в result_store. Поддерживает интеграцию с верхним оркестратором DataProcessor через CLI аргументы.

**Расположение**: `TextProcessor/run_cli.py`

### rlp.py
**Краткое описание**: REPL (Read-Eval-Print Loop) для интерактивной работы с TextProcessor. Функция `run_once()` для быстрого запуска обработки одного документа, функция `main()` для интерактивного режима. Используется для разработки и отладки extractors.

**Расположение**: `TextProcessor/rlp.py`

### src/repl.py
**Краткое описание**: Дополнительные утилиты для REPL режима. Содержит вспомогательные функции для интерактивной работы с TextProcessor, загрузки документов, визуализации результатов.

**Расположение**: `TextProcessor/src/repl.py`

---

## Зависимости

### requirements.txt
**Краткое описание**: Список Python зависимостей для TextProcessor. Включает sentence-transformers, torch, numpy, transformers, scikit-learn, faiss-cpu/faiss-gpu (опционально), и другие библиотеки для работы с текстом, эмбеддингами, метриками. Используется для установки окружения через pip.

**Расположение**: `TextProcessor/requirements.txt`

---

## Структура проекта

TextProcessor организован в модульную структуру:

- **`src/core/`**: Основные компоненты (MainProcessor, BaseExtractor, model_registry, metrics, utilities)
- **`src/extractors/`**: Все 22 extractor'а, каждый в отдельной директории с `main.py`, `README.md`, `render.py`
- **`src/schemas/`**: Схемы данных (VideoDocument, ExtractorResult)
- **`src/utils/`**: Вспомогательные утилиты (meta_builder, validate_emb)
- **`config/`**: Конфигурация (config.py)
- **`scripts/`**: Скрипты для тестирования и бенчмарков
- **`docs/`**: Документация (MAIN_INDEX.md, BATCH_PROCESSING_PLAN.md, LAST_FULL_RUN_LOG.md)

---

## Интеграция с DataProcessor

TextProcessor интегрирован в общий пайплайн DataProcessor:

- **Конфигурация**: через `DataProcessor/configs/global_config.yaml` и `config_parser.py`
- **Orchestration**: через `DataProcessor/main.py` с поддержкой batch processing флагов
- **Storage**: результаты сохраняются в per-run result_store (`dp_results/<platform_id>/<video_id>/<run_id>/text_processor/`)
- **State management**: через `DataProcessor/state/` для отслеживания прогресса
- **Models**: через `dp_models` для offline моделей (no-network policy)
- **Artifacts**: sub-artifacts (`.npy` файлы) сохраняются в `text_processor/_artifacts/` per-run

---

## Статистика

- **Всего extractors**: 22
- **Tier-0 (baseline; порядок: tags → lexical / ASR proxy)**: 3
- **Tier-1 (embedding extractors)**: 6
- **Tier-2 (aggregation extractors)**: 5
- **Tier-3 (advanced metrics extractors)**: 8
- **GPU extractors**: 7 (TitleEmbedder, DescriptionEmbedder, HashtagEmbedder, TranscriptChunkEmbedder, CommentsEmbedder, SpeakerTurnEmbeddingsAggregatorExtractor, SemanticTopicExtractor)
- **CPU-only extractors**: 15

