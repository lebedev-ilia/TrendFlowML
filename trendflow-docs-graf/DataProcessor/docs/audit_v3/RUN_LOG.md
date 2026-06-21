# Audit v3 — Run log (dev validation)

Этот файл фиксирует **конкретные прогоны** DataProcessor/компонентов (dev), чтобы:

- воспроизводимо подтверждать, что изменения аудита работают,
- иметь ссылки на артефакты/рендеры для ручной проверки качества,
- фиксировать фактический sampling (пока Segmenter не аудирован финально).

---

## Dev: TextProcessor — изолированный смок 22 экстракторов × сценарии Audit v3 (2026-04-01)

- **Скрипт**: [`TextProcessor/scripts/smoke_each_extractor_audit_v3.py`](../../TextProcessor/scripts/smoke_each_extractor_audit_v3.py)
- **Вход**: `example/text_audit_v3_smoke/scenarios/audit_v3_20_scenarios.json` → только **`inference.video_document`** по индексу сценария.
- **Модель**: `intfloat/multilingual-e5-large` из **`${DP_MODELS_ROOT}`** (ожидается дерево `bundled_models`); эмбеддеры форсятся в **CPU** через **`extractor_params`** скрипта.
- **Corpus-фикстуры**: скрипт собирает временный union-root поверх bundle и генерирует недостающие **`text/similar_titles_v1/embeddings.npy`** и полный минимальный **`text/semantic_clusters_v1/`**, чтобы **`TopKSimilarCorpusTitlesExtractor`** и **`SemanticClusterExtractor`** инициализировались без дыр в артефактах.
- **Режимы**:
  - по умолчанию: **один** сценарий (индекс **0**), **22** прогона (по одному целевому экстрактору + минимальные зависимости);
  - **`--all-scenarios`**: все сценарии из JSON × **22** экстрактора;
  - **`--limit-scenarios N`**, **`--scenario-index K`**, **`--quiet`**, **`--keep-union-root`** — см. docstring скрипта и [`example/text_audit_v3_smoke/scenarios/README.md`](../../../example/text_audit_v3_smoke/scenarios/README.md).
- **Критерий OK**: для целевого класса в payload есть непустой **`results_by_extractor[Class].features_flat`**, нет **`errors_by_extractor[Class]`**, **`status`** не **`error`**.
- **Проверка** (2026-04-01, dev): `TextProcessor/.tp_venv`, `DP_MODELS_ROOT=DataProcessor/dp_models/bundled_models` — все **22** экстрактора на **сценарии 0** OK; режим **`--all-scenarios --limit-scenarios 1`** OK. Полный прогон **20×22** выполняется локально при необходимости (долго из-за повторных загрузок модели по прогонам).
- **Не покрывает**: единый E2E **`run_cli.py`** / Segmenter / живой ASR в одном **`run_id`** — это отдельная запись в этом файле при прогоне.

---

## Dev: TextProcessor `cosine_metrics_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `CosineMetricsExtractor` v**1.3.0**
- **Изменения**: **39** ключей **`features_flat`**; extra (**`load_ms`**, **`compute_ms`**, matrix stats) **всегда** в схеме → **NaN** при **`emit_extra_metrics=False`**; one-hot **`tp_cos_transcript_agg_source_*`**; зеркала **`require_*`**; default **`transcript_source_priority`** **`["whisper", "youtube_auto"]`**; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**. Неизвестный **`comments_mode`**: без исключения (NaN / 0/0 mode flags).
- **Схемы**: `cosine_metrics_extractor_output_v1`, [`TextProcessor/src/extractors/cosine_metrics_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/cosine_metrics_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/cosine_metrics_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/cosine_metrics_extractor_AUDIT_V3_REPORT.md).
- **Проверка** (2026-04-01, dev): `TextProcessor/.tp_venv` — `extract()` на временном `artifacts_dir` с `.npy` (title/description/transcript agg/comments agg/matrix) и синтетическим **`tp_artifacts`**: порядок и состав **`features_flat`** совпадают с `cosine_metrics_extractor_output_v1.json`; **`aggregates`** / **`matrix`** / **`emit_extra_metrics`** (числа vs NaN); неизвестный **`comments_mode`** → флаги режима **0/0**, transcript↔comments mean — **NaN**, без исключения.

### Audit v3 acceptance (cosine_metrics_extractor, smoke)

- ✅ **Schema**: `cosine_metrics_extractor_output_v1` ↔ фиксированные **39** ключей в коде
- ✅ **Extra block**: тайминги и matrix-derived поля — **NaN** при **`emit_extra_metrics=False`**
- ✅ **Transcript source**: one-hot по **`transcript_source_priority`** (проверено: whisper раньше youtube_auto)
- ✅ **Unknown `comments_mode`**: без fail-fast; TC mean — **NaN**
- ⏳ **Полный E2E** через `DataProcessor/main.py` + реальный пайплайн эмбеддингов: не в этой записи (изолированный smoke достаточен для контракта `features_flat`)

---

## Dev: TextProcessor `title_embedding_cluster_entropy_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `TitleEmbeddingClusterEntropyExtractor` v**1.3.0**
- **Изменения**: **24** фиксированных ключа **`features_flat`**; кламп **`top_k_slots`** ≤ **8** + **`tp_titleclent_top_k_slots_requested`** / **`tp_titleclent_top_k_slots_clamped`**; зеркала политик; extra-блок (**dims**, **margin_top2**, **compute_ms**) → **NaN** при **`emit_extra_metrics=False`** или empty; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**; **`entropy_norm`** при **K≤1** = **0.0**.
- **Схемы**: `title_embedding_cluster_entropy_extractor_output_v1`, [`TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md).
- **Проверка** (2026-04-01, dev): `DP_MODELS_ROOT=dp_models/bundled_models`, `TextProcessor/.tp_venv` — совпадение **`features_flat`** ↔ JSON; smoke **clamp** (`requested=12` → **`top_k_slots=8`**); empty doc; **`emit_extra_metrics=False`** → extra **NaN**.

### Audit v3 acceptance (title_embedding_cluster_entropy_extractor, smoke)

- ✅ **Schema**: `title_embedding_cluster_entropy_extractor_output_v1` ↔ **24** ключа в коде
- ✅ **Кламп top‑K** + флаги requested/clamped
- ✅ **Empty semantics** + **`require_title_embedding`** (без регрессии контракта ключей)
- ⏳ **Полный E2E** с ASR + title embedder в одном run — отдельная запись при прогоне

---

## Dev: TextProcessor `title_to_hashtag_cosine_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `TitleToHashtagCosineExtractor` v**1.2.0**
- **Изменения**: **11** фиксированных ключей **`tp_titlehashcos_*`**; убраны legacy **`tp_title_hashtag_cosine_*`** и внутренний гейт **`enabled`** в контракте (`**kwargs` поглощает устаревшие ключи); **`tp_titlehashcos_unsafe_relpath_flag`** vs **`tp_titlehashcos_*_embed_missing_flag`**; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**.
- **Схемы**: `title_to_hashtag_cosine_extractor_output_v1`, [`TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md).
- **Проверка** (2026-04-01, dev): `.tp_venv` — совпадение JSON ↔ код; smoke cosine=1 на коллинеарных векторах; missing hashtag file; unsafe `relpath`.

### Audit v3 acceptance (title_to_hashtag_cosine_extractor, smoke)

- ✅ **Schema**: `title_to_hashtag_cosine_extractor_output_v1` ↔ **11** ключей.
- ✅ **Флаги**: unsafe отделён от missing/bad_file.
- ⏳ **Полный E2E** TextProcessor — отдельная запись при прогоне.

---

## Dev: TextProcessor `semantic_cluster_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `SemanticClusterExtractor` v**1.3.0**
- **Изменения**: **31** ключ **`tp_semclust_*`**; зеркала **`require_*` / `use_faiss` / `emit_extra_metrics`**; one-hot **`tp_semclust_config_primary_*`**; **`_*_present`** = успешная загрузка **`.npy`**; **`tp_semclust_unsafe_relpath_flag`** vs **`tp_semclust_*_embed_missing_flag`**; extra-блок — **NaN** при **`emit_extra_metrics=False`**; **`semantic_cluster_meta.backend`** на всех ветках; **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`**, **`gpu_peak_mb`**; DAG: **`HashtagEmbedder`** в зависимостях оркестратора.
- **Схемы**: `semantic_cluster_extractor_output_v1`, [`TextProcessor/src/extractors/semantic_cluster_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/semantic_cluster_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/semantic_cluster_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/semantic_cluster_extractor_AUDIT_V3_REPORT.md).
- **Проверка** (2026-04-01, dev): smoke с **`artifacts_dir`**, совпадающим с каталогом **`.npy`**, и **`DP_MODELS_ROOT`** — состав **`features_flat`** ↔ JSON; **`emit_extra_metrics`** → числа vs **NaN**.

### Audit v3 acceptance (semantic_cluster_extractor, smoke)

- ✅ **Schema**: `semantic_cluster_extractor_output_v1` ↔ **31** ключ в коде
- ✅ **Meta**: **`backend`** + digest/version на всех ветках **`extract()`**
- ✅ **Диагностика**: unsafe vs embed-missing; presence по факту загрузки файла
- ⏳ **Полный E2E** через `run_cli` + реальные embedder’ы — отдельная запись при прогоне

---

## Dev: TextProcessor `topk_similar_titles_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `TopKSimilarCorpusTitlesExtractor` v**1.3.0**
- **Изменения**: **29** ключей **`tp_topktitles_*`**; **`tp_topktitles_title_embed_missing_flag`** (нет файла / ошибка **`np.load`**); **`corpus`** в **`topk_similar_corpus_titles`** на всех ветках; **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`** после загрузки корпуса в **`__init__`**, **`gpu_peak_mb`**; зафиксирована **приближённость HNSW** (см. `SCHEMA.md`).
- **Схемы**: `topk_similar_titles_extractor_output_v1`, [`TextProcessor/src/extractors/topk_similar_titles_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/topk_similar_titles_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/topk_similar_titles_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/topk_similar_titles_extractor_AUDIT_V3_REPORT.md).
- **Проверка** (dev): `DP_MODELS_ROOT`, корпус **`similar_titles_corpus_v1`**, **`artifacts_dir`** + title **`.npy`** — состав **`features_flat`** ↔ JSON; **`require_title_embedding`** smoke.

### Audit v3 acceptance (topk_similar_titles_extractor, smoke)

- ✅ **Schema**: `topk_similar_titles_extractor_output_v1` ↔ **29** ключей в коде
- ✅ **Corpus meta** на всех ветках **`extract()`**
- ⏳ **Полный E2E** preflight (22 экстрактора) — отдельная запись при прогоне

---

## Dev: TextProcessor `embedding_shift_indicator_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `EmbeddingShiftIndicatorExtractor` v**1.3.0**
- **Изменения**: **27** ключей **`tp_embshift_*`**; **`tp_embshift_chunk_embed_missing_flag`**; **`tp_embshift_emit_extra_metrics_enabled`**; **`load_ms`/`compute_ms`** → **NaN** при **`emit_extra_metrics=False`**; hotfix ветки missing-file (**`sys_after`**); выбор relpath по **`transcripts`[][]** без обязательного **`transcript_chunks`**; **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`**, **`gpu_peak_mb`**.
- **Схемы**: `embedding_shift_indicator_extractor_output_v1`, [`TextProcessor/src/extractors/embedding_shift_indicator_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/embedding_shift_indicator_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md).

### Audit v3 acceptance (embedding_shift_indicator_extractor, smoke)

- ✅ **Schema**: `embedding_shift_indicator_extractor_output_v1` ↔ **27** ключей
- ✅ **Тайминги**: gated **`emit_extra_metrics`**
- ⏳ Полный E2E с ASR + chunk embedder — отдельная запись

---

## Dev: TextProcessor `embedding_source_id_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `EmbeddingSourceIdExtractor` v**1.3.0**
- **Изменения**: **13** фиксированных ключей **`tp_embid_*`** (`tp_embid_strict_missing_primary_enabled`, три диагностических флага); **`strict_missing_primary`** единственно задаёт fail-fast vs soft empty + **`embedding_source_id.error`** после выбора relpath (включая unsafe/missing/load/empty/non-finite); разведены **`model_name`** и **`model_version`** во вложенном dict; верхний уровень **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`**, **`gpu_peak_mb`**.
- **Схемы**: `embedding_source_id_extractor_output_v1`, [`TextProcessor/src/extractors/embedding_source_id_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/embedding_source_id_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/embedding_source_id_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/embedding_source_id_extractor_AUDIT_V3_REPORT.md).

### Audit v3 acceptance (embedding_source_id_extractor, smoke)

- ✅ **Schema**: `embedding_source_id_extractor_output_v1` ↔ **13** ключей
- ✅ **Soft path**: `strict_missing_primary=False` → полный **`features_flat`** + **`error`**
- ⏳ Полный E2E preflight (22 экстрактора) — см. отдельную запись при прогоне

---

## Dev: TextProcessor `embedding_stats_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `EmbeddingStatsExtractor` v**1.2.0**
- **Изменения**: **39** ключей **`features_flat`**; **8** фиксированных `tp_embstats_topvar_*`; кламп **`top_k_slots`** с **`_requested`/`_clamped`**; фиксированные **`tp_embstats_source_used_{whisper,youtube_auto}`**; приоритет источника по умолчанию **`["whisper"]`**; **`emit_extra_metrics`** → **`load_ms`/`compute_ms`** или **NaN**; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**; DAG: **`TranscriptChunkEmbedder`** только; энтропия тем по upstream **`topic_probs`** (in-memory).
- **Схемы**: `embedding_stats_extractor_output_v1`, [`TextProcessor/src/extractors/embedding_stats_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/embedding_stats_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/embedding_stats_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/embedding_stats_extractor_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `semantics_topics_keyphrases` — Audit v3 (2026-04-01)

- **Компонент**: `SemanticTopicExtractor` v**2.1.0**
- **Изменения**: **116** ключей **`features_flat`** (фиксированный порядок: entropy/style‑блок до topic‑слотов, затем kp‑слоты, затем score summary и extra); **8** topic-слотов (id/score/prob) и **16** keyphrase-слотов; кламп **`top_k_slots`** / **`keyphrase_slots`** с **`_requested`** / **`_clamped`**; one-hot **`transcript_source_policy`**; **`emit_extra_metrics`** — блок **`tp_topics_extra_*`** всегда в схеме (**NaN** при выкл. или пропущенной ветке); **`_init_metrics`**, **`gpu_peak_mb`**; lazy **`get_model_with_meta`** в **`extract()`**; верхний уровень **`model_name` / `model_version` / `weights_digest`** (**`null`** при **`enabled=False`** или пустом тексте); сырой список ключевых фраз только в **`result.tp_topics_keyphrases_raw`** (не в **`features_flat`**).
- **Схемы**: `semantics_topics_keyphrases_output_v1`, [`TextProcessor/src/extractors/semantics_topics_keyphrases/SCHEMA.md`](../../TextProcessor/src/extractors/semantics_topics_keyphrases/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/semantics_topics_keyphrases_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/semantics_topics_keyphrases_AUDIT_V3_REPORT.md).
- **Примечание**: полный smoke — **`DP_MODELS_ROOT`**, taxonomy + encode.

---

## Dev: TextProcessor `embedding_pair_topk_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `EmbeddingPairTopKExtractor` v**1.3.0**
- **Изменения**: **69** ключей **`features_flat`**; **8** фиксированных слотов; кламп **`top_k_slots`** ≤ **8**; **`emit_extra_metrics`** всегда присутствующий блок (NaN при выкл.); **`_init_metrics`**, **`gpu_peak_mb`**; **`model_name`/`weights_digest`** = **null**.
- **Схемы**: `embedding_pair_topk_extractor_output_v1`, [`TextProcessor/src/extractors/embedding_pair_topk_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/embedding_pair_topk_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/embedding_pair_topk_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/embedding_pair_topk_extractor_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `qa_embedding_pairs_extractor` — Audit v3 (2026-04-01)

- **Компонент**: `QAEmbeddingPairsExtractor` v**1.3.0**
- **Изменения**: **34** ключа **`features_flat`**; **`emit_extra_metrics`** — строго **NaN** для rate/centroid при выкл.; valid empty + extra → **0.0** `/` **NaN** по правилам SCHEMA; **`tp_qa_max_chars_per_comment`** всегда в шаблоне; **`model_name`**, **`model_version`**, **`weights_digest`** на верхнем уровне; **`_init_metrics`**, **`gpu_peak_mb`**; default **`intfloat/multilingual-e5-large`** (**`global_config.yaml`**).
- **Схемы**: `qa_embedding_pairs_extractor_output_v1`, [`TextProcessor/src/extractors/qa_embedding_pairs_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/qa_embedding_pairs_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `comments_aggregator` — Audit v3 (2026-04-01)

- **Компонент**: `CommentsAggregationExtractor` v**1.3.0**
- **Изменения**: **39** стабильных ключей в **`features_flat`** (три семейства префиксов); **`emit_extra_metrics`** → **`tp_commentsagg_agg_*_ms`** или **NaN**; **`dp_models.resolve`** в **`__init__`**; верхний уровень **`model_name`**, **`model_version`**, **`weights_digest`**; **`_gpu_peak_mb`**; пустая ветка с тем же legacy-набором, что и успех.
- **Схемы**: `comments_aggregator_output_v1`, [`TextProcessor/src/extractors/comments_aggregator/SCHEMA.md`](../../TextProcessor/src/extractors/comments_aggregator/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/comments_aggregator_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/comments_aggregator_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `transcript_aggregator` — Audit v3 (2026-04-01)

- **Компонент**: `TranscriptAggregatorExtractor` v**1.3.0**
- **Изменения**: **19** ключей `tp_tragg_*`; **9** extra при **`emit_extra_metrics=False`** → **NaN**; std-слоты **NaN** при **`compute_std=False`**; **`dp_models.resolve`** в **`__init__`** (без inference); default **`intfloat/multilingual-e5-large`**; **`model_name`**, **`weights_digest`** на верхнем уровне; **`_init_metrics`**, **`gpu_peak_mb`**.
- **Схемы**: `transcript_aggregator_output_v1`, [`TextProcessor/src/extractors/transcript_aggregator/SCHEMA.md`](../../TextProcessor/src/extractors/transcript_aggregator/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `speaker_turn_embeddings_aggregator` — Audit v3 (2026-04-01)

- **Компонент**: `SpeakerTurnEmbeddingsAggregatorExtractor` v**1.3.0**
- **Изменения**: **17** стабильных ключей `tp_spkemb_*`; **`emit_extra_metrics=False`** → **NaN** для 5 tuning-полей; **`get_model_with_meta`** + **`_init_metrics`**; **`gpu_peak_mb`**; **`model_name`** / **`weights_digest`** на верхнем уровне ответа; default **`intfloat/multilingual-e5-large`**.
- **Схемы**: `speaker_turn_embeddings_aggregator_output_v1`, [`TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md`](../../TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md).
- **Примечание**: полный smoke — **DP_MODELS_ROOT**, **ASR** + **speaker diarization** (см. preflight §4).

---

## Dev: TextProcessor `comments_embedder` — Audit v3 (2026-04-01)

- **Компонент**: `CommentsEmbedder` v**1.3.0**
- **Изменения**: исправлен **`return`** в **`extract()`** на успешной ветке; **`features_flat`** — **18** ключей; **`emit_extra_metrics=False`** → **NaN** для 10 diagnostic/timing полей; в **`extract_batch`** при **`emit_extra_metrics=True`** **`tp_commentsemb_cache_hit`** = **NaN**; **`encode_ms`** / **`timings_s.encode`** — доля от общего batch-encode по числу комментариев документа; default **`model_name`** = **`intfloat/multilingual-e5-large`**; **`gpu_peak_mb`** из снимков GPU.
- **Схемы**: `comments_embedder_output_v1`, [`TextProcessor/src/extractors/comments_embedder/SCHEMA.md`](../../TextProcessor/src/extractors/comments_embedder/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md).
- **Примечание**: smoke с реальным encode — `DP_MODELS_ROOT` и локальная модель.

---

## Dev: TextProcessor `transcript_chunk_embedder` — Audit v3 docs (2026-04-01)

- **Компонент**: `TranscriptChunkEmbedder` v**1.3.0**
- **Изменения**: стабильные **16** ключей `tp_tchunk_*` в `features_flat` на всех ветках; **`extract_batch`** согласован с **`extract`** по **`emit_confidence_metrics`** / **`emit_extra_metrics`** (при выключенных флагах — **0 / NaN**, ключи сохраняются); в **`result`** всегда **`model_name`**, **`model_version`**, **`weights_digest`**.
- **Схемы**: `transcript_chunk_embedder_output_v1`, [`TextProcessor/src/extractors/transcript_chunk_embedder/SCHEMA.md`](../../TextProcessor/src/extractors/transcript_chunk_embedder/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md).
- **Примечание**: полный smoke encode требует `DP_MODELS_ROOT` и локальной sentence-transformers модели; порядок ключей можно сверить с `_build_features_flat` в `main.py` или с JSON-схемой.

---

## Dev: TextProcessor `hashtag_embedder` — Audit v3 (2026-04-01)

- **Компонент**: `HashtagEmbedder` v**1.2.0**
- **Изменения**: default **`strict_missing_hashtags=false`** (конфиг **`require_hashtags: false`** соблюдается); **`extract_batch`** согласован с **`extract`** при **`require_hashtags`**; **`model_name`** / **`weights_digest`** в пустых ветках **`result`**.
- **Схемы**: `hashtag_embedder_output_v1` (23 keys), [`TextProcessor/src/extractors/hashtag_embedder/SCHEMA.md`](../../TextProcessor/src/extractors/hashtag_embedder/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/hashtag_embedder_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/hashtag_embedder_AUDIT_V3_REPORT.md).

---

## Dev: TextProcessor `description_embedder` — Audit v3 docs (2026-04-01)

- **Компонент**: `DescriptionEmbedder` v**1.2.0**
- **Схемы**: `description_embedder_output_v1` (19 keys в `features_flat`), [`TextProcessor/src/extractors/description_embedder/SCHEMA.md`](../../TextProcessor/src/extractors/description_embedder/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/description_embedder_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/description_embedder_AUDIT_V3_REPORT.md).
- **Примечание**: smoke encode требует `DP_MODELS_ROOT`, модели и `shared_tokenizer_v1`; ключи сверены с `main.py`.

---

## Dev: TextProcessor `title_embedder` — Audit v3 docs (2026-04-01)

- **Компонент**: `TitleEmbedder` v**1.2.0**
- **Схемы**: `title_embedder_output_v1` (16 keys в `features_flat`), [`TextProcessor/src/extractors/title_embedder/SCHEMA.md`](../../TextProcessor/src/extractors/title_embedder/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/title_embedder_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/title_embedder_AUDIT_V3_REPORT.md).
- **Примечание**: smoke encode требует `DP_MODELS_ROOT` и локальной модели; контракт ключей сверен с `main.py`.

---

## Dev: TextProcessor `asr_text_proxy_audio_features` — Audit v3 (2026-04-01)

- **Компонент**: `ASRTextProxyExtractor` v**1.2.0**
- **Изменения**: `require_asr_text` / `strict_document_duration`; duration fallback из payload + флаг деградации; token-decode path достижим при пустых `segments`; `tp_asrproxy_token_decode_failed_flag`; `tp_asrproxy_speech_rate_wpm_ratio_to_baseline`; флаги политик в `features_flat`.
- **Схемы**: `asr_text_proxy_audio_features_output_v1` (37 keys), [`TextProcessor/src/extractors/asr_text_proxy_audio_features/SCHEMA.md`](../../TextProcessor/src/extractors/asr_text_proxy_audio_features/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/asr_text_proxy_audio_features_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/asr_text_proxy_audio_features_AUDIT_V3_REPORT.md).
- **Проверка**: `.tp_venv` — совпадение ключей schema ↔ `features_flat`; smoke `strict_document_duration`, `require_asr_text`, payload duration.

---

## Dev: TextProcessor `lexico_static_features` — Audit v3 baseline (2026-04-01)

- **Компонент**: `LexicalStatsExtractor` v**1.2.0**
- **Изменения**: default `enable_emoji=true`, `emoji_policy=optional` (NaN при отсутствии пакета `emoji`); `require_transcript` opt-in fail-fast; канон full audit: **`transcript_source_policy=asr_only`** (см. preflight).
- **Схемы**: `lexico_static_features_output_v1`, [`TextProcessor/src/extractors/lexico_static_features/SCHEMA.md`](../../TextProcessor/src/extractors/lexico_static_features/SCHEMA.md)
- **Проверка**: совпадение ключей schema ↔ `features_flat` (67 keys); `require_transcript=true` падает на `fixtures/doc_basic_no_asr.json`.

---

## Dev: TextProcessor `tags_extractor` — smoke после Audit v3 (2026-04-01)

- **Компонент**: `TagsExtractor` v**1.2.0**
- **Проверка**: импорт под `DataProcessor/TextProcessor/.tp_venv`, `extract()` на `fixtures/doc_tags_basic.json` + merge `hashtags` из JSON (`#api`, `ExtraTag` → уникальные `casefold` в `doc.hashtags`).
- **Контракт**: `tags_extractor_output_v1`, human [`TextProcessor/src/extractors/tags_extractor/SCHEMA.md`](../../TextProcessor/src/extractors/tags_extractor/SCHEMA.md), отчёт [`TextProcessor/docs/audit_v3/components/tags_extractor_AUDIT_V3_REPORT.md`](../../TextProcessor/docs/audit_v3/components/tags_extractor_AUDIT_V3_REPORT.md).
- **Полный E2E** (Segmenter → ASR → TextProcessor smoke из preflight): не запускался в этой записи.

---

## Run: `youtube/video1_fixed/audit3_audio_smoke_pack1` — Segmenter + AudioProcessor smoke (OK, no-audio empty)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video1_fixed.mp4`
- **Платформа / video_id / run_id**: `youtube / video1_fixed / audit3_audio_smoke_pack1`
- **Компоненты**: `Segmenter` → `AudioProcessor`
- **Набор extractors (requested)**: `clap, tempo, loudness, spectral, quality, mfcc, mel, chroma`

### Ключевой результат

- `Segmenter.extract_audio`: `audio_present=false` (видео без audio stream) → записан `frames_dir/audio/segments.json` с `audio_present=false`.
- `AudioProcessor`: корректно записывает **empty NPZ artifacts** для всех requested компонентов (без падения и без “missing result” ошибок в meta).

### Команда запуска (факт)

```bash
python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том/TrendFlowML/example/example_videos/video1_fixed.mp4" \
  --output "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/_frames_audit3_audio" \
  --rs-base "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results" \
  --platform-id youtube \
  --video-id video1_fixed \
  --run-id audit3_audio_smoke_pack1 \
  --sampling-policy-version v1 \
  --dataprocessor-version audit3_dev \
  --visual-cfg-path "/media/ilya/Новый том/TrendFlowML/configs/visual_triton_baseline_gpu_local.yaml" \
  --run-audio \
  --audio-device auto \
  --audio-extractors "clap,tempo,loudness,spectral,quality,mfcc,mel,chroma" \
  --no-run-visual
```

### Артефакты

- frames_dir: `DataProcessor/dp_results/_frames_audit3_audio/video1_fixed/`
  - `audio/segments.json` содержит `audio_present=false`
- result_store:
  - `DataProcessor/dp_results/youtube/video1_fixed/audit3_audio_smoke_pack1/manifest.json`
  - `DataProcessor/dp_results/youtube/video1_fixed/audit3_audio_smoke_pack1/<component>/*.npz` (status=`empty`)

## Run: `youtube/video2_fixed/audit3_audio_smoke_pack2` — Segmenter + AudioProcessor smoke (OK, no-audio empty)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video2_fixed.mp4`
- **Платформа / video_id / run_id**: `youtube / video2_fixed / audit3_audio_smoke_pack2`
- **Компоненты**: `Segmenter` → `AudioProcessor`
- **Набор extractors (requested)**: `clap, tempo, loudness, spectral, asr`
- **Ключевой результат**: `audio_present=false` → audited empty semantics работают (аналогично pack1).

## Run: `youtube/video3_fixed/audit3_audio_smoke_pack3b` — Segmenter + AudioProcessor smoke (OK, no-audio empty, render OK)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video3_fixed.mp4`
- **Платформа / video_id / run_id**: `youtube / video3_fixed / audit3_audio_smoke_pack3b`
- **Компоненты**: `Segmenter` → `AudioProcessor`
- **Набор extractors (requested)**: `tempo, loudness`

### Ключевой результат

- `audio_present=false` → NPZ статус `empty`.
- Рендер для empty артефактов не падает на `int(NaN)` (фикс в `tempo_extractor/render.py` и `loudness_extractor/render.py`).

---

## Run: `youtube/video1/audit3_audio_present_pack1b` — Segmenter + AudioProcessor validation (OK, audio present)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / video1 / audit3_audio_present_pack1b`
- **Компоненты**: `Segmenter` → `AudioProcessor`
- **Набор extractors (requested)**: `clap, tempo, loudness, asr`

### Ключевой результат

- `Segmenter.extract_audio`: `audio_present=true`, пишет `frames_dir/audio/audio.wav` и `frames_dir/audio/segments.json` (`schema_version=audio_segments_v1`).
- `AudioProcessor` успешно извлекает Tier‑0 extractors + ASR в offline режиме (ModelManager‑only), пишет NPZ + manifest.

### Замечания

- `spectral` намеренно **не включали** в этом запуске, потому что он требует явного включения feature flags (`--spectral-enable-basic-features` и т.п.), а DataProcessor entrypoint пока не пробрасывает эти флаги напрямую (будет решено на этапе аудита `spectral_extractor`).
- tempo: наблюдается warning от librosa (`n_fft=2048 is too large for input signal of length=1`) — требует отдельного разбора (скорее всего, сверхкороткий сегмент/сэмпл‑округление).

---

## Run: `youtube/video3_fixed/audit3_loudness_schema_check_empty` — loudness schema rollout (OK, known schema)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video3_fixed.mp4` (no audio stream)
- **Платформа / video_id / run_id**: `youtube / video3_fixed / audit3_loudness_schema_check_empty`
- **Компоненты**: `Segmenter` → `AudioProcessor(loudness)`

### Ключевой результат

- `schema_version` для loudness переведён на **per-extractor**: `loudness_extractor_npz_v1`
- machine schema размещена в: `DataProcessor/AudioProcessor/schemas/loudness_extractor_npz_v1.json`
- валидатор (`validate_npz`) теперь ищет схемы в:
  - `VisualProcessor/schemas/`
  - `AudioProcessor/schemas/`
  - `TextProcessor/schemas/`
- подтверждено: `validate_npz(..., require_known_schema=True)` → `ok=True` для loudness артефакта

## Run: `youtube/test_1_audit_3/test_1_audit_3` — `core_clip` smoke (OK)

- **Дата**: `2026-02-14`
- **Видео**: `/media/ilya/Новый том1/TrendFlowML/example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / test_1_audit_3 / test_1_audit_3`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_clip)`

### Команда запуска (факт)

Запускался верхний entrypoint:

```bash
python3 main.py \
  --video-path "/media/ilya/Новый том1/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config configs/global_config.yaml \
  --run-audio \
  --platform-id youtube \
  --video-id test_1_audit_3 \
  --run-id test_1_audit_3 \
  --output-dir "/media/ilya/Новый том1/TrendFlowML/DataProcessor/dp_results"
```

### Конфиг (важный фрагмент)

Фрагмент `DataProcessor/configs/global_config.yaml` (Audit v3):

- `visual.inline_config.core_clip.runtime="triton"`
- **ModelManager-only specs**:
  - `triton_image_model_spec: "clip_image_224_triton"`
  - `triton_text_model_spec: "clip_text_triton"`
- `batch_size=16`
- render включён (dev):
  - `enable_render=true`, `enable_html_render=true`

Также важно (Audit v3 infra):

- `visual.inline_config.global.triton_http_url: "http://localhost:8000"`
  - VisualProcessor экспортирует это значение в env subprocess’ов как `TRITON_HTTP_URL` (если env не задан вручную).
- VisualProcessor также выставляет `DP_MODELS_ROOT=<DataProcessor>/dp_models/bundled_models` в env subprocess’ов (если env не задан вручную),
  чтобы работали bundled assets (например Places365 категории) и кеш text-embeddings.

### Результаты / артефакты

- **Manifest**:  
  `dp_results/youtube/test_1_audit_3/test_1_audit_3/manifest.json`
- **NPZ (source-of-truth)**:  
  `dp_results/youtube/test_1_audit_3/test_1_audit_3/core_clip/embeddings.npz`
  - `schema_version=core_clip_npz_v2`
  - `producer_version=2.1`
  - `models_used=[clip_image_224_triton, clip_text_triton]`
  - legacy top-level scalars (`version/created_at/model_name/total_frames`) **отсутствуют** (только `meta.*`)
- **Render (dev-only)**:
  - `dp_results/.../core_clip/_render/render_context.json`
  - `dp_results/.../core_clip/_render/render.html`

### Быстрая валидация артефакта (ключевые проверки)

- **Keys присутствуют**: `frame_indices`, `times_s`, `frame_embeddings`, prompt-наборы, а также backend-proxy ключи:
  - `consecutive_cosine_prev`
  - `places365_topk_indices/scores`, `places365_video_topk_indices/scores`
  - `*_scores` для prompt-наборов (`shot_quality`, `scene_*`, `cut_detection_transition`, `popularity_topic`)
- **Time-axis alignment**: `times_s` соответствует `union_timestamps_sec[frame_indices]` (макс. ошибка ~ `1e-6` из-за float32).

### Sampling (проверка политики)

Фактический sampling для этого короткого видео (`duration≈28.8s`, `fps=30`):

- `union_frames = 115`
- разности `diff(frame_indices)` = `{7,8}` (среднее ~ `7.56`)
- эффективный `gap_sec≈0.252` → `rate_fps≈3.97`

**Важно**: это **приближает** требование “stride=7” для `duration<=120s`, но сейчас Segmenter фактически реализует **gap-based** выборку (`target_gap_sec≈0.25`), а не строго stride=7.
Финальное выравнивание политики/реализации делаем при аудите Segmenter (в конце Audit v3).

## Run: `youtube/test_3_audit_3/test_3_audit_3` — `core_depth_midas` (OK, schema v2)

- **Дата**: `2026-02-14`
- **Видео**: `/media/ilya/Новый том1/TrendFlowML/example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / test_3_audit_3 / test_3_audit_3`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_clip, core_depth_midas)`

### Ключевой результат (что хотели подтвердить)

- `core_depth_midas` пишет `depth.npz` в **`core_depth_midas_npz_v2`** без legacy top-level scalar keys.
- При использовании `triton_model_spec: "midas_256_triton"`:
  - `meta.models_used[].model_name == "midas_256_triton"` (identity через ModelManager),
  - дополнительно пишем `meta.triton_model_spec="midas_256_triton"` и `meta.triton_model_name="midas_256"` (имя модели в Triton repo).

### Артефакты / рендеры

- NPZ (source-of-truth):  
  `dp_results/youtube/test_3_audit_3/test_3_audit_3/core_depth_midas/depth.npz`
- Render (dev-only):
  - `dp_results/.../core_depth_midas/_render/render_context.json`
  - `dp_results/.../core_depth_midas/_render/render.html`

## Run: `youtube/test_4_audit_3/test_4_audit_3` — `core_depth_midas` (OK, backend proxies, schema v3)

- **Дата**: `2026-02-14`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_clip, core_depth_midas)`
- **Артефакт**:
  - `dp_results/youtube/test_4_audit_3/test_4_audit_3/core_depth_midas/depth.npz`
  - `schema_version=core_depth_midas_npz_v3`, `producer_version=2.2`
- **Backend-friendly outputs (проверено в NPZ)**:
  - `depth_maps_norm` (0..1)
  - `depth_complexity_score`
  - `foreground_background_separation_proxy`
  - `preview_*` (K=10 равномерных depth карт для UI)

## Run: `youtube/test_5_audit_3/test_5_audit_3` — `core_optical_flow` (OK, backend previews, schema v3)

- **Дата**: `2026-02-14`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_clip, core_depth_midas, core_optical_flow)`
- **Артефакт**:
  - `dp_results/youtube/test_5_audit_3/test_5_audit_3/core_optical_flow/flow.npz`
  - `schema_version=core_optical_flow_npz_v3`, `producer_version=2.2`
- **Backend-friendly outputs (проверено в NPZ)**:
  - `preview_pair_pos` (K=10)
  - `preview_flow_mag_map_norm (K,64,64)` в `[0,1]`
  - `meta.backend_proxy_version="core_optical_flow_backend_proxy_v1"`



## Run: `youtube/audit3_cod_smoke_2/audit3_cod_smoke_2` — `core_object_detections` (OK, schema v2, render OK)

- **Дата**: `2026-02-16`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_cod_smoke_2 / audit3_cod_smoke_2`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections)`

### Команда запуска (факт)

Запускался верхний entrypoint:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том2/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_object_detections.yaml \
  --platform-id youtube \
  --video-id audit3_cod_smoke_2 \
  --run-id audit3_cod_smoke_2 \
  --output-dir "/media/ilya/Новый том2/TrendFlowML/DataProcessor/dp_results"
```

### Результаты / артефакты

- **Manifest**:  
  `dp_results/youtube/audit3_cod_smoke_2/audit3_cod_smoke_2/manifest.json`
- **NPZ (source-of-truth)**:  
  `dp_results/youtube/audit3_cod_smoke_2/audit3_cod_smoke_2/core_object_detections/detections.npz`
  - `schema_version=core_object_detections_npz_v2`
  - `producer_version=2.2`
  - ключи v2 присутствуют: `boxes_norm/centers_norm/areas_frac`, агрегаты (`det_count/person_count/*_area_frac`), `meta_json`
  - `class_names` имеет длину 41 (`0:person` … `40:food_item`)
- **Render (dev-only)**:
  - `dp_results/.../core_object_detections/_render/render_context.json`
  - `dp_results/.../core_object_detections/_render/render.html`

### Sampling (факт)

Для этого короткого видео (`duration≈28.8s`, `fps=30`):

- `union_frames = 115`
- фактический `rate_fps≈4.0` (`target_gap_sec≈0.25`)

## Run: `youtube/audit3_cod_ocr_smoke_1/audit3_cod_ocr_smoke_1` — `ocr_extractor` (OK, schema v2, meta_json, hard tesseract)

- **Дата**: `2026-02-16`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_cod_ocr_smoke_1 / audit3_cod_ocr_smoke_1`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections, ocr_extractor)`

### Команда запуска (факт)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том2/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_object_detections_and_ocr.yaml \
  --platform-id youtube \
  --video-id audit3_cod_ocr_smoke_1 \
  --run-id audit3_cod_ocr_smoke_1 \
  --output-dir "/media/ilya/Новый том2/TrendFlowML/DataProcessor/dp_results"
```

### Результаты / артефакты

- **Manifest**:  
  `dp_results/youtube/audit3_cod_ocr_smoke_1/audit3_cod_ocr_smoke_1/manifest.json`
- **NPZ (source-of-truth)**:
  - `core_object_detections`:
    - `dp_results/.../core_object_detections/detections.npz`
    - `schema_version=core_object_detections_npz_v2`, `producer_version=2.2`
  - `ocr_extractor`:
    - `dp_results/.../ocr_extractor/ocr.npz`
    - `schema_version=ocr_extractor_npz_v2`, `producer_version=0.2`
    - ключи присутствуют: `frame_indices`, `times_s`, `ocr_raw`, `meta`, `meta_json`
    - `meta.retain_raw_ocr_text=true` (включено специально для dev-инспекции)
    - `ocr_raw` содержит 87 OCR строк (list[dict] в object array)
- **Render (dev-only)**:
  - `dp_results/.../ocr_extractor/_render/render_context.json`
  - `dp_results/.../ocr_extractor/_render/render.html`

### Sampling (факт)

Для этого короткого видео (`duration≈28.8s`, `fps=30`):

- `union_frames = 115`
- фактический `rate_fps≈4.0` (`target_gap_sec≈0.25`)
- Segmenter alignment log: `ocr_extractor ⊆ core_object_detections` (OCR использует shared sampling group)

## Run: `youtube/audit3_brand_semantics_smoke_1/audit3_brand_semantics_smoke_1` — `brand_semantics` (contract smoke, FAIL-FAST: empty DB)

- **Дата**: `2026-02-17`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_brand_semantics_smoke_1 / audit3_brand_semantics_smoke_1`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections, brand_semantics)`

### Команда запуска (факт)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_object_detections_and_brand_semantics.yaml \
  --platform-id youtube \
  --video-id audit3_brand_semantics_smoke_1 \
  --run-id audit3_brand_semantics_smoke_1 \
  --output-dir "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results"
```

### Конфиг

- Visual cfg: `DataProcessor/configs/audit_v3/visual_core_object_detections_and_brand_semantics_only.yaml`
- Profile: `DataProcessor/configs/audit_v3/profile_core_object_detections_and_brand_semantics.yaml`
- `brand_semantics.embedding_service_url="http://localhost:8005"`

### Результаты

- `core_object_detections` — **OK**:
  - NPZ: `dp_results/youtube/audit3_brand_semantics_smoke_1/audit3_brand_semantics_smoke_1/core_object_detections/detections.npz`
  - Render: `.../core_object_detections/_render/render_context.json` + `render.html`
- `brand_semantics` — **FAIL-FAST (ожидаемо по контракту)**:
  - Ошибка: `Embedding Service category 'brand' has 0 labels (fail-fast)`
  - Причина: в Embedding Service категория `brand` пустая (`GET /categories/brand/labels` → `count=0`)

### Что нужно для OK прогона (не часть smoke_1)

Чтобы `brand_semantics` отработал **OK** и записал артефакт `brand_semantics_npz_v2`, нужно:

1) Поднять Triton (Embedding Service использует CLIP для `/search` по изображениям):
   - `TRITON_BASE_URL=http://localhost:8000`
2) Засеять категорию `brand` хотя бы несколькими labels:
   - скрипт: `VisualProcessor/core/model_process/core_identity/brand_semantics/sync_known_brands_to_embedding_service.py`
   - после seeding: `GET /categories/brand/labels` должен вернуть `count > 0`
3) Перезапустить прогон (smoke_2), ожидаемые артефакты:
   - NPZ: `.../brand_semantics/brand_semantics.npz` (`schema_version=brand_semantics_npz_v2`, `producer_version=0.2`)
   - Render (dev-only): `.../brand_semantics/_render/render_context.json` + offline `render.html` + `_render/assets/`

## Run: `youtube/audit3_content_domain_smoke_1/audit3_content_domain_smoke_1` — `content_domain` (contract smoke, expected OK if Triton+domain DB present)

- **Дата**: `2026-02-17`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_content_domain_smoke_1 / audit3_content_domain_smoke_1`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_clip, content_domain)`

### Команда запуска (рекомендовано)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_clip_and_content_domain.yaml \
  --platform-id youtube \
  --video-id audit3_content_domain_smoke_1 \
  --run-id audit3_content_domain_smoke_1 \
  --output-dir "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results"
```

### Preconditions (чтобы run был OK)

- **Triton** доступен и `TRITON_HTTP_URL` настроен (или берётся из profile/global):
  - ожидается `http://localhost:8000`
- **Offline domain DB** присутствует:
  - `DataProcessor/dp_models/bundled_models/semantics/content_domain/v1/manifest.json`
  - `DataProcessor/dp_models/bundled_models/semantics/content_domain/v1/domains.jsonl`

### Ожидаемые артефакты

- `core_clip`:
  - NPZ: `.../core_clip/embeddings.npz` (`schema_version=core_clip_npz_v2`)
- `content_domain`:
  - NPZ: `.../content_domain/content_domain.npz`
    - `schema_version=content_domain_npz_v2`, `producer_version=0.2`
    - `meta.db_digest` присутствует (reproducibility)
    - `meta_json` присутствует (cross-venv safe)
  - Render (dev-only):
    - `.../content_domain/_render/render_context.json`
    - `.../content_domain/_render/render.html` (offline, **без CDN**)

### Expected fail-fast cases (по контракту)

- Если domain DB отсутствует/пустая/битая → **error** (run должен упасть, а не писать empty).
- Если Triton недоступен → **error**.

## Run: `youtube/audit3_ppocr_rec_smoke_1/audit3_ppocr_rec_smoke_1` — `ocr_extractor` (`ppocr_rec_onnx`) (OK, schema v2, meta_json, ONNX pack OK)

- **Дата**: `2026-02-16`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_ppocr_rec_smoke_1 / audit3_ppocr_rec_smoke_1`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections, ocr_extractor)`

### Команда запуска (факт)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том2/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_object_detections_and_ocr.yaml \
  --platform-id youtube \
  --video-id audit3_ppocr_rec_smoke_1 \
  --run-id audit3_ppocr_rec_smoke_1 \
  --output-dir "/media/ilya/Новый том2/TrendFlowML/DataProcessor/dp_results"
```

### Ключевой результат (что хотели подтвердить)

- `ocr_extractor` работает на **`engine=ppocr_rec_onnx`** (не tesseract) и загружает локальный ONNX pack через `dp_models.ModelManager`.
- Для Audit v3 профиля важна явная настройка `engine/rec_model_spec` в
  `DataProcessor/configs/audit_v3/visual_core_object_detections_and_ocr_only.yaml`,
  т.к. дефолт CLI у компонента = `engine=tesseract`.

### Результаты / артефакты

- **Manifest**:  
  `dp_results/youtube/audit3_ppocr_rec_smoke_1/audit3_ppocr_rec_smoke_1/manifest.json`
- **NPZ (source-of-truth)**:
  - `core_object_detections`:
    - `dp_results/.../core_object_detections/detections.npz`
    - `schema_version=core_object_detections_npz_v2`, `producer_version=2.2`
  - `ocr_extractor`:
    - `dp_results/.../ocr_extractor/ocr.npz`
    - `schema_version=ocr_extractor_npz_v2`, `producer_version=0.2`
    - ключи присутствуют: `frame_indices`, `times_s`, `ocr_raw`, `meta`, `meta_json`
    - `meta.engine="ppocr_rec_onnx"`, `meta.rec_model_spec="ppocr_rec_onnx_v1_inprocess"`
    - `meta.retain_raw_ocr_text=true` (включено специально для dev-инспекции)
    - `ocr_raw` содержит 90 OCR строк (list[dict] в object array)
- **Render (dev-only)**:
  - `dp_results/.../ocr_extractor/_render/render_context.json`
  - `dp_results/.../ocr_extractor/_render/render.html`

### Валидация контракта (быстро)

- **Time-axis alignment**:
  - `max |times_s - union_timestamps_sec[frame_indices]| ≈ 9e-7`
  - `ocr_extractor.frame_indices == core_object_detections.frame_indices` (строгая проверка включена)

### Sampling (факт)

Для этого короткого видео (`duration≈28.8s`, `fps=30`):

- `union_frames = 115`
- фактический `rate_fps≈4.0` (`target_gap_sec≈0.25`)
- Segmenter alignment log: `ocr_extractor ⊆ core_object_detections` (OCR использует shared sampling group)

## Run: `youtube/video3/audit3_asr_langcode_v1b` — `asr_extractor` (Audit v3 complete, OK, schema v1, token contract strict)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video3.mp4`
- **Платформа / video_id / run_id**: `youtube / video3 / audit3_asr_langcode_v1b`
- **Компоненты**: `Segmenter` → `AudioProcessor(asr)`

### Ключевой результат (Audit v3 завершён)

- `asr_extractor` переведён на **per-extractor schema**: `asr_extractor_npz_v1`
- **Строгий token contract**: `token_ids_by_segment` всегда через `shared_tokenizer_v1` (no fallback на Whisper tokens)
- **Privacy-safe outputs**: `lang_code_by_segment`, `lang_conf_by_segment`, `segment_quality_by_segment` (числовые метрики)
- **Language distribution**: по `lang_code` (analytics), не по сырому `lang_id`
- **Feature gating**: все aggregates явно контролируются через feature flags
- **Schema validation**: проходит runtime validation через `vp_schema_v1` validator

### Команда запуска (факт)

```bash
python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том/TrendFlowML/example/example_videos/video3.mp4" \
  --output "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/_frames_audit3_asr_lang" \
  --rs-base "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results" \
  --platform-id youtube \
  --video-id video3 \
  --run-id audit3_asr_langcode_v1b \
  --sampling-policy-version v1 \
  --dataprocessor-version audit3_dev \
  --visual-cfg-path "/media/ilya/Новый том/TrendFlowML/configs/visual_triton_baseline_gpu_local.yaml" \
  --run-audio \
  --audio-device auto \
  --audio-extractors "asr" \
  --asr-enable-token-sequences \
  --no-run-visual
```

### Артефакты

- **NPZ (source-of-truth)**:
  - `DataProcessor/dp_results/youtube/video3/audit3_asr_langcode_v1b/asr_extractor/asr_extractor_features.npz`
  - `schema_version=asr_extractor_npz_v1`
  - ключи присутствуют:
    - `token_ids_by_segment` (object array, 1D, каждый элемент = int32[T_i])
    - `lang_code_by_segment` (object array, строки: "en", "ru", "", ...)
    - `lang_conf_by_segment` (float32[], confidence или NaN)
    - `segment_quality_by_segment` (object array, dicts с `avg_logprob`, `compression_ratio`, etc.)
    - `lang_distribution` (object, dict по lang_code)
    - `segment_start_sec`, `segment_end_sec`, `segment_center_sec` (analytics)
    - feature-gated aggregates (token_counts, token_total, token_density_per_sec, speech_rate_wpm, etc.)
- **Schema validation**: ✅ проходит (`validate_npz` с `require_known_schema=True`)
- **Render (dev-only)**:
  - `.../asr_extractor/_render/render_context.json`
  - `.../asr_extractor/_render/render.html` (offline, без CDN)

### Audit v3 acceptance criteria (все выполнены)

- ✅ **Schema**: `asr_extractor_npz_v1` (human `SCHEMA.md` + machine JSON schema)
- ✅ **Token contract**: строгий (shared_tokenizer_v1 only, encode failure → error)
- ✅ **Privacy**: raw text debug-only (opt-in), token IDs privacy-sensitive, quality metrics числовые
- ✅ **Language metadata**: нормализованные `lang_code` + confidence, distribution по кодам
- ✅ **Feature gating**: все aggregates явно контролируются через flags
- ✅ **Empty semantics**: корректно обрабатывает `audio_present=false` (empty artifacts)
- ✅ **ModelManager-only**: offline-first, no-network policy
- ✅ **Reproducibility**: `models_used[]`, `model_signature` в meta
- ✅ **Render**: offline HTML, privacy banner для текста
- ✅ **Sampling policy**: управляемые параметры через Segmenter (`--segmenter-asr-sampling-profile`, `--segmenter-asr-window-sec`, etc.)

### Sampling policy (ASR windows)

- **Segmenter contract**: `families.asr` поддерживает `profile` (semantic/proxy), `window_sec`, `stride_sec`, `max_windows`
- **Semantic profile** (default): `window_sec=30.0`, `stride_sec=25.0` (для качественной транскрипции)
- **Proxy profile**: `window_sec=10.0`, `stride_sec=5.0` (для быстрой оценки/фичей без полной транскрипции)
- Проверено на `video3` (proxy profile): `segments_count=3`, параметры записаны в `segments.json`

### TextProcessor integration

- `DataProcessor/main.py` автогенерирует `VideoDocument` из `asr_extractor` NPZ:
  - использует `token_ids_by_segment` (без raw текста по умолчанию)
  - извлекает `audio_duration_sec` из `segments.json`
  - `TextProcessor` может декодировать token IDs transiently через `shared_tokenizer_v1`
- Проверено end-to-end: `asr_extractor` → `TextProcessor` (token input) ✅

## Run: `youtube/video3/audit3_asr_sampling_proxy` — `asr_extractor` (sampling policy validation, OK)

- **Дата**: `2026-02-22`
- **Видео**: `/media/ilya/Новый том/TrendFlowML/example/example_videos/video3.mp4`
- **Платформа / video_id / run_id**: `youtube / video3 / audit3_asr_sampling_proxy`
- **Компоненты**: `Segmenter` → `AudioProcessor(asr)`

### Ключевой результат

- **Sampling policy knobs** работают: `--segmenter-asr-sampling-profile proxy` → `segments.json` содержит `families.asr.profile="proxy"`, `window_sec=10.0`, `stride_sec=5.0`
- `asr_extractor` корректно обрабатывает proxy-сегменты (меньше окон, быстрее)

### Команда запуска (факт)

```bash
python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том/TrendFlowML/example/example_videos/video3.mp4" \
  --output "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/_frames_audit3_asr_sampling" \
  --rs-base "/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results" \
  --platform-id youtube \
  --video-id video3 \
  --run-id audit3_asr_sampling_proxy \
  --sampling-policy-version v1 \
  --dataprocessor-version audit3_dev \
  --visual-cfg-path "/media/ilya/Новый том/TrendFlowML/configs/visual_triton_baseline_gpu_local.yaml" \
  --segmenter-asr-sampling-profile proxy \
  --run-audio \
  --audio-device auto \
  --audio-extractors "asr" \
  --asr-enable-token-sequences \
  --no-run-visual
```

### Артефакты

- `segments.json`: `families.asr.profile="proxy"`, `window_sec=10.0`, `stride_sec=5.0`, `segments_count=3`
- ASR NPZ: корректно обработан, `segments_count=3` в payload

## Run: `youtube/audit3_core_face_landmarks_smoke_2/audit3_core_face_landmarks_smoke_2` — `core_face_landmarks` (OK, schema v2, mini-dashboard render + assets)

- **Дата**: `2026-02-16`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_core_face_landmarks_smoke_2 / audit3_core_face_landmarks_smoke_2`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections, core_face_landmarks)`

### Ключевой результат

- `core_face_landmarks` пишет `landmarks.npz` в **`core_face_landmarks_npz_v2`**:
  - `face_landmarks` (filtered) + `face_landmarks_raw` (QA/debug),
  - person-mask диагностика: `person_present`, `face_mesh_ran`,
  - `empty_reason` различает “лиц нет” vs “не запускали из-за person-mask”.
- Render: offline `render.html` + `_render/assets/*.jpg` (downscaled overlays, privacy banner).

### Артефакты / рендеры

- NPZ (source-of-truth):  
  `dp_results/.../core_face_landmarks/landmarks.npz`
- Render (dev-only):
  - `dp_results/.../core_face_landmarks/_render/render_context.json`
  - `dp_results/.../core_face_landmarks/_render/render.html`
  - `dp_results/.../core_face_landmarks/_render/assets/`

## Run: `youtube/audit3_core_face_landmarks_smoke_3/audit3_core_face_landmarks_smoke_3` — `core_face_landmarks` (OK, QA fixes: NaN-preserving filter + safer overlays)

- **Дата**: `2026-02-16`
- **Ключевой фикс качества**:
  - temporal filter больше не “галлюцинирует” landmarks на кадрах без лица (NaN восстанавливаются),
  - bbox/overlays в рендере строятся только при `face_present=True` и по `*_raw`.

## Run: `youtube/audit3_core_face_landmarks_smoke_4/audit3_core_face_landmarks_smoke_4` — `core_face_landmarks` (OK, recall tuning for sparse sampling)

- **Дата**: `2026-02-16`
- **Видео**: `example/example_videos/video1.mp4`
- **Платформа / video_id / run_id**: `youtube / audit3_core_face_landmarks_smoke_4 / audit3_core_face_landmarks_smoke_4`
- **Компоненты**: `Segmenter` → `VisualProcessor(core_object_detections, core_face_landmarks)`

### Команда запуска (факт)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/main.py \
  --video-path "/media/ilya/Новый том2/TrendFlowML/example/example_videos/video1.mp4" \
  --global-config DataProcessor/configs/global_config.yaml \
  --profile-path DataProcessor/configs/audit_v3/profile_core_face_landmarks.yaml \
  --platform-id youtube \
  --video-id audit3_core_face_landmarks_smoke_4 \
  --run-id audit3_core_face_landmarks_smoke_4 \
  --output-dir "/media/ilya/Новый том2/TrendFlowML/DataProcessor/dp_results"
```

### Ключевой результат

- Для sparse sampling (~4 fps) подняли recall:
  - `face_mesh_static_image_mode=true`
  - `face_mesh_min_*_confidence=0.4`
- Render восстановлен и работает offline (mini-dashboard + assets).
---

## Навигация

[DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
