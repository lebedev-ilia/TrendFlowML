# TextProcessor — Audit v3 (документация прогона)

Единая **preflight** (audit pack, ASR, 22 компонента, модель e5-large, corpus packs, run-log):  
[`DataProcessor/docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`](../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)

Критерии Audit v3 для Text (общий чеклист):  
[`DataProcessor/docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md`](../../../docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md) — раздел **§6 TextProcessor**.

Карта экстракторов и ссылки на README:  
[`TextProcessor/docs/MAIN_INDEX.md`](../MAIN_INDEX.md)

Шаблон VideoDocument для smoke (пример + JSON Schema):  
[`example/text_audit_v3_smoke/_template/`](../../../../example/text_audit_v3_smoke/_template/)

20 сценариев (inference + training_row):  
[`example/text_audit_v3_smoke/scenarios/`](../../../../example/text_audit_v3_smoke/scenarios/)  
Там же в **`scenarios/README.md`** описан скрипт изолированного смока по **22** экстракторам (один или все сценарии): [`scripts/smoke_each_extractor_audit_v3.py`](../../scripts/smoke_each_extractor_audit_v3.py).

Отчёты по компонентам (заполняются по мере аудита):  
[`components/`](components/) — закрыто: [`tags_extractor_AUDIT_V3_REPORT.md`](components/tags_extractor_AUDIT_V3_REPORT.md), [`lexico_static_features_AUDIT_V3_REPORT.md`](components/lexico_static_features_AUDIT_V3_REPORT.md), [`asr_text_proxy_audio_features_AUDIT_V3_REPORT.md`](components/asr_text_proxy_audio_features_AUDIT_V3_REPORT.md), [`title_embedder_AUDIT_V3_REPORT.md`](components/title_embedder_AUDIT_V3_REPORT.md), [`description_embedder_AUDIT_V3_REPORT.md`](components/description_embedder_AUDIT_V3_REPORT.md), [`hashtag_embedder_AUDIT_V3_REPORT.md`](components/hashtag_embedder_AUDIT_V3_REPORT.md), [`transcript_chunk_embedder_AUDIT_V3_REPORT.md`](components/transcript_chunk_embedder_AUDIT_V3_REPORT.md), [`comments_embedder_AUDIT_V3_REPORT.md`](components/comments_embedder_AUDIT_V3_REPORT.md), [`speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md`](components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md), [`transcript_aggregator_AUDIT_V3_REPORT.md`](components/transcript_aggregator_AUDIT_V3_REPORT.md), [`comments_aggregator_AUDIT_V3_REPORT.md`](components/comments_aggregator_AUDIT_V3_REPORT.md), [`qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md`](components/qa_embedding_pairs_extractor_AUDIT_V3_REPORT.md), [`embedding_pair_topk_extractor_AUDIT_V3_REPORT.md`](components/embedding_pair_topk_extractor_AUDIT_V3_REPORT.md), [`semantics_topics_keyphrases_AUDIT_V3_REPORT.md`](components/semantics_topics_keyphrases_AUDIT_V3_REPORT.md), [`embedding_stats_extractor_AUDIT_V3_REPORT.md`](components/embedding_stats_extractor_AUDIT_V3_REPORT.md), [`cosine_metrics_extractor_AUDIT_V3_REPORT.md`](components/cosine_metrics_extractor_AUDIT_V3_REPORT.md), [`title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md`](components/title_embedding_cluster_entropy_extractor_AUDIT_V3_REPORT.md), [`title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md`](components/title_to_hashtag_cosine_extractor_AUDIT_V3_REPORT.md), [`semantic_cluster_extractor_AUDIT_V3_REPORT.md`](components/semantic_cluster_extractor_AUDIT_V3_REPORT.md), [`topk_similar_titles_extractor_AUDIT_V3_REPORT.md`](components/topk_similar_titles_extractor_AUDIT_V3_REPORT.md), [`embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md`](components/embedding_shift_indicator_extractor_AUDIT_V3_REPORT.md), [`embedding_source_id_extractor_AUDIT_V3_REPORT.md`](components/embedding_source_id_extractor_AUDIT_V3_REPORT.md)

Логи сессий (инфраструктура и крупные шаги):  
[`sessions/`](sessions/)

Machine schemas (NPZ):  
[`../../schemas/`](../../schemas/) — в т.ч. `text_npz_v1` для `run_cli.py` / `text_features.npz`.

Декларативный DAG (cross-processor):  
[`DataProcessor/docs/reference/component_graph.yaml`](../../../docs/reference/component_graph.yaml) — стадия `text_processor_tier0` (tags до lexical / ASR proxy / title+description embedder’ов).

Плейсхолдеры corpus / FAISS (до фиксации pack’ов):  
[`../../config/corpus_packs.placeholder.yaml`](../../config/corpus_packs.placeholder.yaml)

Dev run-log репозитория:  
[`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../../docs/audit_v3/RUN_LOG.md)

### Performance (orchestrator, 2026-04)

- **`MainProcessor.run()` (один документ):** при `batch_enable_cpu_parallel=True` подряд идущие **`LexicalStatsExtractor`** и **`ASRTextProxyExtractor`** в конфиге (оба на CPU) выполняются **параллельно** в `ThreadPoolExecutor` — только эта пара (read-only к `VideoDocument`, без мутаций `tp_artifacts`). Остальной порядок и семантика как при последовательном запуске.
- **`run_batch` / уровни DAG:** группировка «GPU vs CPU» для шага использует **эффективное устройство**: учитывается и ключ слота (`gpu`/`cpu`/`cpu2`), и **`device` в `extractor_params`** (раньше слот `cpu2` + `device: cuda` ошибочно относился к CPU-ветке для кэша CUDA).
- Переменные окружения для гигиены памяти между шагами: см. docstring `_text_processor_memory_after_step` в `src/core/main_processor.py` (`DP_TEXT_SKIP_CUDA_EMPTY_CACHE`, и т.д.).

Связанный worklog: [`backend/docs/E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md`](../../../../backend/docs/E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md) (§ 3.8).
---

## Навигация

[TextProcessor](../MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
