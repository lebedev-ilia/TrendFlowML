# TextProcessor audit checklist index (audit_v1)

**Цель**: единый индекс всех audit‑файлов по TextProcessor и его extractors.  
**Source-of-truth критерии**: `TP_AUDIT_CRITERIA.md`.

---

## Как пользоваться

- Каждый extractor аудируем отдельно и создаём файл в:  
  `TextProcessor/docs/audit_v1/components/<extractor_name>_AUDIT.md`
- После создания/обновления audit‑файла добавляем строку в таблицу ниже.
- Статусы:
  - `planned` — запланирован, но аудит не начат
  - `in_review` — аудит идёт (вопросы/дизайн обсуждение)
  - `in_progress` — внедряем изменения в код/README
  - `done` — аудит закрыт, компонент соответствует критериям
  - `blocked` — заблокирован (нет данных/модели/апстрима/контракта)

---

## Индекс audit‑файлов

### Основные документы

- `TP_AUDIT_CRITERIA.md` — критерии аудита (этот каталог)

### Orchestrator/Writer (core components)

| Component | Audit file | Status | Notes |
|-----------|------------|--------|-------|
| `orchestrator_writer` (`run_cli.py` + `main_processor.py`) | `components/orchestrator_writer_AUDIT.md` | `done` | error handling per extractor; status aggregation; features_flat conflict detection; models_used collection; required_extractors; NPZ validation before write; structured logging; orchestrator metrics |

### Extractors (components)

| Extractor | Audit file | Status | Notes |
|----------|------------|--------|-------|
| `asr_text_proxy_audio_features` | `components/asr_text_proxy_audio_features_AUDIT.md` | `done` | stable `tp_asrproxy_*`; strict duration contract; schema/quality flags; resource_costs placeholder added |
| `lexico_static_features` | `components/lexico_static_features_AUDIT.md` | `done` | enabled gating; explicit transcript_source_policy; max_*_chars truncation; stable `tp_lex_*`; placeholder resource_costs |
| `tags_extractor` | `components/tags_extractor_AUDIT.md` | `done` | privacy/no-raw by default; in-memory doc mutations; Unicode hashtag parsing + limits + top-K hashed slots |
| `transcript_aggregator` | `components/transcript_aggregator_AUDIT.md` | `done` | synced with transcript_chunk_embedder canonical tp_artifacts; fixed per-run aggregate artifacts; no fake vectors |
| `transcript_chunk_embedder` | `components/transcript_chunk_embedder_AUDIT.md` | `done` | token-aware chunking via dp_models shared_tokenizer_v1; fixed per-run artifacts; canonical tp_artifacts keys |
| `title_embedder` | `components/title_embedder_AUDIT.md` | `done` | fixed per-run artifact name; split compute/write; dp_models meta; stable `tp_titleemb_*` |
| `description_embedder` | `components/description_embedder_AUDIT.md` | `done` | fixed per-run artifact name; strict tokenizer chunking; stable `tp_descemb_*`; no fake vectors |
| `comments_embedder` | `components/comments_embedder_AUDIT.md` | `done` | stable `tp_commentsemb_*` + cost controls; dp_models meta; split compute/write; no abs paths |
| `comments_aggregator` | `components/comments_aggregator_AUDIT.md` | `done` | safe relpath join; split compute/write; stable `tp_commentsagg_*` + legacy aliases |
| `hashtag_embedder` | `components/hashtag_embedder_AUDIT.md` | `done` | dp_models model meta + stable tp_hashemb_*; fixed per-run artifact name; cache default off |
| `cosine_metrics_extractor` | `components/cosine_metrics_extractor_AUDIT.md` | `done` | canonical transcripts + safe relpath join; valid empty semantics; outputs `tp_cos_*` scalars |
| `title_to_hashtag_cosine_extractor` | `components/title_to_hashtag_cosine_extractor_AUDIT.md` | `done` | safe relpath join; stable `tp_titlehashcos_*` + legacy aliases; no fake metrics (zero-norm/dim mismatch) |
| `topk_similar_titles_extractor` | `components/topk_similar_titles_extractor_AUDIT.md` | `done` | safe relpath join; enabled/require flags; process-level index cache (TTL/LRU); stable `tp_topktitles_*`; export modes+limits |
| `embedding_shift_indicator_extractor` | `components/embedding_shift_indicator_extractor_AUDIT.md` | `done` | safe relpath join; stable `tp_embshift_*`; enabled/require flags; canonical transcripts + legacy fallback |
| `embedding_pair_topk_extractor` | `components/embedding_pair_topk_extractor_AUDIT.md` | `done` | safe relpath join; canonical transcripts + legacy fallback; stable `tp_embpair_*` + aliases; no fake metrics; optional topk idx slots |
| `embedding_stats_extractor` | `components/embedding_stats_extractor_AUDIT.md` | `done` | safe relpath join; canonical transcripts + legacy fallback; stable `tp_embstats_*`; topic_probs validation/normalization; enabled/require flags |
| `semantic_cluster_extractor` | `components/semantic_cluster_extractor_AUDIT.md` | `done` | strict dp_models assets + valid empty semantics; reads via `doc.tp_artifacts`; outputs `tp_semclust_*` |
| `title_embedding_cluster_entropy_extractor` | `components/title_embedding_cluster_entropy_extractor_AUDIT.md` | `done` | dp_models `semantic_clusters_v1`; valid empty semantics; outputs `tp_titleclent_*` |
| `qa_embedding_pairs_extractor` | `components/qa_embedding_pairs_extractor_AUDIT.md` | `done` | dp_models meta (weights_digest); enabled/require policies; stable `tp_qa_*`; transcript source policy; privacy-safe artifacts |
| `speaker_turn_embeddings_aggregator` | `components/speaker_turn_embeddings_aggregator_AUDIT.md` | `done` | dp_models + no raw-derived hashes; deterministic per-run artifacts; features_flat only |
| `embedding_source_id_extractor` | `components/embedding_source_id_extractor_AUDIT.md` | `done` | canonical transcripts+aggregates + safe relpath join; outputs features_flat flags + embedding_source_id dict |
| `semantics_topics_keyphrases` | `components/semantics_topics_keyphrases_AUDIT.md` | `done` | dp_models topics taxonomy spec + weights_digest; cache TTL/limits; stable slots; privacy-safe hashed keyphrases; no raw-derived artifact names |

> Примечание: каталог `components/` может быть пустым до первого завершённого аудита. Это нормально.


