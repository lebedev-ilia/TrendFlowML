# `embedding_stats_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/embedding_stats_extractor/main.py`

## Резюме

Переведён на детерминированные входы через `doc.tp_artifacts` (без glob/mtime, без arbitrary JSON cache). Выдаёт только `features_flat`:
- `tp_embstats_l2_variance`
- `tp_embstats_topvar_*` (fixed slots)
- `tp_embstats_topic_entropy` (+ present flag)
- source tracking `tp_embstats_source_used_*`

## Примечания

- Chunk embeddings читаются из canonical `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` с legacy fallback на `transcript_chunks` (ставится `tp_embstats_used_legacy_key_flag`).
- Safe relpath join (path traversal защита) + `tp_embstats_unsafe_relpath_flag`.
- Topic entropy считается из `doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"]` (in-memory, заполняется `semantics_topics_keyphrases` при `enable_topic_distribution=true`), с валидацией и нормализацией probs.
- Empty semantics: если `n_chunks < min_chunks_required` → `tp_embstats_present=0`, метрики NaN. Fail-fast опционально через `require_chunks=true`.
- Есть feature-gating: `enabled`, `compute_topic_entropy`, `require_topic_distribution`.

## TODO

- `resource_costs` замеры.


