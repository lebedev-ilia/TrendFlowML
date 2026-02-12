# `topk_similar_titles_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/topk_similar_titles_extractor/main.py`

## Резюме

Компонент приведён к production-grade политике:
- corpus загружается **через `dp_models`** (offline, fail-fast, digest/версия фиксируются)
- title embedding берётся детерминированно через `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- missing corpus / missing title embedding / mismatch dim → **RuntimeError** (no-fallback)
- добавлены HNSW параметры и режимы экспорта top-k (`export_topk_mode` + `max_export_k`) для контроля UI-выхода (privacy/NPZ size)
- safe relpath join + `tp_topktitles_unsafe_relpath_flag`
- valid empty semantics при `require_title_embedding=false` (иначе fail-fast)
- process-level cache индекса/корпуса (TTL + max_entries) по ключу `(spec+weights_digest+backend+hnsw params)`
- numpy backend: без повторной нормализации корпуса на каждом запросе + защита от больших корпусов без FAISS

## Контракт

- **Вход**:
  - corpus: `dp_models` spec `similar_titles_corpus_v1` (embeddings.npy + ids.json)
  - title embedding: per-run `.npy` через `doc.tp_artifacts`
- **Выход**:
  - `result.features_flat`: `tp_topktitles_*` (скалярные summary)
  - `result.topk_similar_corpus_titles`:
    - `corpus` (meta: version/digest/size/dim/backend)
    - `topk_similar_ids/scores` только если `export_topk_lists=true`

## TODO

- `resource_costs` (unit=1 query) и лимиты по размеру top-k для UI.


