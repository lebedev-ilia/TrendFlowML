# Audit v3 — `topk_similar_titles_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия компонента**: `1.3.0`  
**Machine schema (`features_flat`)**: `topk_similar_titles_extractor_output_v1`  
**Human schema**: [`src/extractors/topk_similar_titles_extractor/SCHEMA.md`](../../../src/extractors/topk_similar_titles_extractor/SCHEMA.md)

## TL;DR

**29** фиксированных ключей `tp_topktitles_*`; **`tp_topktitles_title_embed_missing_flag`** (файл отсутствует / ошибка загрузки; независимо от dim/NaN/zero-norm); **`corpus`** в **`topk_similar_corpus_titles`** на всех ветках; **`model_*`/`weights_digest`** = **`null`**; **`system.pre_init`/`post_init`** из **`__init__`** (после загрузки корпуса), **`gpu_peak_mb`**. Задокументирована **приближённость HNSW** относительно numpy. Tier полей — **analytics** до фиксации corpus pack.

## Acceptance

- [x] `SCHEMA.md` + `topk_similar_titles_extractor_output_v1.json` (29 keys ↔ `main.py`).
- [x] Corpus meta на пустых / disabled / error-path ветках.
- [x] Preflight: для строгого прогона — `require_title_embedding: true` (комментарий в `global_config.yaml`).

## Файлы

- `DataProcessor/TextProcessor/src/extractors/topk_similar_titles_extractor/main.py`
- `DataProcessor/TextProcessor/src/extractors/topk_similar_titles_extractor/SCHEMA.md`
- `DataProcessor/TextProcessor/schemas/topk_similar_titles_extractor_output_v1.json`

## Run-log

См. [`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../../../docs/audit_v3/RUN_LOG.md) — секция `topk_similar_titles_extractor`.
