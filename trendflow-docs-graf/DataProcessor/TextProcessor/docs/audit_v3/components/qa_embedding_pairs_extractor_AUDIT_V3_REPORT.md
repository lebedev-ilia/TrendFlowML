# Audit v3 — `qa_embedding_pairs_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `QAEmbeddingPairsExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `qa_embedding_pairs_extractor_output_v1`  
**Human schema**: [`src/extractors/qa_embedding_pairs_extractor/SCHEMA.md`](../../../src/extractors/qa_embedding_pairs_extractor/SCHEMA.md)

## TL;DR

Извлечение вопросоподобных строк и эмбеддинги **(N×D)** в **`qa_question_embeddings.npy`**. **`features_flat`**: **34** ключа `tp_qa_*`, фиксированный порядок, **`emit_extra_metrics`** строго управляет производными rate/centroid. Default **`model_name`** = **`intfloat/multilingual-e5-large`**; верхний уровень ответа **`model_name`**, **`model_version`**, **`weights_digest`**; **`_init_metrics`**, **`_gpu_peak_mb`**. Ключ **`tp_qa_max_chars_per_comment`** всегда в шаблоне (конфиг). **`extract_batch`** не добавлялся. Имя «Pairs» — только документация (нет Q–A пар).

## Входы / выходы

- Входы: **`VideoDocument`** (title, description, asr, optional legacy transcripts, comments).
- Выходы: **`result.features_flat`**, артефакты, **`doc.tp_artifacts["qa"]`**.

## Acceptance

- [x] `SCHEMA.md` + `qa_embedding_pairs_extractor_output_v1.json` (34 keys ↔ `main.py`).
- [x] `emit_extra_metrics`, init/GPU peaks, top-level model metadata.
- [ ] Полный smoke с `DP_MODELS_ROOT` — при прогоне запись в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/qa_embedding_pairs_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/qa_embedding_pairs_extractor_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
