# Audit v3 — `comments_aggregator` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `CommentsAggregationExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `comments_aggregator_output_v1`  
**Human schema**: [`src/extractors/comments_aggregator/SCHEMA.md`](../../../src/extractors/comments_aggregator/SCHEMA.md)

## TL;DR

Агрегация эмбеддингов комментариев (**weighted mean** + **покомпонентная median**), артефакты **`comments_agg_mean.npy`** / **`comments_agg_median.npy`**. **`features_flat`**: **39** стабильных ключей — **`tp_commentsagg_*`**, legacy **`tp_comments_agg_*`**, **`tp_cagg_*`**. **`dp_models.resolve`** в **`__init__`** (без inference); на верхнем уровне ответа **`model_name`**, **`model_version`**, **`weights_digest`**. **`emit_extra_metrics`**: **`tp_commentsagg_agg_mean_ms`**, **`tp_commentsagg_agg_median_ms`** (мс) или **NaN**. Пустая и успешная ветки отдают **один и тот же набор ключей** (в т.ч. legacy weights/compute на empty). **`gpu_peak_mb`** через **`_gpu_peak_mb`**. **`extract_batch`** не добавлялся.

## Входы / выходы

- Входы: матрица по **`doc.tp_artifacts`**, опционально индексы и веса документа для mean.
- Выходы: **`result.features_flat`**, векторы в артефактах и **`tp_artifacts`**.

## Acceptance

- [x] `SCHEMA.md` + `comments_aggregator_output_v1.json` (39 keys ↔ `main.py`).
- [x] **`emit_extra_metrics`**, **`_init_metrics`**, **`gpu_peak_mb`**, resolve metadata.
- [ ] Полный smoke с `DP_MODELS_ROOT` — при прогоне запись в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/comments_aggregator/main.py`
- `DataProcessor/TextProcessor/schemas/comments_aggregator_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
