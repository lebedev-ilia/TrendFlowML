# Audit v3 — `comments_embedder` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `CommentsEmbedder.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `comments_embedder_output_v1`  
**Human schema**: [`src/extractors/comments_embedder/SCHEMA.md`](../../../src/extractors/comments_embedder/SCHEMA.md)

## TL;DR

Эмбеддинги выбранных комментариев (L2-norm), артефакты **`comments_embeddings.npy`** + **`comments_selected_indices.npy`**, **`tp_commentsemb_*`**: **18** стабильных ключей. Исправлен отсутствующий **`return`** в **`extract()`** при успехе; **`extract_batch`** согласован по **`emit_extra_metrics`**, **`tp_commentsemb_cache_hit`** в batch при включённых extra — **NaN**, **`encode_ms`** пропорционален доле комментариев в общем batch-encode. Default **`model_name`** = **`intfloat/multilingual-e5-large`**. **`gpu_peak_mb`** считается через снимки GPU как у других эмбеддеров.

## Входы / выходы

- Входы: **`doc.comments`**, опционально **`comments_likes`** / **`comments_recency`** для отбора; **`get_model_with_meta`** (offline).
- Выходы: **`result.features_flat`**, матрица и индексы в артефактах и **`tp_artifacts`** (пути прежние — совместимость с **`comments_aggregator`**).

## Acceptance

- [x] `SCHEMA.md` + `comments_embedder_output_v1.json` (18 keys ↔ `main.py`).
- [x] `extract` / `extract_batch`: **`emit_extra_metrics`**, batch cache_hit / encode share.
- [ ] Полный smoke с `DP_MODELS_ROOT` — при прогоне запись в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/comments_embedder/main.py`
- `DataProcessor/TextProcessor/schemas/comments_embedder_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
