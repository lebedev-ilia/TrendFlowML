# Audit v3 — `description_embedder` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `DescriptionEmbedder.VERSION = 1.2.0`  
**Machine schema (`features_flat`)**: `description_embedder_output_v1`  
**Human schema**: [`src/extractors/description_embedder/SCHEMA.md`](../../../src/extractors/description_embedder/SCHEMA.md)

## TL;DR

Экстрактор строит **один** L2-нормализованный эмбеддинг для **`doc.description`** через chunking (`shared_tokenizer_v1` + `max_chunk_tokens_model`), encode моделью из `dp_models`, pooling по стратегии; опционально пишет **`description_embedding.npy`**, регистрирует **`doc.tp_artifacts["embeddings"]["description"]`**, отдаёт **19** скаляров в **`features_flat`** (`tp_descemb_*`). Пустое описание — валидный empty. Preflight Audit v3: модель **`intfloat/multilingual-e5-large`** задаётся профилем прогона.

## Входы / выходы

- Входы: `VideoDocument.description`, `model_name`, `pooling_strategy`, tokenizer spec, кеш/устройство, флаги `compute_embedding`, `write_artifact`, `compute_raw_norm`.
- Выходы: `result.features_flat`; top-level `model_name`, `model_version`, `weights_digest`, `system`, `timings_s`, `error`; плотный вектор только в `.npy` + `tp_artifacts`.

## Принятые решения

1. Контракт скаляров — JSON-схема; вектор отдельным файлом.
2. One-hot `tp_descemb_pooling_length_weighted` отражает только стратегию `length_weighted_mean`; для mean/max/logsumexp значение **0.0** (детальный one-hot других стратегий в v1 не выводится).
3. `emit_extra_metrics` не расширяет `features_flat` в v1.2.0.

## Acceptance

- [x] `SCHEMA.md` + `description_embedder_output_v1.json` (19 keys).
- [ ] Smoke с `DP_MODELS_ROOT` + tokenizer — вне этой записи; ключи сверены с `main.py`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/description_embedder/main.py`
- `DataProcessor/TextProcessor/schemas/description_embedder_output_v1.json`
