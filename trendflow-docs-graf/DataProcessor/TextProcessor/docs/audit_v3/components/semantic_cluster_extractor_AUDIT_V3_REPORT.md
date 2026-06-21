# Audit v3 — `semantic_cluster_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `SemanticClusterExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `semantic_cluster_extractor_output_v1`  
**Human schema**: [`src/extractors/semantic_cluster_extractor/SCHEMA.md`](../../../src/extractors/semantic_cluster_extractor/SCHEMA.md)

## TL;DR

**31** фиксированный ключ `tp_semclust_*`; зеркала **`require_*` / `use_faiss` / `emit_extra_metrics`**; one-hot **`primary_source`** из конфига; **`_*_present`** = успешная загрузка файла; **unsafe** vs **`*_embed_missing_flag`**; extra-блок — **NaN** при **`emit_extra_metrics=False`**; **`semantic_cluster_meta.backend`** на всех ветках; **`model_*`/`weights_digest`** = **`null`**; граф: **`HashtagEmbedder`** добавлен к зависимостям.

## Acceptance

- [x] `SCHEMA.md` + `semantic_cluster_extractor_output_v1.json` (31 key ↔ `main.py`).
- [x] Dev smoke: `DP_MODELS_ROOT` + `artifacts_dir` + синтетический embedding.
- [ ] Полный E2E TextProcessor — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/semantic_cluster_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/semantic_cluster_extractor_output_v1.json`
- `DataProcessor/TextProcessor/src/core/main_processor.py` (dependencies)
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
