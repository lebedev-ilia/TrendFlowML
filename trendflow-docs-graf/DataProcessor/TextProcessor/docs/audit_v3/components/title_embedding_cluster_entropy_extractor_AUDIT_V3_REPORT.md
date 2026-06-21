# Audit v3 — `title_embedding_cluster_entropy_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TitleEmbeddingClusterEntropyExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `title_embedding_cluster_entropy_extractor_output_v1`  
**Human schema**: [`src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md`](../../../src/extractors/title_embedding_cluster_entropy_extractor/SCHEMA.md)

## TL;DR

**24** фиксированных ключа; **кламп** `top_k_slots` ≤ **8** + `tp_titleclent_top_k_slots_requested` / `*_clamped`; extra-блок всегда в схеме → **NaN** при **`emit_extra_metrics=False`** или empty; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**; **`entropy_norm`** при **K≤1** → **0.0** (избегаем деления на ноль). **`export_topk_distribution`** только в **`title_cluster_entropy_meta.topk`**.

## Acceptance

- [x] `SCHEMA.md` + `title_embedding_cluster_entropy_extractor_output_v1.json` (24 keys ↔ `main.py`).
- [x] Dev smoke: `DP_MODELS_ROOT` → `bundled_models`, `extract()` на синтетическом title `.npy` + empty doc.
- [ ] Полный E2E TextProcessor + ASR — при следующем прогоне в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/title_embedding_cluster_entropy_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/title_embedding_cluster_entropy_extractor_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
