# Audit v3 — `embedding_stats_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `EmbeddingStatsExtractor.VERSION = 1.2.0`  
**Machine schema (`features_flat`)**: `embedding_stats_extractor_output_v1`  
**Human schema**: [`src/extractors/embedding_stats_extractor/SCHEMA.md`](../../../src/extractors/embedding_stats_extractor/SCHEMA.md)

## TL;DR

**39** фиксированных ключей; **8** слотов `tp_embstats_topvar_*` с клампом **`top_k_slots`**; фиксированные флаги **`tp_embstats_source_used_{whisper,youtube_auto}`**; приоритет транскрипта по умолчанию **ASR-first** (`whisper`); **`emit_extra_metrics`** → тайминги **`load_ms`/`compute_ms`** или **NaN**; **`_init_metrics`**, **`gpu_peak_mb`**; верхний уровень **`model_*`/`weights_digest`** = **`null`**. DAG: жёсткая зависимость только от **`TranscriptChunkEmbedder`**. Энтропия тем — по upstream **`topic_probs`**.

## Acceptance

- [x] `SCHEMA.md` + `embedding_stats_extractor_output_v1.json` (39 keys ↔ `main.py`).
- [x] Документация/README/MAIN_INDEX согласованы с фактическим scope (только чанки транскрипта + optional topics).
- [ ] Полный smoke — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/embedding_stats_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/embedding_stats_extractor_output_v1.json`
- `DataProcessor/TextProcessor/src/core/main_processor.py` (зависимости)
