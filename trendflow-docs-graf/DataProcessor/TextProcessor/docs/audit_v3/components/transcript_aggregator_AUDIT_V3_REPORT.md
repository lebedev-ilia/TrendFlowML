# Audit v3 — `transcript_aggregator` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TranscriptAggregatorExtractor.VERSION = 1.3.0`  
**Machine schema**: `transcript_aggregator_output_v1`  
**Human schema**: [`src/extractors/transcript_aggregator/SCHEMA.md`](../../../src/extractors/transcript_aggregator/SCHEMA.md)

## TL;DR

Агрегатор чанковых эмбеддингов (**mean/max**, decay, optional std, combined). **`features_flat`** — **19** стабильных ключей; **9** extra при **`emit_extra_metrics=False`** → **NaN** (std-слоты дополнительно **NaN** при **`compute_std=False`**). **`dp_models.resolve`** в **`__init__`** без inference; default **`intfloat/multilingual-e5-large`**; **`_init_metrics`**, **`gpu_peak_mb`**, top-level **`model_name` / `weights_digest`**.

## Acceptance

- [x] `SCHEMA.md` + JSON ↔ `main.py`.
- [ ] Полный smoke цепочки **TranscriptChunkEmbedder → transcript_aggregator** — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/transcript_aggregator/main.py`
- `DataProcessor/TextProcessor/schemas/transcript_aggregator_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
