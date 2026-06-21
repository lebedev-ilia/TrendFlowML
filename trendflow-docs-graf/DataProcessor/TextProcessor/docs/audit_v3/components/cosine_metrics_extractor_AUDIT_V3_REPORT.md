# Audit v3 — `cosine_metrics_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `CosineMetricsExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `cosine_metrics_extractor_output_v1`  
**Human schema**: [`src/extractors/cosine_metrics_extractor/SCHEMA.md`](../../../src/extractors/cosine_metrics_extractor/SCHEMA.md)

## TL;DR

**39** фиксированных ключей; extra-блок (**load/compute**, matrix stats) **всегда** в схеме → **NaN** при **`emit_extra_metrics=False`**; one-hot выбранного **`agg_mean`** источника транскрипта; default **`transcript_source_priority`** в коде **`["whisper", "youtube_auto"]`**; зеркала **`require_*`**; **`_init_metrics`**, **`gpu_peak_mb`**; **`model_*`/`weights_digest`** = **`null`**. Неизвестный **`comments_mode`**: **NaN**-косины, **0/0** mode flags (без исключения).

## Acceptance

- [x] `SCHEMA.md` + `cosine_metrics_extractor_output_v1.json` (39 keys ↔ `main.py`).
- [x] `global_config.yaml` — приоритет транскрипта ASR-first.
- [ ] Полный smoke — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/cosine_metrics_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/cosine_metrics_extractor_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
