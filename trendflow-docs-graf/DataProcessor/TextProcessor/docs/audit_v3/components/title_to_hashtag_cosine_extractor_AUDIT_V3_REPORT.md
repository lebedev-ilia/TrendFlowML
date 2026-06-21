# Audit v3 — `title_to_hashtag_cosine_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TitleToHashtagCosineExtractor.VERSION = 1.2.0`  
**Machine schema (`features_flat`)**: `title_to_hashtag_cosine_extractor_output_v1`  
**Human schema**: [`src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md`](../../../src/extractors/title_to_hashtag_cosine_extractor/SCHEMA.md)

## TL;DR

**11** фиксированных ключей `tp_titlehashcos_*`; убраны **legacy** `tp_title_hashtag_cosine_*` и внутренний гейт **`enabled`** в контракте; раздельно **`unsafe_relpath`** vs **`_*_embed_missing_flag`**; **`_init_metrics`**, **`gpu_peak_mb`**, **`model_*`/`weights_digest`** = **`null`**. Лишние kwargs (в т.ч. устаревший **`enabled`**) поглощаются в **`__init__`**.

## Acceptance

- [x] `SCHEMA.md` + `title_to_hashtag_cosine_extractor_output_v1.json` (11 keys ↔ `main.py`).
- [x] Dev smoke: временные `.npy`, missing file, unsafe `relpath`.
- [ ] Полный E2E — при прогоне в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/title_to_hashtag_cosine_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/title_to_hashtag_cosine_extractor_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
