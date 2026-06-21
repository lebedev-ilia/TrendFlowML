# Audit v3 — `transcript_chunk_embedder` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TranscriptChunkEmbedder.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `transcript_chunk_embedder_output_v1`  
**Human schema**: [`src/extractors/transcript_chunk_embedder/SCHEMA.md`](../../../src/extractors/transcript_chunk_embedder/SCHEMA.md)

## TL;DR

Чанковые эмбеддинги транскрипта (логические каналы **`whisper`** и **`youtube_auto`**), per-source `.npy` в артефактах. Плоский контракт **`tp_tchunk_*`**: стабильно **16** ключей; при **`emit_confidence_metrics=False`** — **0 / NaN** для четырёх conf-полей; при **`emit_extra_metrics=False`** — **NaN** для пяти tuning-полей. **`extract_batch`** строит **`features_flat`** через тот же **`_build_features_flat`**, что и **`extract`**. В **`result`** на всех путях — **`model_name`**, **`model_version`**, **`weights_digest`**.

## Входы / выходы

- Входы: `doc.asr.segments` (whisper при `use_asr`), `doc.transcripts["youtube_auto"]` (при `use_youtube_auto`), tokenizer **`shared_tokenizer_v1`**.
- Выходы: `result.features_flat` (**16** ключей); матрицы — `transcript_{source}_chunk_embeddings.npy` и canonical пути в `doc.tp_artifacts` (см. README).

## Acceptance

- [x] `SCHEMA.md` + machine JSON (16 keys) + индексы.
- [ ] Полный smoke с `DP_MODELS_ROOT` — см. `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/transcript_chunk_embedder/main.py`
- `DataProcessor/TextProcessor/schemas/transcript_chunk_embedder_output_v1.json`
---

## Навигация

[README](README.md) · [Audit v3 index](../README.md) · [TextProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
