# Audit v3 — `speaker_turn_embeddings_aggregator` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `SpeakerTurnEmbeddingsAggregatorExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `speaker_turn_embeddings_aggregator_output_v1`  
**Human schema**: [`src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md`](../../../src/extractors/speaker_turn_embeddings_aggregator/SCHEMA.md)

## TL;DR

Turn-level эмбеддинги по спикерам (diar+ASR или legacy **`doc.speakers`**), per-speaker **mean/max** `.npy`, **`tp_spkemb_*`** — **17** ключей. **`get_model_with_meta`**, **`_init_metrics`**, **`gpu_peak_mb`**, на всех ветках верхний уровень **`model_name`** / **`weights_digest`**. **`emit_extra_metrics=False`** → последние **5** полей **NaN**. **`extract_batch`**: не реализован (out of scope этого аудита).

## Acceptance

- [x] `SCHEMA.md` + JSON (17 keys ↔ `main.py`).
- [x] Стабильный `features_flat`, preflight default **e5-large**.
- [ ] Полный smoke с **DP_MODELS_ROOT**, **diarization** + **ASR** — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/main.py`
- `DataProcessor/TextProcessor/schemas/speaker_turn_embeddings_aggregator_output_v1.json`
