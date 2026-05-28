# Audit v3 — `semantics_topics_keyphrases` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `SemanticTopicExtractor.VERSION = 2.1.0`  
**Machine schema (`features_flat`)**: `semantics_topics_keyphrases_output_v1`  
**Human schema**: [`src/extractors/semantics_topics_keyphrases/SCHEMA.md`](../../../src/extractors/semantics_topics_keyphrases/SCHEMA.md)

## TL;DR

**116** фиксированных ключей `tp_topics_*`; **8×3** topic-слота и **16×3** keyphrase-слота; кламп `top_k_slots` и `keyphrase_slots` с флагами **requested/clamped**. Единый полный `features_flat` на ветках **disabled**, **пустой текст** и **успех**. **`emit_extra_metrics`**: блок **`tp_topics_extra_*`** всегда в схеме → **NaN** при выкл. или пропущенной ветке. **`_init_metrics`**, **`_gpu_peak_mb`**, **`model_name` / `model_version` / `weights_digest`** на верхнем уровне (**`null`** при выкл. или без текста). Сырой список фраз — только **`result.tp_topics_keyphrases_raw`**.

## Входы / выходы

- Входы: `VideoDocument` (ASR segments, опционально legacy transcripts, title, description), конфиг taxonomy и модели.
- Выходы: скалярный **`features_flat`**; опционально **`tp_topics_keyphrases_raw`** в **`result`**.

## Acceptance
 
- [x] `SCHEMA.md` + `semantics_topics_keyphrases_output_v1.json` (116 keys ↔ `main.py`).
- [x] Фиксированные слоты, extra-блок, init/GPU, кламп слотов, one-hot `transcript_source_policy`.
- [ ] Полный smoke — `RUN_LOG.md` при прогоне с `DP_MODELS_ROOT`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/main.py`
- `DataProcessor/TextProcessor/schemas/semantics_topics_keyphrases_output_v1.json`
