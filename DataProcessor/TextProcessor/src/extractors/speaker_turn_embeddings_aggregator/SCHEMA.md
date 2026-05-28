# `speaker_turn_embeddings_aggregator` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `speaker_turn_embeddings_aggregator` |
| Класс | `SpeakerTurnEmbeddingsAggregatorExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/speaker_turn_embeddings_aggregator_output_v1.json` |
| `schema_version` | `speaker_turn_embeddings_aggregator_output_v1` |
| Версия реализации | `1.3.0` (`SpeakerTurnEmbeddingsAggregatorExtractor.VERSION`) |

## Назначение

Агрегация **эмбеддингов по спикерским turn’ам**: для каждого логического спикера считаются **mean** и/или **max** по L2-нормированным векторам фрагментов текста. В **`result.features_flat`** — ровно **17** ключей `tp_spkemb_*`. Векторы — `speaker_{spkXXX}_mean.npy` / `_max.npy` (фиксированные имена, без content-hash).

## Audit v3 preflight (модель)

Канон — **`intfloat/multilingual-e5-large`** ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)). Для полного прогона рекомендуется **ASR + speaker diarization** в том же run ([preflight §4](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)).

## Входы (режимы)

1. **Предпочтительный:** `doc.speaker_diarization["speaker_segments"]` + `doc.asr["segments"]` с **`start_sec` / `end_sec`**, выравнивание по **пересечению интервалов**. Флаги: **`tp_spkemb_input_mode_diar_asr`**, **`tp_spkemb_asr_present`**, **`tp_spkemb_diar_present`**.
2. **Legacy / dev:** `doc.speakers` (dict с `name` / `description`). Флаг: **`tp_spkemb_input_mode_legacy_doc_speakers`**. Не смешивать с прод-контрактом без явной пометки в `RUN_LOG`.

## `emit_extra_metrics`

**5** полей в конце контракта (`batch_size`, `max_speakers`, `max_turns_per_speaker`, `min_chars_per_turn`, `max_chars_per_turn`): при **`False`** — **NaN** (ключи сохраняются).

## `require_input`

При **`require_input=True`**, если нет ни diar+ASR списков, ни legacy **`doc.speakers`** (`tp_spkemb_input_present` остаётся **0**) → **RuntimeError** (fail-fast).

## Метаданные модели

- Верхний уровень payload: **`model_name`**, **`model_version`**, **`weights_digest`**.
- Дублирование в **`result.speaker_embeddings_meta`** (обратная совместимость).

## GPU

**`system.peaks.gpu_peak_mb`** — max по снимкам init + post-process (как у других эмбеддеров).

## Версионирование

Изменение ключей/семантики → bump **`speaker_turn_embeddings_aggregator_output_v2`** + `RUN_LOG.md` + отчёт.
