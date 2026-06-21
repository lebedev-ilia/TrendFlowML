# `transcript_aggregator` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `transcript_aggregator` |
| Класс | `TranscriptAggregatorExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/transcript_aggregator_output_v1.json` |
| `schema_version` | `transcript_aggregator_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Агрегация **уже посчитанных** чанковых эмбеддингов транскрипта (**`TranscriptChunkEmbedder`**) в векторы **mean** / **max** (L2 после агрегации), опционально **combined** по порядку `sources`. Модель **не запускается**; **`model_name` / `weights_digest`** задают **то же пространство**, что и у чанков (resolve через **`dp_models`** без загрузки весов для forward).

## Входы / артефакты

- Чтение relpath из **`doc.tp_artifacts`**: canonical **`transcripts[source].chunk_embeddings_relpath`**, legacy **`transcript_chunks`**.
- Запись: **`transcript_{source}_agg_mean.npy`**, **`transcript_{source}_agg_max.npy`**, **`transcript_combined_agg_*.npy`**.
- Регистрация: **`transcripts[source].agg_mean_relpath`**, **`agg_max_relpath`**, зеркало в **`transcript_aggregates`**.

## `features_flat` (19 ключей)

- **10 core**: присутствие агрегатов по whisper / youtube_auto (`tp_tragg_present_youtube` = флаг для **`youtube_auto`**), combined, **`decay_rate`**, флаги **`compute_*`**, **`write_artifacts`**.
- **9 extra**: chunk counts и scalar std по whisper / youtube_auto / combined — при **`emit_extra_metrics=False`** все **NaN**; при **`compute_std=False`** поля **`*_mean_std` / `*_max_std`** — **NaN**.

## Ошибки

- Нет **`doc.tp_artifacts`**: **RuntimeError** (fail-fast).
- **`require_chunks=True`** и отсутствует файл по ожидаемому relpath: **RuntimeError**.

## Метаданные

Верхний уровень ответа: **`model_name`**, **`model_version`**, **`weights_digest`** (согласованы с чанковым эмбеддером).

## Версионирование

Смена ключей/семантики → **`transcript_aggregator_output_v2`** + `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
