# `cosine_metrics_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `cosine_metrics_extractor` |
| Класс | `CosineMetricsExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/cosine_metrics_extractor_output_v1.json` |
| `schema_version` | `cosine_metrics_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Косинусная близость между векторами из **`doc.tp_artifacts`**:

- **title** / **description**: `embeddings.title|description.relpath`
- **transcript (mean aggregate)**: `transcripts[source].agg_mean_relpath` или legacy `transcript_aggregates[source].agg_mean_relpath`; выбор **источника** по `transcript_source_priority` (только `whisper`, `youtube_auto`, `combined`)
- **comments**: агрегаты `comments.agg_mean_relpath` / `agg_median_relpath` (и legacy алиасы в `embeddings`) или матрица `embeddings.comments.relpath` при `comments_mode=matrix`

Модель **не** загружается.

## `features_flat` (39 ключей)

- Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** ↔ JSON (`allow_extra_keys: false`).
- **`tp_cos_transcript_agg_source_{whisper,youtube_auto,combined}`**: ровно один **`1.0`** при успешном выборе агрегата; иначе все **`0.0`**.
- **`tp_cos_comments_mode_aggregates` / `tp_cos_comments_mode_matrix`**: зеркало режима (неизвестный `comments_mode` → **0.0 / 0.0**, косины transcript↔comments остаются **NaN** — задокументировано, без fail-fast).

## `emit_extra_metrics`

- **`tp_cos_load_ms`**, **`tp_cos_compute_ms`**, **`tp_cos_tc_n_comments_used`**, **`tp_cos_tc_sims_std`**, **`tp_cos_tc_sims_p95`**: при **`False`** → **NaN**; при **`True`** — числа, если фаза применима (для std/p95/n_comments — в основном режим **`matrix`** и успешные sims).

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`**
- **`system`**: **`_init_metrics`**, **`gpu_peak_mb`** по снимкам.

## Версионирование

Смена ключей → **`cosine_metrics_extractor_output_v2`** + `RUN_LOG.md`.
