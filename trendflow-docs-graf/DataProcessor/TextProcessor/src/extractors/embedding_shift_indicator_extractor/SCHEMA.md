# `embedding_shift_indicator_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `embedding_shift_indicator_extractor` |
| Класс | `EmbeddingShiftIndicatorExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/embedding_shift_indicator_extractor_output_v1.json` |
| `schema_version` | `embedding_shift_indicator_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

**Семантический сдвиг** вдоль транскрипта: косинус между усреднёнными эмбеддингами **первых** и **последних** `win` чанков (`win = min(n_window_chunks, max(1, n_chunks // 2))`). Матрица чанков читается с диска по registry **`doc.tp_artifacts`** (без `glob+mtime`), относительно **`artifacts_dir`**.

**Tier:** все **`tp_embshift_*`** — **analytics**.

## Источник матрицы чанков

Порядок **`transcript_source_priority`** (по умолчанию **whisper → youtube_auto**). Для каждого источника:

1. **Канон:** `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]` — **не** требует наличия ключа **`transcript_chunks`**.
2. **Legacy:** `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]` или `["embeddings_path"]` → **`tp_embshift_used_legacy_key_flag=1`**.

One-hot **`tp_embshift_source_used_{whisper,youtube_auto}`** отражает фактический выбранный слот (другие имена источников в приоритете **не** дают отдельных ключей в v1).

## `features_flat` (27 ключей)

Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** ↔ JSON (`allow_extra_keys: false`).

- Зеркала: **`enabled`**, **`require_transcript_chunks`**, **`require_min_chunks`**, **`emit_extra_metrics`**, **`compute_shift_flag`**, **`compute_extra_cosines`**
- Счётчики/размеры: **`n_chunks`**, **`n_window_chunks`**, **`dim`**, **`cosine_threshold`**
- Метрики: **`cosine_begin_end`**, **`shift_flag`** (NaN если **`compute_shift_flag=false`** или cosine невалиден), **`margin`**, при **`compute_extra_cosines`**: **`cosine_first_last`**, **`mean_cosine_last_to_start_window`** (иначе NaN)
- **`tp_embshift_present`**: **1.0** только если **основной** **`cosine_begin_end`** конечен (валидное число); иначе **0.0** (в т.ч. недостаточно чанков — ранний выход без «фиктивных» метрик)
- Флаги: **`unsafe_relpath`**, **`chunk_embed_missing`** (нет файла или ошибка **`np.load`** при безопасном пути), **`dim_mismatch`**, **`zero_norm`**, **`nan_inf`**
- **`load_ms`**, **`compute_ms`**: при **`emit_extra_metrics=False`** всегда **NaN**; при **`True`** — миллисекунды по соответствующим участкам, если ветка дошла до измерения

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`**
- **`system`**: **`pre_init` / `post_init`** из **`__init__`**, **`post_process`** из **`extract()`**, **`gpu_peak_mb`**, **`ram_peak_mb`**

## DAG

Заявленная зависимость: **`TranscriptChunkEmbedder`**.

## Strict / preflight

По умолчанию **`require_transcript_chunks=false`**: soft empty + флаги. Для полного Text Audit v3 с обязательными чанками см. комментарий в **`global_config.yaml`**.

## Версионирование

Смена ключей → **`embedding_shift_indicator_extractor_output_v2`** + **`RUN_LOG.md`**.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
