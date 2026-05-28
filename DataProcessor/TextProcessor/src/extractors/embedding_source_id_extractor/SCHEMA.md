# `embedding_source_id_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `embedding_source_id_extractor` |
| Класс | `EmbeddingSourceIdExtractor` |
| Machine schema (`features_flat`) | `DataProcessor/TextProcessor/schemas/embedding_source_id_extractor_output_v1.json` |
| `schema_version` | `embedding_source_id_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Детерминированный выбор **primary** эмбеддинга из **`doc.tp_artifacts`**, загрузка **`.npy`**, вычисление переносимого **`vector_id`** (SHA256 первых 12 байт от little-endian **float32** C-contiguous байтов вектора). Метаданные для интеграции с vector store — во вложенном **`result.embedding_source_id`**.

**Tier:** все **`tp_embid_*`** в machine JSON — **analytics**.

Участие в прогоне задаётся оркестратором (**`enabled`** в `global_config`); отдельного ключа **`tp_embid_enabled`** в **`features_flat`** нет.

## Источники primary

- **Title / description:** `embeddings.{title,description}.relpath` (+ опционально **`model_name`**, **`model_version`**, **`weights_digest`** в том же dict).
- **Transcript mean:** `transcripts.{combined,whisper,youtube_auto}.agg_mean_relpath` (канон), затем legacy **`transcript_aggregates.*.agg_mean_relpath`**.

Политика **`primary_source_policy`**: `transcript_first` | `title_first` | `description_first` | `title_only` | `transcript_only`.

## `features_flat` (13 ключей)

Фиксированный порядок ↔ JSON (`allow_extra_keys: false`).

- **`tp_embid_strict_missing_primary_enabled`**: зеркало **`strict_missing_primary`**; при **`True`** все перечисленные ниже ошибки после выбора relpath → **RuntimeError**; при **`False`** → **valid empty** + флаги + **`embedding_source_id.error`**.
- One-hot политики: **`tp_embid_policy_*`**
- One-hot типа выбранного источника: **`tp_embid_primary_is_*`** (если primary не выбран — все **0**)
- **`tp_embid_unsafe_relpath_flag`**, **`tp_embid_primary_embed_missing_flag`** (нет файла / ошибка **`np.load`** / пустой вектор), **`tp_embid_nan_inf_flag`**
- **`tp_embid_present`**: **1.0** только при успешно загруженном конечном векторе без NaN/inf

## `result.embedding_source_id`

**Успех:**

- **`vector_id`**, **`vector_store_uri`**, **`embedding_relpath`** (per-run относительный путь), **`primary_source`**
- **`model_name`**: из upstream meta или **`null`** (например transcript mean без per-field meta)
- **`model_version`**: из upstream **`model_version`** либо fallback конфига (**не** подмена **`model_name`**)
- **`weights_digest`**: из upstream или строка **`unknown`**

**Ошибка (только при `strict_missing_primary=False` на соответствующей ветке):** ключ **`error`** с кодом: `no_embedding_found` | `unsafe_relpath` | `embedding_file_missing` | `embedding_load_failed` | `embedding_empty` | `embedding_non_finite`

## Ответ `extract` (верхний уровень)

- **`model_name` / `model_version` / `weights_digest`**: **`null`** (дубликат не дублируем; см. вложенный блок)
- **`system`**: **`pre_init`/`post_init`** из **`__init__`**, **`post_process`**, peaks (**`gpu_peak_mb`**)

## DAG

**TitleEmbedder**, **DescriptionEmbedder**, **TranscriptAggregatorExtractor** (заявлено в `MainProcessor`).

## Версионирование

Смена ключей **`features_flat`** → **`embedding_source_id_extractor_output_v2`** + **`RUN_LOG.md`**.
