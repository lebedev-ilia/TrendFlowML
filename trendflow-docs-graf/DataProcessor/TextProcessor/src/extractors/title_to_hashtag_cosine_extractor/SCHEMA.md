# `title_to_hashtag_cosine_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `title_to_hashtag_cosine_extractor` |
| Класс | `TitleToHashtagCosineExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/title_to_hashtag_cosine_extractor_output_v1.json` |
| `schema_version` | `title_to_hashtag_cosine_extractor_output_v1` |
| Версия реализации | `1.2.0` |

## Назначение

**Cosine similarity** между **L2-нормированными** эмбеддингами **title** и **hashtag** (`doc.tp_artifacts["embeddings"]["title|hashtag"]["relpath"]`). Модель **не** загружается.

**Политика tier:** все поля в схеме — **analytics** (Q7 baseline).

## Включение в прогоне

Выключается **только** оркестратором (например `enabled: false` в `global_config.yaml` для секции экстрактора — ключ **не** передаётся в `__init__` и **не** дублируется в `features_flat`).

## `features_flat` (11 ключей)

Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** ↔ JSON (`allow_extra_keys: false`).

- **Зеркала:** `tp_titlehashcos_require_title_embedding_enabled`, `tp_titlehashcos_require_hashtag_embedding_enabled`
- **Сигнал:** `tp_titlehashcos_present`, `tp_titlehashcos_cosine` (NaN при empty)
- **Входы:** `tp_titlehashcos_title_present`, `tp_titlehashcos_hashtag_present`
- **Диагностика:**
  - `tp_titlehashcos_unsafe_relpath_flag` — path traversal при resolve (`safe_join` исключение)
  - `tp_titlehashcos_title_embed_missing_flag` / `tp_titlehashcos_hashtag_embed_missing_flag` — задан `relpath`, путь безопасен, но файла нет или файл не читается / пустой после reshape
  - `tp_titlehashcos_dim_mismatch_flag`, `tp_titlehashcos_zero_norm_flag`

**Legacy-алиасы** `tp_title_hashtag_cosine_*` **не** эмитятся (v1.2.0).

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`**
- **`system`**: **`pre_init` / `post_init`** из **`__init__`**, **`post_process`**, **`gpu_peak_mb`**

## Версионирование

Смена ключей → **`title_to_hashtag_cosine_extractor_output_v2`** + `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
