# `semantic_cluster_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `semantic_cluster_extractor` |
| Класс | `SemanticClusterExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/semantic_cluster_extractor_output_v1.json` |
| `schema_version` | `semantic_cluster_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

**Ближайший центроид** (cosine → inner product на L2-нормированных векторах) в пространстве **PCA** над эмбеддингом **title** / **description** / **hashtag**. Ассеты **PCA**, **centroids**, **`clusters.jsonl`** — строго **`dp_models`** (`clusters_spec_name`, по умолчанию `semantic_clusters_v1`). Sentence-transformers в рантайме **не** исполняется.

**Tier:** все поля **`tp_semclust_*`** в JSON схеме — **analytics** до отдельного ML-контракта (taxonomy pack, preflight §7).

## Включение в прогоне

Оркестратор (**`enabled`** в `global_config` для секции экстрактора) задаёт только участие в списке; в **`features_flat`** полей **`enabled` нет**.

## `features_flat` (31 ключ)

Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** ↔ JSON (`allow_extra_keys: false`).

- **Зеркала:** `tp_semclust_require_primary_source_enabled`, `tp_semclust_require_embedding_enabled`, `tp_semclust_use_faiss_enabled`, `tp_semclust_require_faiss_enabled`, `tp_semclust_emit_extra_metrics_enabled`
- **Конфиг primary:** `tp_semclust_config_primary_{title,description,hashtag}` — ровно одно **1.0**
- **`tp_semclust_{title,description,hashtag}_present`:** **1.0** только если соответствующий **`.npy`** **успешно** загружен (размер > 0)
- **Использованный источник:** `tp_semclust_source_*` — какой слот реально выбран политикой primary+fallback
- **`tp_semclust_fallback_used`:** **1.0** если выбран не `primary_source`
- **Диагностика загрузки:** `tp_semclust_unsafe_relpath_flag`; `tp_semclust_*_embed_missing_flag` — для слота задан `relpath`, join безопасен, но файла нет / ошибка чтения / пустой вектор (**не** выставляется при unsafe для того же слота)
- **Метрики:** `tp_semclust_present`, `tp_semclust_id`, `tp_semclust_similarity`, `tp_semclust_distance` — **NaN** при empty / dim mismatch
- **Extra-блок** (`n_clusters`, `model_*_dim`, `embedding_dim`, `margin_top2`, `compute_ms`): при **`emit_extra_metrics=False`** или нерелевантной ветке → **NaN**; при **dim mismatch** и **`emit_extra_metrics=True`** — частично числа (см. `main.py` `_apply_extra_block`)

## `semantic_cluster_meta`

Всегда (все ветки **`extract()`**): `clusters_spec_name`, `clusters_spec_version`, `clusters_weights_digest`, `cluster_db_version`, **`backend`** (`faiss_ip` | `numpy_cosine`).

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`**
- **`system`**: **`pre_init` / `post_init`** из **`__init__`**, **`gpu_peak_mb`**

## DAG

Заявленные зависимости: **TitleEmbedder**, **DescriptionEmbedder**, **HashtagEmbedder** (последний нужен при `primary_source` / fallback **`hashtag`**).

## Версионирование

Смена ключей → **`semantic_cluster_extractor_output_v2`** + `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
