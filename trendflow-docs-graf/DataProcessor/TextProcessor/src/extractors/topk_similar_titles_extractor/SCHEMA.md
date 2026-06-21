# `topk_similar_titles_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `topk_similar_titles_extractor` |
| Класс | `TopKSimilarCorpusTitlesExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/topk_similar_titles_extractor_output_v1.json` |
| `schema_version` | `topk_similar_titles_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

**Top-K** ближайших заголовков корпуса по **cosine similarity** (inner product на L2-нормированных векторах) между эмбеддингом **title** текущего видео и строками матрицы корпуса. Корпус (`embeddings.npy` + `ids.json`) загружается **только** через **`dp_models`** (`corpus_spec_name`). Запросный вектор читается с диска по **`doc.tp_artifacts["embeddings"]["title"]["relpath"]`** относительно **`artifacts_dir`** (safe join).

**Tier:** все поля **`tp_topktitles_*`** в machine JSON — **analytics** до фиксации corpus pack (`pack_version` / `pack_digest`, preflight §7).

## Поисковый backend

- **`faiss_hnsw_ip`**: `IndexHNSWFlat` + inner product на нормированных векторах. Результат **приближенный** (параметры HNSW: `m`, `efConstruction`, `efSearch`); порядок/скор может отличаться от полного перебора (numpy).
- **`numpy_cosine`**: точный top-K через матричный matmul по всему корпусу (ограничения по размеру — см. конфиг).

## Включение в прогоне

Оркестратор задаёт **`enabled`**; в **`features_flat`** отдельного ключа `enabled` нет — есть зеркало **`tp_topktitles_enabled`**.

## `features_flat` (29 ключей)

Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** в `main.py` ↔ JSON (`allow_extra_keys: false`).

- **Политика / зеркала:** `tp_topktitles_disabled_by_policy`, `tp_topktitles_enabled`, `tp_topktitles_require_title_embedding_enabled`, флаги FAISS / numpy / cache, one-hot **`export_topk_mode`** (`*_ids_only`, `*_ids_and_scores`, `*_none`), `tp_topktitles_max_export_k`, `tp_topktitles_k`, размеры корпуса и dim.
- **Экспорт в payload:** `tp_topktitles_export_k_used`, `tp_topktitles_export_k_truncated_flag` (если \(k >\) `max_export_k`).
- **Диагностика title artifact:** `tp_topktitles_unsafe_relpath_flag`; `tp_topktitles_title_embed_missing_flag` — безопасный путь, но **нет файла** или **ошибка чтения** `.npy` (не строка для dim mismatch / NaN / zero-norm — там свои флаги).
- **Качество вектора:** `tp_topktitles_dim_mismatch_flag`, `tp_topktitles_zero_norm_flag`, `tp_topktitles_nan_inf_flag`.
- **Сводные скоры:** `tp_topktitles_top1_score`, `tp_topktitles_topk_mean_score` — **NaN** при `present=0`.
- **`tp_topktitles_present`:** успешный поиск (1.0), иначе 0.0.

Списки **`topk_similar_ids`** / **`topk_similar_scores`** не входят в `features_flat`; они опционально лежат в **`result.topk_similar_corpus_titles`** (см. `export_topk_mode`).

## `result.topk_similar_corpus_titles`

На **всех** ветках **`extract()`** присутствует ключ **`corpus`** с метаданными: `corpus_spec_name`, `corpus_version`, `corpus_weights_digest`, `id_kind`, `corpus_size`, `dim`, **`backend`** (`faiss_hnsw_ip` | `numpy_cosine`), блок **`hnsw`** (параметры из конфига; при numpy backend не используется для поиска, но остаётся для трассировки).

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`** (sentence-transformer в рантайме не гоняется; идентификация корпуса — в **`corpus`** и digest в meta корпуса).
- **`system`**: **`pre_init` / `post_init`** после загрузки корпуса в **`__init__`**, **`post_process`** после **`extract()`**; **`gpu_peak_mb`**, **`ram_peak_mb`**.

## DAG

Заявленная зависимость: **TitleEmbedder**.

## Empty / strict

- По умолчанию **`require_title_embedding=False`**: отсутствие relpath / файла / невалидный файл → **valid empty** с флагами (без исключения).
- При **`require_title_embedding=True`**: соответствующие случаи → **RuntimeError** (fail-fast).

Для полного Text Audit v3 preflight рекомендуется профиль с **`require_title_embedding: true`** там, где title embedding обязан после пайплайна.

## Версионирование

Смена ключей или семантики → **`topk_similar_titles_extractor_output_v2`** + запись в `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
