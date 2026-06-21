# `title_embedding_cluster_entropy_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `title_embedding_cluster_entropy_extractor` |
| Класс | `TitleEmbeddingClusterEntropyExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/title_embedding_cluster_entropy_extractor_output_v1.json` |
| `schema_version` | `title_embedding_cluster_entropy_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Энтропия распределения **title embedding** по общей таксономии кластеров: **PCA** и **центроиды** загружаются строго из **`dp_models`** (`clusters_spec_name`, по умолчанию `semantic_clusters_v1`). Входной вектор — **`doc.tp_artifacts["embeddings"]["title"]["relpath"]`** (per-run `.npy`). Отдельная **sentence-transformers** модель в рантайме **не** исполняется.

**ML политика (preflight §7):** сигнал опирается на замороженную taxonomy/digest в meta; до явной фиксации corpus pack как `model_facing` трактуйте как **analytics** downstream.

## `features_flat` (24 ключа)

Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** ↔ JSON (`allow_extra_keys: false`).

- **Зеркала конфигурации:** `tp_titleclent_emit_extra_metrics_enabled`, `tp_titleclent_require_title_embedding_enabled`, `tp_titleclent_use_faiss_enabled`, `tp_titleclent_require_faiss_enabled`, `tp_titleclent_export_topk_distribution_enabled`.
- **Top‑K:** `tp_titleclent_schema_top_k_slots_max` = **8** (канонический потолок Audit v3); `tp_titleclent_top_k_slots_requested` — из конфига; `tp_titleclent_top_k_slots` — после клампа; `tp_titleclent_top_k_slots_clamped` — **1.0** если запрошено > 8.
- **Метрики:** `tp_titleclent_entropy_raw`, `tp_titleclent_entropy_norm` (для **K≤1** нормировка **0.0**), `tp_titleclent_perplexity`, `tp_titleclent_top_k_used`, `tp_titleclent_distinct_clusters_topk` — **NaN** на empty / до успешного счёта.
- **Extra-блок** (`tp_titleclent_n_clusters`, `tp_titleclent_model_orig_dim`, `tp_titleclent_model_reduced_dim`, `tp_titleclent_margin_top2`, `tp_titleclent_compute_ms`): при **`emit_extra_metrics=False`** или на **empty** ветках → **NaN**.

## `title_cluster_entropy_meta` (в `result`)

`clusters_spec_name`, `clusters_spec_version`, `clusters_weights_digest`, `cluster_db_version`, `backend` (`faiss_ip` | `numpy_cosine`). При **`export_topk_distribution=True`** — поле **`topk`** (`cluster_ids`, `probs`, `scores`), без сырого текста.

## Ответ `extract`

- **`model_name` / `model_version` / `weights_digest`**: **`null`** (кластерная база не есть weights embedding-модели в этом контракте).
- **`system`**: **`pre_init` / `post_init`** из **`__init__`**, **`post_process`** после `extract`, **`gpu_peak_mb`** из снимков.

## Версионирование

Смена ключей или смысла полей → **`title_embedding_cluster_entropy_extractor_output_v2`** + запись в `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
