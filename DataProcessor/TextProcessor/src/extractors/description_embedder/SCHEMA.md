# `description_embedder` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `description_embedder` |
| Класс | `DescriptionEmbedder` |
| Machine schema | `DataProcessor/TextProcessor/schemas/description_embedder_output_v1.json` |
| `schema_version` (логический контракт `features_flat`) | `description_embedder_output_v1` |
| Версия реализации | `1.2.0` (см. `DescriptionEmbedder.VERSION`) |

## Назначение

Считать **один L2-нормализованный эмбеддинг** для **`doc.description`**: длинный текст режется **token-aware** чанками через **`shared_tokenizer_v1`**, чанки кодируются моделью, затем **pooling** (`length_weighted_mean`, `mean`, `max`, `logsumexp` — параметр `pooling_strategy`). Итоговый вектор — в **`description_embedding.npy`** (опционально) + метрики в **`result.features_flat`** (`tp_descemb_*`, **19** ключей).

## Audit v3 preflight (модель)

Как у title/description embedder’ов: в полном TextProcessor preflight каноническая модель — **`intfloat/multilingual-e5-large`** ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)). Фактическая модель задаётся **`model_name`** в прогоне / конфиге.

## Upstream

- **`TagsExtractor`** (если включён раньше): описание может быть **очищено от inline `#тегов`**; этот экстрактор **не** мутирует `doc`.
- Пустое описание: **валидный empty** (без фейк-вектора, без записи в `tp_artifacts`). Отдельного **`require_description`** в текущей версии **нет** (в отличие от опционального `require_title` у `TitleEmbedder`).

## Артефакты vs `features_flat`

| Что | Где |
|-----|-----|
| Вектор dim D, float32, L2-normalized | `description_embedding.npy` при `write_artifact=true` + `doc.tp_artifacts["embeddings"]["description"]` |
| Скаляры | `result.features_flat` |

В **`result`** нет сырого вектора.

## Полный перечень `features_flat`

Source of truth: `main.py` → `_stable_features_template()` и `features_flat.update(...)` после encode/pool.

| Ключ | Смысл |
|------|--------|
| `tp_descemb_present` | 1.0 если эмбеддинг посчитан |
| `tp_descemb_dim` | Размерность или NaN |
| `tp_descemb_norm_raw` | L2 норма **до** финальной нормализации pooled вектора, если `compute_raw_norm`; иначе NaN |
| `tp_descemb_l2_norm` | L2 норма итогового вектора (ожидается ≈1) |
| `tp_descemb_description_present` | 1.0 если описание непустое после нормализации |
| `tp_descemb_compute_enabled` | `compute_embedding` |
| `tp_descemb_write_artifact_enabled` | `write_artifact` |
| `tp_descemb_artifact_written` | артефакт записан |
| `tp_descemb_cache_enabled` | дисковый кеш |
| `tp_descemb_cache_hit` | 0/1 или NaN по веткам |
| `tp_descemb_fp16` | fp16 на GPU |
| `tp_descemb_device_cuda` | CUDA |
| `tp_descemb_model_digest_u24` | префикс `weights_digest` как float |
| `tp_descemb_pooling_length_weighted` | 1.0 если `pooling_strategy == length_weighted_mean` (one-hot для этой стратегии; другие стратегии → 0.0) |
| `tp_descemb_n_chunks` | число чанков после chunking |
| `tp_descemb_avg_chunk_tokens` | среднее число токенов на чанк |
| `tp_descemb_chunk_ms` | время chunk (мс) |
| `tp_descemb_encode_ms` | время encode (мс) |
| `tp_descemb_pool_ms` | время pool (мс) |

Параметр **`emit_extra_metrics`** в v1.2.0 **не добавляет** ключей в `features_flat`.

## Версионирование

Изменение набора или смысла ключей → bump **`description_embedder_output_v2`** + `RUN_LOG.md` + отчёт компонента.
