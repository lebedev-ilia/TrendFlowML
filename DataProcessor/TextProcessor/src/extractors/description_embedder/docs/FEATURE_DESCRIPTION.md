# `description_embedder` — описание фич и артефактов

**Компонент:** `DescriptionEmbedder` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **19** скаляров `tp_descemb_*` в `text_processor/text_features.npz`. Плотный вектор — в `description_embedding.npy`, не в `feature_values`.  
**Контракт:** [`../../../../schemas/description_embedder_output_v1.json`](../../../../schemas/description_embedder_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.2.0 (`DescriptionEmbedder.VERSION`).

---

## 1. Назначение

Один **L2-нормализованный** эмбеддинг для `doc.description`: token-aware чанки (`shared_tokenizer_v1`), encode через `SentenceTransformer` (`dp_models`), pooling (`length_weighted_mean` | `mean` | `max` | `logsumexp`). Параметр `emit_extra_metrics` в конструкторе **не** добавляет отдельных ключей в `features_flat` (v1.2.0).

---

## 2. Полный перечень (19)

| Ключ | Смысл |
|------|--------|
| `tp_descemb_present` | 1.0, если вектор реально посчитан (`compute_embedding` и непустое описание после нормализации) |
| `tp_descemb_dim` | `D`; **NaN** при valid-empty или `compute_embedding=False` без encode |
| `tp_descemb_norm_raw` | L2 норма **до** финальной нормализации pooled; **NaN**, если `compute_raw_norm=False` или нет compute |
| `tp_descemb_l2_norm` | L2 норма итогового вектора (ожидается ≈ **1**, если `present=1`) |
| `tp_descemb_description_present` | 1.0, если после `normalize_whitespace` строка непустая |
| `tp_descemb_compute_enabled` | конфиг `compute_embedding` |
| `tp_descemb_write_artifact_enabled` | конфиг `write_artifact` ∧ `write_embedding_artifact` |
| `tp_descemb_artifact_written` | успешная запись `description_embedding.npy` |
| `tp_descemb_cache_enabled` | дисковый кеш |
| `tp_descemb_cache_hit` | с кеша: 0/1; на пустой ветке из шаблона — **NaN**; при `cache_enabled=False` на успешном encode ветке кладётся **0.0** (не «неизвестно») |
| `tp_descemb_fp16` | 0/1 (актуально при CUDA) |
| `tp_descemb_device_cuda` | 0/1 |
| `tp_descemb_model_digest_u24` | `int(weights_digest[:6], 16)` |
| `tp_descemb_pooling_length_weighted` | 1.0 iff `pooling_strategy == length_weighted_mean` |
| `tp_descemb_n_chunks` | число чанков; **NaN**, если encode не выполнялся |
| `tp_descemb_avg_chunk_tokens` | среднее токенов на чанк; **NaN**, если не применимо |
| `tp_descemb_chunk_ms` / `encode_ms` / `pool_ms` | фазы chunk / encode / pool в **мс**; **NaN** на ветках без соответствующего шага |

**Тайминги:** в ответе `extract()` есть `timings_s.total`; per-phase только в `tp_descemb_*_ms`. Агрегированный `text_features.meta` не хранит отдельно тайминги экстрактора.

---

## 3. Нормальные диапазоны (`--ranges`)

| Группа | Ожидание |
|--------|----------|
| Флаги 0/1 (при finite) | present, description_present, compute/write/cache/fp16/cuda/artifact/pooling_length_weighted |
| `cache_hit` (finite) | 0 или 1 |
| `tp_descemb_dim` (finite) | > 0 |
| `tp_descemb_l2_norm` (при `present=1`) | ≈ 1.0 (допуск ~1e-3) |
| `tp_descemb_norm_raw` (finite) | > 0 |
| `model_digest_u24` | ≥ 0 |
| `n_chunks` (finite) | ≥ 1 при multi-chunk; ≥ 0 |
| `avg_chunk_tokens` (finite) | ≥ 0 |
| `chunk_ms`, `encode_ms`, `pool_ms` (finite) | [0, 1e7] мс |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_description_embedder_text_npz.py`](../utils/validate_description_embedder_text_npz.py)  
- HTML: `text_processor/_render/description_embedder_report.html` ([`../render.py`](../render.py))

---

## 5. Чеклист

1. `meta.status=ok` → **19** имён из схемы в `feature_names`.  
2. `len(feature_values)==len(feature_names)`.
