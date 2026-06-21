# `comments_aggregator` — описание фич и артефактов

**Компонент:** `comments_aggregator` (`CommentsAggregationExtractor`, [`../main.py`](../main.py))  
**Вклад в NPZ:** **39** скаляров в `text_processor/text_features.npz` — три семейства имён (canonical + legacy), порядок в коде: [`_FEATURES_FLAT_KEYS`](../main.py).  
**Контракт:** [`../../../../schemas/comments_aggregator_output_v1.json`](../../../../schemas/comments_aggregator_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия реализации:** 1.3.0 (`CommentsAggregationExtractor.VERSION`).

---

## 1. Назначение

Агрегация **уже посчитанной** матрицы эмбеддингов комментариев (`CommentsEmbedder` → `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`) в два вектора: **взвешенное среднее** (опциональные веса через `selected_indices` и `comments_likes` / `comments_authority` / `comments_recency`) и **помедианный по осям**; L2-нормализация векторов. Forward модели **нет**; `model_name` в `dp_models` — только согласование пространства с эмбеддером.

---

## 2. Полный перечень полей (39)

### 2.1. `tp_commentsagg_*` (22, canonical)

| Ключ | Смысл |
|------|--------|
| `tp_commentsagg_present` | 0/1: есть агрегат (вектор mean и/или median реально вычислен) |
| `tp_commentsagg_count` | `N` строк эмбеддингов, участвовавших в mean (и в median при успехе — совпадает) |
| `tp_commentsagg_dim` | `D` размерности; **NaN** на ветке valid-empty (нет валидной матрицы) |
| `tp_commentsagg_mean_std` | при `compute_std` и `compute_mean`: mean по осям от `std` по комментариям; иначе **NaN** |
| `tp_commentsagg_median_std` | аналогично для median |
| `tp_commentsagg_compute_mean_enabled` | конфиг: `compute_mean` |
| `tp_commentsagg_compute_median_enabled` | конфиг: `compute_median` |
| `tp_commentsagg_compute_std_enabled` | конфиг: `compute_std` |
| `tp_commentsagg_write_artifacts_enabled` | конфиг: `write_artifacts` |
| `tp_commentsagg_require_comment_embeddings_enabled` | конфиг: `require_comment_embeddings` |
| `tp_commentsagg_artifact_mean_written` | 0/1: записан `comments_agg_mean.npy` |
| `tp_commentsagg_artifact_median_written` | 0/1: записан `comments_agg_median.npy` |
| `tp_commentsagg_weights_applied` | 0/1: хотя бы один вес (likes/auth/recency) реально вошёл в произведение весов |
| `tp_commentsagg_weights_mask_likes` / `authority` / `recency` | 0/1: какой компонент применялся (при выравнивании индексов) |
| `tp_commentsagg_weights_align_present` | 0/1: прочитан `selected_indices_relpath` |
| `tp_commentsagg_weights_align_shape_ok` | 0/1: `len(idx)==N` и N>0 |
| `tp_commentsagg_dim_mismatch_flag` | 1.0, если `np.ndarray` был, но форма не (N>0, D>0) |
| `tp_commentsagg_unsafe_relpath_flag` | 1.0: relpath вне `artifacts_dir` (path traversal) |
| `tp_commentsagg_agg_mean_ms` / `tp_commentsagg_agg_median_ms` | мс агрегации при `emit_extra_metrics=True`; иначе **NaN**; **NaN**, если соответствующий `compute_*` выключен |

### 2.2. `tp_comments_agg_*` (12) и `tp_cagg_*` (5)

Полные зеркала canonical-полей для обратной совместимости (`tp_comments_agg_compute_std` / `compute_mean` / `compute_median` — **флаги**, не std-значения).  
Семантически: `tp_commentsagg_count` = `tp_comments_agg_count` = `tp_cagg_count`, и т.д. для пары std-слотов и present.

---

## 3. Тайминги

| Где | Что |
|-----|-----|
| `result.timings_s` (ответ `extract()`) | `total`, и при успехе `mean` / `median` (сек) для шагов агрегации |
| `tp_commentsagg_agg_*_ms` | только при **`emit_extra_metrics=True`**; в **агрегированном** `text_features.meta` **нет** per-extractor таймингов — см. `schema_version: text_npz_v1` |

---

## 4. Нормальные диапазоны (`--ranges`)

При **`meta.status=ok`**, валидные finite-значения:

| Группа | Ожидание |
|--------|----------|
| Флаги `0/1` | `present`, `*_enabled`, `artifact_*_written`, `weights_*`, `*_mismatch*`, `unsafe_relpath` |
| `tp_commentsagg_count` (и зеркала) | ≥ 0 |
| `tp_commentsagg_dim` (и зеркала) | при `present=1` — finite **> 0**; при valid-empty — **NaN** |
| `*_mean_std`, `*_median_std` | **NaN** или ≥ 0 (при `compute_std=false` — **NaN**) |
| `tp_commentsagg_agg_mean_ms` / `agg_median_ms` | **NaN** или ≥ 0 (и разумно &lt; 1e7 мс) |

**Согласованность зеркал:** при full slice `tp_commentsagg_count` == `tp_comments_agg_count` == `tp_cagg_count` (и аналогично для `present`, `dim`, std-слотов).

**Пустой срез** при `meta.status=error`: ожидаемо, если пайплайн не заполнил `feature_names` (см. audit v4).

---

## 5. Инструменты

- L2 stats: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)
- Валидатор: [`../utils/validate_comments_aggregator_text_npz.py`](../utils/validate_comments_aggregator_text_npz.py)
- HTML: `text_processor/_render/comments_aggregator_report.html` ([`../render.py`](../render.py))

---

## 6. Чеклист

1. `meta.status=ok` → **ровно 39** имён из `comments_aggregator_output_v1.json` в `feature_names`.
2. `len(feature_values)==len(feature_names)`.
3. Три семейства — взаимные копии по смыслу для count/present/dim/std.
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
