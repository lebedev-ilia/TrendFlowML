# `comments_embedder` — описание фич и артефактов

**Компонент:** `comments_embedder` (`CommentsEmbedder`, [`../main.py`](../main.py))  
**Вклад в NPZ:** **18** скаляров `tp_commentsemb_*` в `text_processor/text_features.npz` (`text_npz_v1`).  
**Контракт:** [`../../../../schemas/comments_embedder_output_v1.json`](../../../../schemas/comments_embedder_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия реализации:** 1.3.0 (`CommentsEmbedder.VERSION`).

---

## 1. Назначение

L2-нормализованные эмбеддинги **отобранных** комментариев (`VideoDocument.comments`) через `SentenceTransformer` из **`dp_models`** (без сети). Опционально: кэш по хешу, `extract` / `extract_batch`. Артефакты: `comments_embeddings.npy`, `comments_selected_indices.npy` (индексы исходных комментариев для `comments_aggregator`).

---

## 2. Два слоя: core (8) и extra (10)

| Слой | Ключи | Gating |
|------|-------|--------|
| **Core** | `present`, `count`, `dim`, `n_input`, `n_deduped`, `n_selected`, `total_chars_used`, `truncated_by_total_chars_flag` | Всегда в `features_flat`; при valid-empty / без encode часть — **0** / **NaN** (см. ниже) |
| **Extra** | `cache_enabled`, `cache_hit`, `fp16`, `device_cuda`, `model_digest_u24`, `compute_enabled`, `write_artifact_enabled`, `artifact_written`, `select_ms`, `encode_ms` | При **`emit_extra_metrics=False`** (типичный прогон) — все **10** **NaN** (после `_finalize_commentsemb_features_flat`) |

Особенности (см. [`SCHEMA.md`](../SCHEMA.md)):

- **`extract_batch`** + `emit_extra_metrics=True`: `tp_commentsemb_cache_hit` = **NaN** (единый encode, per-doc кеш не используется; не путать с «miss»).
- **`extract_batch`**: `tp_commentsemb_encode_ms` — **доля** wall-time batch-encode в мс, пропорциональная **числу** закодированных комментариев документа; **`timings_s.encode`** — доля в секундах.
- **`extract`**: `encode_ms` — полное wall-time encode документа в мс (при `emit_extra_metrics=True`).

---

## 3. Полный перечень (18)

| Ключ | Смысл |
|------|--------|
| `tp_commentsemb_present` | 1.0, если **есть** вычисленные эмбеддинги (путь encode с ненулевой выборкой); иначе 0 |
| `tp_commentsemb_count` | `N` строк в матрице; 0 на empty; **NaN**, если `compute_embeddings=False` (отбор был, encode не вызывали) |
| `tp_commentsemb_dim` | `D`; **NaN** при `present=0` (empty или без encode) |
| `tp_commentsemb_n_input` | комментариев после нормализации/фильтра, до дедупа |
| `tp_commentsemb_n_deduped` | после дедупа |
| `tp_commentsemb_n_selected` | выбрано политикой (до `max_total_chars` / `max_comments`) |
| `tp_commentsemb_total_chars_used` | суммарно символов в выбранных |
| `tp_commentsemb_truncated_by_total_chars_flag` | 0/1: обрезка по `max_total_chars` |
| `tp_commentsemb_cache_enabled` / `cache_hit` | extra; hit 0/1 в `extract` при кеше; batch → hit **NaN** при extra on |
| `tp_commentsemb_fp16` | 0/1 (fp16 только с CUDA) |
| `tp_commentsemb_device_cuda` | 0/1 |
| `tp_commentsemb_model_digest_u24` | `int(weights_digest[:6], 16)` — идентификатор весов |
| `tp_commentsemb_compute_enabled` / `write_artifact_enabled` | отражают конструктор |
| `tp_commentsemb_artifact_written` | 0/1 — успешно записан `comments_embeddings.npy` |
| `tp_commentsemb_select_ms` | время отбора (мс), extra |
| `tp_commentsemb_encode_ms` | мс encode (см. extract vs batch выше) |

**Тайминги** в `timings_s` (`extract` / `extract_batch`) **не** дублируются в `text_features.meta`; в NPZ только `tp_commentsemb_*_ms` при `emit_extra_metrics=True`.

---

## 4. Нормальные диапазоны (`--ranges`)

При `meta.status=ok` и **finite** значениях:

| Поле / группа | Ожидание |
|----------------|----------|
| `tp_commentsemb_present`, `truncated_*`, (extra, если не NaN) `cache_enabled`, `fp16`, `device_cuda`, `compute_enabled`, `write_artifact_enabled`, `artifact_written` | 0/1; `cache_hit` 0/1 или **NaN** (batch) |
| `n_input`, `n_deduped`, `n_selected`, `count` (если finite) | ≥ 0 |
| `total_chars_used` | ≥ 0 |
| `tp_commentsemb_dim` (если finite) | > 0 |
| `model_digest` | ≥ 0 (часто &lt; 2²⁴) |
| `select_ms`, `encode_ms` (если finite) | ≥ 0, &lt; 1e7 мс |

**Согласованность:** при `present=1` и finite: `count` ≈ `n_selected` (один и тот же набор комментариев).

**Пустой срез** при `meta.status=error`: валидно (пайплайн не дошёл до заполнения).

---

## 5. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)
- Валидатор: [`../utils/validate_comments_embedder_text_npz.py`](../utils/validate_comments_embedder_text_npz.py)
- HTML: `text_processor/_render/comments_embedder_report.html` ([`../render.py`](../render.py))

---

## 6. Чеклист

1. `status=ok` → ровно **18** имён схемы в `feature_names`.  
2. `emit_extra_metrics=false` → все **10** extra в NPZ **NaN** (как в референс-run).  
3. `len(feature_values)==len(feature_names)`.
