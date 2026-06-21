# `embedding_shift_indicator_extractor` — описание фич и артефактов

**Компонент:** `EmbeddingShiftIndicatorExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **27** скаляров `tp_embshift_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/embedding_shift_indicator_extractor_output_v1.json`](../../../../schemas/embedding_shift_indicator_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.3.0 (`EmbeddingShiftIndicatorExtractor.VERSION`).

---

## 1. Назначение

Косинус **усреднённого начала** и **усреднённого конца** матрицы эмбеддингов чанков транскрипта (`cosine_begin_end`); бинарный **`shift_flag`**, если `cosine < cosine_threshold`. Опционально: косинус **первый↔последний** чанк и среднее косинусов **хвоста к start window**. Модель не грузится; векторы читаются из `*.npy` по `doc.tp_artifacts`.

---

## 2. Группы полей

| Группа | Ключи (идея) | Заметки |
|--------|----------------|--------|
| Сводка | `tp_embshift_present` | **1.0** только если **`cosine_begin_end`** конечен; иначе **0.0** |
| Порог / косинус / флаг | `cosine_threshold`, `cosine_begin_end`, `shift_flag`, `margin` | `margin` = `cosine_begin_end - cosine_threshold` (оба конечны); косинусы **∈ [−1, 1]** при конечных значениях |
| Окна | `n_window_chunks` | `min(config.n_window_chunks, max(1, n_chunks // 2))` |
| Gating (конфиг) | `*_enabled`, `require_min_chunks` | `emit_extra_metrics` → `tp_embshift_emit_extra_metrics_enabled` |
| Доп. косинусы | `cosine_first_last`, `mean_cosine_last_to_start_window` | **NaN**, если `compute_extra_cosines=False` |
| Источник | `source_used_whisper`, `source_used_youtube_auto`, `used_legacy_key_flag` | Только `whisper` / `youtube_auto` получают **1.0** на «своём» флаге; иной источник → **0/0** (сумма **0** или **1**) |
| Диагностика | `unsafe_relpath`, `chunk_embed_missing`, `dim_mismatch`, `zero_norm`, `nan_inf` | 0/1 |
| Тайминги в фичах | `load_ms`, `compute_ms` | **NaN** при **`emit_extra_metrics=False`**; иначе **≥ 0** (мс) при выставлении |

**Тайминг в ответе `extract`:** `timings_s.total` (сек.); в NPZ — только `tp_embshift_load_ms` / `tp_embshift_compute_ms` (если extra).

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (при finite) | `enabled`, `disabled_by_policy`, флаги require/compute/emit, диагностика, `source_used_*`, `used_legacy` |
| `tp_embshift_shift_flag` (finite) | **0.0** или **1.0** |
| `tp_embshift_present` (finite) | **0.0** или **1.0** |
| Косинусы: `cosine_begin_end`, `cosine_first_last`, `mean_cosine_last_to_start_window` | **[-1, 1]** |
| `tp_embshift_margin` (finite) | согласован с `cosine_begin_end - cosine_threshold` (допуск **1e-3**) |
| `tp_embshift_cosine_threshold` (tip.) | чаще **[0, 1]**, в конфиге может быть иначе |
| `tp_embshift_require_min_chunks` | **≥ 1** |
| `tp_embshift_n_chunks` | **≥ 0** |
| `tp_embshift_n_window_chunks`, `tp_embshift_dim` (finite) | **≥ 1** |
| `tp_embshift_load_ms`, `tp_embshift_compute_ms` | при **`emit_extra_metrics_enabled=0`** → **NaN**; при **1** и finite → **[0, 1e7]** мс |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_embedding_shift_indicator_extractor_text_npz.py`](../utils/validate_embedding_shift_indicator_extractor_text_npz.py)

---

## 5. Чеклист

1. `meta.status=ok` и успешный merge → **27** имён схемы в `feature_names` для среза `tp_embshift_*`.  
2. `len(feature_values) == len(feature_names)`.  
3. `tp_embshift_source_used_whisper + tp_embshift_source_used_youtube_auto` ∈ **{0, 1}** (при обоих finite).
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
