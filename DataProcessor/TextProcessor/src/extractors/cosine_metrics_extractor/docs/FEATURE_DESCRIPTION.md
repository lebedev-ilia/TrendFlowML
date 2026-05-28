# `cosine_metrics_extractor` — описание фич и артефактов

**Компонент:** `CosineMetricsExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **39** скаляров `tp_cos_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/cosine_metrics_extractor_output_v1.json`](../../../../schemas/cosine_metrics_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.3.0 (`CosineMetricsExtractor.VERSION`).

---

## 1. Назначение

Косинус **между L2-нормализованными** (или сопоставимыми) векторами из `doc.tp_artifacts`: title / description / transcript `agg_mean` (выбор источника по `transcript_source_priority`: `whisper`, `youtube_auto`, `combined`) / comments (агрегаты mean&median **или** матрица N×D при `comments_mode=matrix`). **Модель не загружается.**

---

## 2. Порядок ключей

Фиксирован: `_build_features_flat_keys()` / `_FEATURES_FLAT_KEYS` в [`main.py`](../main.py) — совпадает с JSON (`allow_extra_keys: false`).

---

## 3. Группы полей

### 3.1. Presence (4)

`tp_cos_title_present`, `tp_cos_desc_present`, `tp_cos_transcript_present`, `tp_cos_comments_present` — 0/1, наличие загруженного вектора/матрицы.

### 3.2. Включение пар (4)

`tp_cos_*_enabled` для `title_desc`, `title_transcript`, `desc_transcript`, `transcript_comments_mean`, `transcript_comments_median` — зеркалят `compute_*` конструктора.

### 3.3. Require / политика (5)

`tp_cos_require_any_metric_enabled`, `require_title`, `require_description`, `require_transcript`, `require_comments_for_tc_enabled` — при `True` и отсутствии входов → **RuntimeError** (см. код).

### 3.4. «Пусто» при включённых метриках (4)

`tp_cos_empty_no_title`, `empty_no_desc`, `empty_no_transcript`, `empty_no_comments` — 0/1, только если соответствующий блок метрик **включён** и вектор отсутствует.

### 3.5. Диагностика (5)

`tp_cos_zero_norm_flag`, `tp_cos_dim_mismatch_flag`, `tp_cos_pair_dim_mismatch_flag`, `tp_cos_tc_dim_mismatch_flag`, `tp_cos_unsafe_relpath_flag` — 0/1 (traversal/размеры/норма).

### 3.6. Косинусы (5)

`tp_cos_title_desc`, `title_transcript`, `desc_transcript` — **NaN**, если вектор отсутствует, пары отключена, или dim/zero-norm.  
`tp_cos_transcript_comments_mean` / `median` — в режиме **aggregates**: косинус с mean/median; в **matrix**: агрегат по ряду sims (mean/median/nan* по коду), либо **NaN** при несовместимости.

Значения при finite: **∈ [−1, 1]**.

### 3.7. One-hot источника транскрипта (3)

`tp_cos_transcript_agg_source_{whisper,youtube_auto,combined}` — ровно **одна** 1.0, если `agg_mean` взят с этим ключом; иначе **все 0.0** (транскрипта нет / не выбран).

### 3.8. Режим комментариев (2)

`tp_cos_comments_mode_aggregates`, `tp_cos_comments_mode_matrix` — для `aggregates` / `matrix` соответственно 1.0, другой 0.0; **неизвестный** `comments_mode` → **0.0 / 0.0** (косины transcript↔comments — **NaN**).

### 3.9. Extra timing / matrix-статистика (5)

`tp_cos_emit_extra_metrics_enabled` (0/1)  
`tp_cos_load_ms`, `tp_cos_compute_ms` — при `emit_extra_metrics=False` → **NaN**; иначе мс.  
`tp_cos_tc_n_comments_used`, `tp_cos_tc_sims_std`, `tp_cos_tc_sims_p95` — при `emit_extra_metrics=False` → **NaN**; в **aggregates** или без matrix обычно **NaN**; в **matrix** с успешными sims — числа (std ≥ 0, p95 в [−1, 1] для косинусов); см. схему.

**`timings_s.total`** есть в ответе экстрактора, в `text_features.meta` per-компонент **нет**.

---

## 4. Нормальные диапазоны (`--ranges`)

| Группа | Ожидание |
|--------|----------|
| Флаги 0/1 (presence, enabled, require, empty_*, *flag*, one-hot, emit, comments_mode) | {0, 1} |
| Косинус-метрики (5 шт.) при finite | [−1, 1] |
| Сумма 3 one-hot transcript source | 0 **или** 1 |
| `tp_cos_comments_mode_aggregates` + `tp_cos_comments_mode_matrix` | оба 0, или ровно один 1 (типично) |
| `load_ms`, `compute_ms` при finite | ≥ 0, &lt; 1e7 |
| `tp_cos_tc_n_comments_used` при finite | ≥ 0 |
| `tp_cos_tc_sims_std` при finite | ≥ 0 |
| `tp_cos_tc_sims_p95` при finite | [−1, 1] |

---

## 5. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_cosine_metrics_extractor_text_npz.py`](../utils/validate_cosine_metrics_extractor_text_npz.py)  
- HTML: `text_processor/_render/cosine_metrics_extractor_report.html` ([`../render.py`](../render.py))

---

## 6. Чеклист

1. `meta.status=ok` → **39** имён в `feature_names` по схеме.  
2. `len(feature_values)==len(feature_names)`.  
3. При `emit_extra_metrics=False` в NPZ: `load_ms`, `compute_ms`, `tc_*` (три) — **NaN** (как в типичном прогоне).
