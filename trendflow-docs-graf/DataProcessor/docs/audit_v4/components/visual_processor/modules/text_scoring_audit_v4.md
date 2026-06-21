# Audit v4 — `text_scoring` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/text_scoring_l2/text_scoring_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `text_scoring` (A+B)  
**Контракт:** [`VisualProcessor/schemas/text_scoring_npz_v2.json`](../../../../../VisualProcessor/schemas/text_scoring_npz_v2.json) · [`modules/text_scoring/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/text_scoring/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ось | ✓ | Segmenter **`text_scoring.frame_indices`**, **`times_s`** из `union_timestamps_sec` ([`SCHEMA.md`](../../../../../VisualProcessor/modules/text_scoring/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `text_scoring_npz_v2` | ✓ | Только разрешённые; опциональные **`ocr_raw`**, **`ocr_unique_elements`** присутствуют (пустые); **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N** | ✓ | **N=120** (своя политика семплинга модуля, не обязана совпадать с **N=48** у других VP-модулей) |
| **`text_present`** | ✓ | Скаляр **`True`** на **A** при разреженном тексте (**`text_presence`** true **~2.5%**) — глобальный флаг «в ролике был OCR», не «на каждом кадре» |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Последовательности | ✓ | **`frame_indices`**, **`times_s`**, **`text_presence`**, **`text_count_per_frame`**: без NaN |
| **`feature_values`** | ◐ | **10/35 ≈ 28.6%** NaN на **A** |
| Блок **CTA** (**`cta_timestamp`** … **`cta_last_position`**) | ✓ | Все NaN при **`cta_presence=0`** — ожидаемо |
| **`ocr_language_entropy`**, **`text_movement_speed`**, **`text_emphasis_peaks_count`** | ✓ | NaN при **`enable_language_entropy` / `enable_text_movement_speed` / `enable_text_peaks`** = **False** в **`meta`** |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.4 — Privacy / debug

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`retain_raw_ocr_text`**, **`store_debug_objects`** | ✓ | Оба **False** на **A**; **`ocr_raw`**, **`ocr_unique_elements`** длины **0**; счётчики **`ocr_raw_count`**, **`ocr_unique_elements_count`** при этом **>0** (агрегаты без хранения строк) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **пустой список** (OCR/эвристики могут не попадать в запись — стоит унифицировать трассировку, если в других модулях список заполняется) |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **согласованы**, `manifest.status=ok`, `schema_version=text_scoring_npz_v2`, `producer_version=2.0.1`.  
Разреженный текст отражён в `text_presence` (ratio **0.025…0.1333** на A+B); `ocr_raw`/`ocr_unique_elements` пустые на всех 5 (privacy defaults).  
NaN в `feature_values` на A+B стабильно **10/35** (CTA блок при `cta_presence=0` + отключённые фичефлаги peaks/entropy/speed).

**Оценка:** **~8.1 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; сценарии с **`store_debug_objects=true`**, включёнными peaks/entropy/speed.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **600** (по **120** на run)
- **F**: **35** на всех
- **text_present**: `True` на всех 5
- **text_presence ratio**: min **0.025**, max **0.1333**
- **text_count_sum_total**: **77**
- **feature_values NaN**: **50/175** (то есть **10/35** на каждый run)
- **ocr_raw / ocr_unique_elements len**: **0 / 0** на всех 5

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 120 |
| `text_presence` true ratio | ~0.025 |
| `text_count_per_frame` sum | 3 |
| F | 35 |
| `feature_values` NaN count | 10 |
| `ocr_raw` / `ocr_unique_elements` len | 0 / 0 |
| `ocr_raw_count` / `ocr_unique_elements_count` | 3 / 2 |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
