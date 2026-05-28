# Audit v4 — `source_separation_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/source_separation_extractor_l2/source_separation_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/source_separation_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/source_separation_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `source_separation_extractor_npz_v2.json`, `npz_savers/source_separation.py` |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Путь + `run_id` | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`) |
| **B** | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** | ✗ | TODO |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular **F=11** | ✓ | Совпадает с замороженным списком в `SCHEMA.md` |
| `share_mean`[4] / `source_order`[4] | ✓ | Согласованы с tabular долями на **A** |
| `dominant_source_id` | ✓ | Float в tabular, значение **0…3** (на **A**: **3** = `other`) |

#### §4.1a — Строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN в tabular на **A+B** | ◐ | На 4/5 прогонов `status=empty` (`empty_reason=audio_silent`) → много NaN по долям/доминированию; на **A** (ok) NaN = 0 |
| Строки в tabular | ✓ | Нет; **`device_used`**, **`model_name`**, **`weights_digest`** в **`meta`** |

#### §4.2 — Суммы / доли

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Четыре доли в tabular | ✓ | Сумма ≈ **1.0** на **A** (~1.000) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N** сегментов | ◐ | На **A**: **N=1** (family `source_separation`); ось **[0, ~12.03]** с |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | TODO |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Baseline / контракты | **Частично** — отдельный torch in-process stem; `models_used` / `model_signature` в **`meta`** по пайплайну |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/source_separation_extractor_l2/source_separation_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/source_separation_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/source_separation_extractor_l2/figures/tabular_corr_heatmap.png`

Замечание по набору B: большая доля прогонов имеет `status=empty` из-за `empty_reason=audio_silent`, поэтому агрегации по долям/доминированию следует интерпретировать с учётом mask по `meta.status`.

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `source_separation_extractor_npz_v2` |
| `source_separation_contract_version` | `source_separation_contract_v1` |
| `device_used` | `cuda` |
| `features_enabled` | `share_sequence`, `share_std` |
| `model_name` / `weights_digest` | присутствуют в meta |

---

## 2. Tabular и массивы (на **A**)

**Имена:** `share_vocals_mean` … `share_other_mean`, `dominant_source_id`, `dominant_source_share`, `source_balance_score`, `source_transitions_count`, `source_stability_score`, `segments_count`, `sample_rate` (44100).

**Массивы:** `share_mean` (4), `share_std` (4), `share_sequence` (1,4), `source_distribution_ratio`, `source_duration_sec`, `source_segments_count`, `source_order` (object).

---

## 3. Код

Исправлений класса «строка / категория в float tabular» **не потребовалось**. Уточнены **`SCHEMA.md`** (§5 meta) и **`README.md`** (Audit v4).

---

## 4. Audit 4.2 — engineering log (после L2)

[`../audit_4_2/audio_processor/source_separation_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/source_separation_extractor_engineering_log_v4_2.md)

---

## 5. Вердикт

**Плюсы:** дисциплинированный tabular; доли и доминирование явны; `source_order` зафиксирован; masking policy в схеме.

**Минусы / наблюдения:** на одном и том же video/run **N** может отличаться от других экстракторов из-за family — задокументировано; L1 без B/C и golden.

---

## 6. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / типы | **9** |
| Полезность для анализа микса | **8** |

**Итог: ~8.5/10** при §4.8 и L2.
