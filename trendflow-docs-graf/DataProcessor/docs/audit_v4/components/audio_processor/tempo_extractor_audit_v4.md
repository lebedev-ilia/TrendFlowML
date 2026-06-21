# Audit v4 — `tempo_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/tempo_extractor_l2/tempo_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/tempo_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/tempo_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + вердикт | ✓ | Отчёт, `SCHEMA.md` / `README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `tempo_extractor_npz_v1`, `npz_savers/tempo.py`, `tempo_extractor/__init__.py` |

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

#### §4.1a — Tabular, строки

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **F=11**, NaN **0** на **A** | ✓ | |
| Строки в tabular | ✓ | Нет; **`device_used`** в **meta** |

#### §4.2 — Ось и полнотрековые ряды

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=12** | ✓ | `bpm_by_segment`, маска все true на **A** |
| `tempo_estimates` | ◐ | Длина **519** — глобальная оценка librosa; не путать с **N** окон Segmenter |

#### §4.5 — `duration_sec`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Семантика | ◐ | На **A** совпадает с **max(segment_end_sec)** (~12.03 с); при расхождении с другими экстракторами сверять источник WAV / загрузчик |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | TODO |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Baseline | В [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) **tempo** входит в минимальный набор |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/tempo_extractor_l2/tempo_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/tempo_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/tempo_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `tempo_extractor_npz_v1` |
| `device_used` | `cpu` |
| `features_enabled` | *отсутствует* |
| `tempo_contract_version` | *отсутствует* |

---

## 2. Tabular (на **A**)

`tempo_bpm` ~107.67, агрегаты mean/median/std, `tempo_confidence` ~0.85, `duration_sec` ~12.03, `sample_rate` 22050, статистики по `bpm_by_segment`, `segments_count` 12.

---

## 3. Код

Исправлений не потребовалось. Обновлены **`docs/SCHEMA.md`**, **`docs/README.md`** (Audit v4).

---

## 4. Вердикт

**Плюсы:** стабильный baseline BPM; чистый tabular; canonical axis + пер-трековые `tempo_estimates` для отладки.

**Минусы / наблюдения:** meta без `features_enabled`/`contract` — меньше единообразия с другими компонентами; L1 без B/C.

---

## 4.5. Audit 4.2 — engineering log (после L2)

- Engineering log: [`../audit_4_2/audio_processor/tempo_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/tempo_extractor_engineering_log_v4_2.md)

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / baseline-роль | **9** |

**Итог: ~8.5/10** при §4.8 и выравнивании meta при необходимости.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
