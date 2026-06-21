# Audit v4 — `spectral_entropy_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/spectral_entropy_extractor_l2/spectral_entropy_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/spectral_entropy_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/spectral_entropy_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `spectral_entropy_extractor_npz_v2`, `npz_savers/spectral_entropy.py` |

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

#### §4.1 / §4.1a

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular **F=2** | ✓ | `spectral_entropy_mean`, `spectral_entropy_std` |
| NaN в tabular на **A** | ✓ | **0** |
| Строки в tabular | ✓ | Нет; **`device_used`** в **`meta`** |

#### §4.2 — Per-segment

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=12**, mask all true на **A** | ✓ | `entropy_mean_by_segment`, `entropy_std_by_segment` (12,) |
| Опциональные ряды | N/A **A** | `features_enabled`: только `basic_stats` |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | TODO |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Тяжёлые DL в NPZ | **Нет** — классические спектральные признаки |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/spectral_entropy_extractor_l2/spectral_entropy_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/spectral_entropy_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/spectral_entropy_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `spectral_entropy_extractor_npz_v2` |
| `spectral_entropy_contract_version` | `spectral_entropy_contract_v1` |
| `device_used` | `cpu` |
| `features_enabled` | `basic_stats` |

В **`meta`** также echo: `sample_rate`, `n_fft`, `hop_length`, `n_mels`, и т.д. (см. `SCHEMA.md` §4).

---

## 2. Tabular (на **A**)

| Имя | Значение (пример) |
|-----|-------------------|
| `spectral_entropy_mean` | ~4.36 |
| `spectral_entropy_std` | ~1.62 |

---

## 3. Код

Савер **не** кладёт строки в `feature_values`. Исправлений не потребовалось. Обновлены **`SCHEMA.md`** (статус, §4 meta) и **`README.md`** (Audit v4).

---

## 4. Вердикт

**Плюсы:** очень компактный tabular + богатая per-segment ось без склейки длинных рядов; параметры в meta.

**Минусы:** L1 только; на **A** не задействованы опциональные per-segment flatness/spread.

---

## 4.5. Audit 4.2 — engineering log (после L2)

[`../audit_4_2/audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/spectral_entropy_extractor_engineering_log_v4_2.md)

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / выравнивание оси | **9** |
| Полезность для спектральной аналитики | **8** |

**Итог: ~8.5/10** при §4.8 и L2.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
