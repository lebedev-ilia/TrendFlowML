# Audit v4 — `loudness_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/loudness_extractor/loudness_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/loudness_extractor_l2/loudness_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`

**Engineering log 4.2 (после L2):** `DataProcessor/docs/audit_v4/components/audit_4_2/audio_processor/loudness_extractor_engineering_log_v4_2.md`

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `loudness_extractor_npz_v2.json`, `npz_savers/loudness.py` |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Компонент + run **A** | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Тот же `run_id` |
| **B** | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** | ✗ | Клипы без pyloudnorm, пустые окна, `audio_present=false` |

#### §3.1 — Уровень отчёта

| Критерий | Статус |
|----------|--------|
| L2 product stats | ✓ |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular **F=18**, порядок как в [`loudness.py`](../../../../AudioProcessor/src/core/npz_savers/loudness.py) | ✓ | Совпадает с `SCHEMA.md` |
| Сегменты **N=48** | ✓ | Family **`primary`** (не путать с N=12 у `chroma` на том же run) |
| `lufs_present` scalar | ✓ | **True** на **A** |

#### §4.1a — Строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN в `feature_values` на **A** | ✓ | **0** (все поля числовые) |
| `loudness_lufs` | ✓ | Конечно (~−22.7); при отсутствии LUFS — политика NaN + `lufs_present=false` в савере |

#### §4.2 — Сегментные массивы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `segment_mask` | ✓ | Все **48** `true` на **A** |
| NaN в `segment_rms` / `segment_lufs` / `segment_dbfs` | ✓ | **0** на **A** |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Перцентили на **A+B** | ✓ | `storage/audit_v4/loudness_extractor_l2/loudness_extractor_audit_v4_stats.json` |
| **A** | ✓ | `loudness_dbfs` ≈ −22.86; глобальный RMS ≈ 0.072 |

#### §4.4 — object / meta

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки в tabular | ✓ | Нет |
| `models_used` | ✓ | `[]` |

#### §4.5 — Временная ось / смысл метрик

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Глобальные + frame stats | ◐ | После загрузки **полного** wav (`run_segments`); **duration_sec** / **frames_count** относятся к полному клипу |
| Сегментные ряды | ✓ | По окнам **primary**; `segments_count` в tabular = **48** |

#### §4.6–§4.12, §7–§8

| Критерий | Статус |
|----------|--------|
| Корреляции (tabular A+B) | ✓ | `storage/audit_v4/loudness_extractor_l2/figures/tabular_corr_heatmap.png` |
| Golden, полный DoD, межкомпонентные корреляции | ✗ / ◐ | L3 / C |

#### §5 — Документация

| Подпункт | Статус |
|----------|--------|
| README / SCHEMA | ◐ | Таблица Audit v4 |

##### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|-------|
| В [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) audio trio? | **Да** (`loudness_extractor` вместе с `clap_extractor`, `tempo_extractor`) |
| Tabular / encoder | **Да** | Плотный скалярный вектор + ось **N** для token-ready пайплайнов |
| Segmenter | **Да** | `families.primary` |

#### §6 — Verdict

**Плюсы:** Tier‑0 baseline; чистый tabular на **A**; LUFS доступен; строгая маска и выравнивание **N**.

**Минусы:** L1 без **B/C**; число окон **primary** на run может сильно отличаться от других семейств — downstream должен опираться на свои **N** и маску.

#### §6.1 — Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность / контракт | **9** |
| Tabular для моделей | **9** |
| Encoder (с осью сегментов) | **9** |
| Аналитика | **8** |

**Итог: ~9/10** для продукта как обязательного аудио-сигнала громкости.

---

## 1. Мета (**A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `loudness_extractor_npz_v2` |
| `producer_version` | `2.1.0` |
| `status` | `ok` |

## 2. Tabular (**A**, выборочно)

| Имя | Значение (порядок) |
|-----|---------------------|
| `loudness_rms` | ~0.072 |
| `loudness_peak` | ~0.419 |
| `loudness_dbfs` | ~−22.86 |
| `loudness_lufs` | ~−22.69 |
| `frames_count` | **515** |
| `segments_count` | **48** |

## 3. Сверка с кодом

1. **`run_segments`**: окна → `_compute_from_np`; агрегаты RMS по **valid** маске; повторная загрузка **полного** трека для глобальных метрик и frame-wise статистик.
2. **Савер**: переименование `rms`→`loudness_rms` и т.д.; `lufs` + `lufs_present` согласованы с политикой NaN.

## 4. Код

На **A** дефектов уровня string→NaN в tabular **не** выявлено; изменений коду не вносилось.
