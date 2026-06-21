# Audit v4 — `onset_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/onset_extractor/onset_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/onset_extractor_l2/onset_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`
**Engineering log 4.2:** [`../audit_4_2/audio_processor/onset_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/onset_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `onset_extractor_npz_v2.json`, `npz_savers/onset.py` |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Путь + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` |
| **B** ≥5 видео | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** edge | ✗ | |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| L2 product stats | ✓ | `RUN_LOG`: L2 закрыт по **A+B** |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сегментная ось + tabular | ✓ | N=12, mask all true (семантика см. `SCHEMA.md`) |
| Сверка JSON-схемы | ◐ | Добавлен `meta.optional_keys.backend` |

#### §4.1a — Строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| До фикса | ◐ | **1 NaN** — `backend` (`librosa`/`essentia` → `as_float`) |
| После фикса | ✓ | `backend` в **`meta`**; tabular **F=19** (на **A** было **F=20**) |
| Старый NPZ без `meta.backend` | ◐ | Рендер: `unknown` до повторного прогона |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Числовые поля на **A** кроме `backend` | ✓ | 0 NaN в остальном tabular |

#### §4.4 — Категориальные

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки вне float-вектора | ✓ | После фикса |

#### §4.8 — Golden (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | После повторного **A** |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F≈19 |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|-------|
| Только текущий клип? | Да |
| Глобальная нормализация? | Нет |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| ML в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) | **N/A** — сигнальная обработка, `models_used` пустой |

#### §8 — DoD

**Не закрыт:** B+C, golden, полный §4 плана.

---

## 1. Мета (набор **A**, фрагмент)

| Поле | Значение |
|------|----------|
| `schema_version` | `onset_extractor_npz_v2` |
| `onset_contract_version` | `onset_contract_v1` |
| `device_used` | `cpu` |
| `features_enabled` | `basic_features`, `interval_stats`, `rhythmic_metrics`, `time_series` |
| `backend` (до фикса в meta) | *отсутствует* |

---

## 2. Tabular (на **A**, до фикса)

Примеры: `onset_count` 67, `onset_density_per_sec` ~3.03, интервальная статистика, ритмические метрики, `sample_rate` 22050, `hop_length` 512, `duration` ~22.09, `segments_count` 12, **`backend` = NaN**.

---

## 3. Код

1. **`npz_savers/onset.py`:** убран `add("backend", …)`; `backend` пишется в **`meta_extra`** строкой.
2. **`utils/render.py`:** если `features["backend"]` нет / `None`, используется **`meta.backend`**.
3. **`schemas/onset_extractor_npz_v2.json`:** `backend` в **`meta.optional_keys`**.

---

## 4. Вердикт

**Плюсы:** компактный rhythm-слой; явный контракт; большие `onset_times` вне NPZ (`onset_times_npy`).

**Минусы:** категориальный `backend` ошибочно шёл в tabular (исправлено); README ранее ссылался на устаревшую схему — выровнено на `onset_extractor_npz_v2`.

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / стабильность | **8** |
| Полезность для rhythm / ML | **8** |

**Итог: ~8/10** при повторном **A** и заполнении `meta.backend` + §4.8.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
