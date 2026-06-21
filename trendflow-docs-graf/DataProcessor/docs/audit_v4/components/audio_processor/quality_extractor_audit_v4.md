# Audit v4 — `quality_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/quality_extractor/quality_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/quality_extractor_l2/quality_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`
**Engineering log 4.2:** [`../audit_4_2/audio_processor/quality_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/quality_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `quality_extractor_npz_v2.json`, `npz_savers/quality.py` |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Путь + `run_id` | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` |
| **B** ≥5 видео | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** edge | ✗ | |

#### §4.1 / §4.1a — Типы, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сегментная ось | ✓ | N=12, mask all true на **A** |
| Tabular до фикса | ◐ | **1 NaN** — `device_used` |
| После фикса | ✓ | **`device_used`** только **`meta`**; **F=15** (на **A** было **F=16**) |

#### §4.2 — Прочие NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Остальной tabular на **A** | ✓ | Конечные значения |

#### §4.4 — Строки

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Вне float-вектора | ✓ | После фикса |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash на **A** | ✗ | После повторного прогона |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F≈15 |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| ML в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) | **N/A** — инженерные метрики, без inference-моделей в типичном смысле |

#### §8 — DoD

**Не закрыт:** B+C, golden, полный §4.

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `quality_extractor_npz_v2` |
| `quality_contract_version` | `quality_contract_v1` |
| `device_used` | `cpu` |
| `features_enabled` | `basic_metrics`, `dynamic_metrics` |

---

## 2. Tabular (на **A**, до фикса), **F=16**

`sample_rate` … `dynamic_range_stability`; позиция **1** — **`device_used` = NaN** (устранено: убрано из `add()` в савере).

---

## 3. Код

- **`npz_savers/quality.py`:** удалён `add("device_used", …)`; устройство остаётся в **`meta`** через общий пайплайн `build_meta`.
- **`utils/render.py`:** summary уже берёт **`device_used`** из **`meta`** (tabular не используется для этого поля).

---

## 4. Вердикт

**Плюсы:** лёгкий quality-слой; понятные метрики; серии вынесены в `.npy` по путям в meta.

**Минусы:** тот же класс ошибки, что у mel/mfcc — строка в tabular (исправлено).

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / полезность | **8** |

**Итог: ~8/10** после повторного **A** и §4.8.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
