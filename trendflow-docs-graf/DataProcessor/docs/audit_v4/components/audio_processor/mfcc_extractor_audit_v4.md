# Audit v4 — `mfcc_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/mfcc_extractor/mfcc_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/mfcc_extractor_l2/mfcc_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`
**Engineering log 4.2:** [`../audit_4_2/audio_processor/mfcc_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/mfcc_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | `SCHEMA.md`, `schemas/mfcc_extractor_npz_v2.json`; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первая волна AudioProcessor | ✓ | `mfcc_extractor` |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` |
| **B** ≥5 видео | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** edge | ✗ | |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2 product stats**, не L3 | ✓ | `RUN_LOG`: L2 закрыт по **A+B** |
| Нет полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `dtype` / `shape` | ✓ | `mfcc_*` (13); сегменты (12, 13), (12,) |
| Сверка с `mfcc_extractor_npz_v2.json` | ◐ | Логически совпадает |

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular float32 | ◐ | На **A до фикса**: **1 NaN** — `device_used` (строка → `as_float`) |
| После фикса | ✓ | `device_used` только **`meta`**; **F=13** tabular (на **A** было **F=14**) |
| Render старых NPZ | ✓ | `utils/render.py`: NaN в tabular для `device_used` → fallback на `meta` |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Массивы на **A** | ✓ | Без NaN при полной маске |
| Старый tabular | ◐ | Устранённый `device_used` |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 | ✓ | `storage/audit_v4/mfcc_extractor_l2/mfcc_extractor_audit_v4_stats.json` |
| **A** | ✓ | См. §2 отчёта |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки не в float-векторе | ✓ | После фикса |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=12**, family `mfcc` | ✓ | `segment_*` + per-segment ряды |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Внутренние (между tabular на A+B) | ✓ | `storage/audit_v4/mfcc_extractor_l2/figures/tabular_corr_heatmap.png` |
| ρ с mel/spectral | ✗ | L3 / C / далее |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| §4.7 плана | ◐ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | **TODO** после повторного прогона **A** |

#### §4.9 — Sampling

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Два прогона policy | ✗ | **B** |

#### §4.10 — empty (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Артефакты | ✗ | |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F≈13 |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|-------|
| Только текущий клип? | Да |
| Глобальная нормализация по датасету? | Нет |
| Онлайн API? | Нет |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| README / SCHEMA | ◐ | Обновлены под `device_used` → meta |

##### §5.3 — Models

| Вопрос | Ответ | Комментарий |
|--------|-------|-------------|
| В минимальном Baseline? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) не перечисляет `mfcc_extractor`. |
| Encoder / dense | **Частично** | `mfcc_mean_by_segment` (12,13), статистики (13) |

#### §6 — Verdict

| Критерий | Статус |
|----------|--------|
| Блок §6 | ✓ |

#### §8 — DoD

**Не закрыт:** B+C, golden §4.8 на свежем **A**.

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `mfcc_extractor_npz_v2` |
| `mfcc_contract_version` | `mfcc_contract_v1` |
| `device_used` | `cuda` |
| `features_enabled` | `basic_features`, `deltas`, `time_series` |
| `mfcc_npy` | присутствует в meta (offline pointer) |

---

## 2. Tabular (артефакт **до** фикса): **F=14**, **1 NaN** — `device_used`

| Имя | Значение (пример) |
|-----|-------------------|
| `sample_rate` … `duration_sec` | как в NPZ |
| ~~`device_used`~~ | ~~NaN~~ (убрано из tabular) |
| `segments_count` | 12 |
| `mfcc_energy` … `mfcc_stability` | см. NPZ |

**Массивы:** `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max` (13); `mfcc_mean_by_segment` (12,13); `mfcc_energy_by_segment` (12); `delta_mean_by_segment` (12,13).

---

## 3. Код

1. **`npz_savers/mfcc.py`:** удалён `add("device_used", ...)`.
2. **`utils/render.py`:** если tabular `device_used` — float NaN, берётся `meta.device_used`.

---

## 4. Вердикт

**Плюсы:** компактный MFCC-блок; сегментная ось согласована; дельты и per-segment ряды при включённых флагах.

**Минусы:** тот же класс бага, что у mel/pitch — строка в tabular; вне минимального Baseline; L1 без B/C.

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / стабильность | **8** |
| Полезность для ML | **8** |
| Encoder | **7** |

**Итог: ~7.7 / 10** (≈ **8/10** после повторного **A** и §4.8).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
