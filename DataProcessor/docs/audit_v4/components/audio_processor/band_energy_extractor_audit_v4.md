# Audit v4 — `band_energy_extractor`

**Дата:** 2026-04-06 (L2: 2026-04-06)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A** + **B**, ≥5 видео; **C** и полный §8 — не закрыты).  
**Reference A (воспроизводимый):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/band_energy_extractor/band_energy_extractor_features.npz`  
*Путь L1 (`…/4c3bf25b-e300-47b3-915e-4699c72ab190/…`) в текущем `result_store` отсутствует; для регрессий и отчёта используется доступный прогон того же `video_id`.*  
**Статистика L2 (JSON + figures):** `storage/audit_v4/band_energy_extractor_l2/` (`band_energy_extractor_audit_v4_stats.json`, `figures/`).  
**Tooling:** `DataProcessor/.data_venv/bin/python` `src/extractors/band_energy_extractor/scripts/audit_v4_npz_stats.py --seed 0` (список `--npz` — в JSON `paths`).

**Audit 4.2 (после L2):** инженерный журнал (профилирование/ускорение/мета): [`../audit_4_2/audio_processor/band_energy_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/band_energy_extractor_engineering_log_v4_2.md).

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | `SCHEMA.md`, код; Models — таблица §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первая волна AudioProcessor | ✓ | `band_energy_extractor` |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` (см. шапку; старый `4c3bf25b-…` недоступен) |
| **B** ≥5 видео | ✓ | 5 прогонов — пути в `band_energy_extractor_audit_v4_stats.json` → `paths` |
| **C** edge | ✗ | `audio_present=false`, все сегменты <0.5s, нет валидных окон |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2**, не L3 | ✓ | `RUN_LOG`: `in_progress (v4 L2)`; `passed` только после **C** + §8 |
| Нет заявления полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `dtype` / `shape` | ✓ | `band_edges_hz` (3,2), `band_energy_shares` (3), tabular из 3 имён |
| Сверка с `band_energy_extractor_npz_v1.json` | ◐ | Логически совпадает; CI/runtime validation не приложена |
| Сумма долей | ✓ | Код: [0.99, 1.01]; на **A** ≈ 1.0 |

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Три доли — непрерывные, сумма 1 | ✓ | Согласованы `band_energy_shares` и tabular |
| Строки → `as_float` / молчаливый NaN | ✓ | На профиле **A** строк в tabular нет |
| `band_dominant_band` как float при balance_metrics | N/A **A** | Индекс 0…2 в float — по `SCHEMA.md` |
| Ветка `<3` полос в савере | ◐ | NaN в tabular при неконсистентном размере — на **C**/багах |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN/Inf в shares/tabular на **A** | ✓ | 0 |
| NaN/Inf в shares/tabular на **B** (5 run) | ✓ | 0 |
| Нули vs missing | ✓ | Доли ≥ 0; missing не маскируется нулём (см. политику NaN в схеме) |
| При `time_series` и `segment_mask=false` | N/A **A** | Ожидаются NaN в строках матрицы |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 по полям на нескольких run | ✓ | `aggregate.tabular_per_feature` в JSON (**B**, 5 run) |
| Значения на **A** | ✓ | low ≈0.199, mid ≈0.734, high ≈0.066 |
| Линейная зависимость трёх долей | N/A | Одна степень свободы; избыточность осознанна для tabular API |

**Сводка по **B** (tabular, агрегат по видео):**

| Фича | min | max | mean | p50 |
|------|-----|-----|------|-----|
| `band_share_low` | 0.050 | 0.283 | 0.155 | 0.143 |
| `band_share_mid` | 0.619 | 0.928 | 0.778 | 0.792 |
| `band_share_high` | 0.022 | 0.098 | 0.068 | 0.066 |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки в `meta` | ✓ | `method`, `producer`, … не проталкиваются в float-вектор |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сегментные массивы в NPZ | N/A **A** | `time_series` выключен |
| `meta.duration` | ◐ | В артефактах L2 (segment e2e) было **`None`**; в коде после `2.1.1` заполнено span по сегментам (см. engineering log) |
| Union `frame_indices` | N/A | Аудио spectral family |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| ρ между tabular на **B** | ✓ | Pearson: `low↔mid` ≈ **−0.98**, `low↔high` ≈ **0.71**, `mid↔high` ≈ **−0.83** — ожидаемо при сумме долей ≈ 1; см. `figures/tabular_corr_heatmap.png` |
| ρ с другими экстракторами | ✗ | Не измерялось в этом прогоне |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Таблица наблюдений → выводы плана §4.7 | ◐ | §5–6 отчёта |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура долей / hash | ◐ | Tooling готово (без запуска): `AudioProcessor/src/extractors/band_energy_extractor/scripts/audit_v4_npz_stats.py --golden-npz <A.npz> --golden-out storage/audit_v4/band_energy_extractor_l2/golden_A.json --golden-round 8` |

#### §4.9 — Sampling, N, перекрытие

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Агрегат `nanmean` по валидным сегментам | ✓ | Код `run_segments` |
| Таблица duration × N × доли | ◐ | Разнообразие долей на **B** есть; явная таблица duration×N — **TODO** (в L2 `meta.duration` был `null`, после `2.1.1` ожидается заполнение) |
| Два прогона sampling policy | ✗ |
| Shared STFT (`spectral_extractor`) | ◐ | Поддержано в коде; на **A** не проверяли |

#### §4.10 — `empty` / `empty_reason` (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Нет «тихого ok» при пустом вводе | ✗ | Нужны артефакты **C** (тишина/нет речи/слишком короткое аудио/ошибка чтения). После появления NPZ: прогнать тем же скриптом `audit_v4_npz_stats.py` и проверить `meta.status`, `empty_reason`, отсутствие «ок» с бессмысленными долями. |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F=3 без balance_metrics |

#### §4.12 — Anti-leakage

| Вопрос (план) | Ответ | Комментарий |
|-----------------|-------|-------------|
| Только текущее видео/аудио? | Да | |
| Глобальная нормализация по датасету? | Нет | Peak normalization локально |
| Онлайн API? | Нет | |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1–§5.2 `docs/README.md` | ◐ | Каталог Audit v4; полный шаблон колонок плана — по желанию |
| §5.3 | ◐ | Ниже |

##### §5.3 — Сверка с Models (`band_energy_extractor`)

| Вопрос (план §5.3) | Ответ | Комментарий |
|--------------------|-------|-------------|
| В минимальном Baseline v1.0? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md): `clap_extractor`, `loudness_extractor`, `tempo_extractor`. |
| Tabular FeatureSpec / расширенный набор | **Частично** | Три доли — малый устойчивый вектор. |
| Dense / [`ENCODER_CONTRACT.md`](../../../../../Models/docs/contracts/ENCODER_CONTRACT.md) | **Частично** | Без `time_series` нет временной сетки в NPZ. |
| Segmenter / время | **Да** | `families.spectral`; маска коротких окон. |
| UI / analytics | **Да** | Наглядный баланс полос. |

#### §6 — Verdict

| Критерий | Статус |
|----------|--------|
| Блок §6 плана | ✓ |

#### §6.1 — Оценка, операционно

| Критерий | Статус |
|----------|--------|
| Баллы 0–10 | ✓ |
| Wall-time / RAM | ✗ |

#### §7 — Порядок работ

| Шаг | Статус |
|-----|--------|
| Идентификация + NPZ + статистика | ✓ |
| Скрипт `audit_v4_npz_stats.py`, **seed 0** | ✓ |
| §4 на L3 | ✗ |

#### §8 — Definition of Done

**Не закрыт (Level 3 / `passed`):** набор **C**, **§4.8** golden, **§4.10**, полная §4.9 (два sampling policy). Для **L2** (A+B) — статистика и артефакты зафиксированы в `storage/audit_v4/band_energy_extractor_l2/`, журнал и commit — в `RUN_LOG.md`.

#### §9–§11

| Раздел | Статус | Заметка |
|--------|--------|---------|
| §9 blocked | N/A | |
| §10 журнал | ◐ | **Git commit**, **B**, JSON/figures — в `RUN_LOG.md`; **C** — TODO |
| §11 | N/A | |

---

## 1. Мета (фрагмент)

| Поле | Значение |
|------|----------|
| `schema_version` | `band_energy_extractor_npz_v1` |
| `producer_version` | `2.1.0` |
| `status` | `ok` |
| `method` | `librosa` |
| `sample_rate` | 22050 |
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `features_enabled` | `[]` |
| `duration` (meta) | **`None`** |
| `models_used` | `[]` |

---

## 2. Tabular и массивы (набор **A**)

**`feature_names` / `feature_values` (3 скаляра):**

| Имя | Значение |
|-----|----------|
| `band_share_low` | 0.1993 |
| `band_share_mid` | 0.7342 |
| `band_share_high` | 0.0665 |

**`band_edges_hz`:** \([0,200)\), \([200,2000)\), \([2000, 11025]\) Hz (nyquist при 22050).

**`band_energy_shares`:** совпадает с tabular в пределах float; сумма ≈ **1.0**.

---

## 2b. Набор **B** (L2)

Пять NPZ перечислены в `storage/audit_v4/band_energy_extractor_l2/band_energy_extractor_audit_v4_stats.json` (`paths`). Гистограммы tabular и heatmap корреляций: `storage/audit_v4/band_energy_extractor_l2/figures/`.

---

## 3. Сверка с кодом

1. **STFT:** `|librosa.stft|²` по кадрам; частоты `librosa.fft_frequencies`.
2. **Маски полос:** `lo <= f < hi` (верхняя граница последней полосы вплоть до nyquist по конфигу).
3. **Энергия полосы:** сумма по частотным бинам × сумма по времени (`mask_matrix.T @ S`).
4. **Доли:** энергии / сумма энергий; валидация суммы ∈ [0.99, 1.01].
5. **`run_segments`:** короткие/битые сегменты — `segment_mask=false` и NaN в матрице (при экспорте time_series); глобальные shares — `nanmean` по оси сегментов.
6. **Audit v3:** `basic_stats` / `extended` / `dynamics` / mel / non-librosa — **fail-fast**.

---

## 4. Заметки по качеству артефакта

- Доминирует **mid** (~73%), типично для «полного» микса с речью/музыкой в среднем спектре.
- **`meta.duration=None` в L2:** для e2e сегментного пути длительность в meta не дублировалась. Не баг вычисления долей, но **пробел наблюдаемости** для downstream, который читает только meta. В коде после `2.1.1` это закрыто (см. engineering log).

---

## 5. Вердикт

**Плюсы:** дешёвые, интерпретируемые 3 фичи; строгий контракт; сумма долей контролируется; совместимость с общим STFT через `shared_features`.

**Минусы:** вне минимального Baseline v3 аудио-набора; без `time_series` нет оси для encoder; **C** (empty/короткое аудио) не закрыт; **duration** в meta для сегментного режима пустой; кросс-корреляции с loudness/spectral не измерялись.

---

## 6. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность / контракт | **9** |
| Полезность tabular для моделей | **7** |
| Полезность для encoder (dense) | **4** (без time_series) |
| Аналитика / UI | **8** |

**Итог: 7.5 / 10** (округлённо **8/10** для продукта как лёгкий спектральный сигнал; **L2** по **A+B** закрыт, для **L3** — **C**, golden §4.8, при необходимости `meta.duration`).
