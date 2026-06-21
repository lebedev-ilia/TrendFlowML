# Audit v4 — `chroma_extractor`

**Дата:** 2026-04-06 (L2: 2026-04-12)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A** + **B**, ≥5 видео; **C** и полный §8 — не закрыты).  
**Reference A (воспроизводимый):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/chroma_extractor/chroma_extractor_features.npz`  
*Путь L1 (`…/4c3bf25b-e300-47b3-915e-4699c72ab190/…`) в текущем `result_store` может отсутствовать; для регрессий и отчёта используется доступный прогон того же `video_id`.*  
**Статистика L2 (JSON + figures):** `storage/audit_v4/chroma_extractor_l2/` (`chroma_extractor_audit_v4_stats.json`, `figures/`).  
**Tooling:** `DataProcessor/.data_venv/bin/python` `src/extractors/chroma_extractor/scripts/audit_v4_npz_stats.py --seed 0` (список `--npz` — в JSON `paths`).

**Audit 4.2 (после L2):** инженерный журнал (profiling/мета): [`../audit_4_2/audio_processor/chroma_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/chroma_extractor_engineering_log_v4_2.md).

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | `SCHEMA.md`, `chroma_extractor_npz_v1.json`; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первая волна AudioProcessor | ✓ | `chroma_extractor` |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190` |
| **B** ≥5 видео | ✓ | 5 прогонов — пути в `chroma_extractor_audit_v4_stats.json` → `paths` |
| **C** edge | ✗ | `audio_present=false`, все сегменты пустые, срыв CQT |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2**, не L3 | ✓ | `RUN_LOG`: `in_progress (v4 L2)`; `passed` только после **C** + §8 |
| Нет заявления полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `dtype` / `shape` | ✓ | Tabular F=16; `chroma_mean` (12); сегменты (12,12), mask (12,) |
| Сверка с `chroma_extractor_npz_v1.json` | ◐ | Ключи совпадают; CI/runtime validation не приложена |
| Согласованность `dominant` | ✓ | `argmax(chroma_mean)==chroma_dominant_class` на **A** |

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `chroma_dominant_class` int vs float tabular | ✓ | Класс только в `int32` массиве; tabular без string/as_float артефактов |
| NaN в tabular на **A** | ✓ | 0 |
| Сумма `chroma_mean` (сегментный e2e) | ◐ | На **A** = **1.0** (взвешенная агрегация L1-кадров); на `run()` может отличаться |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN/Inf в основных массивах на **A** | ✓ | 0 в `chroma_mean`, scalars, tabular |
| `chroma_mean_by_segment` при mask | N/A **A** | Все 12 сегментов `true`; **C** — ожидаемы NaN в строках |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 на нескольких run | ✗ | **B** |
| Значения на **A** | ✓ | entropy ≈ 2.48; stability ≈ 0.97; contrast ≈ 0.038 |

**Сводка по **B** (tabular, агрегат по видео):**

| Фича | min | max | mean | p50 |
|------|-----|-----|------|-----|
| `chroma_entropy` | 2.423 | 2.480 | 2.464 | 2.474 |
| `chroma_harmonic_stability` | 0.972 | 0.986 | 0.978 | 0.976 |
| `chroma_contrast` | 0.025 | 0.122 | 0.055 | 0.041 |
| `chroma_dominant_energy` | 0.096 | 0.168 | 0.118 | 0.108 |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки в `meta` | ✓ | `chroma_type`, `normalize`, `producer`, … не в float-векторе |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сегментные массивы | ✓ | `segment_centers_sec`, `durations`, mask, `chroma_mean_by_segment` |
| Полный `chroma[12,T]` | N/A **A** | Режим `run_segments()` — ключа `chroma` нет по контракту; `features_enabled` всё же `['time_series']` |
| `meta.duration_sec` | ✓ | На **A** ≈ 22.09 с (сумма длительностей сегментов в payload) |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| ρ внутри tabular на **B** | ✓ | `storage/audit_v4/chroma_extractor_l2/figures/tabular_corr_heatmap.png` (16 фич) |
| ρ с `key`/`pitch`/spectral | ✗ | Не измерялось в этом прогоне |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Таблица наблюдений → §4.7 | ◐ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Хеш вектора / метаданных | ✗ | TODO |

#### §4.9 — Sampling, N, перекрытие

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Взвешивание по `segment_durations_sec` | ✓ | Код `run_segments` |
| Два прогона sampling policy | ✗ |
| Shared STFT | N/A | librosa CQT/STFT внутри компонента |

#### §4.10 — `empty` / `empty_reason` (**О**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Артефакты пустого ввода | ✗ | **C** |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | ◐ | F=16 tabular + 12-вектор; при желании dense — сегментная матрица |

#### §4.12 — Anti-leakage

| Вопрос (план) | Ответ | Комментарий |
|-----------------|-------|-------------|
| Только текущее видео/аудио? | Да | |
| Глобальная нормализация по датасету? | Нет | L1 по кадрам локально |
| Онлайн API? | Нет | |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1–§5.2 `docs/README.md` | ◐ | Таблица Audit v4; разделение `run()` / `run_segments` для ключа `chroma` |
| §5.3 | ◐ | Ниже |

##### §5.3 — Сверка с Models (`chroma_extractor`)

| Вопрос (план §5.3) | Ответ | Комментарий |
|--------------------|-------|-------------|
| В минимальном Baseline v1.0? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md): `clap_extractor`, `loudness_extractor`, `tempo_extractor`. |
| Tabular / FeatureSpec | **Частично** | 12 классов + скаляры — устойчивый компактный вектор. |
| Dense / encoder | **Частично** | С `chroma_mean_by_segment` возможна ось сегментов; без полного `chroma` в segment-режиме — только агрегаты по окнам. |
| Segmenter | **Да** | `families.chroma`; маска и NaN по контракту. |
| UI / analytics | **Да** | Доминирующий класс, энтропия, стабильность. |

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

**Не закрыт (Level 3 / `passed`):** набор **C**, golden §4.8, §4.9 (sampling policy), кросс‑корреляции с другими компонентами. Для **L2** (A+B): статистика и артефакты — `storage/audit_v4/chroma_extractor_l2/`, журнал — `RUN_LOG.md`.

#### §9–§11

| Раздел | Статус | Заметка |
|--------|--------|---------|
| §9 blocked | N/A | |
| §10 журнал | ◐ | commit, B/C — TODO |
| §11 | N/A | |

---

## 1. Мета (фрагмент)

| Поле | Значение |
|------|----------|
| `schema_version` | `chroma_extractor_npz_v1` |
| `producer_version` | `2.1.0` |
| `status` | `ok` |
| `chroma_type` | `cqt` |
| `normalize` | `l1` |
| `features_enabled` | `['time_series']` |
| `chroma_time_series_omitted` | `False` (нет inline `chroma` — см. §2) |
| `segments_count` | `12` |
| `duration_sec` | ≈ 22.095 |
| `tuning_failed` | `False` |
| `models_used` | `[]` |

---

## 2. Tabular и массивы (набор **A**)

**`feature_names` / `feature_values` (F=16):** 12× `chroma_mean_*`, затем `chroma_entropy`, `chroma_harmonic_stability`, `chroma_contrast`, `chroma_dominant_energy`. **`chroma_dominant_class` в tabular отсутствует** (analytics, `int32` отдельным массивом).

**Канонические массивы:** как в `SCHEMA.md`; `tuning_estimate` = −0.5.

**Сегменты:** N=12; все `segment_mask=true`; `chroma_mean_by_segment` shape `(12, 12)`.

**Ключ `chroma`:** отсутствует — ожидаемо для `run_segments()` (полная спектрограмма не сериализуется).

---

## 3. Сверка с кодом

1. **Тюнинг:** один раз на развёртке полного клипа; сбой → `0.0`, `tuning_failed`.
2. **Хрома:** `librosa` CQT или STFT; L1 по кадрам в audited контракте.
3. **Сегменты:** per-segment chroma → строки `chroma_mean_by_segment`; глобальный `chroma_mean` — вес по `segment_durations_sec` и маске.
4. **Савер:** `npz_savers/chroma.py` — tabular без `chroma_dominant_class`; optional `chroma` только если ndarray в payload.
5. **`meta.chroma_time_series_omitted`:** выставляется в `True` только при отбрасывании большого `chroma` в `run()`; при `run_segments()` без ключа `chroma` флаг остаётся `False` — **семантическая тонкость** (см. README / SCHEMA).

---

## 4. Заметки по качеству артефакта

- Компактный **12-классовый** профиль + интерпретируемые скаляры; на **A** нет «тихих» NaN в tabular.
- **Несоответствие ожиданий:** `features_enabled` содержит `time_series`, но **нет** `chroma` — для segment-режима это норма; документация обновлена.

---

## 5. Вердикт

**Плюсы:** строгий контракт, librosa-only, явный `chroma_type`, segment-aligned матрица при time_series на сегментах, `dominant` согласован с `chroma_mean`.

**Минусы:** вне minimal Models baseline; на L1 не проверены корреляции и edge-кейсы; семантика `chroma_time_series_omitted` vs отсутствие ключа `chroma` в `run_segments()` требует внимания потребителя (зафиксировано в docs).

---

## 6. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность / контракт | **9** |
| Полезность tabular для моделей | **8** |
| Полезность для encoder (dense) | **6** (есть `chroma_mean_by_segment` на сегментах, нет полного `chroma` в этом режиме) |
| Аналитика / UI | **8** |

**Итог: ~8/10** при условии L2/L3 и ясного контракта для downstream.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
