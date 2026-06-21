# Audit v4 — `mel_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/mel_extractor/mel_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/mel_extractor_l2/mel_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`
**Engineering log 4.2:** [`../audit_4_2/audio_processor/mel_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/mel_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | `SCHEMA.md`, `schemas/mel_extractor_npz_v2.json`; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первая волна AudioProcessor | ✓ | `mel_extractor` |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` |
| **B** ≥5 видео | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** edge | ✗ | `audio_present=false`, короткие сегменты, empty |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L2 product stats**, не L3 | ✓ | `RUN_LOG`: L2 закрыт по **A+B** |
| Нет заявления полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `dtype` / `shape` | ✓ | `mel_*` по 128; `mel_stats_vector` (512); сегментные (12,·) / (12,128) |
| Сверка с `mel_extractor_npz_v2.json` | ◐ | Логически совпадает; CI/runtime validation не приложена |

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular — float32 | ◐ | На **A до фикса**: **1 NaN** у `device_used` (строка проталкивалась через `as_float`) |
| После фикса | ✓ | `device_used` только в **`meta`**; tabular **F=22** (на **A** ранее **F=23** с псевдо-полем) |
| Строки в meta | ✓ | `device_used`: **`cuda`** в meta при NaN в старом tabular — рендер уже брал fallback из meta |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN/Inf в числовых сериях на **A** | ✓ | Массивы без NaN (маска все true) |
| Старый tabular | ◐ | Только устранённый `device_used` |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 на нескольких run | ✓ | `storage/audit_v4/mel_extractor_l2/mel_extractor_audit_v4_stats.json` |
| Значения на **A** | ✓ | См. §2 отчёта |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Строки не в float-векторе | ✓ | После фикса — да |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сегментная ось `N=12` | ✓ | family `mel`; `segment_*` + per-segment ряды |
| Union `frame_indices` | N/A | Spectral family, не token |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Внутренние (между tabular на A+B) | ✓ | `storage/audit_v4/mel_extractor_l2/figures/tabular_corr_heatmap.png` |
| ρ с mfcc/spectral | ✗ | L3 / C / далее |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Таблица наблюдений → выводы §4.7 | ◐ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash tabular / ключевых сумм | ✗ | **TODO** после повторного прогона **A** (новый NPZ без `device_used` в tabular) |

#### §4.9 — Sampling, N, перекрытие

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Два прогона policy | ✗ | **B** |
| Shared STFT | ◐ | Поддержка в экосистеме; на **A** не изолировали |

#### §4.10 — `empty` / `empty_reason` (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Артефакты **C** | ✗ | |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F≈22 после фикса |

#### §4.12 — Anti-leakage

| Вопрос (план) | Ответ | Комментарий |
|-----------------|-------|--------------|
| Только текущее видео/аудио? | Да | |
| Глобальная нормализация по датасету? | Нет | Локально по клипу / сегменту |
| Онлайн API? | Нет | |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1–§5.2 | ◐ | README / SCHEMA обновлены под `device_used` → meta |
| §5.3 | ◐ | Ниже |

##### §5.3 — Сверка с Models (`mel_extractor`)

| Вопрос (план §5.3) | Ответ | Комментарий |
|--------------------|-------|-------------|
| В минимальном Baseline v1.0? | **Нет** | [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) не перечисляет `mel_extractor`. |
| Tabular / dense | **Частично** | Сильный блок: `mel_stats_vector` (512) + скаляры; при `time_series` — матрицы по сегментам. |
| Encoder | **Частично** | [`ENCODER_CONTRACT.md`](../../../../../Models/docs/contracts/ENCODER_CONTRACT.md) — по фактическому использованию в пайплайне. |
| Segmenter | **Да** | `families.spectral` / mel-сегменты. |

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
| Фикс tabular `device_used` | ✓ | `npz_savers/mel.py` |
| §4 на L3 | ✗ |

#### §8 — Definition of Done

**Не закрыт:** **B+C**, golden §4.8 на свежем **A**, commit в `RUN_LOG` при необходимости.

#### §9–§11

| Раздел | Статус | Заметка |
|--------|--------|---------|
| §10 журнал | ◐ | `RUN_LOG` |

---

## 1. Мета (фрагмент, набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `mel_extractor_npz_v2` |
| `mel_contract_version` | `mel_contract_v1` |
| `status` | `ok` |
| `device_used` | `cuda` |
| `features_enabled` | `basic_features`, `statistics`, `spectral_features`, `time_series`, `stats_vector` |

---

## 2. Tabular и массивы (набор **A**, артефакт **до** удаления `device_used` из савера)

**`feature_names` / `feature_values`:** **F=23**, один **NaN** по имени **`device_used`** (исправлено в коде).

| Имя | Значение (пример) |
|-----|-------------------|
| `sample_rate` | 22050 |
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `n_mels` | 128 |
| `fmin` | 0 |
| `fmax` | 11025 |
| `power` | 2 |
| `duration_sec` | ~22.09 |
| ~~`device_used`~~ | ~~NaN~~ (убрано из tabular) |
| `segments_count` | 12 |
| `mel_shape_0` / `mel_shape_1` | 128 × 962 |
| `mel_elements` | 123136 |
| `mel_energy` … `mel_stability` | см. NPZ |

**Массивы:** `segment_*` (N=12); `mel_mean`/`std`/`min`/`max` (128); `mel_stats_vector` (512); `mel_mean_by_segment` (12,128); per-segm. `mel_energy_by_segment`, `mel_centroid_mean_by_segment`, `mel_bandwidth_mean_by_segment`.

**Offline:** в `meta` возможны пути `mel_spectrogram_npy` / `mel_series_npy` (большие массивы не в NPZ).

---

## 3. Сверка с кодом

1. **Савер:** `save_mel_npz` — признаки по `features_enabled`; tabular через `add` → `float32`.
2. **Фикс audit v4:** не вызывать `add("device_used", …)` — строка давала NaN; `device_used` остаётся в `meta` через общий `build_meta` / `extra_meta` пайплайна.
3. **Рендер:** `utils/render.py` — `get_feature("device_used", meta.get("device_used", "cpu"))` корректен после фикса.

---

## 4. Вердикт

**Плюсы:** богатый NPZ (вектор 512 + спектральные ряды + per-segment mel); явная `mel_contract_version`; совместимость с feature-gates.

**Минусы:** до фикса — молчаливый NaN в tabular по `device_used` (класс как у pitch/key); вне минимального Baseline; L1 без B/C и golden.

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность / контракт | **8** (−1 за исторический tabular-баг, устранён) |
| Полезность для encoder / dense | **8** |
| Tabular для лёгких моделей | **7** |
| Аналитика | **8** |

**Итог: ~7.8 / 10** (округлённо **8/10** при условии повторного **A** и §4.8).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
