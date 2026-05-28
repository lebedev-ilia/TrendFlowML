# Audit v4 — `cut_detection` (VisualProcessor)

**Дата:** 2026-04-06 (обновление: 2026-04-13)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A + B**, 5 run).  

**Артефакты (один run — два NPZ):**

1. **Analytics / оглавление (набор A, фактический в `storage/result_store`):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/cut_detection/cut_detection_features_2026-04-07_01-25-18-232636_0a25687f.npz`  
2. **Model-facing (набор A):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/cut_detection/cut_detection_model_facing_2026-04-07_01-25-18-189575_5dfad6b7.npz`  

**Контракты:** [`cut_detection_npz_v1.json`](../../../../../VisualProcessor/schemas/cut_detection_npz_v1.json), [`cut_detection_model_facing_npz_v1.json`](../../../../../VisualProcessor/schemas/cut_detection_model_facing_npz_v1.json) · [`docs/SCHEMA.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA.md), [`docs/SCHEMA_MODEL_FACING.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA_MODEL_FACING.md)  

**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** / **◐** / **✗** / **N/A** — как в других отчётах VP.

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Два файла на компонент | ✓ | `features` (`cut_detection_npz_v1`) + `model_facing` (`cut_detection_model_facing_npz_v1`); связь через `model_facing_npz_path` в «толстом» NPZ |
| Имена файлов | ◐ | Суффикс **timestamp + hash** — для §4.8 фиксировать полный путь в `RUN_LOG` / golden |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs схемы, `allow_extra_keys=false` | ✓ | Для **обоих** NPZ на **A**: лишних/отсутствующих полей нет (сверка множеств ключей) |
| `manifest.notes` | ✓ | **`null`** |
| **N**, пары, события | ✓ | **N=48** кадров на оси; **N−1=47** пар; **E=7** событий (`event_*` массивы согласованы) |
| `model_facing_npz_path` | ✓ | Строка с **относительным** путём к второму NPZ (как в [`SCHEMA.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA.md)) |

#### §4.1a — Маски и «мертвые» ветки сигналов

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **Deep** | ✓ | `deep_valid_mask` **все False** → **`deep_cosine_dist`** **100% NaN** — согласовано (ветка не активна / нет эмбеддингов) |
| **SSIM** | ◐ | `ssim_valid_mask` **true ~25.5%**; **`ssim_drop`** **~74.5% NaN** там, где маска false — ожидаемая политика |
| **Flow / hist / hard_score** | ✓ | NaN **0%** на **A** для этих рядов (конечные значения в разумных диапазонах) |
| **`threshold_deep`** | ◐ | Весь массив **NaN** (при выключенном deep); **`meta.thresholds['deep']=0.0`** — не дублирует порог по элементам |

#### §4.2 — Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Пары / события | ✓ | **0** Inf в проверенных float-массивах |

#### §4.3 — Распределения (**A**, model-facing)

| Ряд | Заметка |
|-----|--------|
| `hist_diff_l1` | min **~0.005**, max **~0.52** |
| `flow_mag` | max **~2.87** |
| `hard_score` | **0 … 3** |
| `pair_dt_s` | положительные, **~0.23…0.27 s** |

#### §4.4 — Analytics NPZ (`features` / `detections`)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `features` | ✓ | **75** верхнеуровневых ключей; **float NaN: 0** на **A** |
| Примеры | ✓ | `hard_cuts_count=3`, `hard_cuts_per_minute≈15.08` |
| `detections` | ✓ | **18** ключей (позиции, индексы, soft_events, shot/scene boundaries и т.д.) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Невозрастающие шаги **≥0** |
| `pair_dt_s` | ✓ | Все **> 0** |
| `union_timestamps_sec` vs `times_s` | ✓ | Оба **(N,)**, float32 |

#### §4.6 — Корреляции

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Между `hist_diff_l1`, `flow_mag`, `hard_score` | ◐ (JSON) | `storage/audit_v4/cut_detection_l2/cut_detection_audit_v4_stats.json` содержит per-run summaries; отдельную матрицу ρ добавить при расширении B (N сейчас мал) |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Два NPZ + явная ссылка | Удобно для модели (узкий model-facing) vs аналитика |
| Deep-канал «пустой», но поля есть | Нормально при **mask=false**; encoder должен уважать маски |
| Имена файлов с timestamp | Усложняет автопоиск артефакта без manifest — opора на `manifest.json` |

#### §4.8 / §4.10

| Критерий | Статус |
|----------|--------|
| Golden **A** | ✗ |
| Edge / empty | ✗ | См. [`SCHEMA.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA.md) (empty не ожидается) |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по видео + core deps? | Да (`flow_source`: `core_optical_flow` в meta model-facing) |
| Deep при отсутствии модели | Нет сигнала — только NaN + маска |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`cut_detection_model_facing_npz_v1`**: пары **+** события **E** + опциональные soft/motion ряды |
| Tabular | **`features`** dict (**75** полей) — на L2 проверить избыточность |

#### §6 — Verdict

**Итог L2 (A+B, 5 run):** оба артефакта **совпадают со схемами**, `model_facing_npz_path` корректно связывает два NPZ на каждом run. По статистике A+B: **N_total=543**, **pairs_total=538**, **E_total=53**, `deep_valid_ratio_mean=0.0`, `ssim_valid_ratio_mean≈0.254`, `flow_valid_ratio_mean=1.0`.

**Оценка:** **~8.5 / 10** до закрытия **C** и **golden (§4.8)**.

#### §8 — DoD

**Не закрыт:** **C**, golden **§4.8** и полный DoD (§8).

---

## 2. L2 stats (A+B, 5 run) — артефакт

- JSON: `storage/audit_v4/cut_detection_l2/cut_detection_audit_v4_stats.json`
- Итог по этим 5 run: **N_total=543**, **pairs_total=538**, **E_total=53**

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N (`frame_indices`) | 48 |
| Пар (N−1) | 47 |
| Событий E | 7 |
| `event_type_id` (int16) | `[1,1,1,2,3,8,7]` |
| `deep_valid_mask` true% | 0 |
| `ssim_valid_mask` true% | ≈25.5 |
| `flow_valid_mask` true% | 100 |
| `features` ключей | 75 |
| `detections` ключей | 18 |
