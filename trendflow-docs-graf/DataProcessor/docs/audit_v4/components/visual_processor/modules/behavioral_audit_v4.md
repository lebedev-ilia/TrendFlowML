# Audit v4 — `behavioral` (VisualProcessor)

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L1 — draft** (набор **A** только).  
**Артефакт:** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/behavioral/behavioral_features.npz`  
**Код / контракт:** `DataProcessor/VisualProcessor/modules/behavioral/` · machine schema: [`VisualProcessor/schemas/behavioral_npz_v1.json`](../../../../../VisualProcessor/schemas/behavioral_npz_v1.json) · [`docs/SCHEMA.md`](../../../../../VisualProcessor/modules/behavioral/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | VP schema; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| VisualProcessor | ✓ | Зависимость от `core_face_landmarks` (см. `SCHEMA.md`) |
| Путь артефакта + `run_id` | ✓ | [`RUN_LOG.md`](../../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | e2e reference |
| **B** ≥5 видео | ✗ | Распределения `landmarks_present`, жесты, агрегаты |
| **C** edge | ✗ | `status=empty`, нет лиц, очень мало кадров |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1 draft** | ✓ | `in_progress` в журнале |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи NPZ vs `behavioral_npz_v1.json` | ✓ | Набор `seq_*`, ось, `aggregated`, object-поля — совпадает; **`allow_extra_keys: false`** — на **A** лишних ключей нет |
| `manifest.json` (этот run) | ✓ | `notes: null` — валидация артефакта не ругалась (в отличие от `action_recognition`) |
| **N** | ✓ | **250** кадров на оси; `meta.processed_frames=250`, `total_frames=338` |
| `aggregated` | ✓ | `dtype=object`, shape `()` — **dict** с **33** скалярными/вложенными полями на **A** |

#### §4.1a — Семантика типов

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `seq_num_hands` как **float32** | ◐ | На конечных кадрах значения **0, 1, 2** — по смыслу дискретный счётчик; для §4.1a зафиксировать как «float-обёртка», см. [`SCHEMA.md`](../../../../../VisualProcessor/modules/behavioral/docs/SCHEMA.md) |
| `seq_blink_flag` / `seq_self_touch_flag` | ◐ | **0/1** в float — ок при явной политике |
| Строки в tabular | ✓ | Жесты — в `hand_gestures` / `frame_results` (object), не в `seq_*` |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Inf в `seq_*` | ✓ | **0** |
| NaN vs `landmarks_present` | ✓ | Для **`seq_num_hands`**: **214** NaN, **36** конечных значений; **`landmarks_present=True`** ровно на **36** кадрах; **0** NaN при `True`, **0** конечных при `False` — строгое соответствие маске |
| Вторичная пропажа подмаски | ◐ | **`seq_num_hands`**: при `landmarks_present=True` **36/36** конечны (строго). **`seq_mouth_*`**: только **19/36**. **`seq_arm_openness` / `seq_pose_expansion` / `seq_body_lean_angle`**: **27/36** конечны при `True` (**9** кадров — NaN). В `SCHEMA.md` в одном месте указано «NaN when `landmarks_present=false`» для ряда полей — для production лучше явно описать **иерархию опор** (лицо → pose/руки → рот) |
| `aggregated` | ◐ | На **A**: **`early_engagement_mean`** = NaN; в **`early_late_ratios.engagement`** — NaN; остальное конечно — вероятно мало опорных кадров/деление |

#### §4.3 — Распределения (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ось времени | ✓ | `times_s`: **0 … ~11.93 s**, монотонно; `frame_indices`: **0 … 337**, монотонно |
| `seq_timestamp_norm` | ✓ | [0, 1], без NaN на всей оси |
| Доля «активных» кадров | ✓ | **36/250 = 14.4%** с полными pose/hand seq (по маске) |
| Константы на конечной подвыборке | ◐ | **`seq_body_lean_angle`** на всех конечных кадрах **= 1.0** — похоже на вырождение/плейсхолдер; проверить на **B** и в коде [`behavior_analyzer.py`](../../../../../VisualProcessor/modules/behavioral/utils/behavior_analyzer.py) |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `hand_gestures`, `frame_results` | ◐ | L1: структура есть; частоты жестов — **B** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Согласованность с sampling | ◐ | **A**: `analysis_fps=30`, индексы разрежены (union 250 из 338) |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Избыточность жестовых prob | ✗ | Много каналов `seq_gesture_prob_*` |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Чистая маска NaN ↔ `landmarks_present` для основного блока seq | Контракт соблюдается; удобно для encoder + mask |
| Рот глубже маски лица | Нужна явная документация missing policy |
| `meta.producer_version` = **`unknown`** | Шероховатость метаданных (не блокер модели) |
| `models_used` = **`[]`** | Ожидаемо для эвристик/landmarks-upstream; не путать с «нет модели» |

#### §4.8 — Golden **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура агрегатов / доля покрытия | ✗ | TODO после выбора ключей |

#### §4.9 — Sampling (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Минимум кадров с `landmarks_present` | ✗ | На **A** 36 — порог надёжности для `aggregated` зафиксировать на **B/C** |

#### §4.10 — `empty` (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Нет лиц | ✗ | Артефакты **C** |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `aggregated` | ◐ | **33** поля верхнего уровня — на **B** имеет смысл §4.11 по подполям |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Только текущее видео? | Да |
| Глобальная нормализация по датасету? | Нет в явном виде на **A** |
| Онлайн API? | Нет |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Вход encoder | Временные ряды `seq_*` **[N, F]** + маска `landmarks_present` + video-level `aggregated` |
| Риск | Высокая разрежённость оси на контенте без лица — нужен pooling / маска |

#### §6 — Verdict

**Итог L1:** артефакт на **A** **согласован со схемой** и **валидируется** в manifest; политика **NaN строго привязана к `landmarks_present`** для основного блока признаков — сильная сторона. Замечания: **вторичные NaN** для mouth при наличии лица, **NaN в части `aggregated`**, константа **`seq_body_lean_angle`**, **`producer_version: unknown`**.

**Оценка:** **~8.5 / 10** (на уровне L1; до **9+** — уточнить mouth/aggregated missing policy в docs, проверить lean на B, L2 корреляции жестов).

#### §8 — DoD

**Не закрыт:** **B+C**, §4.6, §4.8, commit в `RUN_LOG`.

---

## 1. Снимок **A** (кратко)

| Величина | Значение |
|----------|-----------|
| N | 250 |
| `landmarks_present` True | 36 (14.4%) |
| NaN `seq_num_hands` при LP | 0 |
| Mouth finite при LP | 19 / 36 |
| Pose (`seq_arm_openness` и др.) finite при LP | 27 / 36 |
| `times_s` max | ≈ 11.93 |
| Inf в float seq | 0 |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
