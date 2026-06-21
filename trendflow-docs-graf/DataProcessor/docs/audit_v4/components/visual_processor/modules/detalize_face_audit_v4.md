# Audit v4 — `detalize_face` (VisualProcessor)

**Дата:** 2026-04-06 (обновление: 2026-04-13)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A + B**, 5 run).  
**Артефакт (набор A, фактический в `storage/result_store`):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/detalize_face/detalize_face.npz`  
**Контракт:** [`VisualProcessor/schemas/detalize_face_npz_v3.json`](../../../../../VisualProcessor/schemas/detalize_face_npz_v3.json) · [`modules/detalize_face/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/detalize_face/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

Краткие метки **✓** / **◐** / **✗** — как в других VP-отчётах.

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Один NPZ, фиксированное имя | ✓ | `detalize_face.npz` |
| Зависимости | ✓ | `core_face_landmarks`, ось Segmenter (см. SCHEMA) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `detalize_face_npz_v3.json` | ✓ | Нет лишних и нет пропусков обязательных; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** |
| **N** | ✓ | **250** = длина оси Segmenter для модуля |
| `primary_compact_features` | ✓ | **`(250, 40)`** float32 |
| Опциональные `primary_*` кривые | ◐ | В **`meta`**: `write_primary_curves: false` — поэтому **нет** массивов `primary_gaze_*` и т.д. в NPZ (допустимо по схеме **`required: false`**) |

#### §4.1a — Нули vs NaN vs маски

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Политика compact | ✓ | При **`primary_valid=False`** все **40** компонент **ровно 0** (**231** кадр на **A**); при **`True`** строки **без NaN** |
| Маски | ✓ | На **A** **`face_present` == `processed_mask` == `primary_valid`** (все **7.6%** true) — согласованно: лица редки, и каждый face-кадр обработан |
| **`face_count`** | ◐ | float32 с значениями **0** или **1** на **A** — дискретный счётчик в float (см. §4.1a плана) |
| **`primary_tracking_id`** | ✓ | **-1** при отсутствии primary; иначе положительные id (**1…19** на **A**) |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `primary_compact_features` | ✓ | **0%** NaN, **0%** Inf |
| `aggregated` векторная статистика | ✓ | `compact_mean`, `compact_std`, `compact_p10`, `compact_p90` — **0%** NaN |

#### §4.3 — Распределения (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Разрежённость лица | ✓ (JSON) | На **A+B** `primary_valid` True **73** из **1250** (**~5.84%**), `compact_zero_row_ratio≈94.16%` |
| `aggregated.compact_l2_*` | ◐ | Нормы **~2100–2700** (крупная шкала — не L2=1; encoder должен нормализовать или использовать маску) |

#### §4.4 — `faces_agg` / `summary`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `faces_agg` | ✓ | Dict с ключами **tracking_id** (**1…19** на **A**) |
| `summary` | ✓ | **9** полей; имена **`frames_with_faces_total` / `frames_with_faces_processed`** — в human [`SCHEMA.md`](../../../../../VisualProcessor/modules/detalize_face/docs/SCHEMA.md) перечислены как `frames_with_faces` (дрейверка формулировок, не NPZ) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.6 — Корреляции

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Между колонками compact | ◐ | На текущем наборе A+B valid-ось маленькая (valid=73); корреляции разумно считать на B с более частыми лицами |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| **0-fill** при невалидных кадрах | Encoder **обязан** использовать **`primary_valid`** (или `processed_mask`), иначе «пустые» кадры выглядят как нулевой вектор |
| `meta.processed_frames` (**19**) vs **N=250** | **19** — число реально обработанных face-кадров; **250** — длина оси; не смешивать |

#### §4.8 / §4.10

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Golden **A** | ✗ | TODO |
| Empty (`no_faces`) | ✗ | На **A** `status=ok` |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по видео? | Да |
| `models_used` | **`[]`** — эвристики поверх landmarks |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`(N, 40)`** + boolean маски; при необходимости нормализовать по `aggregated` или batch |
| Tabular | **`aggregated`** (**15** верхнеуровневых полей + массивы статистики по 40) |

#### §6 — Verdict

**Итог L2 (A+B, 5 run):** схема и файл **совпадают**, строки manifest для `detalize_face` — `status=ok`, `schema_version=detalize_face_npz_v3`, `producer_version=2.0.2`. По A+B: `primary_valid` True **73/1250 (~5.84%)**, `processed_mask` согласован с `face_present`/`primary_valid`; `primary_compact_features` 0-fill при невалидных кадрах устойчиво.

**Оценка:** **~8.5 / 10** до закрытия **C** и **golden (§4.8)**.

#### §8 — DoD

**Не закрыт:** **C**, golden **§4.8** и полный DoD (§8).

---

## 2. L2 stats (A+B, 5 run) — артефакт

- JSON: `storage/audit_v4/detalize_face_l2/detalize_face_audit_v4_stats.json`
- Итог по этим 5 run: **N_total=1250**, `primary_valid` True **73** (**~5.84%**), `processed_mask_true_total=73`.

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 250 |
| `primary_valid` true | 19 (7.6%) |
| `primary_compact_features` | (250, 40), NaN 0 |
| Нулевые строки compact при `~primary_valid` | 100% |
| `summary.processed_frames` | 19 |
| `meta.total_frames` | 338 |
| `meta.write_primary_curves` | false |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
