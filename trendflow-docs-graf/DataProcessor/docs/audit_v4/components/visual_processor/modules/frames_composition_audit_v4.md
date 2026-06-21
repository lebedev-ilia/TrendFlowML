# Audit v4 — `frames_composition` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** run).  
**Артефакт (A):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/frames_composition/frames_composition.npz`  
**JSON stats (A+B):** `storage/audit_v4/frames_composition_l2/frames_composition_audit_v4_stats.json`  
**Контракт:** [`VisualProcessor/schemas/frames_composition_npz_v1.json`](../../../../../VisualProcessor/schemas/frames_composition_npz_v1.json) · [`modules/frames_composition/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/frames_composition/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Один NPZ | ✓ | `frames_composition.npz` |
| Зависимости | ✓ | depth / face / objects — см. SCHEMA |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `frames_composition_npz_v1` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N, D, F** | ✓ | **N=48**, **D=32** per-frame, **F=217** video-level |
| Выравнивание shape | ✓ | `feature_values (F,)`, `frame_feature_values (N,D)`, `frame_feature_present_ratio (D,)` |

#### §4.1a — NaN и `present_ratio`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Политика SCHEMA | ✓ | Per-frame **NaN если метрика не определена**; есть **`frame_feature_present_ratio`** |
| Доля NaN в матрице | ✓ | **~7.03%** элементов **(N,D)**; по столбцам максимум **~37.5%** у face-зависимых полей (`face_center_*`, `face_area_ratio`, `anchor_*`, `thirds_alignment`, …) — согласуется с отсутствием лица на части кадров |
| `frame_feature_present_ratio` | ✓ | **min 0.625**, **max 1.0**, без NaN — алгебра «доля finite» выглядит правдоподобно |

#### §4.2 — Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_feature_values`, `feature_values` | ✓ | **0%** Inf |

#### §4.3 — Video-level tabular (**A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `feature_values` | ✓ | **217** скаляров, **без NaN** на **A** |
| Структура имён | ◐ | Префиксы `__mean/__std/__p10/...` по базовым композиционным фичам + `style_*`, `has_faces`, `frames_n`, и т.д. |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.6 / §4.11 — Много scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **F=217** | ✓ | На **L2** собраны cross-run корреляции video-level фич (top‑pairs) для навигации по избыточности (см. JSON stats) |
| **D=32** попарно | ◐ | На **L2** подтверждена стабильность D и корректность `present_ratio`; корреляции per-frame (D×D) не строились (N=5 run) |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Два уровня (per-frame + video) | Удобно для encoder (ось) и tabular head (агрегаты) |
| Частичные NaN в столбцах | Использовать **`frame_feature_present_ratio`** + маски при обучении |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Локально по видео? | Да |
| `models_used` | Пусто в типичном meta ([`SCHEMA.md`](../../../../../VisualProcessor/modules/frames_composition/docs/SCHEMA.md)) |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`(N, D)`** с учётом NaN |
| Tabular | **`(F,)`** — после feature selection на **B** |

#### §6 — Verdict

**Итог L2:** схема и NPZ **совпадают** на **5 run** (A+B), `axis_ok_all=true`; NaN в per-frame **объяснимы** (в основном face‑зависимые столбцы) и **сопровождаются** `frame_feature_present_ratio`, который численно совпадает с долей finite. Video-level вектор **плотный** (без NaN на этих 5 run), избыточность по F частично подсвечена через top корреляции (ожидаемые связи типа `negative_space_ratio` ↔ `object_bbox_coverage_ratio`).

**Оценка:** **~8.8 / 10** на L2 (всё ещё не `passed` до L3/§8).

#### §8 — DoD

**Не закрыт:** C, §4.8 (golden), L3/§8.

---

## 1. Снимок **A**

| Величина | Значение |
|---------|----------|
| N | 48 |
| D | 32 |
| F | 217 |
| NaN в `frame_feature_values` | ~7.0% |
| NaN в `feature_values` | 0 |
| `meta.processed_frames` | 48 |
| `meta.total_frames` | 338 |
| `meta.feature_set` | `default` |

---

## 4.3b — L2 stats (A+B, 5 run)

- **JSON**: `storage/audit_v4/frames_composition_l2/frames_composition_audit_v4_stats.json`
- **Итоги**:
  - **N_total**: **543**
  - **D**: **32** (стабильно на всех run)
  - **F**: **217** (стабильно на всех run)
  - `present_ratio_max_abs_diff_vs_computed_max`: ~**2e‑8**
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
