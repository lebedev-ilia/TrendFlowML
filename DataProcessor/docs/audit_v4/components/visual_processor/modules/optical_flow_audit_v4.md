# Audit v4 — `optical_flow` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** run).  
**Артефакт (A):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/optical_flow/optical_flow.npz`  
**JSON stats (A+B):** `storage/audit_v4/optical_flow_l2/optical_flow_audit_v4_stats.json`  
**Контракт:** [`VisualProcessor/schemas/optical_flow_npz_v3.json`](../../../../../VisualProcessor/schemas/optical_flow_npz_v3.json) · [`modules/optical_flow/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/optical_flow/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль модуля | ✓ | **Потребитель** `core_optical_flow/flow.npz`, без локального RAFT ([`SCHEMA.md`](../../../../../VisualProcessor/modules/optical_flow/docs/SCHEMA.md)) |
| Манифест run | ◐ | У компонента в manifest **`device_used: cpu`** (агрегация); **`models_used`** ссылается на **RAFT/Triton cuda** — отражает upstream, не обязательно процесс модуля |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `optical_flow_npz_v3` | ✓ | Совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** |
| Размеры | ✓ | **N=250**, **D=16**, **F=9** |

#### §4.1a — NaN и `missing_frame_ratio`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| POLICY | ✓ | Нет покрытия в `core_optical_flow` → **NaN** ([`SCHEMA.md`](../../../../../VisualProcessor/modules/optical_flow/docs/SCHEMA.md)) |
| На **A** | ✓ | **`motion_norm_per_sec_mean`**: **~85.6%** NaN |
| Согласованность | ✓ | **`missing_frame_ratio` = 0.856** в `feature_values` — совпадает с долей NaN по оси |
| Матрица | ✓ | **`frame_feature_values`**: **~86.0%** NaN; **первый столбец** совпадает с **`motion_norm_per_sec_mean`** по маске NaN и значениям |

#### §4.2 — Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Проверенные float-массивы | ✓ | **0%** Inf на **A** |

#### §4.3 — Конечные кадры (**A**)

| Ряд | min / max / mean (finite sample) |
|-----|----------------------------------|
| `motion_norm_per_sec_mean` | **0 … ~2.87**, mean **~0.20** |

#### §4.4 — Video-level **`feature_values`**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN | ✓ | **0 / 9** на **A** |
| Набор имён | ✓ | Совпадает с перечислением в [`SCHEMA.md`](../../../../../VisualProcessor/modules/optical_flow/docs/SCHEMA.md) (`motion_curve_*`, `missing_frame_ratio`, camera/flow агрегаты) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.11 — Много scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **F=9** | N/A | Порог **24** не превышен |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Источник сигнала | Только **текущее видео** через **core_optical_flow** |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | **`motion_norm_per_sec_mean (N,)`** + **`frame_feature_values (N,16)`** с учётом высокой доли **NaN** или маски покрытия |
| Tabular | **`feature_values (9,)`** |

#### §6 — Verdict

**Итог L1:** схема и NPZ **совпадают**, manifest **чистый**; **высокая доля NaN** на оси **согласована** с `missing_frame_ratio` и контрактом consumer-only. Для моделей важно не трактовать отсутствие flow как «ноль движения».

**Оценка:** **~8.5 / 10** на L1.

#### §8 — DoD

**Не закрыт:** C, §4.8 (golden).

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 250 |
| D | 16 |
| F | 9 |
| NaN в `motion_norm_per_sec_mean` | ~85.6% |
| `missing_frame_ratio` | 0.856 |
| `feature_values` NaN | 0 |

---

## 4.3b — L2 stats (A+B, 5 run)

- **JSON**: `storage/audit_v4/optical_flow_l2/optical_flow_audit_v4_stats.json`
- **Итоги**:
  - **N_total**: **1250**
  - **D**: **16** (стабильно)
  - **F**: **9** (стабильно)
  - `missing_ratio_curve_mean`: **~0.886**
  - `missing_ratio_matrix_mean`: **~0.890**
  - `missing_ratio_max_abs_diff_max`: **~0.00375**
