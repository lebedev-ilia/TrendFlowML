# Audit v4 — `core_face_landmarks` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**, 5 run).  
**Артефакты (A+B):** см. `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json` (пути `npz_path` внутри).  
**Контракт:** [`VisualProcessor/schemas/core_face_landmarks_npz_v2.json`](../../../../../VisualProcessor/schemas/core_face_landmarks_npz_v2.json) · [`core/model_process/core_face_landmarks/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/core_face_landmarks/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | MediaPipe: лица **468×3**, опционально **pose** и **hands**; маски **`face_present`** / **`pose_present`** / **`hands_present`** |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `core_face_landmarks_npz_v2` | ✓ | Только разрешённые; опциональные **pose/hands** и дублирующие debug-поля присутствуют на **A**; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, **FACES** | ✓ | **N=48**, **FACES=1**; **`face_landmarks` / `face_landmarks_raw`**: **(48, 1, 468, 3)** |
| **HANDS** | ✓ | **`hands_landmarks`**: **(48, 2, 21, 3)** |
| **Pose** | ✓ | **`pose_landmarks`**: **(48, 33, 4)** |

#### §4.2 — NaN / маски

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **Лицо** | ✓ | При **`face_present=False`** слот заполнен **NaN** (**100%** в матрице 468×3); при **`True`** — **без NaN** на **A** |
| Доля NaN в **`face_landmarks`** | ✓ | **~37.5%** ячеек — ровно **18/48** кадров без лица (**1−0.625**) |
| **Руки** | ✓ | **`hands_landmarks`**: **~82.3%** NaN на **A**; при отсутствии руки в слоте — **все NaN**; **`hands_present`** true **~17.7%** по слотам |
| **Pose** | ✓ | **`pose_landmarks`** **0%** NaN на **A**; **`pose_present`** везде **True** |
| Inf | ✓ | **0** в проверенных float-массивах |

#### §4.1a — Флаги аналитики

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`has_any_face`** | ✓ | **`True`** на **A** |
| **`face_mesh_ran`** | ✓ | На **A**: true **38/48** (**~79.2%**). **`face_present` ⇒ `face_mesh_ran`** (0 противопримеров); **`face_mesh_ran` ∧ ¬`face_present`**: **8** кадров (mesh проходил, лица нет → **NaN** в слоте) |
| **`empty_reason` / `face_empty_reason`** | ✓ | **`None`** на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **1** запись (**mediapipe** **0.10.14**, **cpu**) |
| `person_mask_enabled` | **`True`** в **`meta`** |

#### §6 — Verdict (L2)

**Итог L2:** на 5 run (A+B) shape‑инварианты **стабильны**: `FACES=1`, `HANDS=2`. NaN‑политика слотов **строго соблюдена**: для face/hands «absent ⇒ all‑NaN», «present ⇒ no‑NaN» (violations=0). Встречается `face_mesh_ran ∧ ¬face_present` (mesh запускался, но лицо не найдено) — ожидаемо при гейтинге по person‑mask/детектору. **~8.8 / 10** на L2 (до L3/§8 не `passed`).

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; сценарий **FACES>1**.

---

## 2. L2 stats (A+B)

JSON: `storage/audit_v4/core_face_landmarks_l2/core_face_landmarks_audit_v4_stats.json`

Коротко по агрегатам:

- `N_total=543`, `FACES_set=[1]`, `HANDS_set=[2]`
- `face_present_ratio` варьирует **~0.090…0.865** (контент/кадры с человеком)
- NaN‑инварианты соблюдены (all violations = 0)

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 48 |
| FACES | 1 |
| `face_present` true ratio | 0.625 |
| `face_mesh_ran` true ratio | ~0.792 (38/48) |
| `hands_present` true ratio (по слотам) | ~0.177 |
| `face_landmarks` NaN (доля ячеек) | ~37.5% |
