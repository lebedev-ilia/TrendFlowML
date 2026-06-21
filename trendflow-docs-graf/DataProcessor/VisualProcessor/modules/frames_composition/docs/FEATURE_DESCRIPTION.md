# `frames_composition` — описание фич и артефакта

**Компонент:** `frames_composition` (композиция кадра: баланс, depth, объекты, стиль и т.д.)  
**NPZ:** `frames_composition/frames_composition.npz`  
**Schema:** `frames_composition_npz_v1` — `docs/SCHEMA.md`  
**Код:** `utils/balance_composition.py` (`FramesCompositionModule`, `VERSION` / `SCHEMA_VERSION`)

Семантика групп фич и зависимостей — в **`FEATURES_DESCRIPTION.md`**. Ниже — структура артефакта, meta, диапазоны для QA / `--ranges`, melt.

---

## 1. Зависимости и ось

- **Hard deps (aligned `frame_indices`):** `core_object_detections`, `core_face_landmarks` (допустимы валидные empty), `core_depth_midas` (**обязан `status=ok`**).
- **Ось:** `metadata["frames_composition"].frame_indices`, **`times_s` = `union_timestamps_sec[frame_indices]`** (no-fallback).
- **Пусто:** при отсутствии лиц на всём ролике — `status=empty`, `empty_reason=no_faces_in_video` (см. `SCHEMA.md`); при этом сохраняются согласованные массивы (часто `face_present` + NaN-столбцы).

---

## 2. Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Ось N | `frame_indices (N,) int32`, `times_s (N,) float32` — строго возрастающие индексы, неубывающие времена |
| Per-frame | `frame_feature_names (D,)`, `frame_feature_values (N,D) float32`, `frame_feature_present_ratio (D,)` — доля finite по столбцу, **∈ [0, 1]** |
| Video-level | `feature_names (F,)`, `feature_values (F,) float32` |
| Контракт | `meta` (`feature_set`, `features`, `num_workers`, `stage_timings_ms`, …) |

`D` и `F` **не фиксированы** в коде (зависят от `feature_set` / списка групп); валидатор проверяет **согласованность длин**, а не конкретные имена.

---

## 3. `stage_timings_ms` → CSV

Ключи (мс, без `_ms` в имени внутри dict): **`init`**, **`axis`**, **`load_deps`**, **`per_frame`**, **`aggregate`**, **`total`**.  
В плоском CSV: `meta_timing_init`, `meta_timing_axis`, …

---

## 4. Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `frame_indices` | Строго возрастает, уникальны |
| `times_s` | Не убывает |
| `frame_feature_present_ratio` | **[0, 1]** (с небольшим допуском) |
| `meta.feature_set` (если строка) | Обычно `default` / `ml` / `all` |

Полные правила для melt-HTML: `storage/result_store/view_csv_feature_qa.json` → **`frames_composition`**.

---

## 5. Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/frames_composition/utils/validate_frames_composition.py \
  <path/to/frames_composition.npz> --struct --qa --ranges
```

Батч по `result_store` (обход `**/frames_composition/frames_composition.npz`):

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/frames_composition/utils/validate_frames_composition.py \
  --results-base storage/result_store --platform-id youtube
```

При успехе в однофайловом `--struct` печатается строка вида: `Structure OK (N=…, D=…, F=…, frames_composition_npz_v1)`.

---

## 6. CSV / melt / RU

- `view_csv_melt_interesting.json` → `frames_composition` (`add_all_meta_timing: true`)
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*` и (при melt) к столбцам

---

## 7. Сверка с прогоном (пример)

Проверено на:  
`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/frames_composition/frames_composition.npz`  
— ключи совпадают с контрактом; `stage_timings_ms`: `init`, `axis`, `load_deps`, `per_frame`, `aggregate`, `total`.
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
