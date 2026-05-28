# `color_light` — описание фич и артефактов

**Компонент:** `color_light` (цвет / свет, union-domain)  
**producer** в `meta`: `color_light`  
**producer_version (код `utils/processor.py`):** **2.0.2**  
**schema_version NPZ:** **`color_light_npz_v2`**

**NPZ:** `color_light/color_light_features.npz`  
**Schema:** [SCHEMA.md](SCHEMA.md), `DataProcessor/VisualProcessor/schemas/color_light_npz_v2.json`  
**Код:** `utils/processor.py` (`ColorLightProcessor`, **`FRAME_COMPACT_KEYS`**)

Семантика уровней frame/scene/video — в **[FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md)**. Ниже — ключи артефакта, meta → CSV, валидатор, `--ranges`.

---

## 1. Назначение и зависимости

- **Hard dependency:** `scene_classification` (сцены + `indices`), ось — `frame_indices` сегментера, время — `union_timestamps_sec[frame_indices]`.
- **`store_debug_objects`:** при `false` в NPZ пустые/минимальные `frames` / `scenes` (см. `SCHEMA.md`).

---

## 2. Ключи NPZ (сводно)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,)`, `times_s (N,)` |
| Sequence / compact | `sequence_frame_indices`, `sequence_times_s`, `frame_compact_features (M,16)`, `frame_compact_feature_names (16,)`, `frame_compact_frame_indices (M,)` |
| Compat / debug | `sequence_inputs` (object) |
| Агрегаты | `video_features`, `aggregated` (object) |
| Сцены / кадры | `scenes`, `frames` (object) |
| Контракт | `meta` |

Имена **16** компактных фич — **`FRAME_COMPACT_KEYS`**: в т.ч. нормированные hue/sat/val/contrast, **энтропии hue могут быть > 1**; доли `*_ratio` и `soft_light_prob` — **[0, 1]**; **`dominant_lab_a_norm` / `dominant_lab_b_norm`** — в пилоте ожидаются в **[-2.5, 2.5]** (не как остальные «norm» в [0,1]).

---

## 3. Meta → wide CSV

`flatten_meta` → `meta_*`; `stage_timings_ms` (мс): `initialization`, `load_deps`, `process_frames`, `post_process`, `save`, `total` → `meta_timing_*`.

Repro / highlights: `store_debug_objects`, `hue_hist_bins`, `palette_*`, `max_frames_per_scene`, `stride`, `frame_compact_dim`, `module_sampling_policy_version`, …

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → `color_light` (`add_all_meta_timing`)
- `view_csv_feature_qa.json` → `color_light`
- `view_csv_feature_descriptions_ru.json`

---

## 5. Нормальные диапазоны (`--ranges` в валидаторе)

| Проверка | Ожидание |
|----------|----------|
| `times_s`, `sequence_times_s` | Неубывающие ряды (finite) |
| `skin_tone_ratio`, `overexposed_ratio`, `underexposed_ratio`, `vignetting_score_norm`, `soft_light_prob` | **[0, 1]** по выборке |
| `dominant_lab_a_norm`, `dominant_lab_b_norm` | Пилот **[-2.5, 2.5]** |
| `meta` | `processed_frames` ≤ `total_frames` |

Video-level поля в `video_features` могут быть **NaN** при узких данных (например gini) — **не** считаются ошибкой структуры.

---

## 6. Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/color_light/utils/validate_color_light.py \
  <path/to/color_light_features.npz> --struct --qa --ranges
```

Скан по дереву `result_store`:

```bash
.../validate_color_light.py --results-base storage/result_store --platform-id youtube
```

Старый режим отчёта (`run_id == video_id` в тестовых деревьях): **`--results-base ... --legacy-report`**.

---

## 7. Сверка с прогоном (пример)

`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/color_light/color_light_features.npz`
