# `video_pacing` — описание фич и артефактов

**Компонент:** `video_pacing` (темп монтажа, плотность склейок, motion/semantic/color кривые)  
**producer** в `meta`: `video_pacing`  
**producer_version (код `utils/video_pacing.py`):** **2.0.1**  
**schema_version NPZ:** **`video_pacing_npz_v3`**

**NPZ:** `video_pacing/video_pacing_features.npz` (`ARTIFACT_FILENAME`)  
**Schema human:** [SCHEMA.md](SCHEMA.md), **machine:** `DataProcessor/VisualProcessor/schemas/video_pacing_npz_v3.json`  
**Код:** `utils/video_pacing.py` (`VideoPacingModule`, **`_FEATURE_NAMES_V1`**)

Смысл серий и табличных метрик — в **[FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md)**. Ниже — структура артефакта, meta → CSV, QA, валидатор.

---

## 1. Назначение и зависимости

- Ось: union-domain `frame_indices (N,)`; время — `union_timestamps_sec[frame_indices]`.
- **Hard deps (no-fallback):** `cut_detection`, `core_optical_flow`, `core_clip` согласованы по индексам с сэмплингом модуля.
- Минимум кадров: **`min_frames`** (по умолчанию 30) — fail-fast при нарушении.

---

## 2. Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,)`, `times_s (N,)` |
| Монтаж | `shot_boundary_frame_indices (S,)` (начала шотов, union-domain) |
| Кривые (N) | `motion_norm_per_sec_mean`, `semantic_change_rate_per_sec`, `color_change_rate_per_sec` — `float32` |
| Таблица | `feature_names` / `feature_values` — фиксированный порядок **`_FEATURE_NAMES_V1`**, **57** скаляров (в т.ч. 5+8 бинов гистограмм/плотностей) |
| Контракт | `meta` (`ui_payload`, `stage_timings_ms`, config highlights, …) |

`cut_density_map_8bins_*` — **плотности по времени** (не обязаны суммироваться в 1); `shot_length_histogram_5bins_*` при включённых гистограммах — вероятностный вектор (может быть NaN, если блок выключен).

---

## 3. Meta → wide CSV

`flatten_meta` → `meta_*`:

- `downscale_factor`, `min_shot_length_seconds`, `shot_detect_k`, `min_frames`, флаги `enable_*` (в CSV как 0/1);
- `stage_timings_ms`: обычно **`frame_manager_ms`**, **`process_ms`**, опционально **`save_ms`**, **`total_ms`** (мс) → `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_save_ms`, `meta_timing_total_ms`.

Точный набор ключей в сохранённом `meta` зависит от `BaseModule.save_results` (после записи `stage_timings_ms` дополняется `save_ms`/`total_ms`).

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → `video_pacing` (`add_all_meta_timing: true`)
- `view_csv_feature_qa.json` → `video_pacing`
- `view_csv_feature_descriptions_ru.json`

---

## 5. Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|----------|
| `motion_norm_per_sec_mean` | **≥ 0** для finite (кривая от core) |
| Доли | `short_shot_fraction`, `share_of_high_motion_frames`, `share_of_high_motion_shots`, `high_change_frames_ratio` **∈ [0,1]** (finite) |
| `meta` | `processed_frames` ≤ `total_frames` (когда оба int ≥ 0) |

---

## 6. Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/video_pacing/utils/validate_video_pacing.py \
  <path/to/video_pacing_features.npz> --struct --qa --ranges
```

Батч (схема + struct):

```bash
.../validate_video_pacing.py --results-base storage/result_store --platform-id youtube
```

## 7. Сверка с прогоном (пример)

`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/video_pacing/video_pacing_features.npz`  
— **57** имён в `feature_names` как в `_FEATURE_NAMES_V1`; `meta.producer_version` = **2.0.1**.
