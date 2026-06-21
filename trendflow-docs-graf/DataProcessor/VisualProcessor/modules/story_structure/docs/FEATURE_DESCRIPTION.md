# `story_structure` — описание фич и артефактов

**Компонент:** `story_structure` (hook / climax / story energy, Tier‑0 baseline)  
**NPZ:** `story_structure/story_structure.npz`  
**Schema:** `story_structure_npz_v3` — `docs/SCHEMA.md`  
**Код:** `utils/story_structure.py` (`StoryStructureBaselineModule`, `_FEATURE_NAMES_V1`)

Семантика серий и табличных скаляров — в **`FEATURES_DESCRIPTION.md`**. Ниже — структура артефакта, meta → CSV / melt / QA.

---

## 1. Назначение и зависимости

- Ось: `frame_indices (N,)`; время — `union_timestamps_sec[frame_indices]` (no-fallback).
- **Hard deps:** `core_clip`, `core_optical_flow`, `core_face_landmarks` на тех же `frame_indices`.
- Границы: **`min_frames`** (по умолчанию 30), **`max_frames`** (по умолчанию 200) — fail-fast при нарушении.

---

## 2. Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,) int32`, `times_s (N,) float32` |
| Кривые (N) | `story_energy_curve`, `motion_norm_per_sec_mean`, `embedding_change_rate_per_sec`, `topic_shift_curve` — `float32` |
| Пары соседей (N−1) | `embedding_sim_next`, `embedding_diff_next` — `float32` |
| Маски / качество | `any_face_present (N,) bool`, `frame_feature_present_ratio (N,) float32` |
| Сводка 128 | `story_energy_curve_downsampled_128 (128,) float32` |
| Пики энергии | `story_energy_peaks_idx`, `story_energy_peaks_times_s`, `story_energy_peaks_values_z` — согласованные 1D |
| Текст / тема | `topic_shift_curve_present` (скаляр bool в NPZ), `topic_shift_peaks_idx (K,) int32` |
| Таблица | `feature_names (F,) object`, `feature_values (F,) float32` — строгий порядок **`_FEATURE_NAMES_V1`** |
| Контракт | `meta` (в т.ч. `ui_payload`, `stage_timings_ms`, config highlights) |

---

## 3. Имена табличных фич

См. **`_FEATURE_NAMES_V1`** в `utils/story_structure.py` (hook, climax, персонажи/face, `topic_shift_*`). `topic_shift_curve_present` в таблице — как 0/1.

---

## 4. Meta → wide CSV

`flatten_meta` → `meta_*`, стадии `stage_timings_ms` → `meta_timing_*` (значения в **мс**):  
`deps_ms`, `compute_curves_ms`, `hooks_climax_ms`, `text_ms`, `load_deps_ms`, `process_ms`, `save_ms`, `total_ms`.

Config highlights: `min_frames`, `max_frames`, `energy_smoothing_sigma`, `min_energy_peak_sep_sec`, `text_mode`, `ocr_max_chars_per_frame`, `clip_text_model_spec`, `clip_text_batch_size` и т.д. (см. `run()` / `save_metadata` в `story_structure.py`).

---

## 5. Melt / QA / RU

- `view_csv_melt_interesting.json` → `story_structure` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → `story_structure`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*` (в т.ч. тайминги)

---

## 6. Проверка артефакта

```bash
python3 DataProcessor/VisualProcessor/modules/story_structure/utils/validate_story_structure.py \
  /path/to/story_structure.npz --struct
python3 DataProcessor/VisualProcessor/modules/story_structure/utils/validate_story_structure.py \
  /path/to/story_structure.npz --qa
```

Нужны `numpy` и `DataProcessor/qa` на `PYTHONPATH`.
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
