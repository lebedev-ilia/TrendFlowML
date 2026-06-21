# `text_scoring` — описание фич и артефактов

**Компонент:** `text_scoring` (OCR → табличные скоры + per-frame плотность текста)  
**NPZ:** `text_scoring/text_scoring.npz`  
**Schema:** `text_scoring_npz_v2` — `docs/SCHEMA.md`, `DataProcessor/VisualProcessor/schemas/text_scoring_npz_v2.json`  
**Код:** `utils/text_scoring.py` (`TextScoringModule`, `_FEATURE_NAMES_V1`) · **CLI:** `main.py`

Подробные определения скалярных метрик (синхронизация с движением, CTA, continuity) см. **`FEATURES_DESCRIPTION.md`**. Ниже — структура артефакта и связь с wide CSV / melt / QA.

---

## 1. Назначение

- Потребитель **OCR NPZ** (сам OCR в модуле не выполняется). Нет артефакта OCR → **валидный empty**: `meta.status=empty`, `empty_reason=dependency_missing`, `text_present=false`, нули в per-frame полях.
- Ось: union-domain `frame_indices (N,)` + `times_s (N,)`; сегментер обязан задать `text_scoring.frame_indices` (no-fallback в `run()` на пустом списке кадров).

---

## 2. Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,) int32`, `times_s (N,) float32` |
| Per-frame | `text_present` (scalar bool, 0d), `text_presence (N,) bool`, `text_count_per_frame (N,) int32` |
| Таблица (ML) | `feature_names (F,) object` — фиксированный порядок **`_FEATURE_NAMES_V1`**, `feature_values (F,) float32` (булевы фичи как 0.0/1.0) |
| Отладка | `ocr_raw (M,)`, `ocr_unique_elements (K,)` object; пустые массивы, если `store_debug_objects` выключен или нет сырых данных |
| Контракт | `meta` |

**Табличные имена (F, порядок фиксирован):** `text_present`, `text_frames_ratio`, `text_count_mean`, `text_count_p95`, `num_unique_texts`, `text_action_sync_score`, `text_motion_alignment`, `text_motion_alignment_windowed`, `multimodal_attention_boost_score`, `multimodal_attention_boost_position`, `text_on_screen_continuity*`, `text_switch_rate`, `time_to_first_text_sec`, `time_to_first_text_position`, `text_area_fraction`, CTA-поля, `text_readability_score`, `ocr_language_entropy`, `text_movement_speed`, `text_emphasis_peaks_count`, `ocr_raw_count`, `ocr_unique_elements_count` — полный список в `text_scoring.py` → `_FEATURE_NAMES_V1`.

---

## 3. Meta → wide CSV

`batch_runs_feature_report` кладёт в строку компонента только **`flatten_meta(meta)`** → `meta_*` (без склейки `feature_values` в колонки, если не добавлено отдельно).

- `stage_timings_ms`: **`frame_manager_ms`**, **`process_ms`**, **`save_ms`**, **`total_ms`** → `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_save_ms`, `meta_timing_total_ms` (мс, без суффикса `_ms` в имени колонки).
- Настройки: `use_face_data`, `use_motion_data`, `alignment_window_seconds`, `motion_weight`, `face_weight`, `audio_weight`, `min_ocr_confidence`, `retain_raw_ocr_text`, `store_debug_objects`, `enable_text_peaks`, `enable_language_entropy`, `enable_text_movement_speed`, `ocr_npz` (строка пути при наличии).
- ID прогона: `total_frames`, `processed_frames`, `analysis_fps` / `analysis_width` / `analysis_height`, `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`.

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → `text_scoring` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → `text_scoring`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*` для модуля

---

## 5. Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `text_frames_ratio`, `text_area_fraction`, `text_on_screen_continuity_normalized` (в `feature_values`) | **[0, 1]** (finite) |
| `text_present`, `cta_presence`, `persistent_cta_flag` (как 0/1 float в таблице) | **∈ [0, 1]** (finite) |
| `meta.processed_frames` / `meta.total_frames` | `processed ≤ total` |
| `meta.processed_frames` | **N = len(frame_indices)** |
| `meta.stage_timings_ms` | **≥ 0** |
| `meta.empty_reason` при `status=empty` (если задан) | Канон: **`dependency_missing`**, **`no_text_available`**, (редко) унифицированные `no_text` / `ocr_empty` |

`meta.status=error`: в `--struct` / батче — краткое сообщение; в `--ranges` — проверки пропускаются.

## 6. Проверка артефакта (single-file + батч)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/text_scoring/utils/validate_text_scoring.py \
  <path/to/text_scoring.npz> --struct --qa --ranges
```

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/text_scoring/utils/validate_text_scoring.py \
  --results-base storage/result_store --platform-id youtube
```

`--qa` сливает `flatten_meta` и **`feature_names` / `feature_values`** для сверки с `view_csv_feature_qa` → `text_scoring`.

Нужны `numpy` и импорт `DataProcessor/qa` (см. другие `validate_*.py` в репозитории).
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
