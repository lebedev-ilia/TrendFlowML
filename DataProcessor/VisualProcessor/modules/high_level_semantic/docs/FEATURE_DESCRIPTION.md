# `high_level_semantic` — описание фич и артефактов

**Компонент:** `high_level_semantic` (модуль VisualProcessor, агрегатор сцен / dense per-frame / события)  
**NPZ:** `high_level_semantic/high_level_semantic.npz`  
**Schema:** `high_level_semantic_npz_v2` — `docs/SCHEMA.md`, `DataProcessor/VisualProcessor/schemas/high_level_semantic_npz_v2.json`  
**Код:** `main.py`, `utils/hl_semantic.py` · **render:** `utils/render.py`

Подробный разбор dense-колонок `frame_feature_names` и taxonomic событий см. `FEATURES_DESCRIPTION.md` (синхрон с кодом v1/v2). Здесь — сводка + связь с wide CSV / melt / QA.

---

## 1. Назначение

- **Ось:** union-domain кадры `frame_indices (N,)` + `times_s (N,)`; источник времени — `metadata.json.union_timestamps_sec`.
- **Без сети в модуле:** CLIP не грузится; читается `core_clip/embeddings.npz`.
- **Сцены:** только `cut_detection` (жёсткая зависимость по согласованию индексов).
- **Выходы:** `scene_*`, `frame_features` + `frame_feature_present_ratio`, поток `event_*`, копия text snapshot (`text_feature_*`), словарь `features`, `ui` (карта типов событий, `upstream` presence), `meta`.

---

## 2. Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Время / кадр | `frame_indices`, `times_s`, `scene_id` |
| Сцены | `scene_embeddings`, `scene_start_*`, `scene_end_*`, `scene_duration_s`, `scene_representative_frame_idx`, `scene_embedding_mean_norm` |
| Dense | `frame_feature_names`, `frame_features`, `frame_feature_present_ratio` |
| События | `event_times_s`, `event_type_id`, `event_strength`, `event_frame_pos` |
| Text snapshot | `text_feature_names`, `text_feature_values` |
| Прочее | `features` (object: скалярный summary), `ui`, `meta` |

`event_type_id`: см. `ui.event_type_map` (пример: `1` hard cut, `200` semantic jump, `210` emotion keyframe — как в `FEATURES_DESCRIPTION.md`).

---

## 3. Meta → wide CSV (batch_runs_feature_report)

`flatten_meta` даёт `meta_*`:

- `meta_status`, `meta_empty_reason`, `meta_producer`, `meta_producer_version`, `meta_schema_version`, `meta_dataprocessor_version`, run identity: `meta_platform_id`, `meta_video_id`, `meta_run_id`, `meta_sampling_policy_version`, `meta_config_hash`, `meta_created_at`
- `meta_total_frames`, `meta_processed_frames`, `meta_analysis_fps`, `meta_analysis_width`, `meta_analysis_height`, `meta_frames_dir`
- Конфиг-настройки: `meta_feature_groups`, `meta_require_cut_detection_model_facing`, `meta_require_text_processor`, `meta_require_audio_loudness`, `meta_require_audio_tempo`, `meta_require_audio_clap`, `meta_progress_every_frames`, `meta_semantic_jump_topk_events`, `meta_semantic_jump_min_strength`, `meta_semantic_jump_min_distance_frames`
- `stage_timings_ms` (мс) → **`meta_timing_<stage>`** без суффикса `_ms` в имени колонки; стадии в коде: `load_metadata`, `load_artifacts`, `core_compute`, `frame_features`, `events`, `finalize`, `total`

Булевы в meta в CSV плоского вывода: **0/1**.

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → блок `high_level_semantic` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → блок `high_level_semantic` (нормальные диапазоны для подсветки в `--melt-qa`)
- `view_csv_feature_descriptions_ru.json` — краткие пояснения к колонкам

---

## 5. Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `frame_feature_present_ratio` | **∈ [0, 1]** (для finite) |
| `meta.processed_frames` / `meta.total_frames` | `processed ≤ total` |
| `meta.processed_frames` | **N = len(frame_indices)** (если задано) |
| `meta.stage_timings_ms` | **≥ 0** |
| `scene_duration_s` | **≥ 0** (finite) |
| `event_strength` | **≥ 0** (finite) |

`meta.status=error`: в `--struct` / батче — краткое сообщение; в `--ranges` — проверки пропускаются.

## 6. Проверка артефакта (single-file + батч)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/high_level_semantic/utils/validate_high_level_semantic.py \
  <path/to/high_level_semantic.npz> --struct --qa --ranges
```

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/high_level_semantic/utils/validate_high_level_semantic.py \
  --results-base storage/result_store --platform-id youtube
```

`--qa` подмешивает в плоский слой **`text_feature_*`**, **скаляры из `features`**, плюс **`flatten_meta`**, для сверки с `view_csv_feature_qa` → `high_level_semantic`.

Требуется `numpy` и доступ к `DataProcessor/qa` (как в других `validate_*.py` в репозитории).
