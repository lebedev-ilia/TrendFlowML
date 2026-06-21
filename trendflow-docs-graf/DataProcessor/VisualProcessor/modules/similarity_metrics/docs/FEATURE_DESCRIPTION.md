# `similarity_metrics` — фичи и артефакт (audit)

**schema_version:** `similarity_metrics_npz_v3`  
**Файл:** `similarity_metrics/results.npz` (см. [SCHEMA.md](SCHEMA.md), [README.md](../README.md), [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md))

## Назначение

Сопоставление ролика с **референс-сетом** (опционально, `dp_models` + `reference_set_id`) и **внутрироликовая когерентность** по эмбеддингам **core_clip**: кривые `centroid_sims`, `temporal_sim_next`, агрегаты + табличный вектор `feature_names` / `feature_values`. **Hard dep:** `core_clip/embeddings.npz` с **строгим равенством** `frame_indices` сегментеровского списка. Опционально: CLAP, text_processor, video_pacing, shot_quality, micro_emotion (пропуски → NaN в агрегатах, `status=ok`).

## Топ-уровень NPZ (ось N = |frame_indices|)

| Ключ | Форма | Смысл |
|------|--------|--------|
| `frame_indices` | (N,) int32 | Семпл Segmenter (sorted+unique) |
| `times_s` | (N,) float32 | `union_timestamps_sec[...]` |
| `centroid_sims` | (N,) float32 | cos-sim кадра к L2-центроиду нормированных CLIP-эмбеддингов |
| `temporal_sim_next` | (N−1,) float32 | cos-sim соседних кадров; при **N=1** — пустой массив |
| `reference_present` | scalar bool | Был ли задан `reference_set_id` и загружен пак |
| `feature_names` | (F,) object | Стабильный список скаляров для downstream |
| `feature_values` | (F,) float32 | Значения в том же порядке |
| `meta` | object | Канонический meta |

`ui_payload` в корне **нет**: `run()` кладёт его в `meta.ui_payload` (`similarity_metrics_ui_v1`).

## `meta` (wide CSV)

Плоский вывод: `meta_producer`, `meta_schema_version`, `meta_status`, run identity, `meta_analysis_*`, `meta_stage_timings_ms` → `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_total_ms`, `meta_models_used` / `meta_model_signature`. В **meta** также кладутся (если заданы в config): `top_n` → `meta_top_n`, `reference_set_id` → `meta_reference_set_id`, `enable_overall_score` → `meta_enable_overall_score` (bool в CSV как 0/1). Словарь `overall_weights` в плоский flatten **не** входит.

## Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `times_s` | Не убывает (при N>1) |
| `centroid_sims`, `temporal_sim_next` | Finite значения **∈ [−1, 1]** (cos-sim) |
| `meta.processed_frames` / `meta.total_frames` (если оба int) | `processed_frames ≤ total_frames` |
| `meta.processed_frames` | Совпадает с **N = len(frame_indices)** (если поле задано) |
| `meta.stage_timings_ms` | Значения **≥ 0** |

`meta.status=error`: в `--struct` / батче — краткое сообщение, payload не разбирается; в `--ranges` — проверки пропускаются.

## Валидатор (single-file)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/similarity_metrics/utils/validate_similarity_metrics_npz.py \
  <path/to/results.npz> --struct --qa --ranges
```

Батч по `result_store` (обход `**/similarity_metrics/results.npz`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/similarity_metrics/utils/validate_similarity_metrics_npz.py \
  --results-base storage/result_store --platform-id youtube
```

Полные правила meta/табличных полей: `storage/result_store/view_csv_feature_qa.json` → **`similarity_metrics`**.

**Датасетный/мульти-видео** валидатор (другой сценарий): `utils/validate_similarity_metrics.py` (база run’ов, не заменяет single-file NPZ-валидатор).
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
