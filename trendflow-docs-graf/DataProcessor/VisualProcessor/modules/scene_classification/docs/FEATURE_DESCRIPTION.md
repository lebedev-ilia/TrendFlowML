# `scene_classification` — фичи и артефакт (audit)

**schema_version:** `scene_classification_npz_v2`  
**Файл:** `scene_classification/scene_classification_features.npz` (см. [SCHEMA.md](SCHEMA.md), [README.md](../README.md))

## Назначение

Сегментация видео на **сцены** с Places365+семантикой из **core_clip**: per-frame top-K, энтропия, `frame_scene_id`, табличные поля по сценам, `scenes` / `scenes_raw`, `aggregated` логика в `scenes` + summary. **Hard deps:** `core_clip/embeddings.npz`, **cut_detection** (границы кадров), `union_timestamps_sec`, `metadata[scene_classification].frame_indices` ( ⊆ индексов core_clip ).

## Снимок `label_fusion` (top-level NPZ)

В NPZ дублируется лёгкий снимок конфига: `label_fusion` (str), `min_scene_seconds` (float) — согласованы с meta.

## Per-frame (ось N)

| Ключ | Форма | Смысл |
|------|--------|--------|
| `frame_indices` | (N,) int32 | Семпл Segmenter |
| `times_s` | (N,) float32 | `union_timestamps_sec[...]` |
| `frame_topk_ids` / `frame_topk_probs` | (N, 5) | Top-5 Places365 |
| `frame_entropy`, `frame_top1_prob`, `frame_top1_top2_gap` | (N,) | |
| `frame_scene_id` | (N,) int32 | Индекс сцены 0..S-1 (≥0) |

## Per-scene (ось S = число `scene_ids`)

`scene_label`, `fusion_mode`, `start_frame`/`end_frame`, длительности, метрики Places, aesthetic/luxury, atmosphere, stability; `indices`, `dominant_places_topk_ids/probs` — object-массивы переменной длины.

## Словари / object

- `scenes`, `scenes_raw` — один и тот же dict сцен (совместимость)
- `scene_*_prompts`, `places365_prompts` — репромпты/таксономия
- `summary` — в т.ч. `stage_timings_ms`: `infer_ms`, `aggregate_ms`; в **meta** после `run()` добавляется `total_ms`

## `meta` (wide CSV)

Плоский вывод: `meta_label_fusion`, `meta_min_scene_seconds`, `meta_min_scene_length_frames`, `meta_runtime`, `meta_batch_size`, `meta_input_size`, `meta_temporal_smoothing` (0/1), `meta_smoothing_window`, `meta_use_tta`, `meta_use_multi_crop`, `meta_use_timm`, `meta_enable_advanced_features`, `meta_prefer_cut_detection_boundaries`, `meta_module_sampling_policy_version`, `meta_triton_model_spec` (пусто в inprocess), `meta_model_arch`, `meta_timing_infer_ms`, `meta_timing_aggregate_ms`, `meta_timing_total_ms`, …

## Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `times_s` | Не убывает (N>1) |
| `min_scene_seconds` | **≥ 0** |
| `frame_topk_probs`, `frame_top1_prob` | Finite **∈ [0, 1]** |
| `frame_entropy` | **≥ 0** (для finite) |
| `fraction_high_confidence_frames`, `aesthetic_frac_high` (per-scene) | Finite **∈ [0, 1]** |
| `meta.processed_frames` / `meta.total_frames` | `processed ≤ total` |
| `meta.processed_frames` | Совпадает с **N = len(frame_indices)** (если задано) |
| `meta.stage_timings_ms`, `summary.stage_timings_ms` | **≥ 0** |

`meta.status=error`: в `--struct` / батче — краткое сообщение; в `--ranges` проверки пропускаются.

## Валидатор (single-file)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/scene_classification/utils/validate_scene_classification_npz.py \
  <path/to/scene_classification_features.npz> --struct --qa --ranges
```

Батч по `result_store` (`**/scene_classification/scene_classification_features.npz`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/scene_classification/utils/validate_scene_classification_npz.py \
  --results-base storage/result_store --platform-id youtube
```

Полные meta-ограничения: `storage/result_store/view_csv_feature_qa.json` → **`scene_classification`**.

Детальные имена внутри `scenes` и расшифровка метрик: [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md).
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
