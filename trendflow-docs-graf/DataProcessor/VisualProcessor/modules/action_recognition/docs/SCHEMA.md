# Schema: `action_recognition_npz_v2`

- **producer**: `action_recognition`
- **artifact_kind**: `npz`
- **baseline artifact**: `action_recognition/action_recognition_features.npz` (старые прогоны: `action_recognition_emb.npz`)
- **allow_extra_keys**: `False`
- **allowed_extra_key_prefixes**: `metric__` — плоские массивы `(T,)` только для скалярных per-track полей (полные структуры — в `results_json`)
- **schema_system_version**: `vp_schema_v1`

## Meta

### Required meta keys

- `producer`
- `producer_version`
- `schema_version`
- `created_at`
- `platform_id`
- `video_id`
- `run_id`
- `config_hash`
- `sampling_policy_version`
- `dataprocessor_version`
- `status`
- `empty_reason`
- `models_used`
- `model_signature`
- `stage_timings_ms`

### Optional meta keys

- `total_frames`
- `processed_frames`
- `frames_dir`
- `clip_len`
- `stride`
- `batch_size`
- `embedding_dim`
- `model_name`
- `processed_tracks`
- `ui_payload`
- `segment_gap_sec`
- `min_person_confidence`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `tracks` | `True` | `model_facing` | `int32` | `(T)` | Track IDs (segment IDs from person detections) |
| `embeddings` | `True` | `model_facing` | `object` | `(T)` | Per-track embeddings: `embedding_normed_256d` arrays `[num_clips, 256]` float32 |
| `results_json` | `True` | `analytics` | `object` | `(T)` | Per-track metrics and diagnostics (dict per track) |

### Per-track results_json structure

Each element in `results_json` is a dict with the following keys:

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `embedding_normed_256d` | `True` | `model_facing` | `float32` | `(num_clips, 256)` | L2-normalized embeddings for each clip (sequence features for VisualTransformer) |
| `max_temporal_jump` | `True` | `analytics` | `float32` | `()` | Maximum jump between adjacent clips (L2 distance on normed embeddings) |
| `mean_temporal_jump` | `True` | `analytics` | `float32` | `()` | Mean jump between adjacent clips |
| `stability` | `True` | `analytics` | `float32` | `()` | Action stability (PCA+KMeans, longest run fraction, 0-1) |
| `stability_centroid_dist` | `True` | `analytics` | `float32` | `()` | Alternative stability metric: mean distance to cluster centroid (lower = more stable) |
| `num_switches` | `True` | `analytics` | `int32` | `()` | Number of cluster switches (from KMeans labels) |
| `num_clips` | `True` | `analytics` | `int32` | `()` | Number of clips for this track |
| `track_frame_count` | `True` | `analytics` | `int32` | `()` | Number of frames in track |
| `embedding_dim` | `True` | `debug` | `int32` | `()` | Embedding dimension (always 256) |
| `clip_center_frame_indices` | `True` | `debug` | `int32` | `(num_clips)` | Center frame indices (union-domain) |
| `clip_center_times_s` | `True` | `debug` | `float32` | `(num_clips)` | Center times in seconds (if `union_timestamps_sec` available) |
| `temporal_jumps` | `True` | `debug` | `float32` | `(num_clips-1)` | Per-clip temporal jumps (L2 distances) |
| `clip_frame_indices` | `True` | `debug` | `object` | `(num_clips)` | Frame indices for each clip (list of lists) |

## Notes

- **Sequence features** (`embedding_normed_256d`): Used by VisualTransformer for temporal pattern learning.
- **Aggregate features** (`max_temporal_jump`, `mean_temporal_jump`, `stability`, `stability_centroid_dist`, `num_switches`): Used by MLP/Tabular Head for action dynamics analysis.
- **Debug features** (`clip_center_frame_indices`, `clip_center_times_s`, `temporal_jumps`, `clip_frame_indices`): For QA, visualization, and diagnostics.
- Tracks are generated from person detections (class_id=0) by grouping consecutive frames with person detections.
- Segmentation uses temporal gap threshold (`segment_gap_sec`, default 0.5s) and minimum person confidence (`min_person_confidence`, default 0.3).
- Empty outputs are valid if no person detections found (`status="empty"`, `empty_reason="no_person_detections"`).

## ⚠️ Статус качества

**ВНИМАНИЕ**: Модуль требует **доработки качества** и **более тщательной проверки**. Все поля схемы генерируются корректно, но алгоритмы требуют валидации на репрезентативных датасетах:

- **Метрики стабильности**: Требуется проверка корректности PCA+KMeans кластеризации на различных типах действий
- **Временные метрики**: Валидация корректности вычисления `temporal_jumps` и согласованности с `clip_center_times_s`
- **Сегментация**: Проверка логики группировки person детекций на edge cases
- **Эмбеддинги**: Валидация нормализации и проекции 2048d → 256d

См. подробности в [README.md](./README.md#-статус-качества-и-доработки).
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
