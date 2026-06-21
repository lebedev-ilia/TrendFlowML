# `high_level_semantic_npz_v2` — NPZ schema (Audit v3)

- **producer**: `high_level_semantic`
- **producer_version**: `2.0.2`
- **schema_version**: `high_level_semantic_npz_v2`
- **Source-of-truth**: NPZ (`high_level_semantic/high_level_semantic.npz`)
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/high_level_semantic_npz_v2.json`

## Key ideas

- **Axis**: Segmenter-owned union-domain frames: `frame_indices (N,)` with `times_s (N,)`.
- **No-network**: module does not load CLIP; consumes `core_clip/embeddings.npz`.
- **Scenes**: strictly from `cut_detection` (hard dependency), indices must match.
- **Dense model-facing**: `frame_features (N,F)` + `frame_feature_names (F,)`.
- **Events stream**: `event_*` arrays for UI/analytics (tier=`analytics`).

## Fields

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | union-domain indices |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `scene_id` | True | model_facing | int32 | `(N,)` | per-frame scene id |
| `scene_embeddings` | True | model_facing | float32 | `(S,D)` | mean core_clip embeddings per scene, L2-normalized |
| `scene_start_frame_idx` | True | model_facing | int32 | `(S,)` | union-domain |
| `scene_end_frame_idx` | True | model_facing | int32 | `(S,)` | union-domain (end proxy from cut_detection) |
| `scene_start_time_s` | True | model_facing | float32 | `(S,)` | |
| `scene_end_time_s` | True | model_facing | float32 | `(S,)` | |
| `scene_duration_s` | True | model_facing | float32 | `(S,)` | |
| `scene_representative_frame_idx` | True | model_facing | int32 | `(S,)` | union-domain |
| `scene_embedding_mean_norm` | True | analytics | float32 | `(S,)` | norm of per-scene mean embedding before L2-normalization (quality proxy) |
| `frame_feature_names` | True | model_facing | object | `(F,)` | column names for `frame_features` |
| `frame_features` | True | model_facing | float32 | `(N,F)` | NaN for missing optional modalities |
| `frame_feature_present_ratio` | True | model_facing | float32 | `(F,)` | доля finite по каждой колонке `frame_features` |
| `event_times_s` | True | analytics | float32 | `(E,)` | sorted by time |
| `event_type_id` | True | analytics | int16 | `(E,)` | taxonomy v1 (see `ui.event_type_map`) |
| `event_strength` | True | analytics | float32 | `(E,)` | |
| `event_frame_pos` | True | analytics | int32 | `(E,)` | 0..N-1 |
| `text_feature_names` | True | analytics | object | `(T,)` | optional copy from TextProcessor (may be empty) |
| `text_feature_values` | True | analytics | float32 | `(T,)` | |
| `features` | True | analytics | object | scalar | small scalar summary dict |
| `ui` | True | debug | object | scalar | event taxonomy, upstream presence flags, enabled groups |

## `meta` (baseline contract + highlights)

Required baseline keys include:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash`
- `dataprocessor_version`, `status`, `empty_reason`
- `models_used`, `model_signature`
- `stage_timings_ms` (dict)

Config highlights written to `meta` (optional keys in schema):
- `feature_groups`
- `require_cut_detection_model_facing`, `require_text_processor`, `require_audio_loudness`, `require_audio_tempo`, `require_audio_clap`
- `progress_every_frames`
- `semantic_jump_topk_events`, `semantic_jump_min_strength`
- `semantic_jump_min_distance_frames`
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
