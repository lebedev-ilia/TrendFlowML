# `optical_flow_npz_v3` — NPZ schema (Audit v3)

- **producer**: `optical_flow`
- **producer_version**: `2.0.2`
- **schema_version**: `optical_flow_npz_v3`
- **Source-of-truth**: NPZ (`optical_flow/optical_flow.npz`)
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/optical_flow_npz_v3.json`

## Key ideas

- **Consumer-only**: модуль НЕ считает RAFT и не запускает модели; читает `core_optical_flow/flow.npz`.
- **Axis**: Segmenter-owned union-domain frames: `frame_indices (N,)` и `times_s (N,)`.
- **NaN policy**: если `frame_indices` не покрыт `core_optical_flow.frame_indices` → пишем `NaN` (и учитываем это в `missing_frame_ratio`).
- **Model-facing**: per-frame кривая + tabular агрегаты `feature_names/feature_values`.

## Fields

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | union-domain indices |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `motion_norm_per_sec_mean` | True | model_facing | float32 | `(N,)` | px/sec (from `core_optical_flow`) |
| `frame_feature_names` | True | model_facing | object | `(D,)` | имена per-frame compact фич |
| `frame_feature_values` | True | model_facing | float32 | `(N,D)` | per-frame compact фичи (NaN если не определено) |
| `feature_names` | True | model_facing | object | `(F,)` | имена video-level агрегатов |
| `feature_values` | True | model_facing | float32 | `(F,)` | значения video-level агрегатов |

### `feature_names` (fixed set)

- `motion_curve_mean`
- `motion_curve_median`
- `motion_curve_p90`
- `motion_curve_variance`
- `missing_frame_ratio`
 - `cam_shake_std_mean`
 - `cam_rotation_abs_mean`
 - `cam_translation_abs_mean`
 - `flow_consistency_mean`

## `meta` (baseline contract)

Required baseline keys include:
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash`
- `dataprocessor_version`, `status`, `empty_reason`
- `models_used`, `model_signature`
- `stage_timings_ms` (dict)

Optional:
- `ui_payload` (dict): privacy-safe payload for UI rendering (chart data).
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
