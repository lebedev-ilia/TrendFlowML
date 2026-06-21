# `similarity_metrics` — NPZ schema (`similarity_metrics_npz_v3`)

- **producer**: `similarity_metrics`
- **producer_version**: `2.0.2`
- **schema_version**: `similarity_metrics_npz_v3`
- **artifact**: `results.npz`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/similarity_metrics_npz_v3.json`

## Purpose

Модуль вычисляет:
- **intra-video coherence** (покадровые кривые на `core_clip`): `centroid_sims`, `temporal_sim_next`
- **reference similarity** (опционально): агрегированные метрики сходства с reference set из `dp_models`

## Dependencies / axis

Hard deps (no-fallback):
- `core_clip` (`core_clip/embeddings.npz`)

Optional modalities (best-effort, отсутствие допустимо → NaN в агрегатах):
- `clap_extractor`, `text_processor`, `video_pacing`, `shot_quality`, `micro_emotion`, …

Axis (Audit v3 strict):
- `frame_indices`: строго от Segmenter (`metadata.json["similarity_metrics"]["frame_indices"]`)
- **требование**: `core_clip.frame_indices` **строго совпадает** с `frame_indices` (иначе error)
- `times_s = union_timestamps_sec[frame_indices]` (no-fallback)

## Empty / error semantics

- `empty` не используется: если optional модальности отсутствуют, модуль всё равно пишет `status="ok"`, а соответствующие агрегаты `NaN`.
- Любая проблема с axis/deps/union_timestamps_sec ⇒ **error**.

## Output keys

Обозначения: `N=len(frame_indices)`, `F=len(feature_names)`.

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | sorted+unique |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `centroid_sims` | True | model_facing | float32 | `(N,)` | cosine similarity to centroid |
| `temporal_sim_next` | True | model_facing | float32 | `(N-1,)` | similarity of consecutive frames |
| `reference_present` | True | model_facing | bool | `()` | True if reference_set_id enabled |
| `feature_names` | True | model_facing | object | `(F,)` | stable list (sorted keys) |
| `feature_values` | True | model_facing | float32 | `(F,)` | aligned to `feature_names` |
| `meta` | True | debug | object | `()` | canonical meta contract |

## Meta

### Required meta keys

- `producer`, `producer_version`, `schema_version`, `created_at`
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status`, `empty_reason`
- `models_used`, `model_signature`
- `stage_timings_ms`

### Optional meta keys

- `ui_payload` (dict): top‑K reference matches etc (`similarity_metrics_ui_v1`)
- config highlights: `top_n`, `reference_set_id`, `enable_overall_score`, `overall_weights`
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
