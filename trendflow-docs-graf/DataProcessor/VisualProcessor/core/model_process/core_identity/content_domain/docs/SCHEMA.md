# Schema: `content_domain_npz_v2`

- **producer**: `content_domain`
- **artifact_kind**: `npz`
- **allow_extra_keys**: `False`
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
- `db_name`
- `db_version`
- `db_digest`

### Optional meta keys

- `db_path`
- `threshold_global`
- `threshold_per_label`
- `top_k`
- `confidence_threshold_top1`
- `clip_text_model_spec`
- `domain_db_dir`
- `core_clip_model_signature`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `frame_is_confident_top1` | `True` | `analytics` | `bool` | `(N)` |  |
| `frame_topk_ids` | `True` | `analytics` | `int32` | `(N, K)` |  |
| `frame_topk_scores` | `True` | `analytics` | `float32` | `(N, K)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `meta_json` | `True` | `debug` | `str` | `()` | meta as JSON string (cross-venv safe) |
| `semantic_label_names` | `True` | `model_facing` | `str` | `(A)` | label space: 'id:name' from domain db (ids are stable within db_digest) |
| `threshold_per_label_arr` | `True` | `debug` | `float32` | `(A)` | per-label thresholds aligned with semantic_label_names (NaN if not set) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
| `track_ids` | `True` | `model_facing` | `int32` | `(1)` | video aggregate pseudo-track id (=0) |
| `track_is_confident_top1` | `True` | `analytics` | `bool` | `(1)` |  |
| `track_present_mask` | `True` | `model_facing` | `bool` | `(1)` |  |
| `track_topk_ids` | `True` | `model_facing` | `int32` | `(1, K)` |  |
| `track_topk_scores` | `True` | `model_facing` | `float32` | `(1, K)` |  |
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
