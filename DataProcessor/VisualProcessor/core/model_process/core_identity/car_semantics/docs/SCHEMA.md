# Schema: `car_semantics_npz_v2`

- **producer**: `car_semantics`
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
- `embedding_service_url`
- `category`
- `top_k`
- `confidence_threshold_top1`
- `proposal_classes`
- `proposal_class_ids`
- `max_tracks`
- `max_dets_per_frame`
- `labels_count`
- `tracks_total`
- `tracks_present`
- `dets_present`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `det_is_confident_top1` | `True` | `analytics` | `bool` | `(N, M_det)` |  |
| `det_present_mask` | `True` | `analytics` | `bool` | `(N, M_det)` |  |
| `det_topk_ids` | `True` | `analytics` | `int32` | `(N, M_det, K)` |  |
| `det_topk_scores` | `True` | `analytics` | `float32` | `(N, M_det, K)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `frame_is_confident_top1` | `True` | `analytics` | `bool` | `(N)` |  |
| `frame_topk_ids` | `True` | `analytics` | `int32` | `(N, K)` |  |
| `frame_topk_scores` | `True` | `analytics` | `float32` | `(N, K)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `meta_json` | `True` | `debug` | `str` | `()` | meta as JSON string (cross-venv safe) |
| `semantic_label_make` | `True` | `analytics` | `str` | `(A)` | best-effort make parsed from label name (dev-only taxonomy) |
| `semantic_label_model` | `True` | `analytics` | `str` | `(A)` | best-effort model parsed from label name (dev-only taxonomy) |
| `semantic_label_names` | `True` | `model_facing` | `str` | `(A)` | canonical label space: 'int:name' (int is stable within db_digest) |
| `semantic_object_ids` | `True` | `debug` | `str` | `(A)` | original object UUIDs from Embedding Service, aligned with semantic_label_names |
| `threshold_per_label_arr` | `True` | `debug` | `float32` | `(A)` | per-label thresholds (NaN if not set) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
| `track_best_bbox_xyxy` | `True` | `debug` | `float32` | `(T, 4)` |  |
| `track_best_class_id` | `True` | `debug` | `int32` | `(T)` |  |
| `track_best_det_idx` | `True` | `debug` | `int32` | `(T)` |  |
| `track_best_det_score` | `True` | `debug` | `float32` | `(T)` |  |
| `track_best_frame_pos` | `True` | `debug` | `int32` | `(T)` |  |
| `track_ids` | `True` | `model_facing` | `int32` | `(T)` |  |
| `track_is_confident_top1` | `True` | `analytics` | `bool` | `(T)` |  |
| `track_present_mask` | `True` | `model_facing` | `bool` | `(T)` |  |
| `track_topk_ids` | `True` | `model_facing` | `int32` | `(T, K)` |  |
| `track_topk_scores` | `True` | `model_facing` | `float32` | `(T, K)` |  |
