# Schema: `franchise_recognition_npz_v2`

- **producer**: `franchise_recognition`
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
- `embedding_service_url`
- `franchise_category`
- `topk`
- `similarity_threshold`
- `threshold_global`
- `num_franchises`
- `num_frames`
- `db_name`
- `db_version`
- `db_digest`

### Optional meta keys

- `db_path`
- `embedding_model`
- `franchises_found_count`
- `ocr_npz`
- `ocr_events_used`
- `ocr_hits`
- `ocr_candidate_names`
- `ocr_evidence_frames`
- `core_clip_model_signature`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | Frame indices (strictly = `core_clip.frame_indices`) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | Time stamps from `union_timestamps_sec[frame_indices]` |
| `semantic_label_names` | `True` | `model_facing` | `str` | `(A)` | Canonical label space: `"id:name"` (id is stable within db_digest) |
| `semantic_object_ids` | `True` | `debug` | `str` | `(A)` | Original object UUIDs from Embedding Service, aligned with semantic_label_names |
| `threshold_per_label_arr` | `True` | `debug` | `float32` | `(A)` | Per-label thresholds (NaN if not set) |
| `track_ids` | `True` | `model_facing` | `int32` | `(1)` | Video-level aggregate track (always `[0]`) |
| `track_present_mask` | `True` | `model_facing` | `bool` | `(1)` | Video-level present mask (always `[True]`) |
| `track_topk_ids` | `True` | `model_facing` | `int32` | `(1, K)` | Video-level top-K franchise IDs (max over time per franchise) |
| `track_topk_scores` | `True` | `model_facing` | `float32` | `(1, K)` | Video-level top-K similarity scores (max over time) |
| `track_is_confident_top1` | `True` | `analytics` | `bool` | `(1)` | Video-level confidence flag (top-1 score >= threshold_global) |
| `track_topk_evidence_frame_indices` | `True` | `debug` | `int32` | `(1, K)` | Union frame indices where similarity is maximum for each top-K franchise |
| `frame_topk_ids` | `True` | `analytics` | `int32` | `(N, K)` | Per-frame top-K franchise IDs |
| `frame_topk_scores` | `True` | `analytics` | `float32` | `(N, K)` | Per-frame top-K similarity scores (0.0-1.0, NaN where no result) |
| `frame_is_confident_top1` | `True` | `analytics` | `bool` | `(N)` | Per-frame confidence flag (top-1 score >= threshold_global) |
| `meta` | `True` | `debug` | `object` | `()` | Metadata dictionary |
| `meta_json` | `True` | `debug` | `str` | `()` | Meta as JSON string (cross-venv safe) |

## Notes

- **K=5**: Fixed top-K for semantic-head v1 contract
- **NaN-policy**: 
  - `frame_topk_ids`: `-1` where no result
  - `frame_topk_scores`: `NaN` where no result
  - `track_topk_ids`: `-1` where no result
  - `track_topk_scores`: `NaN` where no result
- **Label-space**: Deterministic mapping from UUID to int32 via `semantic_label_names` and `semantic_object_ids`
- **Frame alignment**: Strictly aligned with `core_clip.frame_indices` (no-fallback policy)
- **Video-level aggregate**: `track_topk_ids/scores` computed as max over time per franchise
- **Evidence frames**: `track_topk_evidence_frame_indices` points to union frame indices where each top-K franchise has maximum similarity

