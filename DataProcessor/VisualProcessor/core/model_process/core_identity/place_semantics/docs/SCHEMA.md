# Schema: `place_semantics_npz_v2`

- **producer**: `place_semantics`
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
- `place_category`
- `topk`
- `similarity_threshold`
- `threshold_global`
- `min_track_length`
- `max_gap_sec`
- `num_tracks`
- `num_places`
- `num_frames`
- `db_name`
- `db_version`
- `db_digest`

### Optional meta keys

- `embedding_model`
- `places_found_count`
- `core_clip_model_signature` (provenance chaining: model signature from upstream core_clip component)

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | Frame indices (strictly = `core_object_detections.frame_indices`) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | Time stamps from `union_timestamps_sec[frame_indices]` |
| `semantic_label_names` | `True` | `model_facing` | `str` | `(A)` | Canonical label space: `"id:name"` (id is stable within db_digest) |
| `semantic_object_ids` | `True` | `debug` | `str` | `(A)` | Original object UUIDs from Embedding Service, aligned with semantic_label_names |
| `threshold_per_label_arr` | `True` | `debug` | `float32` | `(A)` | Per-label thresholds (NaN if not set) |
| `track_ids` | `True` | `model_facing` | `int32` | `(T)` | Track IDs (temporal segments, one per detected place) |
| `track_present_mask` | `True` | `model_facing` | `bool` | `(T)` | Track present mask (always `True` for valid tracks) |
| `track_topk_ids` | `True` | `model_facing` | `int32` | `(T, K)` | Per-track top-K place IDs (aggregated over track frames) |
| `track_topk_scores` | `True` | `model_facing` | `float32` | `(T, K)` | Per-track top-K similarity scores (max over track frames) |
| `track_is_confident_top1` | `True` | `analytics` | `bool` | `(T)` | Per-track confidence flag (top-1 score >= threshold_global) |
| `track_topk_evidence_frame_indices` | `True` | `debug` | `int32` | `(T, K)` | Union frame indices where similarity is maximum for each top-K place in each track |
| `frame_topk_ids` | `True` | `analytics` | `int32` | `(N, K)` | Per-frame top-K place IDs |
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
- **Frame alignment**: Strictly aligned with `core_object_detections.frame_indices` (no-fallback policy)
- **Temporal segmentation**: Frames are grouped into tracks based on place recognition results:
  - Consecutive frames with same top-1 place are grouped into tracks
  - Tracks are merged if gap between frames ≤ `max_gap_sec`
  - Tracks shorter than `min_track_length` are filtered out
- **Track-level aggregation**: `track_topk_ids/scores` computed as max similarity over time per place within each track
- **Evidence frames**: `track_topk_evidence_frame_indices` points to union frame indices where each top-K place has maximum similarity in each track
- **Place recognition**: Uses frame embeddings from `core_clip/embeddings.npz` (required by schema) and place embeddings from Embedding Service for direct cosine similarity comparison (10-50x faster than HTTP requests per frame)
- **Upstream dependencies**: Requires `core_clip/embeddings.npz` with frame embeddings covering all required `frame_indices` (fail-fast, no-fallback)
- **Provenance chaining**: Includes `core_clip_model_signature` in meta for tracking upstream model versions

