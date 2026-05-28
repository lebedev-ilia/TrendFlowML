# Schema: `core_face_identity_npz_v2`

- **producer**: `core_face_identity`
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

- `embedding_service_url`
- `category`
- `embedding_model`
- `top_k`
- `similarity_threshold`
- `n_frames`
- `total_faces_processed`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | Frame indices (only frames with faces) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | Time stamps from `union_timestamps_sec[frame_indices]` |
| `semantic_label_names` | `True` | `model_facing` | `str` | `(A)` | Canonical label space: `"int:name"` (int is stable within db_digest) |
| `semantic_object_ids` | `True` | `debug` | `str` | `(A)` | Original object UUIDs from Embedding Service, aligned with semantic_label_names |
| `face_ids` | `True` | `model_facing` | `int32` | `(N, K)` | Face identity indices (int32 from semantic_label_names), -1 where no result |
| `face_names` | `True` | `analytics` | `str` | `(N, K)` | Face names (human-readable), "" where no result |
| `face_similarities` | `True` | `model_facing` | `float32` | `(N, K)` | Similarity scores (0.0-1.0), 0.0 where no result |
| `face_bbox_xyxy` | `True` | `debug` | `float32` | `(N, 4)` | Bbox for top-1 face per frame (x1, y1, x2, y2), NaN where no face (for render assets) |
| `meta` | `True` | `debug` | `object` | `()` | Metadata dictionary |
| `meta_json` | `True` | `debug` | `str` | `()` | Meta as JSON string (cross-venv safe) |

## Notes

- **K=5**: Fixed top-K for semantic-head v1 contract
- **NaN-policy**: 
  - `face_ids`: `-1` where no result
  - `face_similarities`: `0.0` where no result
  - `face_names`: `""` where no result
- **Label-space**: Deterministic mapping from UUID to int32 via `semantic_label_names` and `semantic_object_ids`
- **Frame filtering**: Only frames with faces (from `core_face_landmarks.face_present`) are included
- **Deduplication**: Per-frame results are deduplicated by name (best similarity kept)

