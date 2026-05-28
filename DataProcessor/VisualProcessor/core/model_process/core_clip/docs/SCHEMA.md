# Schema: `core_clip_npz_v2`

- **producer**: `core_clip`
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

### Optional meta keys

- `model_name`
- `total_frames`
- `batch_size`
- `runtime`
- `device`
- `prompts_version`
- `backend_proxy_version`
- `export_prompt_scores`
- `places365_topk_k`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `consecutive_cosine_prev` | `True` | `analytics` | `float32` | `(N)` |  |
| `cut_detection_transition_prompts` | `True` | `model_facing` | `object` | `(P_cut_transition)` |  |
| `cut_detection_transition_scores` | `True` | `analytics` | `float32` | `(N, P_cut_transition)` |  |
| `cut_detection_transition_text_embeddings` | `True` | `debug` | `float32` | `(P_cut_transition, D)` |  |
| `frame_embeddings` | `True` | `debug` | `float32` | `(N, D)` | raw CLIP image embeddings (not backend-facing) |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | union-domain frame indices |
| `meta` | `True` | `debug` | `object` | `()` | boxed dict |
| `places365_prompts` | `True` | `model_facing` | `object` | `(P_places365)` |  |
| `places365_text_embeddings` | `True` | `debug` | `float32` | `(P_places365, D)` |  |
| `places365_topk_indices` | `True` | `analytics` | `int32` | `(N, K_places365)` |  |
| `places365_topk_scores` | `True` | `analytics` | `float32` | `(N, K_places365)` |  |
| `places365_video_topk_indices` | `True` | `analytics` | `int32` | `(K_places365)` |  |
| `places365_video_topk_scores` | `True` | `analytics` | `float32` | `(K_places365)` |  |
| `popularity_topic_prompts` | `True` | `model_facing` | `object` | `(P_pop_topic)` |  |
| `popularity_topic_scores` | `True` | `analytics` | `float32` | `(N, P_pop_topic)` |  |
| `popularity_topic_text_embeddings` | `True` | `debug` | `float32` | `(P_pop_topic, D)` |  |
| `scene_aesthetic_prompts` | `True` | `model_facing` | `object` | `(P_scene_aesthetic)` |  |
| `scene_aesthetic_scores` | `True` | `analytics` | `float32` | `(N, P_scene_aesthetic)` |  |
| `scene_aesthetic_text_embeddings` | `True` | `debug` | `float32` | `(P_scene_aesthetic, D)` |  |
| `scene_atmosphere_prompts` | `True` | `model_facing` | `object` | `(P_scene_atmosphere)` |  |
| `scene_atmosphere_scores` | `True` | `analytics` | `float32` | `(N, P_scene_atmosphere)` |  |
| `scene_atmosphere_text_embeddings` | `True` | `debug` | `float32` | `(P_scene_atmosphere, D)` |  |
| `scene_luxury_prompts` | `True` | `model_facing` | `object` | `(P_scene_luxury)` |  |
| `scene_luxury_scores` | `True` | `analytics` | `float32` | `(N, P_scene_luxury)` |  |
| `scene_luxury_text_embeddings` | `True` | `debug` | `float32` | `(P_scene_luxury, D)` |  |
| `shot_quality_prompts` | `True` | `model_facing` | `object` | `(P_shot_quality)` |  |
| `shot_quality_scores` | `True` | `analytics` | `float32` | `(N, P_shot_quality)` |  |
| `shot_quality_text_embeddings` | `True` | `debug` | `float32` | `(P_shot_quality, D)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | union_timestamps_sec[frame_indices] |
