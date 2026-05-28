# Schema: `behavioral_npz_v1`

- **producer**: `behavioral`
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

- `total_frames`
- `processed_frames`
- `frames_dir`
- `analysis_fps`
- `analysis_width`
- `analysis_height`
- `ui_payload`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | Frame indices in union-domain (from Segmenter) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | `union_timestamps_sec[frame_indices]` (strict time axis) |
| `landmarks_present` | `True` | `model_facing` | `bool` | `(N)` | Mask: whether `core_face_landmarks` provided data for a frame |
| `hand_gestures` | `True` | `analytics` | `object` | `(N)` | Per-frame recognized gesture labels (list[str] per frame, best-effort) |
| `frame_results` | `True` | `debug` | `object` | `(N)` | Per-frame dicts with rich diagnostics (`hand_gestures`, `body_language`, `speech_behavior`, `stress`, `sequence_features`, …) |
| `aggregated` | `True` | `model_facing` | `object` | `()` | Video-level aggregate metrics (dict; baseline/tabular head) |
| `seq_num_hands` | `True` | `model_facing` | `float32` | `(N)` | Sequence feature (NaN when `landmarks_present=false`) |
| `seq_hands_visibility` | `True` | `model_facing` | `float32` | `(N)` | 0/1 visibility proxy (NaN when missing) |
| `seq_hand_motion_energy` | `True` | `model_facing` | `float32` | `(N)` | Hand motion energy proxy (pixel-space wrist motion) |
| `seq_arm_openness` | `True` | `model_facing` | `float32` | `(N)` | Arm openness proxy |
| `seq_pose_expansion` | `True` | `model_facing` | `float32` | `(N)` | Pose expansion proxy |
| `seq_body_lean_angle` | `True` | `model_facing` | `float32` | `(N)` | Body lean angle proxy |
| `seq_balance_offset` | `True` | `model_facing` | `float32` | `(N)` | Left/right balance proxy |
| `seq_shoulder_angle` | `True` | `model_facing` | `float32` | `(N)` | Shoulder angle (deg) |
| `seq_shoulder_angle_velocity` | `True` | `model_facing` | `float32` | `(N)` | Shoulder angle velocity proxy |
| `seq_head_position_x_norm` | `True` | `model_facing` | `float32` | `(N)` | Head center x in [0..1] (NaN if missing) |
| `seq_head_position_y_norm` | `True` | `model_facing` | `float32` | `(N)` | Head center y in [0..1] (NaN if missing) |
| `seq_head_motion_energy` | `True` | `model_facing` | `float32` | `(N)` | Head motion energy proxy |
| `seq_head_stability` | `True` | `model_facing` | `float32` | `(N)` | Stability proxy = 1/(1+motion) |
| `seq_mouth_width_norm` | `True` | `model_facing` | `float32` | `(N)` | Mouth width (normalized) |
| `seq_mouth_height_norm` | `True` | `model_facing` | `float32` | `(N)` | Mouth height (normalized) |
| `seq_mouth_area_norm` | `True` | `model_facing` | `float32` | `(N)` | Mouth area (normalized) |
| `seq_mouth_velocity` | `True` | `model_facing` | `float32` | `(N)` | Mouth area velocity |
| `seq_mouth_open_ratio` | `True` | `model_facing` | `float32` | `(N)` | Mouth open ratio |
| `seq_speech_activity_proxy` | `True` | `model_facing` | `float32` | `(N)` | Speech activity proxy in [0..1] |
| `seq_blink_flag` | `True` | `model_facing` | `float32` | `(N)` | Blink flag proxy (0/1) |
| `seq_blink_rate_short` | `True` | `model_facing` | `float32` | `(N)` | Short-window blink rate proxy |
| `seq_self_touch_flag` | `True` | `model_facing` | `float32` | `(N)` | Self-touch proxy (0/1) |
| `seq_fidgeting_energy` | `True` | `model_facing` | `float32` | `(N)` | Fidgeting proxy |
| `seq_timestamp_norm` | `True` | `model_facing` | `float32` | `(N)` | Normalized time in [0..1] |
| `seq_gesture_prob_pointing` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities (one key per gesture) |
| `seq_gesture_prob_open_palm` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_hands_on_hips` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_self_touch` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_fist` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_thumbs_up` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_thumbs_down` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_victory` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_ok` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_rock` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_call_me` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `seq_gesture_prob_love` | `True` | `model_facing` | `float32` | `(N)` | Soft gesture probabilities |
| `meta` | `True` | `debug` | `object` | `()` | Boxed meta dict |

## Empty semantics

- Если `core_face_landmarks` не нашёл лиц/позы/рук (или дал `status="empty"`), `behavioral` пишет валидный NPZ с:
  - `meta.status="empty"`,
  - `meta.empty_reason="no_faces_in_video"` (или проксируется из `core_face_landmarks`),
  - `landmarks_present=false` и NaN в sequence features.

## Notes

- `frame_results` и `hand_gestures` — **debug/analytics** поля для QA и интерпретации. Источник истины для моделей — `seq_*` и `aggregated`.
- Для production можно включить “лёгкий режим” без изменения схемы:
  - `store_debug_objects=false` ⇒ `frame_results` сохраняется как пустые dict’ы, `hand_gestures` — как пустые списки (ключи остаются, чтобы проходила schema validation).
- Полное описание смысла фичей: [FEATURES_DESCRIPTION.md](./FEATURES_DESCRIPTION.md).


