# `emotion_face` — NPZ schema (`emotion_face_npz_v3`)

Human‑readable контракт артефакта `.npz`, который пишет модуль `emotion_face`.

- **Producer**: `emotion_face`
- **producer_version**: `2.0.2`
- **schema_version**: `emotion_face_npz_v3`
- **Artifact filename (baseline)**: `emotion_face.npz`
- **Source-of-truth**: NPZ. HTML/JSON рендер — только для dev‑QA.

## Назначение

Компонент вычисляет эмоции на лицах (EmoNet) и производит time‑series сигналы + события keyframes.

## Входы / зависимости

- **Hard dependency (no-fallback)**: `core_face_landmarks/landmarks.npz`
- **Hard requirement**: `frames_dir/metadata.json` должен содержать `union_timestamps_sec` (time-axis).
- **Model**: EmoNet через **ModelManager** (`dp_models`, no-network). Legacy `emo_path` допускается только как debug override.

## Sampling / axis policy

- **Axis**: output выровнен по `metadata[emotion_face].frame_indices` (Segmenter contract; union-domain).
- **Fallback (legacy)**: если ключа нет, axis берётся из `core_face_landmarks.frame_indices` (warning).
- **Compute gating**: inference выполняется только на кадрах, где `core_face_landmarks.face_present` имеет хотя бы одно лицо.
- **Internal sampling**: среди face-кадров применяется:
  - stride `face_frame_stride`
  - cap `max_frames`
- `meta.module_sampling_policy_version` и `meta.face_frames_sampling_policy_version` фиксируют политику.

## NPZ keys

| key | required | tier | dtype | shape | описание |
|---|---:|---|---|---|---|
| `meta` | True | analytics | `object` | `()` | метаданные (см. ниже) |
| `frame_indices` | True | model_facing | `int32` | `(N,)` | axis индексы кадров (union-domain) |
| `times_s` | True | model_facing | `float32` | `(N,)` | time-axis `union_timestamps_sec[frame_indices]` |
| `face_present` | True | model_facing | `bool` | `(N,)` | наличие хотя бы одного лица на кадре |
| `processed_mask` | True | model_facing | `bool` | `(N,)` | кадр реально прошёл inference (после internal sampling) |
| `face_count` | True | model_facing | `int16` | `(N,)` | число лиц по core_face_landmarks |
| `valence` | True | model_facing | `float32` | `(N,)` | NaN если `processed_mask=false` |
| `arousal` | True | model_facing | `float32` | `(N,)` | NaN если `processed_mask=false` |
| `intensity` | True | model_facing | `float32` | `(N,)` | NaN если `processed_mask=false` |
| `emotion_confidence` | True | model_facing | `float32` | `(N,)` | NaN если `processed_mask=false` |
| `emotion_probs` | True | model_facing | `float32` | `(N,8)` | NaN если `processed_mask=false` |
| `dominant_emotion_id` | True | model_facing | `int8` | `(N,)` | `-1` если `processed_mask=false` |
| `sequence_features` | True | debug | `object` | `()` | legacy + UI/debug (дублирует time-series) |
| `keyframes` | True | analytics | `object` | `(K,)` | события (peaks/transitions), список dict |
| `summary` | True | analytics | `object` | `()` | summary + `stage_timings_ms` |
| `features` | True | analytics | `object` | `()` | агрегированные фичи (legacy) |
| `advanced_features` | True | analytics | `object` | `()` | доп. фичи (gated, legacy) |
| `axis_source` | False | debug | `object` | `()` | `"emotion_face"` или `"core_face_landmarks"` |

### `sequence_features` (object, debug)

Axis‑aligned массивы длины `N`:

- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `face_present (N,) bool` — наличие хотя бы одного лица
- `processed_mask (N,) bool` — кадр реально прошёл inference (после internal sampling)
- `face_count (N,) int16` — число лиц по core_face_landmarks
- `valence (N,) float32` (NaN если `processed_mask=false`)
- `arousal (N,) float32`
- `intensity (N,) float32`
- `emotion_confidence (N,) float32`
- `emotion_probs (N,8) float32` (Ekman order фиксирован)
- `dominant_emotion_id (N,) int8` (`-1` если `processed_mask=false`)

Multi-face (в пределах `max_faces_per_frame`, NaN padding):

- `valence_faces (N, max_faces)`
- `arousal_faces (N, max_faces)`
- `emotion_confidence_faces (N, max_faces)`
- `emotion_probs_faces (N, max_faces, 8)`

## `meta` required keys (Audit v3)

- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status`, `empty_reason`
- `models_used`, `model_signature`
- `stage_timings_ms`


