# `micro_emotion` — NPZ schema (`micro_emotion_npz_v3`)

Human‑readable контракт артефакта `.npz`, который пишет модуль `micro_emotion`.

- **Producer**: `micro_emotion`
- **producer_version**: `2.0.2`
- **schema_version**: `micro_emotion_npz_v3`
- **Artifact filename (baseline)**: `micro_emotion.npz`
- **Source‑of‑truth**: NPZ. Рендер (`_render/*`) — dev‑QA.

## Inputs / dependencies

Hard (no-fallback):

- `frames_dir/metadata.json`:
  - `union_timestamps_sec` (time-axis)
  - `metadata[micro_emotion].frame_indices` (Segmenter axis, union-domain)
- `core_face_landmarks/landmarks.npz`:
  - `frame_indices`, `face_present` (для face-gating)
- Docker + локальный image OpenFace:
  - `docker` доступен на машине
  - docker image `openface/openface:latest` (или альтернативный через `docker_image`) **должен быть заранее загружен** (компонент сам не делает pull)

## Sampling / axis policy

- **Segmenter‑owned axis**: модуль использует `metadata[micro_emotion].frame_indices`.
- **Compute gating**: OpenFace запускается **только** на кадрах, где `core_face_landmarks.face_present_any=true`.
- **Alignment**: output остаётся выровненным по axis:
  - для не-face кадров → `NaN` в числовых массивах, `face_present_any=false`.

## NPZ keys

| key | required | tier | dtype | shape | описание |
|---|---:|---|---|---|---|
| `meta` | True | analytics | object | `()` | meta contract (см. ниже) |
| `frame_indices` | True | model_facing | int32 | `(N,)` | union-domain индексы |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `face_present_any` | True | model_facing | bool | `(N,)` | наличие лица (any-face) по `core_face_landmarks` |
| `frame_feature_names` | True | model_facing | object | `(F,)` | имена wide per-frame фич |
| `frame_features` | True | model_facing | float32 | `(N,F)` | wide per-frame фичи (NaN где нет лица/нет OpenFace) |
| `compact22` | True | model_facing | float32 | `(N,22)` | **фиксированный** compact per-frame вектор (для encoder/transformer) |
| `compact22_feature_names` | True | model_facing | object | `(22,)` | стабильные имена compact‑координат |
| `event_times_s` | True | analytics | float32 | `(K,)` | события микроэмоций (timestamps) |
| `event_type_id` | True | analytics | int16 | `(K,)` | тип события (0 unknown, 1 smile, 2 surprise, 3 frown, 4 disgust) |
| `event_strength` | True | analytics | float32 | `(K,)` | сила события |
| `feature_names` | True | model_facing | object | `(V,)` | имена video-level scalar features (фиксированный набор) |
| `feature_values` | True | model_facing | float32 | `(V,)` | значения video-level scalar features (NaN если недоступно) |
| `microexpr_features` | True | analytics | object | `()` | подробные фичи микроэмоций (debug/analytics) |
| `summary` | True | analytics | object | `()` | summary + `stage_timings_ms` |

## `meta` required keys (Audit v3)

- identity: `producer`, `producer_version`, `schema_version`, `created_at`
- run: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- status: `status`, `empty_reason`
- models: `models_used`, `model_signature`
- perf: `stage_timings_ms`

## Empty semantics

- Если на всём axis нет кадров с лицами (`face_present_any` all-false) ⇒
  - `status="empty"`, `empty_reason="no_faces_in_video"`
  - массивы per-frame остаются выровненными по axis (NaN/пустые события).
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
