# `detalize_face` — NPZ schema (`detalize_face_npz_v3`)

Этот документ описывает **human‑readable** контракт артефакта `.npz`, который пишет модуль `detalize_face`.

- **Producer**: `detalize_face`
- **producer_version**: `2.0.2`
- **schema_version**: `detalize_face_npz_v3`
- **Artifact filename (baseline)**: `detalize_face.npz`
- **Source-of-truth**: NPZ. HTML/JSON рендер — только для dev‑QA.

## Назначение

Компонент вычисляет производные face‑features на основе `core_face_landmarks` (MediaPipe FaceMesh) и кадров из `frames_dir`.

Важно: большинство `primary_*` метрик являются **heuristic / proxy** (эвристические прокси), а не “истинные” вероятности. Их стоит воспринимать как стабильные baseline‑сигналы для ML/QA, но не как ground truth.

Ключевая особенность sampling/axis:

- модуль **выравнивает выход по Segmenter axis**: `metadata[detalize_face].frame_indices`;
- вычисления делаются только для кадров, где `core_face_landmarks` нашёл лица (`face_present=true`);
- если face-кадров слишком много, допускается **внутренняя uniform‑выборка** среди face-кадров (см. `meta.face_frames_sampling_policy_version`);
- в output есть `processed_mask`, чтобы downstream могли отличать “нет лица” от “не считали”.

## Входы / зависимости

- **Hard dependency (no-fallback)**: `core_face_landmarks/landmarks.npz`
- **Hard requirement**: `frames_dir/metadata.json` должен содержать `union_timestamps_sec` (time-axis).

Если `core_face_landmarks` отсутствует или невалиден → **error** (fail-fast).
Если `core_face_landmarks` валиден, но лиц нет → **valid empty**:
`meta.status="empty"`, `meta.empty_reason="no_faces_in_video"`.

## NPZ keys

### Top-level keys

| key | required | tier | dtype | shape | описание |
|---|---:|---|---|---|---|
| `meta` | True | analytics | `object` | `()` | метаданные (см. ниже) |
| `summary` | True | analytics | `object` | `()` | агрегаты по результату модуля |
| `frame_indices` | True | model_facing | `int32` | `(N,)` | индексы кадров (subset union-domain), где считались фичи |
| `times_s` | True | model_facing | `float32` | `(N,)` | time-axis (из `union_timestamps_sec[frame_indices]`) |
| `face_present` | True | model_facing | `bool` | `(N,)` | флаг наличия лица по core_face_landmarks (до внутренних фильтров) |
| `processed_mask` | True | model_facing | `bool` | `(N,)` | флаг, что фичи для кадра были реально вычислены данным модулем |
| `primary_valid` | True | model_facing | `bool` | `(N,)` | флаг, что для кадра найден primary face (после фильтров качества / трекинга) |
| `face_count` | True | model_facing | `float32` | `(N,)` | число лиц в кадре (после фильтров качества) |
| `primary_tracking_id` | True | model_facing | `int32` | `(N,)` | tracking id primary лица (`-1` если недоступно) |
| `primary_compact_features` | True | model_facing | `float32` | `(N,40)` | compact embedding primary лица (рекомендовано для Encoder/Transformer; 0 если нет данных) |
| `aggregated` | True | model_facing | `object` | `()` | агрегаты по видео для tabular/baseline моделей (см. ниже) |
| `primary_gaze_at_camera_prob` | False | analytics | `float32` | `(N,)` | прокси-вероятность “взгляд в камеру” для primary лица (heuristic) |
| `primary_blink_rate` | False | analytics | `float32` | `(N,)` | прокси-частота морганий (heuristic) |
| `primary_attention_score` | False | analytics | `float32` | `(N,)` | прокси-внимание (heuristic) |
| `primary_quality_proxy_score` | False | analytics | `float32` | `(N,)` | прокси-качество (heuristic) |
| `primary_face_sharpness` | False | analytics | `float32` | `(N,)` | резкость ROI (heuristic) |
| `primary_occlusion_proxy` | False | analytics | `float32` | `(N,)` | прокси-окклюзия (heuristic) |
| `primary_speech_activity_prob` | False | analytics | `float32` | `(N,)` | прокси-речевая активность (heuristic) |
| `faces_agg` | True | analytics | `object` | `()` | агрегаты по трекам (`tracking_id` → dict) |

### `summary` (object)

Минимальный ожидаемый набор полей (может расширяться без изменения schema только при bump `schema_version`):

- `total_frames` (int)
- `processed_frames` (int)
- `frames_with_faces` (int)
- `total_faces` (int)
- `primary_faces` (int)
- `avg_faces_per_frame` (float)
- `stage_timings_ms` (dict): тайминги стадий **внутри** компонента (дублируются в `meta.stage_timings_ms`)

### `meta` (object)

#### Required meta keys (Audit v3 / vp_schema_v1)

- `producer` (str)
- `producer_version` (str)
- `schema_version` (str)
- `created_at` (str, ISO)
- `platform_id` (str)
- `video_id` (str)
- `run_id` (str)
- `config_hash` (str)
- `sampling_policy_version` (str) — run-level sampling policy (Segmenter)
- `dataprocessor_version` (str)
- `status` (`"ok"|"empty"|"error"`)
- `empty_reason` (str or null)
- `models_used` (list[dict])
- `model_signature` (str)
- `stage_timings_ms` (dict)

#### Optional meta keys

- `total_frames` (int)
- `processed_frames` (int)
- `frames_dir` (str)
- `analysis_fps` (float)
- `analysis_width` (int)
- `analysis_height` (int)
- `ui_payload` (dict): backend/UI payload (dev-friendly), `schema_version="detalize_face_ui_v2"`
- `module_sampling_policy_version` (str): `"segmenter_axis_v1"`
- `face_frames_sampling_policy_version` (str): `"faces_all_v1"` или `"faces_uniform_v1_max_<K>"`
- `write_primary_curves` (bool): если `true`, в NPZ будут записаны `primary_*` curves (эвристики)
- `write_primary_compact_features` (bool): если `true`, `primary_compact_features` заполняется для `processed_mask=true` кадров

### `aggregated` (object, model_facing)

Стабильный объект с агрегированными статистиками, безопасный для baseline/tabular head.

- `schema_version`: `"detalize_face_aggregated_v1"`
- `valid_frames` (int): число кадров, где `processed_mask=true` и `primary_valid=true`
- `axis_frames` (int)
- `face_present_ratio` (float)
- `processed_ratio` (float)
- `primary_valid_ratio` (float)
- `compact_dim` (int): `40`
- `compact_mean` (np.ndarray float32, shape `(40,)`)
- `compact_std` (np.ndarray float32, shape `(40,)`)
- `compact_p10` (np.ndarray float32, shape `(40,)`)
- `compact_p90` (np.ndarray float32, shape `(40,)`)
- `compact_l2_mean/std/p10/p90` (float)


