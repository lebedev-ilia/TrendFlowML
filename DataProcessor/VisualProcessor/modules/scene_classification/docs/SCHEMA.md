# `scene_classification` — NPZ schema (`scene_classification_npz_v2`)

Human‑readable контракт артефакта `.npz`, который пишет модуль `scene_classification`.

- **Producer**: `scene_classification`
- **producer_version**: `2.0.1`
- **schema_version**: `scene_classification_npz_v2`
- **Artifact filename (baseline)**: `scene_classification_features.npz`
- **Source‑of‑truth**: NPZ. Render (`_render/*`) — dev‑QA.

## Inputs / dependencies (hard, no-fallback)

- `frames_dir/metadata.json`:
  - `union_timestamps_sec` (time-axis)
  - `metadata[scene_classification].frame_indices` (Segmenter axis)
- `core_clip/embeddings.npz`:
  - `frame_indices`, `frame_embeddings`
  - `places365_text_embeddings` (required when `label_fusion=clip`)
  - `scene_*_text_embeddings` (aesthetic/luxury/atmosphere)
- `cut_detection`:
  - `detections.shot_boundaries_frame_indices` (precision segmentation)

**Audit v3 decisions (FINAL)**:
- `cut_detection` is **hard dependency** (no-fallback).
- default `label_fusion="places"`.
- `enable_advanced_features` is **on** by default (tier=`analytics`).
- prompt lists in NPZ are **debug** (kept for reproducibility).

## Sampling / axis policy

- **Segmenter‑owned**: модуль не выбирает sampling самостоятельно; он принимает `frame_indices` из metadata.
- **Subset constraint**: `scene_classification.frame_indices ⊆ core_clip.frame_indices` (иначе fail-fast).

## NPZ keys (high level)

| key | required | tier | dtype | shape | описание |
|---|---:|---|---|---|---|
| `meta` | True | analytics | object | `()` | meta contract (см. ниже) |
| `frame_indices` | True | model_facing | int32 | `(N,)` | union-domain индексы |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `frame_*` | True | model_facing | numeric | `(N, …)` | per-frame уверенность/entropy/top-k + `frame_scene_id` |
| `scene_*` | True | analytics | numeric/object | `(S, …)` | табличные агрегаты по сценам |
| `scenes` | True | analytics | object | `()` | dict `scene_id → scene_dict` (canonical mapping) |
| `summary` | True | analytics | object | `()` | summary + `stage_timings_ms` (тайминги также дублируются в `meta.stage_timings_ms`) |

## Per-frame (model_facing)

- `frame_topk_ids (N,5) int32`
- `frame_topk_probs (N,5) float32`
- `frame_entropy (N,) float32`
- `frame_top1_prob (N,) float32`
- `frame_top1_top2_gap (N,) float32`
- `frame_scene_id (N,) int32` — индекс сцены (0..S-1), `-1` запрещён (error)

## Per-scene (analytics)

Сцены индексируются `scene_ids (S,)` (например `s0000`, `s0001`, …).

Табличные поля:

- `scene_label (S,) object` (Places365 label)
- `start_frame/end_frame (S,) int32`
- `start_time_s/end_time_s (S,) float32`
- `length_frames (S,) int32`
- `length_seconds (S,) float32`
- Places aggregates: `mean_score`, `class_entropy_mean`, `top1_prob_mean`, `top1_vs_top2_gap_mean`, `fraction_high_confidence_frames`
- Semantics (from `core_clip`): `mean_aesthetic_score`, `aesthetic_std`, `aesthetic_frac_high`, `mean_luxury_score`, `mean_cozy`, `mean_scary`, `mean_epic`, `mean_neutral`, `atmosphere_entropy`
- Stability: `scene_change_score`, `label_stability`

Debug/variable-length:

- `indices (S,) object` — list of union frame indices per scene
- `dominant_places_topk_ids/probs (S,) object` — list per scene

## `meta` required keys (Audit v3)

Обязательные:

- identity: `producer`, `producer_version`, `schema_version`, `created_at`
- run: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- status: `status`, `empty_reason`
- models: `models_used`, `model_signature`
- perf: `stage_timings_ms`

Дополнительно (optional, но ожидается после Audit v3):

- `ui_payload` (`schema_version="scene_classification_ui_v1"`)
- config highlights: `label_fusion`, `min_scene_seconds`, `min_scene_length_frames`, `runtime`, `model_arch`, `input_size`, `batch_size`, `temporal_smoothing`, …


