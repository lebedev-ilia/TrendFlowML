# `scene_classification`

Scene segmentation + classification on **Places365** with CLIP-based semantics computed strictly from `core_clip`.

## Inputs

- **Primary**: frames from `frames_dir` via `FrameManager` (union domain indices).
- **Time-axis**: `metadata.json` must include `union_timestamps_sec` (source-of-truth).
- **Dependencies** (fail-fast):
  - `core_clip` must exist in `rs_path/core_clip/embeddings.npz` and provide:
    - `frame_indices`, `frame_embeddings`
    - `scene_aesthetic_text_embeddings`, `scene_luxury_text_embeddings`, `scene_atmosphere_text_embeddings`
    - `places365_text_embeddings` (**required** when `label_fusion=clip`)
  - `cut_detection` must exist and provide:
    - `detections.shot_boundaries_frame_indices` (used for precision segmentation)

## Models (ModelManager)

Places365 model is loaded **only via** `dp_models` (no URLs, no `pretrained=True`).

Required env:
- `DP_MODELS_ROOT=/abs/path/to/local/models`
  - For local dev in this repo, you can point it to `DataProcessor/dp_models/bundled_models`.

Supported `model_arch`:
- **Places365 ResNet**: `resnet18`, `resnet50` → specs `places365_resnet18`, `places365_resnet50`
- **timm backbones** (when `use_timm=true`): `efficientnet_*`, `convnext_*`, `vit_*`, `regnetx_*`, `resnet50`, `resnet101`
  - resolved as specs `places365_timm_<arch>`

## Outputs (NPZ)

Saved to: `rs_path/scene_classification/scene_classification_features.npz`

**Version**: 2.0.1  
**Schema**: `scene_classification_npz_v2`  
**Artifact filename**: `scene_classification_features.npz`

Schemas:
- Human: `DataProcessor/VisualProcessor/modules/scene_classification/SCHEMA.md`
- Machine: `DataProcessor/VisualProcessor/schemas/scene_classification_npz_v2.json`

Canonical keys:
- `frame_indices (N,) int32` — union-domain indices processed by this module
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]` (source-of-truth)
- `label_fusion (str)` — runtime config snapshot: `places|clip`
- **`scenes`**: dict mapping `scene_id -> scene_dict` where `scene_id` is `s0000`, `s0001`, ...
  - `scene_dict` includes:
    - `scene_label` (Places365 label)
    - `fusion_mode` (`places|clip`) — how label was chosen (segment majority)
    - `indices` (list of union frame indices in this scene)
    - `start_frame`, `end_frame`, `length_frames`, `length_seconds`
    - `start_time_s`, `end_time_s`
    - Places365 aggregates: `mean_score`, `class_entropy_mean`, `top1_prob_mean`, `top1_vs_top2_gap_mean`, `fraction_high_confidence_frames`
    - Ontology aggregates: `mean_indoor`, `mean_outdoor`, `mean_nature`, `mean_urban`
    - CLIP semantics (from `core_clip`): `mean_aesthetic_score`, `aesthetic_std`, `aesthetic_frac_high`, `mean_luxury_score`,
      `mean_cozy`, `mean_scary`, `mean_epic`, `mean_neutral`, `atmosphere_entropy`
    - Stability: `scene_change_score`, `label_stability`
    - `dominant_places_topk_ids`, `dominant_places_topk_probs`
- Flat arrays (`scene_ids`, `scene_label`, `start_frame`, …) duplicated for NPZ-friendly tabular access.
- `meta` includes `models_used[]` and `model_signature` (Places365 + upstream `core_clip`).

## Semantics / scene count rules

- **One scene is valid** (e.g. if all processed frames have the same dominant Places label).
- `frame_indices` **must be >= 2**, иначе `error` (no-fallback).
- Scenes are formed by consecutive frames with the same predicted label, then filtered by `min_scene_seconds` (or `min_scene_length / fps`).

## Downstream usage

- `color_light` consumes `scene_classification.scenes` and treats each `scene_id` as a unique segment (label collisions are handled).

## Sampling / units-of-processing requirements (Segmenter-owned)

- `scene_classification` **не владеет sampling**: `frame_indices` приходят строго из `frames_dir/metadata.json` (Segmenter).
- По графу зависимостей sampling гарантируется как **subset**: `scene_classification ⊆ core_clip` (иначе модуль fail-fast на missing embeddings).
- Бюджеты по умолчанию задаёт Segmenter (см. `Segmenter/_build_default_component_budgets()`), и их можно переопределить в `VisualProcessor` config через секцию:
  - `scene_classification.sampling.{min_frames,target_frames,max_frames}`

Short-video note:
- For ~30–60s clips, Segmenter uses denser primary sampling (≈0.25s gap) to better capture multi-scene content.

## Label fusion (Places365 + CLIP)

This module supports two modes for the **final scene label** (heuristic mixing is forbidden):
- `label_fusion=places` (default): use Places365 top‑1 (supervised)
- `label_fusion=clip`: use CLIP zero-shot over the **same 365 labels** (requires `core_clip.places365_text_embeddings`)

**Note**: The `fused` mode (heuristic mixing of Places365 and CLIP) is not supported. Only `places` and `clip` modes are available.

Notes:
- CLIP text embeddings for Places365 are computed in `core_clip` and stored in `core_clip/embeddings.npz`.
- When `runtime=triton`, `clip_text` commonly has `max_batch_size=64`; `core_clip` chunks prompt batches accordingly.

## Cut-aware segmentation (precision policy)

For higher precision, `scene_classification` uses **hard shot boundaries** from:
- `cut_detection.detections.shot_boundaries_frame_indices`

Scenes are then built by accumulating consecutive shots until duration ≥ `min_scene_seconds` (default **2.0s**).

## Models

### GPU (Triton) — baseline

Places365 ResNet50 branches (fixed spatial shape, **batch-enabled**):
- `places365_resnet50_224` (224×224)
- `places365_resnet50_336` (336×336)
- `places365_resnet50_448` (448×448)

ModelManager specs:
- `dp_models/spec_catalog/vision/places365_resnet50_{224,336,448}_triton.yaml`

Triton contract (ensemble, external IO):
- input: `INPUT__0` (`UINT8`, `[B,S,S,3]`, RGB NHWC)
- output: `OUTPUT__0` (`FP32`, `[B,365]` logits)

## Parallelization

- **Internal batching**: при `runtime=triton` модуль отправляет кадры батчами `--batch-size` (scheduler knob) → один infer на батч.
- **External parallelism**: можно запускать на разных видео (разные `run_id`), ограничение — VRAM/throughput Triton.

## Performance characteristics

Источник правды:
- `docs/models_docs/resource_costs/scene_classification_costs_v1.json` (unit-cost, `model_batch_size=1`)
- `docs/models_docs/resource_costs/scene_classification_costs_b8_v1.json` (throughput, `model_batch_size=8`)

Единица обработки: `frame`

## Progress reporting (backend)

Компонент пишет `state_events.jsonl` прогресс (unit=`frame`) и stage transitions (start/load_deps/infer/aggregate/saved/done).

## UI payload

Компонент формирует `meta.ui_payload` (JSON, без картинок) для backend/UI:
- список сцен + ключевые агрегаты
- `thumb_frame_indices` (backend может нарезать thumbnails из frames_dir)
- per-frame кривые: entropy/top1_prob/gap + `frame_scene_id`

## Quality validation & human-friendly inspection

Для ручной проверки сегментации/лейблов и CLIP-семантики:
- `scripts/baseline/demo_scene_classification_quality.py`

## Audit v3 — Decisions (FINAL)

- **cut_detection**: **hard dependency** (no-fallback). Нет shot boundaries ⇒ `error`.
- **label_fusion default**: `places` (supervised Places365 top‑1). `clip` — опционально.
- **advanced features** (`enable_advanced_features`): **on by default** (tier=`analytics`).
- **prompts in NPZ** (`places365_prompts`, `scene_*_prompts`): **keep** как `debug` для воспроизводимости/QA.

## Render (dev-only, offline)

NPZ — source-of-truth. Рендеры — только для QA/аудита.

### Артефакты рендера

- **Render-context JSON**: `result_store/<platform_id>/<video_id>/<run_id>/scene_classification/_render/render_context.json`
- **HTML mini-dashboard**: `result_store/.../scene_classification/_render/render.html` (работает **offline**, без CDN)

### Что смотреть (QA)

- **Timeline**:
  - `frame_top1_prob`: провалы = “неуверенные кадры” (часто motion blur / смена сцены / плохой кадр)
  - `frame_entropy`: пики = сильная неопределённость / смешанные сцены
  - `frame_top1_top2_gap`: малые значения = ambiguity
- **Top / Anti-top**:
  - top entropy → кандидаты на ошибки/переходы/смешанные сцены
  - anti-top top1_prob → самые слабые кадры для классификации
- **Scenes table**:
  - адекватность `scene_label`
  - `length_seconds` не слишком маленькие (контроль `min_scene_seconds`)
  - `label_stability` и `scene_change_score` логично коррелируют с монтажом

### Где смотреть время выполнения

- `meta.stage_timings_ms` (и дублируется в `summary.stage_timings_ms`)
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
