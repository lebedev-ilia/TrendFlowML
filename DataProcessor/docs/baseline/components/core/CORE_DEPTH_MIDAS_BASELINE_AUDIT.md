## Baseline audit: `core_depth_midas` (Tier‑0 core provider)

**Status**: ✅ CLOSED

### Ratings (1–10)

- **code & algo quality**: **7/10**
- **algorithm logic**: **7/10**
- **global interaction logic**: **8/10**
- **optimizations (parallelism, batching)**: **8/10**

### Scope / goal

`core_depth_midas` вычисляет depth maps (MiDaS family) на primary выборке кадров (union-domain) и сохраняет артефакт `depth.npz` для downstream модулей (в первую очередь `shot_quality`, также `frames_composition`).

### Evidence (files)

- **Component code**: `VisualProcessor/core/model_process/core_depth_midas/main.py`
- **Component README**: `VisualProcessor/core/model_process/core_depth_midas/README.md`
- **Baseline criteria**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Microbench evidence**:
  - B=1: `storage/reports/out/checklist-midas-b1/`
  - B=8: `storage/reports/out/checklist-midas-b8/`
- **Resource costs**:
  - B=1: `docs/models_docs/resource_costs/core_depth_midas_costs_v1.json`
  - B=8: `docs/models_docs/resource_costs/core_depth_midas_costs_b8_v1.json`
- **Human-friendly demo**: `scripts/baseline/demo_core_depth_midas_quality.py`

---

## 1) Architecture / contracts

### Runtime policy

- `runtime=triton` only (GPU-only inference via Triton).
- Sampling is **Segmenter-owned** (no fallback): reads `metadata.json["core_depth_midas"]["frame_indices"]`.

### Input contract (strict)

- `FrameManager.get(idx)` returns **RGB uint8** `HxWx3`
- `frame_indices` come from metadata (union-domain)
- time axis: `union_timestamps_sec` is required (no fallback)

### Output artifact (NPZ)

Path: `result_store/<platform_id>/<video_id>/<run_id>/core_depth_midas/depth.npz`

Keys:
- `frame_indices (N,) int32`
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `depth_maps (N, out_h, out_w) float32`
- `depth_mean (N,) float32`
- `depth_std (N,) float32`
- `depth_p05 (N,) float32` — 5-й перцентиль depth_maps[i] (по finite значениям; robust scale)
- `depth_p95 (N,) float32` — 95-й перцентиль depth_maps[i] (по finite значениям; robust scale)
- `meta` (dict as object-array)

### Meta compliance

Implemented:
- `dataprocessor_version` always present (defaults to `"unknown"` if not provided)
- `models_used[]` and `model_signature` via `apply_models_meta`

### Stage timings + progress (state_events)

Implemented:
- Component measures stage timings and stores them in `meta.stage_timings_ms` as milliseconds for:
  - `initialization`, `depth_inference_total`, `saving`, `total` (минимальный набор, может быть расширен)
- Component emits progress events to `state_events.jsonl` with stages:
  - `start → load_deps → process_frames → post_process → save → done`
- For `process_frames` it emits **granular** progress:
  - `progress ∈ [0,1]`, `done`, `total` (number of processed `frame_indices`)

### Atomic save + runtime validation

Implemented:
- Atomic NPZ save (`temp -> os.replace`)
- Post-save validation: `artifact_validator.validate_npz()` fail-fast (remove file + raise)

---

## 2) Global interaction / dependencies

### Downstream consumers

- `shot_quality` requires strict alignment: `frame_indices(core_depth_midas) == frame_indices(shot_quality)` (enforced in `shot_quality.py`)
- `frames_composition` now reads `core_depth_midas/depth.npz` (legacy `depth.json` removed from baseline path)

### Shared sampling group

`core_depth_midas` is part of the primary visual shared sampling group controlled by Segmenter (Tier‑0 core providers).

---

## 3) Triton models / batching

### Presets (mandatory)

- `midas_256`
- `midas_384`
- `midas_512`

### Batch-enabled

All presets are configured with `max_batch_size > 0` and support `batch_size >= 1`.

**Important note (implementation detail)**:
In the current environment, `DPT_Hybrid` export was not stable for dynamic batching, so for `384/512` we use `MiDaS_small` ONNX exports to keep batch-enabled behavior and consistent contracts.

**Precision (dp_models)**:
Текущие MiDaS_small ONNX веса — `fp32`, поэтому в `dp_models/spec_catalog/vision/midas_*_triton.yaml` зафиксировано `precision: fp32` (чтобы `models_used`/meta не врали).

---

## 4) Performance / resource costs

Источник: JSON files in `docs/models_docs/resource_costs/` (unit = `frame`).

### Unit-cost (B=1)

Источник: `docs/models_docs/resource_costs/core_depth_midas_costs_v1.json`  
Evidence: `storage/reports/out/checklist-midas-b1/`

| Branch | Latency mean (ms / frame) | p95 (ms) | CPU RSS peak (MB) | Triton VRAM peak (MB) | Triton VRAM delta_run (MB) | Spikes |
|--------|----------------------------|----------|-------------------|------------------------|----------------------------|--------|
| 256    | ~95.2                      | ~124.3   | ~49.1             | ~858                   | ~0                         | True   |
| 384    | ~221.8                     | ~244.1   | ~60.4             | ~858                   | ~0                         | False  |
| 512    | ~363.8                     | ~390.2   | ~66.8             | ~858                   | ~0                         | False  |

Примечание: `vram_triton_*` измеряется по процессу `tritonserver`. При B=1 рост VRAM во время инференса может быть 0 (модель уже загружена).

### Unit-cost (B=8)

Источник: `docs/models_docs/resource_costs/core_depth_midas_costs_b8_v1.json`  
Evidence: `storage/reports/out/checklist-midas-b8/`

| Branch | Latency mean (ms / frame) | Latency mean (ms / batch) | CPU RSS peak (MB) | Triton VRAM peak (MB) | Triton VRAM delta_run (MB) | Drift (MB) | Restart recommended |
|--------|----------------------------|----------------------------|-------------------|------------------------|----------------------------|-----------|---------------------|
| 256    | ~90.5                      | ~723.7                     | ~104.9            | ~2088                  | ~1230                      | ~384      | True                |
| 384    | ~204.4                     | ~1634.9                    | ~146.9            | ~3308                  | ~2066                      | ~768      | True                |
| 512    | ~366.7                     | ~2933.9                    | ~241.9            | ~5220                  | ~3210                      | ~2048     | True                |

Примечание: при B=8 наблюдается VRAM drift → рекомендуется перезапуск Triton между тяжёлыми прогонами на GPU 6GB.

---

## 5) Quality validation

### Human-friendly inspection

Скрипт:
- `scripts/baseline/demo_core_depth_midas_quality.py`

Он строит HTML отчёт с:
- выборкой кадров (thumbnails)
- depth maps (визуализация с percentile-нормализацией)
- базовыми sanity checks по артефакту

Примечание про нормализацию (важно для “оговорок” MiDaS):
- MiDaS даёт **относительную** глубину (не метры), абсолютные значения напрямую не интерпретируются.
- Для воспроизводимой нормализации в артефакте сохранены `depth_p05/depth_p95` (robust scale per-frame).
- Для сравнения цветов между кадрами в demo доступен режим `--global-norm` (единая шкала на весь прогон).

---

## 6) Open questions / follow-ups

- Выбор “оптимального” пресета (256/384/512) для обучения/production — **после сравнения качества/стоимости** на целевых данных.
- Если потребуется именно DPT-family на 384/512, нужно отдельно стабилизировать экспорт ONNX с dynamic batch (torch exporter / model variant).


