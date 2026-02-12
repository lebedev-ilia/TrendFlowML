# ✅ Baseline Audit — `story_structure`

Компонент: `DataProcessor/VisualProcessor/modules/story_structure/`  
Тип: Visual module (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑28)  

---

## Резюме

`story_structure` вычисляет **narrative energy curve** (главный сигнал) и story‑markers (**hook window**, **climax**, **energy peaks**) строго на sampled кадрах от Segmenter.

Hard deps (no‑fallback):
- `core_clip` (`embeddings.npz`) — `frame_embeddings`
- `core_optical_flow` (`flow.npz`) — `motion_norm_per_sec_mean`
- `core_face_landmarks` (`landmarks.npz`) — `face_present` (используется только как overlay/flags; “нет лиц” не делает компонент empty)

Text (baseline v1):
- OCR‑ветка **B1**: `ocr_extractor/ocr.npz` → CLIP text encoder (Triton, `clip_text_triton`) → `topic_shift_curve`.
- По продуктовой политике: если OCR отсутствует/пустой — компонент пишет валидный NPZ, но с `meta.status="empty"`.

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) BaseModule интерфейс
- ✅ Inherits from `BaseModule`
- ✅ Implements `process(frame_manager, frame_indices, config)`
- ✅ `required_dependencies()` → `["core_clip", "core_optical_flow", "core_face_landmarks"]`

### 2) I/O contracts
- ✅ `frame_indices` берётся из Segmenter metadata (no self‑sampling)
- ✅ `times_s` = `union_timestamps_sec[frame_indices]` (no‑fallback)
- ✅ `min_frames=30` fail‑fast, `max_frames=200` fail‑fast (scheduler/Segmenter обязаны обеспечить)
- ✅ Fixed artifact name: `story_structure.npz` via `ARTIFACT_FILENAME`
- ✅ `meta` содержит run identity keys (enforced в `BaseModule.run` override)

### 3) Observability (PR‑5)
- ✅ Progress events (stage‑based): `start → load_deps → compute_curves → save → done`
- ✅ Stage timings: `summary.stage_timings_ms`

### 4) UI contract
- ✅ `meta.ui_payload` (JSON) содержит pointers на кривые (NPZ keys), markers (hook/climax), peaks list, flags.
- ✅ UI‑данные не хранятся как отдельные JSON‑артефакты в `result_store`.

### 5) Models policy
- ✅ OCR text embeddings используют `dp_models.ModelManager` (spec: `clip_text_triton`) и добавляют provenance в `meta.models_used`.
- ✅ SciPy разрешён для baseline (сглаживание/peaks).

---

## Артефакт (NPZ)

Путь: `.../story_structure/story_structure.npz`  
Schema: `story_structure_npz_v1`

Ключи (основное):
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `story_energy_curve (N,) float32`
- `motion_norm_per_sec_mean (N,) float32`
- `embedding_change_rate_per_sec (N,) float32`
- `any_face_present (N,) bool`
- `topic_shift_curve (N,) float32` (NaN если текста нет)
- `features` (dict, object-array)
- `meta` (dict, object-array)

Human demo:
- `VisualProcessor/modules/story_structure/quality_report/demo_story_structure_quality.py`

---

## Известные ограничения / next steps

- Variants A1/A2 (TextProcessor transcript / title&desc) описаны в `modules/story_structure/README.md` (roadmap).
- Segmentation (acts 1/2/3) остаётся вне baseline (риск эвристик/шума) — можно сделать opt‑in в legacy.


