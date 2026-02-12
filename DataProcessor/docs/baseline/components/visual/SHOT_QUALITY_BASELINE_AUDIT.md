# Shot Quality — Baseline Audit (v1)

**Component**: `shot_quality`  
**Type**: Visual module (`BaseModule`)  
**Schema**: `shot_quality_npz_v1`  
**Artifact**: `result_store/<platform_id>/<video_id>/<run_id>/shot_quality/shot_quality.npz`  
**Default enabled**: ❌ No (`configs/visual_config.yaml: modules.shot_quality=false`)

---

## 1) Summary

`shot_quality` computes:
- per-frame quality features (`frame_features`) aligned to Segmenter time-axis (`times_s`)
- per-shot aggregates aligned to `cut_detection` shot boundaries
- CLIP-based zero-shot quality probabilities (`quality_probs`) consuming `core_clip`

It also writes `meta.ui_payload` for website visualization and reports runtime progress to `state_events.jsonl`.

**Special policy**: if **no faces in video** (`core_face_landmarks.has_any_face=false`) → `meta.status="empty"`, `empty_reason="no_faces_in_video"` (non-face metrics are still computed).

---

## 2) Architecture & contracts checklist

### 2.1 BaseModule interface
- ✅ Inherits from `BaseModule`
- ✅ Implements `process(frame_manager, frame_indices, config)`
- ✅ Uses `required_dependencies()` (core providers + `cut_detection`)

### 2.2 Input/output contracts
- ✅ `frame_indices` consumed from Segmenter metadata (no self-sampling)
- ✅ `times_s` saved as `union_timestamps_sec[frame_indices]` (no-fallback)
- ✅ Fixed artifact name (`shot_quality.npz`) supported by `BaseModule` via `ARTIFACT_FILENAME`
- ✅ `meta` includes run identity keys (enforced in `run()` override)

### 2.3 No-fallback policy
- ✅ Missing required deps → raises `RuntimeError`
- ✅ Invalid/missing `union_timestamps_sec` → raises `RuntimeError`
- ✅ Misaligned dependencies or processing failures → raises `RuntimeError`

### 2.4 Validation
- ⚠️ Requires validation via `VisualProcessor/utils/artifact_validator.py` (manual / CI hook pending)

---

## 3) Features / gating

Presets:
- `fast`: entropy-heavy metrics disabled; rolling_shutter/lens disabled
- `default`: entropy enabled; rolling_shutter/lens disabled
- `quality`: rolling_shutter + lens group enabled

Metadata:
- `impl_meta.preset`
- `impl_meta.feature_gating.*`

---

## 4) Progress reporting & stage timings

- ✅ Progress events in `state_events.jsonl`
  - stage transitions: `start`, `load_deps`, `quality_probs`, `frame_features`, `process`, `done`
  - cadence: every N frames via `progress_every_n_frames`
- ✅ `summary.stage_timings_ms`:
  - `frame_manager_ms`, `process_ms`, `total_ms`

---

## 5) Performance (measured resource costs)

**Status**: TODO (must be measured).

Target file (source of truth):
- `docs/models_docs/resource_costs/shot_quality_costs_v1.json`

Until measured:
- README documents knobs impacting cost (`preset`, `analysis_max_dim`, `matmul_chunk_size`)

---

## 6) Quality validation

**Human-friendly inspection**:
- TODO: add demo script `scripts/baseline/demo_shot_quality_quality.py` producing HTML + stats

**UI payload (must-haves)**:
- confidence graph: `meta.ui_payload.quality.frame_confidence`
- entropy graph: `meta.ui_payload.quality.frame_entropy`
- top‑K labels per shot: `meta.ui_payload.shots[].quality_topk_ids` + `quality_topk_probs`
- distribution (video): `meta.ui_payload.quality.video_mean_probs_topk_*`

**Sanity checks** (recommended):
- distributions of key features are finite (except NaNs for gated/face-dependent)
- `quality_probs` sums to ~1 per frame
- `shot_ids` cover all frames and are contiguous

---

## 7) Notes / known gaps

- GPU-only acceleration is not implemented yet (implementation is numpy/CPU); `device=cuda` is informational for now.
- Some “lens obstruction/dirt/glare” heuristics were removed from default output as noisy (available only partially via lens group).

---

## 8) Links

- Criteria: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- Schemas registry: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- Segmenter contract: `docs/contracts/SEGMENTER_CONTRACT.md`
- Module docs: `VisualProcessor/modules/shot_quality/README.md`


