# Similarity Metrics — Baseline Audit (v1)

**Component**: `similarity_metrics`  
**Type**: Visual module (`BaseModule`)  
**Schema**: `similarity_metrics_npz_v1`  
**Artifact**: `result_store/<platform_id>/<video_id>/<run_id>/similarity_metrics/results.npz`  
**Default enabled**: ❌ No (`configs/visual_config.yaml: modules.similarity_metrics=false`)

---

## 1) Summary

Wide baseline v1 computes:
- **Intra-video coherence** (per-frame):
  - `centroid_sims`: similarity(frame, centroid)
  - `temporal_sim_next`: similarity(frame_t, frame_{t+1})
- **Reference similarity** (per-video, multi-modality) using a **dp_models reference pack**:
  - visual (CLIP centroid from `core_clip`)
  - audio (CLAP embedding from `clap_extractor`) — **required**
  - text (TextProcessor `primary_embedding`) — optional if text missing
  - pacing (from `video_pacing`) — optional
  - quality/style (from `shot_quality`) — optional
  - emotion (from `micro_emotion`/faces) — optional

UI is supported via `meta.ui_payload` (top‑K reference matches + graphs).

---

## 2) Architecture & contracts checklist

### 2.1 BaseModule interface
- ✅ Inherits from `BaseModule`
- ✅ Implements `process(frame_manager, frame_indices, config)`
- ✅ Implements `required_dependencies()`: `core_clip` + `clap_extractor` (audio required policy)

### 2.2 I/O contracts
- ✅ `frame_indices` consumed from Segmenter metadata (no self-sampling)
- ✅ `times_s` saved as `union_timestamps_sec[frame_indices]` (no-fallback)
- ✅ Fixed artifact name `results.npz` via `ARTIFACT_FILENAME`
- ✅ `meta` contains run identity keys (enforced by `BaseModule.run`)

### 2.3 No-fallback policy
- ✅ Missing `core_clip` coverage for requested frames → `error`
- ✅ Missing audio (CLAP artifact) → `error`
- ✅ Missing faces/OCR/text content is allowed (features become NaN / omitted from overall)

---

## 3) Reference pack contract (dp_models)

Expected path:
- `dp_models/bundled_models/similarity/reference_sets/<reference_set_id>/`

Required files:
- `manifest.json` with `schema_version="similarity_reference_pack_v1"` and `reference_video_ids`
- `clip_video_embeddings.npy`
- `clap_audio_embeddings.npy`
- `text_primary_embeddings.npy`
- `pacing_features.npy`
- `shot_quality_features.npy`
- `emotion_embeddings.npy`

Policy: if `reference_set_id` is set and the pack is missing/invalid → **error**.

---

## 4) Progress reporting & stage timings

Progress events (stage-based):
- `start`, `coherence`, `load_deps`, `reference_similarity`, `done`

Stage timings:
- `summary.stage_timings_ms.process_ms`

---

## 5) Performance (measured resource costs)

**Status**: TODO.

Target file:
- `docs/models_docs/resource_costs/similarity_metrics_costs_v1.json`

---

## 6) Quality validation

Recommended sanity checks:
- `centroid_sims` finite, in [-1..1]
- `temporal_sim_next` finite, in [-1..1]
- top‑K refs stable for fixed reference pack
- strict errors on missing audio/core_clip coverage

Human-friendly demo:
- `VisualProcessor/modules/similarity_metrics/quality_report/demo_similarity_metrics_quality.py` (HTML: coherence + top‑K refs table)

---

## 7) Links

- Criteria: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- Schemas registry: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- Segmenter contract: `docs/contracts/SEGMENTER_CONTRACT.md`
- Module docs: `VisualProcessor/modules/similarity_metrics/README.md`


