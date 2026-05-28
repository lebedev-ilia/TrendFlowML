# `story_structure` — features (Audit v3)

**Module**: `DataProcessor/VisualProcessor/modules/story_structure/story_structure.py`  
**Schema**: `story_structure_npz_v3`  
**producer_version**: `3.0.2`  

Сводка артефакта, meta → CSV, melt/QA: **`docs/FEATURE_DESCRIPTION.md`**

## What is model-facing?

### Sequence features (aligned to `frame_indices`)

- **`story_energy_curve`**: combined visual “energy” proxy based on CLIP embedding change rate + motion.
- **`motion_norm_per_sec_mean`**: per-second motion magnitude (from `core_optical_flow`).
- **`embedding_change_rate_per_sec`**: per-second CLIP embedding change proxy.
- **`any_face_present`**: face presence mask (from `core_face_landmarks`).
- **`topic_shift_curve`**: text topic-shift proxy (OCR → CLIP-text). If OCR missing/empty → array is NaN and `topic_shift_curve_present=false`.
- **`frame_feature_present_ratio`**: per-frame quality metric (finite ratio across model-facing float curves).

### Tabular scalar features (`feature_names/feature_values`)

Fixed order list in code: `_FEATURE_NAMES_V1`. Includes:

- **Hook**: `hook_visual_surprise_score/std`, motion intensity proxies, face presence in hook window.
- **Climax**: time/index/strength and peak count.
- **Character proxies**: face screen-time and switching rates.
- **Text proxy**: `topic_shift_curve_present`, `topic_shift_peaks_count`.

## Notes / caveats

- All signals are **heuristics / proxies** designed as a Tier‑0 baseline; they are useful for cheap ranking / sanity checks and as inputs to baseline models.
- Strict axis alignment is required: Segmenter must provide consistent `frame_indices`, and dependencies must cover them.


