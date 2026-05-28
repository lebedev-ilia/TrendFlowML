# `uniqueness` — features (Audit v3)

Канон для NPZ / wide CSV / QA: **`docs/FEATURE_DESCRIPTION.md`**.

**Module**: `DataProcessor/VisualProcessor/modules/uniqueness/utils/uniqueness.py`  
**Schema**: `uniqueness_npz_v4`  
**producer_version**: `1.0.2`

## Model-facing arrays

- **`max_sim_to_other (N,)`**: for each sampled frame, max cosine similarity to any other sampled frame (diag excluded).
- **`cos_dist_next (N-1,)`**: cosine distance between consecutive sampled frames (in sampling/time order).

## Model-facing scalars (`feature_names/feature_values`)

Fixed list in code: `_FEATURE_NAMES_V1`. Highlights:

- **Repeat threshold**:
  - `repeat_threshold_is_otsu` (0/1)
  - `repeat_threshold_used`, `repeat_threshold_raw`, `repeat_threshold_quality`, `repeat_threshold_min/max`, `repeat_threshold_bins`
- **Repetition**:
  - `repetition_ratio`
  - `max_sim_to_other_mean`, `max_sim_to_other_p95`
  - `pairwise_sim_mean`, `pairwise_sim_p95`
- **Consecutive similarity**:
  - `cos_dist_next_mean`, `cos_dist_next_p95`
- **Temporal change**:
  - `temporal_change_mean` (per-second)
- **Diversity proxy**:
  - `diversity_score = clip(1 - pairwise_sim_mean, 0..1)`
- **Effective unique frames**:
  - `effective_unique_frames`, `effective_unique_ratio` (по порогу `repeat_threshold_used`)
- **Sampling info**:
  - `n_frames`, `max_frames`

## Notes

- This is an **intra-video** baseline; no reference corpus.
- Strict alignment with `core_clip.frame_indices` is required (Segmenter contract).


