# `text_scoring` — NPZ schema (`text_scoring_npz_v2`)

## Versioning

- **producer**: `text_scoring`
- **producer_version**: `2.0.1`
- **schema_version**: `text_scoring_npz_v2`

## Files

- **Module**: `DataProcessor/VisualProcessor/modules/text_scoring/text_scoring.py`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/text_scoring_npz_v2.json`
- **Artifact filename**: `text_scoring.npz`

## Axis & sampling

- **Axis owner**: Segmenter provides `text_scoring.frame_indices` (union-domain).
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback).
- **OCR filtering**: OCR rows are filtered to requested `frame_indices`. If nothing remains → valid empty output (`status="empty"`).

## Output fields (high-level)

### Model-facing (sequence)

- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `text_present () bool`
- `text_presence (N,) bool`
- `text_count_per_frame (N,) int32`

### Model-facing (tabular scalars)

- `feature_names (F,) object`
- `feature_values (F,) float32` — fixed order (see code: `_FEATURE_NAMES_V1`)

### Debug / analytics

- `ocr_raw (M,) object` — stored only if `store_debug_objects=true` (privacy-safe by default; raw text only if `retain_raw_ocr_text=true`)
- `ocr_unique_elements (K,) object` — same policy
- `meta.ui_payload` — UI hints (markers/flags), no heavy arrays duplication
- `meta.stage_timings_ms`


