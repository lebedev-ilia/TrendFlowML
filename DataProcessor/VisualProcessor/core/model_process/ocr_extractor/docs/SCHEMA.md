# Schema: `ocr_extractor_npz_v2`

- **producer**: `ocr_extractor`
- **artifact_kind**: `npz`
- **allow_extra_keys**: `False`
- **schema_system_version**: `vp_schema_v1`

## Meta

### Required meta keys

- `producer`
- `producer_version`
- `schema_version`
- `created_at`
- `platform_id`
- `video_id`
- `run_id`
- `config_hash`
- `sampling_policy_version`
- `dataprocessor_version`
- `status`
- `empty_reason`
- `models_used`
- `model_signature`
- `stage_timings_ms`

### Optional meta keys

- `engine`
- `tesseract_lang`
- `tesseract_psm`
- `proposal_class`
- `retain_raw_ocr_text`
- `rec_model_spec`
- `ppocr_img_h`
- `ppocr_img_w`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `meta_json` | `True` | `debug` | `str` | `()` | meta as JSON string |
| `ocr_raw` | `True` | `analytics` | `object` | `(R)` | list[dict] rows (may be redacted) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
