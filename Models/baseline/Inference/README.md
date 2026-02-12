# Inference (baseline)

This folder implements **M6** from `Models/docs/plan_dev/BASELINE_DEV_PLAN.md`:
- extract features from per-run artifacts (manifest + NPZ) using the same logic as training
- load trained baseline artifacts
- produce deterministic prediction JSON suitable for backend/UI

## Quick start (example)

From repo root:

```bash
python3 Models/baseline/Inference/predict_baseline.py \
  --rs-base /abs/path/to/result_store \
  --platform-id youtube \
  --video-id NSumhkOwSg \
  --run-id 29003dedf0f2 \
  --model-dir Models/baseline/Training/artifacts/baseline_run_001 \
  --out-json /tmp/prediction.json \
  --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
  --enforce-required-components \
  --required-policy degraded
```

## Output fields

`predict_baseline.py` writes a deterministic JSON with:
- `prediction_status`: `ok|degraded`
- `missing_required_components`: list (only if enforcement enabled)
- `model_version` (from training manifest)
- `feature_schema_version` (copied from dataset metadata during training)
- `predictions_log1p_delta`:
  - bundle format: `views.{7d,14d,21d}`, `likes.{7d,14d,21d}`


