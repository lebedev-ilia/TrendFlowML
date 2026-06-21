# Result Store Structure

Recommended layout for generated analytics artifacts:

- `youtube/<video_id>/<run_id>/...` — raw run outputs (npz/artifacts).
- `batch_features_report_*.csv` — source wide batch exports (input to QA).
- `qa_runs/<label>/` — one full QA pipeline execution.
  - `feature_quality_report_<label>.{json,csv,md}`
  - `feature_batch_drift_<label>.{json,csv,md}` (if baseline provided)
  - `golden_mismatches_<label>.{csv,md}` (if golden compare provided)
  - `text_extractor_validators_report_<label>.{json,md}` (if text validators enabled)
  - `feature_shortlist_<label>.csv`
  - `feature_qa_pipeline_<label>.summary.json`
  - `html/` — rendered melt QA HTML outputs
- `incidents/feature_incidents.json` — global cross-run incident registry.

Related tools:

- `DataProcessor/tools/feature_qa_pipeline.py`
- `DataProcessor/tools/feature_quality_audit.py`
- `DataProcessor/tools/feature_batch_drift.py`
- `DataProcessor/tools/golden_batch_compare.py`
- `DataProcessor/tools/run_text_extractor_validators.py`
- `DataProcessor/tools/feature_incident_registry.py`
---

## Навигация

[Vault](../../docs/MAIN_INDEX.md)
