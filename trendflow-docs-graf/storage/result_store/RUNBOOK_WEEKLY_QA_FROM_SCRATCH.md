# Runbook: Weekly QA From Scratch

This runbook rebuilds batch CSV artifacts and runs full weekly QA pipeline.

## 1) Go to repo root

```bash
cd "/media/ilya/Новый том/TrendFlowML"
```

## 2) Build main batch CSV (20 runs)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/batch_runs_feature_report.py \
  --run-glob "/media/ilya/Новый том/TrendFlowML/storage/result_store/youtube/*/*" \
  --max-runs 20 \
  --output-csv "/media/ilya/Новый том/TrendFlowML/storage/result_store/batch_features_report_20runs.csv"
```

## 3) Build baseline batch CSV (2 runs)

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/batch_runs_feature_report.py \
  --run-glob "/media/ilya/Новый том/TrendFlowML/storage/result_store/youtube/*/*" \
  --max-runs 2 \
  --output-csv "/media/ilya/Новый том/TrendFlowML/storage/result_store/batch_features_report_2runs_baseline.csv"
```

## 4) Run full weekly QA pipeline

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_qa_pipeline.py \
  --batch-csv "/media/ilya/Новый том/TrendFlowML/storage/result_store/batch_features_report_20runs.csv" \
  --batch-label "weekly_2026W17_20runs" \
  --baseline-csv "/media/ilya/Новый том/TrendFlowML/storage/result_store/batch_features_report_2runs_baseline.csv" \
  --golden-compare-csv "/media/ilya/Новый том/TrendFlowML/storage/result_store/batch_features_report_20runs.csv" \
  --run-text-validators-from-batch
```

## 5) Main summary output

By default pipeline writes into:

`/media/ilya/Новый том/TrendFlowML/storage/result_store/qa_runs/weekly_2026W17_20runs/`

Primary summary:

`/media/ilya/Новый том/TrendFlowML/storage/result_store/qa_runs/weekly_2026W17_20runs/feature_qa_pipeline_weekly_2026W17_20runs.summary.json`
---

## Навигация

[Vault](../../docs/MAIN_INDEX.md)
