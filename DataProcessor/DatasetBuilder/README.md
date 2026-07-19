# DatasetBuilder

Single point of assembly: per-run `result_store` NPZ artifacts + HF snapshot
metadata → one flat training table (features + targets + masks) for the baseline
boosting model. Schema-driven by `feature_spec.yaml` (feature_schema_version
`v0-real`). Created by Models Bot (Agent B), 2026-07-19.

> Status of first pass: **pipeline works end-to-end, metrics non-NaN, leakage
> audit passes.** Real trained quality is NOT yet possible — see "The target
> blocker" below. This is the expected outcome of the first pass per
> `Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md` §3.

## Files

| file | stage | what it does |
|------|-------|--------------|
| `feature_spec.yaml`        | schema | v0-real feature schema: which components/columns feed the vector |
| `build_training_table.py`  | C1 | walk `result_store/<platform>/<video>/<run>/manifest.json`, extract flat features (reuses `Models/baseline/common/npz_features.py`) |
| `add_targets.py`           | C2 | join HF snapshot shards, compute `y = log1p(max(x_h - x_0, 0))` for 7/14/21d + masks + snapshot_0 fields |
| `enrichment.py`            | C3 | deterministic `channel_id` for the non-leaking channel-group split |
| `build_full_dataset.py`    | C4 | one-command orchestrator C1→C2→C3→C4 → `dataset.parquet` + `dataset_metadata.json` |
| `make_smoke_dataset.py`    | test fixture | bootstrap real v0 feature schema to N videos × K channels with **synthetic, leakage-free** targets (smoke only) |
| `audit_and_report.py`      | gate | leakage audit + permutation importance + diagnostic PNGs |

## Quickstart

```bash
PY=Models/.venv/bin/python   # pandas/pyarrow/sklearn/matplotlib/pyyaml

# 1) Real dataset from whatever is in result_store now (honest masked targets
#    until follow-up snapshots exist):
$PY DataProcessor/DatasetBuilder/build_full_dataset.py --all-runs \
    --snapshots <hf_snapshot_shards_dir> \
    --out Models/baseline/artifacts/<tag>/dataset_real.parquet

# 2) Smoke fixture (enough videos/channels for a meaningful split) + train:
$PY DataProcessor/DatasetBuilder/make_smoke_dataset.py \
    --features <features.parquet> --out smoke.parquet
$PY Models/baseline/Training/train_baseline.py \
    --dataset smoke.parquet --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
    --out-dir <model_dir> --model-family sklearn

# 3) Leakage audit + importance + viz:
$PY DataProcessor/DatasetBuilder/audit_and_report.py \
    --dataset smoke.parquet --model-dir <model_dir> \
    --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml --target views_21d
```

## v0-real feature schema — what's IN and what's OUT

**IN (15 components):** cut_detection, optical_flow, scene_classification,
shot_quality, story_structure, uniqueness, video_pacing (7 visual);
clap_extractor, loudness_extractor, tempo_extractor (3 audio); core_clip,
core_face_landmarks, core_optical_flow, core_depth_midas, core_object_detections
(5 core) + snapshot_0 fields + temporal (video_age/duration/fps).

**OUT (documented, re-add on v1):**
- `brand_semantics`, `car_semantics`, `place_semantics`, `face_identity`,
  `franchise_recognition` — stub bases, no data (owner decision, portfolio §8).
  Training on them = training on constants/noise.
- **text/logo semantics of `core_object_detections`** (`class_ids`, `text_region`,
  `logo_region`) — detector is on COCO-80 weights, taxonomy re-label in progress
  (portfolio §3.4). Kept only taxonomy-independent geometry / person-density.
- Text components (Title/Comments embedders, cosine metrics, …) — not in baseline
  v1.0 list; comments are post-publication → leakage (portfolio §3.9).
- Structural / metadata columns (`frame_indices*`, `__count`, `__version`,
  `meta_json`, `embedding_dim`, …) pruned globally — no content signal.

## The target blocker (READ THIS before trusting any number)

Real targets need view/like snapshots at day **0, 7, 14, 21** for the SAME videos
we have features for. As of 2026-07 the HF collection (`Ilialebedev/*`, Fetcher
dataset_collector) has only **`snapshot_0`** — follow-up snapshots physically
haven't been taken yet (a week hasn't passed). So:

- `add_targets.py` on real shards → **0 rows with real targets** (reported
  honestly, all horizons masked). This is correct, not a bug.
- The trained numbers in `experiments.csv` use **SYNTHETIC** targets
  (`make_smoke_dataset.py`), constructed as a leakage-free monotone function of a
  few real features + `log(views_0)` + noise. They prove the *code path*, the
  *split*, the *masks*, the *metrics* and the *leakage audit* all work — they are
  **not** real predictive quality and must never be reported as such.

When follow-up snapshots land: re-run `build_full_dataset.py --snapshots <dir>`
(no `--synthetic`); everything downstream is unchanged.

## Leakage audit (mandatory gate)

`audit_and_report.py` asserts, before any metric is trusted:
1. no `target_*`/`mask_*` column entered the feature matrix (exact-set check);
2. no feature name carries future-snapshot / post-publication provenance;
3. every feature traces to an allowed source (v0 component / snapshot_0 /
   temporal) — 0 unmapped.
Result on the smoke run: **passed** (918 features, 0 leaked, 0 forbidden, 0
unmapped). By construction the whole v0 set is leakage-safe: features come only
from video content + `snapshot_0` (state at analysis time), never from a future
snapshot. Suspiciously high smoke Spearman (0.66–0.90) is explained by the
synthetic target design, and the permutation-importance top features are exactly
the synthetic drivers (`clap_magnitude`, `views_0`) — i.e. the audit + importance
correctly recover the planted signal, no target proxy leaked.

## Known gaps / next (see `Models/state/training_tasks.json`)
- `channel_id` currently falls back to `video_id` (no channel metadata in local
  manifests) → hybrid split degrades toward per-video on the real table. Real
  `channel_id` should come from HF metadata / YouTube API (enrichment C3 upgrade).
- catboost/lightgbm not installed (no Python 3.14 wheels) → used sklearn HistGB.
- Scale to the 150–250 video corpus (`AUTOMATED_TEST_CORPUS_PROTOCOL.md`) once
  TrendFlow Bot's full DP runs populate `result_store`.
