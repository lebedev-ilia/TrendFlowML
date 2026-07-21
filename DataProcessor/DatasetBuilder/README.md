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
  `meta_json`, `embedding_dim`, `npz_path`, …) pruned globally — no content signal.

## Audit-driven feature cleaning (2026-07-19, exp_0002)

Beyond dropping whole stub components, the 15 IN components were cross-checked
against their own `DataProcessor/docs/component_reports/<c>/FINAL_REPORT.md` and
confirmed against the real 6-video table. Per-component `column_exclude` now
removes **audit-named dead / poisoned / constant-by-design** columns
(**971 → 882** feature cols, −89):

| component | dropped | evidence |
|-----------|---------|----------|
| `story_structure` | `hook_to_avg_energy_ratio` | **POISON**: −8.9e5 … +1.07e6 in real batch — a feature able to break the model / dominate importance. Other `hook_*` are sane, kept. |
| `story_structure` | `topic_shift*` | present=0, curve NaN (dead) |
| `core_optical_flow`, `optical_flow` | `bg_ratio` | ≡ 0.400 on all 26 real runs (constant by design) |
| `core_clip` | `*_text_embeddings__*` (42) | fixed zero-shot **prompt** vectors — identical for every video → constant, pure dead weight (image `frame_embeddings` kept) |
| `loudness_extractor` | `*lufs*` | 100% NaN (pyloudnorm not installed); dBFS/rms kept |
| `cut_detection` | `deep_valid*`, `deep_cosine*`, `jump_cut*` | deep channel inactive — all-False / 100% NaN / ≡0 |
| `core_face_landmarks` | `*_landmarks_raw__*` | debug raw-point tensors, 81–85% NaN — need pooling not raw coords |
| `video_pacing` | `shot_length_histogram`, `tempo_entropy`, `pace_curve_*`, … | all-NaN in batch by **config-drift** (variant A). **Config-recoverable, not design-dead** → re-add after Agent A re-runs variant B. |

Deliberately **NOT** dropped: the ~166 columns that are merely constant on the
current 6-video table (per strategy §7 — small-N artifact, not design-dead). Those
are left for **importance-based pruning once the corpus scales**, not guessed away.

## REAL targets are now available — `pre_final_data` (2026-07-19, UNBLOCKED)

The "target blocker" below described the *live* Fetcher collection
(`dataset_100k_monthly_shards`), which still only has `snapshot_0`. But there is a
**separate, already-matured** dataset with real multi-horizon labels:
**`Ilialebedev/pre_final_data/main_ready/data_00.json … data_21.json`** (22 shards,
~390 MB each, ~5 000 videos/shard → ~100 k corpus). Top level is
`{video_id: {time_interval, metadata, snapshot_0..3, _enriched, ...}}`; metric fields
(`viewCount`, `likeCount`, … as **strings**) live inside each `snapshot_N`. Measured
snapshot spacing ≈ 7 days → index 1→7d, 2→14d, 3→21d. ~81.5 % of videos carry all 4
snapshots.

**All 6 local content-feature videos are present in this dataset** (their IDs sort to
the top), so both real targets AND real `snapshot_0` inputs now join onto the NPZ
feature table too.

Two real builders:
- `add_targets.py --prefinal <shards>` — primary real-target source (streaming
  `ijson`, string→float parse, edge-case logging). Joins onto any feature table by
  `video_id`. On the 6 content-feature videos: 26/26 rows get real targets + real
  snapshot_0.
- `build_real_from_prefinal.py` — builds a **meaningful** real training table
  directly from pre_final_data using leakage-safe snapshot_0 + metadata features
  (see file header) + real targets. One shard → ~4 950 labelled videos / ~4 200
  channels / 21 features. This is the first table that yields **real** metrics.

Data-integrity (measured on data_00, matches owner's sample):
- non-numeric `viewCount`: 8 / 4 958 (0.16 %) → masked, counted.
- decreasing views between snapshots: 278 / 4 958 (5.6 %) → **logged, not dropped**;
  `y = log1p(max(Δ,0))` already floors negative deltas at 0.

`train_baseline.py` fix required by real data: horizons 14d/21d now respect their
own `mask_14d`/`mask_21d` (real videos can miss an individual follow-up snapshot, so
those targets are NaN for some rows — the old code assumed 14/21d always present and
sklearn choked on NaN `y`). Backward-compatible: smoke masks are all 1.0.

## The (live-collection) target blocker — historical note

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

**Second, separately-verified gap (2026-07-19 re-check):** not only the follow-up
targets but also the **`snapshot_0` numeric inputs** (`views_0`, `likes_0`,
`comments_0`, `channel_subscribers_0`, `channel_total_views_0`,
`channel_total_videos_0`) are absent for the 6 local test videos — they too come
only from the HF snapshot record, and there is **no** viewCount/likeCount/channel-
stats file anywhere in local `storage/` for these ad-hoc DataProcessor test runs
(grep-verified). So the real table currently carries **content features only, zero
snapshot_0 signal**, all six fields NaN (handled natively by boosting). This
matters for interpretation: `views_0` / `channel_subscribers_0` are normally the
dominant baseline predictors, so any future real metric on this exact 6-video set
would understate a properly-fed model. Unlike follow-ups, snapshot_0 is *not*
calendar-blocked — it just needs the corpus videos (which live in
`Ilialebedev/dataset_100k_monthly_shards`, not these test IDs) to flow through
Agent A's pipeline into `result_store` with their snapshot records joinable by
`--snapshots`.

## Leakage audit (mandatory gate)

`audit_and_report.py` asserts, before any metric is trusted:
1. no `target_*`/`mask_*` column entered the feature matrix (exact-set check);
2. no feature name carries future-snapshot / post-publication provenance;
3. every feature traces to an allowed source (v0 component / snapshot_0 /
   temporal) — 0 unmapped.
Result on the cleaned smoke run (exp_0002): **passed** (862 features, 0 leaked, 0
forbidden, 0 unmapped). By construction the whole v0 set is leakage-safe: features
come only from video content + `snapshot_0` (state at analysis time), never from a
future snapshot. High smoke Spearman (0.90–0.95, up from exp_0001's 0.67 simply
because the matrix now carries less dead-column noise) is explained by the
synthetic target design — the permutation-importance top features are exactly the
synthetic drivers — i.e. audit + importance correctly recover the planted signal,
no target proxy leaked. **Not** a real-quality claim.

## Known gaps / next (see `Models/state/training_tasks.json`)
- **Real ceiling is 6 videos right now.** `result_store/youtube/` has 17 dirs but
  only 6 are fully-processed YouTube videos that yield features; `dQw4w9WgXcQ` is a
  dead run (0 ok components, correctly skipped), and the `ar_*` dirs have no
  run-level `manifest.json`. More real videos depend on Agent A (TrendFlow Bot)
  full DP runs — not something Agent B can unblock.
- `channel_id` currently falls back to `video_id` (no channel metadata in local
  manifests) → hybrid split degrades toward per-video on the real table. Real
  `channel_id` should come from HF metadata / YouTube API (enrichment C3 upgrade).
- catboost/lightgbm not installed (no Python 3.14 wheels) → used sklearn HistGB
  (`max_iter=500`, ~2 min/head on 862 feats). Native cat/NaN handling + speed will
  improve once a py3.11 venv or wheels land.
- Permutation importance over 862 features on the tiny test split is slow
  (minutes) with sklearn on py3.14 — run it out-of-band, not inside the 2-min loop.
- Scale to the 150–250 video corpus (`AUTOMATED_TEST_CORPUS_PROTOCOL.md`) once
  TrendFlow Bot's full DP runs populate `result_store`.
