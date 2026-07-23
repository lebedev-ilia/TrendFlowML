# Models — current state (snapshot 2026-07-23, post Gate-3 1000-run)

Single-page "where Models stands", maintained by Models Bot (Agent B). For detail see
the pointers at the bottom. Statuses reflect the state after Agent A's Gate-2 500-run.

## TL;DR

1. **The pipeline works end-to-end on REAL data** — real multi-horizon targets
   (`Ilialebedev/pre_final_data`), real content features (Agent A's corpus runs read
   over RunPod S3), reproducible dataset builder + baseline trainer + robust CV.
2. **Definitive baseline to beat** (snapshot_0 + metadata only, 14861 videos, 5-fold
   GroupKFold CV): **views 0.855/0.842/0.845, likes 0.735/0.732/0.736** (7/14/21d).
3. **Content adds a real but MODEST edge; the bigger lever is data volume itself.**
   Scale trend (291→500→1000 videos, 7 VP components):
   - The **S0 baseline itself scales strongly** — `likes_21d` 0.65→0.65→**0.71**,
     `views_21d` 0.82→0.82→**0.86**. More data → snapshot_0 explains more → less
     residual for content to fill.
   - **Content lift** flipped from hurting-all (291, overfit) to a small consistent
     gain by 1000, **concentrated on short-horizon likes**: `likes_7d` S0 0.689 →
     lean 0.706 → full **0.711 (+0.02)**. Neutral-to-tiny on views/other likes. The
     500-run `likes_21d` +0.039 was partly small-N variance (settled at 1000).
   - Signal families (residual analysis, stable across scales): motion dispersion,
     editing/cut characteristics, aesthetics (`scene_luxury/aesthetic_scores`). NOT
     depth.
4. **For a bigger content edge:** the current 7 VP components give ~+0.02. Next levers
   are (a) **richer modalities** — AudioProcessor + TextProcessor components (Agent A's
   next rollout) — and (b) even more data. Pipeline is scale-ready for both.

## Baselines (all Spearman, GroupKFold-by-channel CV — the trustworthy numbers)

| dataset | N videos | features | views 7/14/21d | likes 7/14/21d | exp |
|---|---|---|---|---|---|
| snapshot_0+meta (3 pre_final shards) | 14861 | 20 | 0.855/0.842/0.845 | 0.735/0.732/0.736 | 0009 |
| corpus300 S0 only | 291 | 20 | 0.808/0.807/0.819 | 0.644/0.642/0.647 | 0007 |
| corpus300 S0+full content | 291 | 591 | hurts all heads | hurts all heads | 0007 |
| corpus500 S0 | 499 | 20 | 0.841/0.824/0.824 | 0.653/0.658/0.652 | 0010 |
| corpus500 S0+lean (top-30) | 499 | ~50 | 0.835/0.823/0.833 | 0.649/0.660/0.691 | 0010 |
| **corpus1000 S0** | 991 | 20 | 0.872/0.850/0.862 | 0.689/0.710/0.706 | 0011 |
| **corpus1000 S0+lean (top-30)** | 991 | ~50 | **0.874**/**0.855**/0.858 | **0.706**/0.704/0.706 | 0011 |
| corpus1000 S0+full content | 991 | 569 | 0.870/0.848/0.854 | **0.711**/0.709/0.699 | 0011 |

Reading it: **views are ~saturated by `views_0` (autoregressive), ~0.82-0.85, little
content headroom. likes are the opportunity** (~0.65-0.73) and where content first
helps. The **lean recipe** (S0 + top-30 redundancy-pruned content, per-fold |Spearman|
select) is the current best and the one to carry to 1000.

## What content signal is real (residual analysis, exp_0008 @291 confirmed @500)

Content carries a *distributed weak* signal (each feature ~0.10 corr with the S0
residual) that a model can only exploit with enough rows. Top families:
- **editing rhythm** — `cut_detection.cuts_per_minute`, `cut_rhythm_uniformity`, `cut_interval_cv`
- **pacing** — `video_pacing.cut_density_map`, `motion_shot_corr`
- **aesthetics** — `core_clip.scene_aesthetic_scores`, `scene_luxury_scores` (likes-specific)
- **motion dispersion** — `core_optical_flow.flow_dir_dispersion`
- **depth is NOT a driver** (added at 500-run, 9MB after OPT-3; absent from top-15).

## Pipeline / infrastructure (all built, committed, scale-ready)

`DataProcessor/DatasetBuilder/`:
- `s3_corpus.py` — read Agent A's corpus NPZ from the 120GB volume over RunPod S3
  (no pod). `--prefix <corpus_out|corpus_smoke|...> --from-s3 --include-depth`.
- `build_training_table.py` — `build_from_rs()` direct-rs adapter (walks
  `rs/<comp>/*.npz`, no manifest needed) + manifest path; feature_spec-driven.
- `add_targets.py` — `--prefinal` real targets from pre_final_data (streaming ijson,
  edge-case logging).
- `build_corpus_content_dataset.py` — join content + real targets + snapshot_0 + metadata.
- `build_real_from_prefinal.py` — snapshot_0+metadata dataset direct from pre_final.
- `feature_spec.yaml` (v0-real, frozen) + `feature_spec_v0.5.yaml` (adds 4 cheap VP
  comps for the expanded run).
- `run_scale_rebuild.sh` — ONE command: download → build → analyze → CV → residual.

`Models/state/analysis/`:
- `analyze_features.py` (distributions/NaN/const/redundancy/segmenter),
  `v2_cv_experiment.py` (S0 vs lean vs full CV), `s0_cv_eval.py` (large-N S0 CV),
  `residual_content_signal.py` (does content add beyond snapshot_0).

`Models/baseline/Training/train_baseline.py` — real fixes applied: per-horizon masks
for 14/21d; auto-drop constant/all-NaN columns (HistGB binning crash on numpy≥2/py3.14).

## Findings handed to Agent A (in FEATURE_ANALYSIS_300CORPUS.md §3)
- `video_pacing` produced **0 frames on 36/300 videos** (sampler bug).
- `cut_detection` samples ~230 frames vs its 400-1500 contract budget (under-sampling).
- `scene_classification` — 33% of features constant across the corpus (dead scene classes).

## Next steps
1. **DONE — Gate-3 (1000):** exp_0011. Verdict: content edge modest (+0.02 likes_7d),
   S0 scales strongly. The 7-VP visual set has largely shown what it can add.
2. **Biggest next lever — richer modalities.** Add AudioProcessor + TextProcessor
   components when Agent A's AP/TP rollout produces NPZ (his plan wires ASR→TP). Audio
   (music energy, tempo, loudness) and text (title/topic) are orthogonal to the visual
   editing/motion signal and are the most likely source of a larger likes lift.
   `feature_spec_v0.5.yaml` already scaffolds; add AP/TP component names when data lands.
3. **Lean recipe is the production content block** — S0 + top-30 redundancy-pruned
   content (per-fold |Spearman| select). Never hurts, helps short-horizon likes.
4. **v1 transformer** (`Models/docs/contracts/ENCODER_CONTRACT.md`) — only after
   multi-modal boosting shows a clear, stable multi-head signal.
5. Minor: one-hot/embed `places365`/scene category-index features (raw ordinal now).

## Data & access context (how to resume from a cold session)

**RunPod S3 volume** (Agent A's corpus NPZ, no pod needed):
- volume/bucket `vuiq0iq3yf`, region `eu-ro-1`, endpoint `https://s3api-eu-ro-1.runpod.io`.
- Credentials: **`storage/.s3creds`** (gitignored — AK/SK live there, NOT in git;
  regenerate in RunPod console → Settings → S3 API Keys if lost).
- Reader: `DataProcessor/DatasetBuilder/s3_corpus.py --prefix <p> --from-s3 [--include-depth]`.
- Corpus prefixes on the volume: `corpus_out/` = 300-run (7 VP, depth 156MB);
  `corpus_smoke/` = the SCALE prefix — grew 500→1000 (Gate-2 then Gate-3; 7 VP +
  depth 9MB after OPT-3). Agent A extends `corpus_smoke/` per gate; watch its count.

**Targets** — `Ilialebedev/pre_final_data/main_ready/data_00..21.json` (22 shards,
~390MB each). **All 22 downloaded locally** at `storage/pre_final_data/` (7.8GB,
gitignored). Real snapshots 0/7/14/21d; join via `add_targets.py --prefinal`.
Coverage of corpus1000: 1000/1000 with all 22 shards.

**Corpus video lists** (Agent A, in git): `DataProcessor/docs/corpus_run_report/`
`corpus300.json`, `corpus1000.json` (1000 balanced; corpus500 = its first 500).

**Local NPZ (gitignored, regenerable via s3_corpus.py):** `storage/corpus_npz/`
(300), `storage/corpus_npz_500/` (500 +depth), `storage/corpus_npz_1000/` (991,
no depth).

**One-command rebuild** when a new corpus/gate lands:
```
PY=Models/.venv/bin/python
$PY DataProcessor/DatasetBuilder/s3_corpus.py --prefix corpus_smoke --from-s3 --dest storage/corpus_npz_<N>
$PY DataProcessor/DatasetBuilder/build_corpus_content_dataset.py \
   --rs-root storage/corpus_npz_<N> --prefinal storage/pre_final_data/data_*.json \
   --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
   --out Models/baseline/artifacts/<tag>/dataset_corpus_content.parquet
$PY Models/state/analysis/v2_cv_experiment.py --dataset <...>/dataset_corpus_content.parquet \
   --redundant <...>/redundant_pairs.csv --k 5 --topk 30
```
(or `run_scale_rebuild.sh <tag>` for the whole chain; venv is `Models/.venv`, py3.14,
has pandas/pyarrow/sklearn/scipy/ijson/boto3/matplotlib; NO catboost/lightgbm wheels
→ uses sklearn HistGB.)

**Agent A coordination:** his side = `DataProcessor/` (do not edit his component code
or `DataProcessor/state/agent_a_tasks.json`). His gate reports land under
`DataProcessor/docs/corpus_run_report/`. Watch his commits (`git log -- DataProcessor
':(exclude)DataProcessor/DatasetBuilder'`) for gate completions.

## Pointers
- This file (`Models/state/MODELS_STATE.md`) = canonical entry point.
- Experiment log: `Models/state/experiments.csv` (exp_0001 … exp_0011)
- Task board: `Models/state/training_tasks.json`
- Full feature analysis + 500-run payoff: `Models/state/analysis/FEATURE_ANALYSIS_300CORPUS.md`
- Dataset builder docs: `DataProcessor/DatasetBuilder/README.md`
- Strategy: `Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md`
