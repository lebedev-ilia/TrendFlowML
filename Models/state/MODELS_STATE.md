# Models — current state (snapshot 2026-07-23)

Single-page "where Models stands", maintained by Models Bot (Agent B). For detail see
the pointers at the bottom. Statuses reflect the state after Agent A's Gate-2 500-run.

## TL;DR

1. **The pipeline works end-to-end on REAL data** — real multi-horizon targets
   (`Ilialebedev/pre_final_data`), real content features (Agent A's corpus runs read
   over RunPod S3), reproducible dataset builder + baseline trainer + robust CV.
2. **Definitive baseline to beat** (snapshot_0 + metadata only, 14861 videos, 5-fold
   GroupKFold CV): **views 0.855/0.842/0.845, likes 0.735/0.732/0.736** (7/14/21d).
3. **Content features now PAY OFF at scale.** At 291 videos they overfit and hurt all
   heads; at **495 videos they start HELPING likes** (lean `likes_21d` 0.652→0.691,
   +0.039 Spearman), ~neutral on views. Driven by editing rhythm / pacing / aesthetics
   / motion — not depth. This is the first evidence the content modality adds real
   signal beyond metadata.
4. **Blocker for more gains:** corpus size. Everything is scale-ready — one command
   rebuilds+re-evaluates when Agent A's Gate-3 (1000) lands.

## Baselines (all Spearman, GroupKFold-by-channel CV — the trustworthy numbers)

| dataset | N videos | features | views 7/14/21d | likes 7/14/21d | exp |
|---|---|---|---|---|---|
| snapshot_0+meta (3 pre_final shards) | 14861 | 20 | 0.855/0.842/0.845 | 0.735/0.732/0.736 | 0009 |
| corpus300 S0 only | 291 | 20 | 0.808/0.807/0.819 | 0.644/0.642/0.647 | 0007 |
| corpus300 S0+full content | 291 | 591 | hurts all heads | hurts all heads | 0007 |
| **corpus500 S0** | 499 | 20 | 0.841/0.824/0.824 | 0.653/0.658/0.652 | 0010 |
| **corpus500 S0+lean (top-30)** | 499 | ~50 | 0.835/0.823/**0.833** | 0.649/0.660/**0.691** | 0010 |
| corpus500 S0+full content | 499 | 631 | 0.843/0.797/0.810 | 0.665/0.669/0.672 | 0010 |

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
1. **Gate-3 (1000):** when Agent A's 1000-run lands — `s3_corpus.py --prefix <new>
   --from-s3` → `run_scale_rebuild.sh`. Expect likes lift to grow, maybe a views lift.
   Watch target coverage (corpus1000 may add scale-fillers not in local pre_final
   shards → download more shards if <95%).
2. **Lean recipe productization:** lock the top-30 pruned content set as the v1 content
   block on top of snapshot_0.
3. **v1 transformer** (`Models/docs/contracts/ENCODER_CONTRACT.md`) — only after boosting
   gives a stable multi-head signal at ≥1000 videos.
4. Optional: isolate depth contribution (residual says minimal); one-hot/embed the
   `places365`/scene category-index features (currently raw ordinal IDs).

## Pointers
- Experiment log: `Models/state/experiments.csv` (exp_0001 … exp_0010)
- Task board: `Models/state/training_tasks.json`
- Full feature analysis + 500-run payoff: `Models/state/analysis/FEATURE_ANALYSIS_300CORPUS.md`
- Dataset builder docs: `DataProcessor/DatasetBuilder/README.md`
- Strategy: `Models/docs/MULTI_AGENT_TRAINING_STRATEGY.md`
