# Feature analysis — Agent A's 300-video corpus run (v0-real VP features)

**Author:** Models Bot (Agent B) · **Date:** 2026-07-21 · Deliverable for
`AGENT_B_WORKPLAN_2026-07-21.md` Phase 1. Written so Agent A can act on it without
asking back.

## 0. What this is / data path

Agent A's first real-corpus run: **300 videos, 7 VisualProcessor components**
(`core_clip`, `core_depth_midas`, `core_optical_flow`, `cut_detection`,
`scene_classification`, `video_pacing`, `uniqueness`), NPZ on the 120GB Network
Volume `vuiq0iq3yf` (EU-RO-1) under `corpus_out/<vid>/rs/<comp>/*.npz`, **no
per-run manifest.json**.

Access without a pod: read over **RunPod S3** (`s3api-eu-ro-1.runpod.io`, bucket =
volume id) — `DataProcessor/DatasetBuilder/s3_corpus.py`. I chose Agent A's option
**(b)**: a **direct-rs adapter** (`build_training_table.build_from_rs` + `--rs-root`)
that walks `rs/<comp>/*.npz` directly — Agent A does NOT need to emit manifests.

**Coverage caveat (my side):** I downloaded the 6 light NPZ families (skipped the
156MB `core_depth_midas/depth.npz` per video — raw depth maps, ~47GB of the 54GB;
pooled depth features are a separate step). 281/300 videos downloaded complete (7
NPZ); ~19 partial (S3 stragglers) → those rows show high per-video NaN that is a
**download artifact, not a component failure**. 291 videos yielded a feature row.
Depth excluded → analysis below covers **6 of 7 components**.

Artifacts: `Models/state/analysis/{feature_stats,const_features,redundant_pairs,segmenter_frames}.csv`
+ PNGs; feature table `Models/baseline/artifacts/v1_corpus_2026-07-21/content_table.parquet`.

## 1. Feature inventory & health (656 content features, 291 videos)

| component | features | constant-across-corpus | mean %NaN |
|---|---|---|---|
| cut_detection | 209 | 14 | 4.5 |
| scene_classification | 181 | **59 (33%)** | 2.8 |
| core_optical_flow | 126 | 6 | 2.4 |
| core_clip | 78 | 1 | 1.0 |
| video_pacing | 42 | 2 | **15.7** |
| uniqueness | 20 | 5 | 1.7 |

- **87 features constant across all 291 videos** (`const_features.csv`) — 13% dead.
  On this large, deliberately-diverse corpus this is a **real drop list**, not the
  small-N artifact we kept-for-later on the earlier 6-video table.
- **0 features >50% NaN** — the 6 components are healthy on real corpus (contrast
  with the old 6-video table where config-drift zeroed whole families). The earlier
  portfolio worries (`video_pacing` all-NaN, etc.) are **largely resolved** on this
  run — but see §3 for `video_pacing`'s 0-frame subset.

### Worst offender: `scene_classification` — 59/181 constant (33%)
A third of scene features never vary across 291 diverse videos. Almost certainly
Places365 scene categories that never fire (rare indoor/outdoor classes) → their
per-scene probability aggregates are ≡0. **For Agent A:** consider collapsing the
365-way head to a top-K active-scene set, or emitting only scenes seen in a
calibration set. **For me:** these 59 are auto-dropped at train time (§4).

## 2. Redundancy (feature-feature |Spearman| ≥ 0.98)

**1352 redundant pairs** (533 within-component, **819 cross-component**). Dominant
cross-component clusters:

| pair | redundant feature pairs |
|---|---|
| core_optical_flow ↔ cut_detection | **358** |
| core_optical_flow ↔ scene_classification | 160 |
| cut_detection ↔ scene_classification | 120 |
| core_clip ↔ core_optical_flow | 100 |
| core_clip ↔ cut_detection | 75 |

Interpretation: `core_optical_flow`, `cut_detection` and `core_clip`'s
consecutive-frame cosine all measure the **same underlying thing** — inter-frame
change/motion/transitions — so their aggregates (mean/std/p50/p90 of per-frame
motion) are near-duplicates. This is expected, not a bug, but it means the effective
content dimensionality is far below 656. **Action (my side, v2):** collapse these to
a handful of representative motion features via importance + a redundancy prune;
don't feed all 3 families raw. This is the biggest lever for the p≫n problem in §4.

## 3. Segmenter analysis (sampled frames per component)

From NPZ `frame_indices` (`segmenter_frames.csv`), frames sampled per component:

| component | median | min | max |
|---|---|---|---|
| core_clip | 231 | 15 | 323 |
| core_optical_flow | 227 | 15 | 323 |
| cut_detection | 230 | 15 | 323 |
| scene_classification | 232 | 15 | 250 |
| uniqueness | 119 | 15 | 120 |
| video_pacing | 119 | **0** | 120 |

Findings **for Agent A**:
1. **`cut_detection` under-samples vs its contract budget.** `SEGMENTER_CONTRACT.md`
   lists cut_detection at **400–1500** frames; observed median **230**, max **323** —
   well below the stated floor. Either the contract budget isn't being applied to
   cut_detection on this profile, or the sampler is capping earlier. Worth checking
   which, because cut detection quality scales with temporal density.
2. **`video_pacing` got 0 frames on 36/291 videos** (`-0g23pjxC7Y`, `-5uh-3Yg8Wc`,
   `-63tdE9tJD0`, … full list in analysis output). The component still produced an
   NPZ but with an empty frame set → its features are degenerate for those videos.
   This is the source of video_pacing's 15.7% mean NaN. **Concrete bug to look at.**
3. `uniqueness`/`video_pacing` are capped at **120** frames (a different, smaller
   budget than the ~230 of the clip/flow/cut/scene group) — consistent, looks
   intentional, just noting the two-tier budget.
4. **min = 15 frames** on the short end: 36 videos get <20 frames on some component
   (very short videos). Feature aggregates on 15 frames are high-variance — a caveat
   for `vshort` strata quality, not necessarily fixable.

## 4. Ran it through the models — content features + REAL targets

Joined content features with **real** multi-horizon targets + snapshot_0 inputs +
video metadata from `pre_final_data` (all 300 corpus videos have real targets;
`data_00..02` shards cover them). Table:
`Models/baseline/artifacts/v1_corpus_2026-07-21/dataset_corpus_content.parquet`
(291 rows, 288 with targets, 656 content + 23 snapshot0/meta feats, channel-group
split, mask 7d=232 / 14d=288 / 21d=288).

**Clean ablation on the SAME 291 videos & split** (test Spearman):

| head | A) snap0+meta (22 feat) | B) +content (591 feat) |
|---|---|---|
| views_7d | 0.44 | 0.35 |
| views_14d | **0.49** | 0.23 |
| views_21d | **0.40** | 0.27 |
| likes_7d | 0.40 | 0.52 |
| likes_14d | **0.35** | 0.08 |
| likes_21d | 0.37 | 0.27 |

**Adding 656 content features HURTS on 5/6 heads** — classic **p ≫ n overfit**
(591 usable features vs 230 training rows). Content only "won" on `likes_7d`
(n=19, noise). Both A and B sit at 0.35–0.49, **below `exp_0004`'s 0.75–0.78** —
because exp_0004 had **4958** videos vs **291** here, the corpus is deliberately
**stratified/diverse** (view buckets v_low…v_viral, all durations, multi-language),
and the channel-group test set is tiny (n≈30 → high-variance Spearman).

Fix that broke on real data (now committed): `train_baseline.py` (a) drops
constant/all-NaN columns before fit — 88+2 here — otherwise HistGB binning throws
`window shape cannot be larger than input array shape` on numpy≥2/py3.14; (b) 14d/21d
heads respect their masks (real videos miss individual snapshots).

## 5. Stratified quality (owner's "where are features better/worse")

Per-video mean NaN fraction by duration bucket (content features):

| bucket | mean NaN frac | max | n |
|---|---|---|---|
| medium | 0.002 | 0.02 | 73 |
| short | 0.011 | 0.59 | 77 |
| long | 0.061 | 0.90 | 69 |
| vshort | 0.082 | 0.90 | 72 |

- **medium videos are cleanest**; **vshort & long worst**. vshort → too few frames
  (§3.4) makes aggregates unstable; the high `max` (~0.9) rows are dominated by my
  **partial downloads**, so treat the *mean* not the *max* as the real signal.
- "Without people" strata (owner's example) **not computable yet** — needs
  `core_face_landmarks`/detector, which weren't in this 7-component run. If this
  cut matters, Agent A running `core_face_landmarks` on a subsample would unblock a
  proper people/no-people quality comparison.

## 6. Conclusions & recommended priorities

1. **Scale the corpus before content pays off.** The single biggest result: at
   N=291 the content features overfit and *reduce* accuracy. They will only help
   once there are enough videos (thousands) for 500+ features. **Highest-value next
   step is more corpus videos through Agent A's pipeline**, not more feature
   engineering. Until then, the snapshot_0+metadata model (`exp_0004`, 4958 vids,
   Spearman ~0.75) remains the real baseline to beat.
2. **Aggressively reduce content dimensionality (my v2).** 819 cross-component
   redundant pairs + 87 constants ⇒ collapse optical_flow/cut_detection/clip-motion
   to a few representative features; keep top content features by importance. Feed a
   *lean* content set (~30–50) on top of snapshot_0, not 656 raw.
3. **For Agent A (component fixes surfaced here):**
   - `video_pacing` 0-frame on 36 videos — degenerate output, looks like a sampler
     bug (see §3.2 list).
   - `cut_detection` sampling ~230 frames vs 400–1500 contract budget (§3.1).
   - `scene_classification` 33% constant features — prune/collapse the dead scene
     classes (§1).
4. **Depth still unmeasured** — `core_depth_midas/depth.npz` (156MB/video) skipped;
   pooled depth stats (mean/std/percentile of depth maps) are a separate extract I'll
   add when content features earn their place (post-scale).

### Experiment log
`exp_0005` (content+snap0, 291 vids) and `exp_0006` (snap0-only ablation, same 291)
in `Models/state/experiments.csv`.
