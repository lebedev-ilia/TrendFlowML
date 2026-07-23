# Batch Report — DataProcessor corpus run

Generated: 2026-07-23T08:07:55.833252

**Videos complete:** 1000 | **partial/failed:** 0
**GPU mem peak (max over videos):** 4004.0 MiB | **GPU util max:** 100.0%
**Per-video wall time:** p50=306.1s p95=504.6s
**NPZ per video:** p50=8 (min 0, max 9)

## Per-component (over complete+partial videos)

| Component | OK | Fail | wall p50/p95 s | CPU% p50/p95 | RSS MB p95 | GPU util p95 | GPU mem MiB p95 |
|---|---|---|---|---|---|---|---|
| download | 1000 | 0 | 5.58/8.91 | None/None | None | None | None |
| segmenter | 995 | 0 | 12.44/104.28 | 196/572 | 934.5 | 100.0 | 2891.0 |
| core_clip | 983 | 2 | 48.96/74.05 | 27/40 | 1219.53 | 100.0 | 2891.0 |
| core_depth_midas | 972 | 3 | 43.35/66.15 | 36/53 | 1576.36 | 100.0 | 3262.0 |
| core_optical_flow | 972 | 4 | 63.38/98.63 | 38/53 | 1378.7 | 100.0 | 3262.0 |
| cut_detection | 977 | 5 | 56.07/88.51 | 58/130 | 1476.71 | 100.0 | 3262.0 |
| scene_classification | 499 | 497 | 45.09/83.86 | 34/171 | 1938.07 | 100.0 | 3633.0 |
| video_pacing | 995 | 4 | 18.43/32.97 | 55/89 | 774.81 | 100.0 | 3262.0 |
| uniqueness | 986 | 13 | 2.37/3.32 | 62/113 | 35.5 | 61.0 | 2891.0 |

## Files

- `per_video_component.csv` — full per-(video,component) metrics (time, CPU user/sys/%, RSS, faults, ctx-sw, per-component GPU util/mem)
- `per_video.csv` — per-video summary (total wall, GPU peak/util/mem, NPZ count, status)
- `batch_report.json` — machine-readable aggregates for TF Agent M
- raw: each `corpus_out/<id>/metrics.jsonl`, `gpu_samples.csv`, `.time_*` (full /usr/bin/time -v)