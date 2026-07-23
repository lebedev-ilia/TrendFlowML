# Batch Report — DataProcessor corpus run

Generated: 2026-07-22T23:54:24.083666

**Videos complete:** 500 | **partial/failed:** 0
**GPU mem peak (max over videos):** 4004.0 MiB | **GPU util max:** 100.0%
**Per-video wall time:** p50=328.5s p95=470.1s
**NPZ per video:** p50=8 (min 4, max 8)

## Per-component (over complete+partial videos)

| Component | OK | Fail | wall p50/p95 s | CPU% p50/p95 | RSS MB p95 | GPU util p95 | GPU mem MiB p95 |
|---|---|---|---|---|---|---|---|
| download | 500 | 0 | 5.55/8.46 | None/None | None | None | None |
| segmenter | 500 | 0 | 11.5/65.96 | 339/637 | 936.15 | 100.0 | 2891.0 |
| core_clip | 500 | 0 | 50.26/65.41 | 34/41 | 1218.31 | 100.0 | 3262.0 |
| core_depth_midas | 500 | 0 | 46.18/61.89 | 47/54 | 1583.25 | 100.0 | 3262.0 |
| core_optical_flow | 500 | 0 | 70.73/92.43 | 49/53 | 1380.6 | 100.0 | 3262.0 |
| cut_detection | 499 | 1 | 58.11/82.8 | 105/134 | 1422.85 | 100.0 | 3262.0 |
| scene_classification | 499 | 1 | 60.64/87.29 | 130/186 | 1943.17 | 100.0 | 4004.0 |
| video_pacing | 499 | 1 | 17.64/30.97 | 74/92 | 764.14 | 97.0 | 3262.0 |
| uniqueness | 500 | 0 | 2.34/2.94 | 97/117 | 35.89 | 16.0 | 2891.0 |

## Files

- `per_video_component.csv` — full per-(video,component) metrics (time, CPU user/sys/%, RSS, faults, ctx-sw, per-component GPU util/mem)
- `per_video.csv` — per-video summary (total wall, GPU peak/util/mem, NPZ count, status)
- `batch_report.json` — machine-readable aggregates for TF Agent M
- raw: each `corpus_out/<id>/metrics.jsonl`, `gpu_samples.csv`, `.time_*` (full /usr/bin/time -v)