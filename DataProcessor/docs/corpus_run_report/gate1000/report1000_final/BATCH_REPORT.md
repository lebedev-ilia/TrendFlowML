# Batch Report — DataProcessor corpus run

Generated: 2026-07-24T11:32:12.448856

**Videos complete:** 1000 | **partial/failed:** 10
**GPU mem peak (max over videos):** 4351.0 MiB | **GPU util max:** 100.0%
**Per-video wall time:** p50=303.2s p95=498.2s
**NPZ per video:** p50=8 (min 0, max 10)

## Per-component (over complete+partial videos)

| Component | OK | Fail | wall p50/p95 s | CPU% p50/p95 | RSS MB p95 | GPU util p95 | GPU mem MiB p95 |
|---|---|---|---|---|---|---|---|
| download | 1000 | 0 | 5.44/9.26 | None/None | None | None | None |
| segmenter | 992 | 4 | 11.56/106.58 | 188/572 | 934.5 | 100.0 | 2891.0 |
| core_clip | 977 | 9 | 48.46/77.29 | 27/40 | 1218.31 | 100.0 | 3203.0 |
| core_depth_midas | 971 | 3 | 42.44/66.57 | 37/53 | 1582.44 | 100.0 | 3262.0 |
| core_optical_flow | 962 | 4 | 62.99/95.22 | 39/53 | 1378.86 | 100.0 | 3262.0 |
| cut_detection | 967 | 7 | 51.81/84.96 | 58/130 | 1476.19 | 100.0 | 3262.0 |
| scene_classification | 973 | 13 | 51.66/89.08 | 95/171 | 1938.5 | 100.0 | 3633.0 |
| video_pacing | 982 | 10 | 16.49/30.98 | 61/89 | 778.48 | 100.0 | 3262.0 |
| uniqueness | 989 | 10 | 2.26/3.65 | 74/113 | 35.48 | 51.0 | 2891.0 |

## Files

- `per_video_component.csv` — full per-(video,component) metrics (time, CPU user/sys/%, RSS, faults, ctx-sw, per-component GPU util/mem)
- `per_video.csv` — per-video summary (total wall, GPU peak/util/mem, NPZ count, status)
- `batch_report.json` — machine-readable aggregates for TF Agent M
- raw: each `corpus_out/<id>/metrics.jsonl`, `gpu_samples.csv`, `.time_*` (full /usr/bin/time -v)