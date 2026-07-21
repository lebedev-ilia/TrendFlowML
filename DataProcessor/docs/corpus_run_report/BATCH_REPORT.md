# Batch Report — DataProcessor corpus run

Generated: 2026-07-21T00:49:52.487576

**Videos complete:** 300 | **partial/failed:** 0
**GPU mem peak (max over videos):** 3575.0 MiB | **GPU util max:** 95.0%
**Per-video wall time:** p50=430.8s p95=535.7s
**NPZ per video:** p50=8 (min 8, max 8)

## Per-component (over complete+partial videos)

| Component | OK | Fail | wall p50/p95 s | CPU% p50/p95 | RSS MB p95 | GPU util p95 | GPU mem MiB p95 |
|---|---|---|---|---|---|---|---|
| download | 300 | 0 | 5.34/9.14 | None/None | None | None | None |
| segmenter | 300 | 0 | 8.97/38.02 | 181/377 | 907.95 | 0.0 | 2427.0 |
| core_clip | 300 | 0 | 62.94/76.0 | 19/24 | 1407.43 | 21.0 | 2427.0 |
| core_depth_midas | 300 | 0 | 61.16/77.47 | 40/54 | 1955.98 | 44.0 | 2513.0 |
| core_optical_flow | 300 | 0 | 62.77/78.7 | 32/40 | 1525.57 | 55.0 | 2513.0 |
| cut_detection | 300 | 0 | 104.59/128.77 | 31/43 | 2091.46 | 0.0 | 2513.0 |
| scene_classification | 299 | 1 | 101.03/114.49 | 229/264 | 1590.6 | 0.0 | 2513.0 |
| video_pacing | 300 | 0 | 20.24/27.92 | 40/55 | 764.61 | 0.0 | 2427.0 |
| uniqueness | 300 | 0 | 2.25/3.06 | 59/72 | 35.98 | 0.0 | 2427.0 |

## Files

- `per_video_component.csv` — full per-(video,component) metrics (time, CPU user/sys/%, RSS, faults, ctx-sw, per-component GPU util/mem)
- `per_video.csv` — per-video summary (total wall, GPU peak/util/mem, NPZ count, status)
- `batch_report.json` — machine-readable aggregates for TF Agent M
- raw: each `corpus_out/<id>/metrics.jsonl`, `gpu_samples.csv`, `.time_*` (full /usr/bin/time -v)