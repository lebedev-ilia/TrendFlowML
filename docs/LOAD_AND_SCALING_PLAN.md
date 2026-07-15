# Load & scaling plan — 200k videos (task 5 / P10)

How to size, run and observe the pipeline for the 200k-video batch on multi-node
Kubernetes. Grounded in the existing capacity tooling and metrics — fill the
parametric numbers with **measured** per-video GPU time, don't trust the examples
as SLA.

## 1. Throughput model

The unit of work is one video. The binding resource is **GPU time per video**.

```
videos_per_hour_per_gpu = 3600 / cuda_seconds_per_video      # p50 and p95
cluster_videos_per_hour = videos_per_hour_per_gpu * num_gpus * gpu_utilization
total_gpu_hours(N)      = N * cuda_seconds_per_video / 3600
time_to_finish(N)       = total_gpu_hours(N) / (num_gpus * gpu_utilization)
```

This is exactly what `DataProcessor/scripts/capacity_report.py` computes from real
`dp_results` manifests:

```bash
python DataProcessor/scripts/capacity_report.py \
  --results-root DataProcessor/dp_results --gpus 32
# -> p50/p90/p95 cuda ms per video, gpu_hours_total,
#    videos_per_hour_per_gpu_p50/p95, videos_per_hour_cluster_p50/p95
```

**Action:** run a representative sample (e.g. 200–500 videos) through the real
profile, then read p50/p95 cuda ms from this report. Use those, not the table below.

## 2. Sizing for 200k (parametric)

Days to finish 200k videos at `gpu_utilization = 0.85`:

| min/video (GPU) | GPU-hours | 8 GPU | 16 GPU | 32 GPU | 64 GPU |
|---|---|---|---|---|---|
| 5  | 16 667  | 102 d | 51 d  | 25 d  | 13 d |
| 10 | 33 333  | 204 d | 102 d | 51 d  | 25 d |
| 20 | 66 667  | 409 d | 204 d | 102 d | 51 d |
| 40 | 133 333 | 817 d | 409 d | 204 d | 102 d |

Per-GPU throughput: `60 / min_per_video` videos/h (5→12, 10→6, 20→3, 40→1.5).

Takeaway: per-video GPU time dominates. Two levers before buying GPUs:
1. **Cut per-video GPU time** — profile with `dataprocessor_component_stage_seconds`,
   move heavy components to Triton (shared GPU, dynamic batching), trim/segment
   sampling (Segmenter), drop low-value extractors for the batch profile.
2. **Raise utilization** — keep the queue full (backpressure below) so GPUs never idle.

## 3. Queue, concurrency & backpressure

- **Ingestion rate** must not outrun GPU capacity. Cap in-flight work so the
  backlog is bounded and memory is safe:
  - `MAX_CONCURRENT_RUNS` per worker (default 2) — concurrency hides I/O, it does
    **not** multiply GPU compute; keep low for GPU-bound profiles.
  - `SUBPROCESS_MEMORY_LIMIT_MB` (default 8000) — OOM guard per run.
  - Namespace `ResourceQuota` caps total `nvidia.com/gpu` (see `k8s/governance.yaml`).
- **Autoscaling** of `dataprocessor-worker`:
  - default: CPU HPA (`k8s/dataprocessor/hpa.yaml`).
  - preferred at scale: **KEDA** on Redis Streams backlog
    (`k8s/dataprocessor/keda-scaledobject.yaml`), `pendingEntriesCount` target per
    replica. `maxReplicaCount` ≤ GPUs available / quota.
- **DynamicBatch** chooses `batch_size` from live resource costs and backs off on
  OOM — keep it as the level-1 scheduler in front of heavy components.

## 4. SLOs (define, then enforce via alerts)

| SLO | Target (example) | Metric source |
|---|---|---|
| Run success rate | ≥ 99% of runs complete without crash | `dataprocessor_crashed_runs_total`, `failures_total` |
| Batch completion | ≥ 95% of a daily batch done in < 24 h | `processing_seconds` + queue depth |
| Queue wait p95 | < 30 min under steady load | `dataprocessor_queue_wait_seconds` |
| API availability | 99.9% (RED) | `backend_http_requests_total`, latency histogram |
| GPU utilization | 75–90% during batch | host/GPU metrics, `active_runs` vs replicas |

## 5. Alerts (thresholds)

- `dataprocessor_queue_length` high (>100) **and** `active_runs` at limit for >15 min → add GPU workers.
- `dataprocessor_queue_wait_seconds` p95 rising while GPU idle → queue/lock bug, not capacity.
- `rate(dataprocessor_failures_total[5m]) > 0` sustained on one `component/error_type` → content/pipeline bug.
- `dataprocessor_crashed_runs_total` increasing → OOM/kill; check `SUBPROCESS_MEMORY_LIMIT_MB`, node memory.
- `backend_http_requests_total{status=~"5.."}` rate up, or latency p95 breach → API regression.
- `backend_celery_queue_length` growing unbounded → ingestion outrunning processing; throttle intake.
- Disk/object-store low (`dataprocessor_host_disk_free_bytes`, MinIO) → can't write artifacts.

## 6. Scaling runbook

1. **Measure first:** run 200–500 videos with the production profile →
   `capacity_report.py --gpus <N>` → record p50/p95 cuda ms and gpu_hours.
2. **Pick GPU count** from §2 for the deadline; set `ResourceQuota requests.nvidia.com/gpu`
   and KEDA `maxReplicaCount` accordingly.
3. **Warm the queue:** enqueue a bounded backlog (don't dump all 200k — keep
   `backend_celery_queue_length` bounded; feed in waves).
4. **Watch the green/yellow/red dashboard:** queue depth, queue wait p95,
   active_runs vs replicas, failures/crashes, GPU util, disk.
5. **Scale out** when queue stays deep with GPUs saturated and util ≥85%; KEDA
   does this automatically up to `maxReplicaCount` — raise the cap + add GPU nodes.
6. **If util is low but queue is deep:** bottleneck is upstream (Fetcher/S3/DB or
   lock), not GPU — investigate before adding GPUs.
7. **Failures:** stuck Stream messages are re-claimed via `xpending` (at-least-once);
   for poison messages add a max-attempts → dead-letter stream (P7-code, pending).
8. **Cost:** `gpu_hours_total` from the report × GPU $/h ≈ batch cost; compare
   profiles (full vs trimmed) before committing the full run.

## 7. Pre-flight checklist for the 200k run

- [ ] Measured p50/p95 per-video GPU time recorded (capacity_report).
- [ ] GPU count + `ResourceQuota` + KEDA `maxReplicaCount` set for the deadline.
- [ ] Result storage on **S3/MinIO** (not fs), lifecycle/TTL for temp set.
- [ ] DB migrated (`backend-migrate` Job), Postgres sized for run/index rows.
- [ ] Dashboards + alerts (§4–5) live; on-call knows the runbook.
- [ ] Retry/DLQ policy decided (or accept at-least-once + manual replay for now).
- [ ] A small canary batch (e.g. 1k) completed clean end-to-end before the full 200k.
