# Production architecture — gap map & backlog (task 5.1)

Audit of the current stack against the production requirements in
`PRODUCT_ROADMAP_TO_PRODUCTION.md` (Phase 5). Date: 2026-06-29.
Status legend: ✅ done · 🟡 partial · ❌ missing.

## Summary

The pipeline logic and **observability for DataProcessor are surprisingly mature**
(rich `dataprocessor_*` metrics, Prometheus/Grafana/Jaeger compose, a real
`DynamicBatch` scheduler with CPU/GPU cost model and OOM backoff). The weak spots
are **orchestration glue**: incomplete k8s, no model provisioning into
containers/pods, no single system-wide compose, no backend metrics, and a
lingering `DP_MODELS_ROOT` convention split.

## Area-by-area

### 1. Containerization 🟡
- ✅ Dockerfiles: `backend/`, `Fetcher/`, `DataProcessor/docker/{api,worker}`.
- ✅ DataProcessor compose (api, worker, retention-cleanup, redis, prometheus, grafana, jaeger).
- ❌ **No model provisioning**: `docker/worker/bootstrap.py` checks Redis+MinIO only; worker image does **not** contain the 3.4 GB weights and nothing downloads them at start. Needs `download_models.py` as an init step (volume/PVC, not baked).
- ❌ No single top-level compose for the whole system (4 separate compose files); `bootstrap.sh` runs host processes via E2E scripts, not containers.
- 🟡 Triton runs only via `e2e_triton_docker.sh`; no first-class Triton service in the system compose.

### 2. Orchestration / Kubernetes ❌🟡
- ✅ Present: `fetcher/*` (per-worker deployments + HPAs — good), `backend/deployment.yaml`, `dataprocessor/deployment.yaml`, `infrastructure/postgres.yaml`.
- ❌ Missing (promised in `k8s/README.md`): `infrastructure/{redis,minio,triton}.yaml`, `backend/{service,ingress}.yaml`, `dataprocessor/{service,hpa,gpu-node-selector}.yaml`, `models/`.
- ❌ DP deployment defects: readiness `httpGet :8080/health` but API serves on **:8000**; liveness via `ps aux | grep main.py` (fragile); no Service; no model PVC/initContainer; hardcoded `nvidia-tesla-v100`.
- ❌ No namespace/kustomize/Helm; secrets only as `secrets-example.yaml`; no resource quotas/PDB/NetworkPolicy.

### 3. Parallelism & scale 🟡
- ✅ `DynamicBatch` scheduler: CPU/GPU split, level-1 batching, batch_size from `resource_costs_*`, OOM retry+backoff, state files, file/DB cost providers (Postgres registry).
- ✅ DataProcessor worker: Redis Streams queue, `MAX_CONCURRENT_RUNS`, subprocess isolation + memory limit.
- 🟡 Task state: needs confirmation it's fully in Redis/DB (roadmap warns against in-memory `TaskManager`).
- ❌ No explicit GPU/CPU **worker pools** split in k8s; no dead-letter queue / retry+DLQ policy documented; no backpressure/quota/priority enforcement at the API edge; video-duration limits.

### 4. Data & storage 🟡
- ✅ SoT = `manifest.json` + NPZ; Postgres `core.*` as index; Alembic migrations; DataProcessor retention-cleanup.
- 🟡 Feature materialization layer (Parquet/DuckDB index, Phase 3) not built.
- ❌ No backup/PITR policy; no object-storage lifecycle/TTL config; migrations not run as a k8s Job.

### 5. Observability 🟡
- ✅ DataProcessor: mature metrics (`queue_length`, `queue_wait`, `processing_seconds`, `failures_total`, `active_runs`, `crashed_runs_total`, `component_stage_seconds`, host CPU/mem/disk), Prometheus+Grafana+Jaeger, `METRICS_REFERENCE.md`, alerts.yml.
- ✅ Fetcher monitoring (Grafana dashboard, alerts).
- ❌ **Backend exposes no Prometheus metrics** (no RED for HTTP, no Celery queue depth/age) — roadmap requires it.
- ❌ No unified cross-service dashboard; no OTel tracing wired in code (Jaeger present but unused); logs not confirmed structured-JSON with `correlation_id`/`run_id` across all services.

### 6. Reliability / ops ❌
- 🟡 `backend/docs/OPERATIONS.md` exists; needs 100k-scale runbook + per-alert runbooks.
- ❌ No SLOs/SLA, no load/capacity plan, no chaos/failure drills, no autoscaling validation.

### 7. Cross-cutting cleanup 🟡
- ❌ `DP_MODELS_ROOT` split: we standardized on `DataProcessor/dp_models`, but `DynamicBatch` and `scripts/download_*`/`save_*` still default to `dp_models/bundled_models`. Align everywhere.
- ❌ `k8s/README.md` documents files that don't exist — bring docs and manifests into sync.

## Prioritized backlog

| # | Item | Area | Impact | Effort |
|---|------|------|--------|--------|
| P1 | Model provisioning: wire `download_models.py` into worker bootstrap + k8s initContainer + PVC | 1,2 | High | M |
| P2 | Fix DP k8s deployment (port, probes, Service) + add `dataprocessor/{service,hpa,gpu-node-selector}.yaml` | 2 | High | M |
| P3 | Add missing infra manifests: `redis.yaml`, `minio.yaml`, `triton.yaml` + backend `service/ingress` | 2 | High | M |
| P4 | Backend Prometheus metrics (RED + Celery depth/age) + unified Grafana dashboard | 5 | High | M |
| P5 | Single system-wide `docker-compose.prod.yml` (all services containerized) | 1 | Med | M |
| P6 | Align `DP_MODELS_ROOT` to `DataProcessor/dp_models` across DynamicBatch + scripts | 7 | Med | S |
| P7 | Retry/DLQ + GPU/CPU pool split + queue/quota policy | 3 | High | L |
| P8 | DB migration Job, backup/TTL/lifecycle policy | 4 | Med | M |
| P9 | Structured JSON logs + OTel tracing across services | 5 | Med | L |
| P10 | 100k load plan + SLOs + ops runbooks | 6 | Med | L |

S ≈ <0.5d, M ≈ 0.5–2d, L ≈ multi-day. Suggested first sprint: **P1→P2→P3→P4** (gets a coherent, observable, model-provisioned deployment), with P6 done opportunistically.

## Progress — first sprint done (2026-06-29)

- **P1 ✅** Model provisioning in k8s: `k8s/infrastructure/models.yaml` (RWX `models-pvc` + `model-download` Job running `download_models.py`); DP pods consume weights via `DP_MODELS_ROOT` and mount semantic bases via subPath.
- **P2 ✅** `k8s/dataprocessor/`: split api(CPU)/worker(GPU) `deployment.yaml` (fixed ports/probes, `wait-for-models` init, model mounts) + `service.yaml` + `configmap.yaml` + `hpa.yaml`.
- **P3 ✅** `k8s/infrastructure/{namespace,redis,minio,triton}.yaml`, `k8s/backend/ingress.yaml`, `k8s/secrets-example.yaml`, `k8s/kustomization.yaml` (whole stack via `kubectl apply -k k8s/`); `k8s/README.md` synced. All 28 manifests parse; kustomization references verified.
- **P4 ✅** Backend Prometheus metrics: `backend/app/metrics.py` (RED HTTP + Celery queue depth) wired in `main.py` at `GET /metrics`; scrape annotations on `k8s/backend/deployment.yaml`; `prometheus-client` added to requirements.
- **P6 ✅** `DP_MODELS_ROOT=DataProcessor/dp_models` enforced via env in bootstrap + all k8s manifests; `download_models.py` is path-canonical and root-independent.

## Progress — second sprint (2026-06-29)

- **P8 ✅** `k8s/backend/migrate-job.yaml` (alembic upgrade head, waits for Postgres), `k8s/backend/configmap.yaml` (`backend-config`), `k8s/dataprocessor/retention-cronjob.yaml` (daily CronJob replacing the in-container cron).
- **P7 (k8s) ✅** `k8s/governance.yaml` (2× PriorityClass, ResourceQuota, LimitRange, 3× PodDisruptionBudget); `k8s/dataprocessor/keda-scaledobject.yaml` (queue-based autoscale on `queue:normal`/`queue:high`, optional, KEDA-gated); GPU worker + Triton get `priorityClassName: trendflow-gpu-high`.
- **Bugs fixed while wiring k8s** (were silently broken):
  - backend container listens on **:8080** (Dockerfile) — k8s deployment had :8000. Fixed ports + probes (`/health/live`, `/health/ready`) + Service targetPort.
  - backend reads `TF_BACKEND_DB_DSN` / `TF_BACKEND_REDIS_URL` (Settings `env_prefix=TF_BACKEND_`) — deployment was setting `DATABASE_URL` / `REDIS_URL` (ignored). Fixed.
  - DataProcessor storage switched to **S3/MinIO** in k8s (`TREND_STORAGE_BACKEND=s3`) — multi-node pods must share `result_store`; local `fs` would not be shared. Added `trendflow` bucket to minio-init.

### Retry / DLQ status (P7 code, not yet changed)
- DataProcessor worker already uses **Redis Streams + consumer groups** with ACK and `xpending` re-claim of stuck messages (reliable at-least-once) — good base.
- Backend Celery tasks have only ad-hoc retry (one manual 429 retry). A proper `autoretry_for` + `acks_late` + dead-letter policy is a **code change deferred to a focused PR** (kept out of this infra pass to avoid touching task semantics blindly).

- **P10 ✅** `docs/LOAD_AND_SCALING_PLAN.md` — throughput model (built on `capacity_report.py`), parametric 200k sizing table, queue/backpressure config, SLOs, alert thresholds, scaling runbook, pre-flight checklist.

## Progress — third pass (2026-06-29, backend code)

See `docs/BACKEND_NADEZHNOST_I_NABLYUDAEMOST.md` (RU).
- **P7-code (partial) ✅** Celery reliability hardening in `backend/app/worker.py`:
  `acks_late`, `reject_on_worker_lost`, `prefetch_multiplier=1`, `task_track_started`,
  and a **6 h Redis `visibility_timeout`** (fixes double-execution of >1 h video tasks).
  Still deferred: `autoretry_for` + explicit dead-letter, GPU/CPU queue routing.
- **P9 ✅** Structured JSON logs + `correlation_id` (`backend/app/logging_setup.py`):
  `CorrelationIdMiddleware` (X-Request-ID) + `set_correlation_id(run_id)` in the two
  Celery work tasks. Wired in `main.py`. Stdlib-only, no new deps.

- **P5 ✅** `docker-compose.prod.yml` (17 services: pg/redis/minio+init/model-download/triton[gpu profile]/backend api+worker+beat+migrate/fetcher api+worker+beat/DP api+worker/prometheus+grafana) + `deploy/postgres-init.sql`. One shared redis/postgres/minio/network. Validated: YAML parses, all build contexts + mounted files exist, merge-anchors resolve. **Final check on your machine:** `docker compose -f docker-compose.prod.yml config`.
- **OTel (backend) ✅** `backend/app/tracing.py` — optional OTLP tracing (no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` set + libs installed), mirrors DP; wired in `main.py`.

- **OTel (fetcher) ✅** `Fetcher/fetcher/tracing.py` + wiring in `fetcher/api.py` — optional OTLP, same pattern as backend/DP. Tracing теперь во всех 3 сервисах (Fetcher уже имел structured logging + metrics + backpressure + celery_queues).

Remaining: **P7-code** (autoretry/DLQ + GPU/CPU queue routing — needs tests/review). Triton ONNX models теперь провижатся из `trendflow_artifact_0_1` (multi-source manifest), новый экспорт ONNX — по необходимости. Task 5 infra is otherwise complete.
