# Развёртывание TrendFlow (канонический гайд)

Единая точка входа по развёртыванию. Подробные фоновые документы:
`DEPLOYMENT_GUIDE.md`, `DEPLOYMENT_QUICKSTART.md`, `K8S_FIRST_APPROACH.md`,
`CONTAINER_GRANULARITY.md` — здесь сведён актуальный рабочий путь.

## Варианты

| Цель | Как | Файлы |
|---|---|---|
| Локально/dev (с нуля после git clone) | `./bootstrap.sh` | `bootstrap.sh` |
| Одна машина / staging (весь стек в контейнерах) | `python deploy/validate_compose.py && docker compose -f docker-compose.prod.yml up -d` | `docker-compose.prod.yml`, `deploy/postgres-init.sql` |
| Прод / мульти-нода | `kubectl apply -k k8s/` | `k8s/` (kustomize) |

## 1. k8s (прод, мульти-нода)

```bash
# 1) реальные секреты (вместо примеров в k8s/secrets-example.yaml)
kubectl create ns trendflow
kubectl -n trendflow create secret generic app-secrets \
  --from-literal=database-url='postgresql://trendflow:trendflow@postgres:5432/trendflow' \
  --from-literal=postgres-password='<...>' \
  --from-literal=redis-url='redis://redis:6379/0' \
  --from-literal=minio-endpoint='http://minio:9000' \
  --from-literal=minio-access-key='<...>' --from-literal=minio-secret-key='<...>' \
  --from-literal=dataprocessor-api-key='<...>' --from-literal=hf-token='<для приватных моделей>'

# 2) RWX storageClass в k8s/infrastructure/models.yaml; registry/теги в k8s/kustomization.yaml (images:)
# 3) линт манифестов (ловит несуществующие secret-ключи/сервисы/PVC) и применить
python k8s/validate_manifests.py
kubectl apply -k k8s/
kubectl -n trendflow get jobs        # дождаться model-download / backend-migrate / minio-init
kubectl -n trendflow get pods -o wide
```

Состав: namespace, postgres/redis/minio(+init)/triton, **embedding-service**,
backend(+migrate/ingress), dataprocessor(api/worker/hpa), models-pvc+download Job,
governance (PriorityClass/Quota/PDB), backups (pg-backup CronJob + minio lifecycle),
retention CronJob. KEDA-автоскейл воркера — опционально
(`k8s/dataprocessor/keda-scaledobject.yaml`).

Требования: RWX StorageClass (NFS/CephFS/Longhorn/EFS/Filestore/Azure Files) для
`models-pvc`; NVIDIA GPU Operator для GPU-нод (worker/triton, label `accelerator=nvidia-gpu`).

## 2. Модели

- Веса проекта — из единого HF-репо `trendflow_models` через `download_models.py`
  (в k8s это делает `model-download` Job; Triton-ONNX из `trendflow_artifact_0_1`,
  для него нужен `HF_TOKEN`). См. `MODELS_UNIFIED_AND_BOOTSTRAP.md`.
- Публичные базовые модели (e5-large, pyannote, source_separation, …) —
  `DataProcessor/scripts/provision_base_models.py` (offline). См. `DataProcessor/docs/BASE_MODELS_PROVISION.md`.

## 3. Наблюдаемость

- Метрики: `dataprocessor_*` (queue/latency/failures/host), backend RED + Celery depth,
  Fetcher, Triton, ES. Prometheus scrape-аннотации на подах.
- Логи: структурный JSON + `correlation_id`/`run_id` (backend), json-логи DP/Fetcher.
- Трейсинг: OTel во всех 3 сервисах (включается `OTEL_EXPORTER_OTLP_ENDPOINT`).
- Дашборды/алерты/SLO и сайзинг под 200k — `LOAD_AND_SCALING_PLAN.md`.

## 4. Очередь и масштаб

Канонический путь обработки — Redis Streams (`api/services/queue`+`worker`),
см. `DataProcessor/docs/DATAPROCESSOR_QUEUE_CANONICAL.md`. Автоскейл воркера — KEDA
по длине очереди; сайзинг GPU/воркеров — `LOAD_AND_SCALING_PLAN.md`.

## 5. Бэкапы / retention

- Postgres: ежедневный `pg-backup` CronJob (backup-PVC, ротация) — `k8s/infrastructure/backups.yaml`.
- MinIO: lifecycle TTL на сырое видео; `result_store` (SoT) — без TTL.
- DataProcessor result_store retention — `dataprocessor-retention-cleanup` CronJob.

## Прод-статус и backlog
Актуальные пробелы и план — `PROD_ARCH_GAP_MAP.md`, `IMPLEMENTATION_PLAN.md`.
