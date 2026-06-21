# Kubernetes манифесты для Fetcher

Этот каталог содержит Kubernetes манифесты для развёртывания Fetcher в production.

## Структура

```
k8s/fetcher/
├── README.md                          # Этот файл
├── orchestrator-deployment.yaml       # API сервис (orchestrator)
├── orchestrator-service.yaml          # Service для orchestrator
├── metadata-worker-deployment.yaml   # Metadata worker
├── download-worker-deployment.yaml    # Video download worker
├── comments-worker-deployment.yaml    # Comments worker
├── finalize-worker-deployment.yaml    # Artifact builder worker
├── beat-deployment.yaml               # Celery Beat (периодические задачи)
├── configmap.yaml                     # ConfigMap для конфигурации
├── secrets-example.yaml               # Пример Secrets (НЕ коммитить реальные секреты!)
├── metadata-worker-hpa.yaml          # HPA для metadata worker
├── download-worker-hpa.yaml          # HPA для download worker
└── comments-worker-hpa.yaml          # HPA для comments worker
```

## Компоненты

### 1. Fetcher Orchestrator (API сервис)

- **Deployment**: `orchestrator-deployment.yaml`
- **Service**: `orchestrator-service.yaml`
- **Роль**: HTTP API для управления ingestion pipeline
- **Ресурсы**: CPU: 0.2-1, RAM: 256MB-512MB
- **Replicas**: 2 (для HA)

### 2. Metadata Worker

- **Deployment**: `metadata-worker-deployment.yaml`
- **HPA**: `metadata-worker-hpa.yaml`
- **Роль**: Обработка метаданных видео
- **Ресурсы**: CPU: 0.5-1, RAM: 512MB-1GB
- **Queue**: `fetch.metadata`
- **Replicas**: 1-20 (авто-масштабирование)

### 3. Download Worker

- **Deployment**: `download-worker-deployment.yaml`
- **HPA**: `download-worker-hpa.yaml`
- **Роль**: Скачивание видео
- **Ресурсы**: CPU: 2-4, RAM: 2GB-4GB
- **Queue**: `fetch.video`
- **Replicas**: 1-10 (авто-масштабирование)

### 4. Comments Worker

- **Deployment**: `comments-worker-deployment.yaml`
- **HPA**: `metadata-worker-hpa.yaml`
- **Роль**: Обработка комментариев
- **Ресурсы**: CPU: 1-2, RAM: 1GB-2GB
- **Queue**: `fetch.comments`
- **Replicas**: 1-15 (авто-масштабирование)

### 5. Finalize Worker

- **Deployment**: `finalize-worker-deployment.yaml`
- **Роль**: Построение manifest.json и запуск DataProcessor
- **Ресурсы**: CPU: 0.5-1, RAM: 512MB-1GB
- **Queue**: `fetch.finalize`
- **Replicas**: 1 (фиксированное)

### 6. Celery Beat

- **Deployment**: `beat-deployment.yaml`
- **Роль**: Периодические задачи (lifecycle cleanup, snapshots)
- **Ресурсы**: CPU: 0.1-0.5, RAM: 128MB-256MB
- **Replicas**: 1 (singleton)

## Развёртывание

### 1. Создать Secrets

**ВАЖНО**: Не коммитьте реальные секреты в git!

```bash
# Создать Secret из файла (для тестирования)
kubectl create secret generic fetcher-secrets \
  --from-literal=postgres-dsn="postgresql+psycopg2://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://host:6379/0" \
  --from-literal=s3-access-key="ACCESS_KEY" \
  --from-literal=s3-secret-key="SECRET_KEY"

# Или использовать External Secrets Operator для production
# См. secrets-example.yaml для примера
```

### 2. Создать ConfigMap

```bash
kubectl apply -f k8s/fetcher/configmap.yaml
```

### 3. Развернуть компоненты

```bash
# Orchestrator
kubectl apply -f k8s/fetcher/orchestrator-deployment.yaml
kubectl apply -f k8s/fetcher/orchestrator-service.yaml

# Workers
kubectl apply -f k8s/fetcher/metadata-worker-deployment.yaml
kubectl apply -f k8s/fetcher/download-worker-deployment.yaml
kubectl apply -f k8s/fetcher/comments-worker-deployment.yaml
kubectl apply -f k8s/fetcher/finalize-worker-deployment.yaml

# Celery Beat
kubectl apply -f k8s/fetcher/beat-deployment.yaml

# HPA
kubectl apply -f k8s/fetcher/metadata-worker-hpa.yaml
kubectl apply -f k8s/fetcher/download-worker-hpa.yaml
kubectl apply -f k8s/fetcher/comments-worker-hpa.yaml
```

### 4. Проверить статус

```bash
# Проверить pods
kubectl get pods -l app=fetcher

# Проверить services
kubectl get svc -l app=fetcher

# Проверить HPA
kubectl get hpa -l app=fetcher

# Проверить логи
kubectl logs -l component=orchestrator
kubectl logs -l component=metadata-worker
```

## Настройка

### Resource Limits

Resource limits настроены согласно чеклисту:

| Component | CPU Request | CPU Limit | RAM Request | RAM Limit |
|-----------|-------------|-----------|-------------|-----------|
| Orchestrator | 0.2 | 1 | 256MB | 512MB |
| Metadata Worker | 0.5 | 1 | 512MB | 1GB |
| Download Worker | 2 | 4 | 2GB | 4GB |
| Comments Worker | 1 | 2 | 1GB | 2GB |
| Finalize Worker | 0.5 | 1 | 512MB | 1GB |
| Beat | 0.1 | 0.5 | 128MB | 256MB |

### Авто-масштабирование

HPA настроен для:
- **Metadata Worker**: 1-20 replicas, CPU 70%, Memory 80%
- **Download Worker**: 1-10 replicas, CPU 70%, Memory 80%
- **Comments Worker**: 1-15 replicas, CPU 70%, Memory 80%

### Health Checks

Все компоненты имеют:
- **Liveness Probe**: Проверка работоспособности
- **Readiness Probe**: Проверка готовности к обработке запросов

## Зависимости

Fetcher требует:
- **PostgreSQL**: База данных для метаданных и состояния
- **Redis**: Очереди задач и rate limiting
- **MinIO/S3**: Object storage для артефактов
- **DataProcessor API**: Для проверки backpressure

Убедитесь, что эти сервисы развёрнуты и доступны перед развёртыванием Fetcher.

## Мониторинг

### Метрики

Fetcher экспортирует Prometheus метрики через `/metrics` endpoint на orchestrator.

### Логи

Логи доступны через:
```bash
kubectl logs -l app=fetcher
```

Или через централизованное логирование (ELK/Loki), если настроено.

## Troubleshooting

### Pod не запускается

1. Проверить логи: `kubectl logs <pod-name>`
2. Проверить события: `kubectl describe pod <pod-name>`
3. Проверить Secrets и ConfigMap: `kubectl get secrets fetcher-secrets`, `kubectl get configmap fetcher-config`

### Worker не обрабатывает задачи

1. Проверить подключение к Redis: `kubectl exec -it <pod-name> -- redis-cli -h redis-service ping`
2. Проверить логи worker'а: `kubectl logs -l component=metadata-worker`
3. Проверить очередь в Redis: `kubectl exec -it <pod-name> -- redis-cli -h redis-service LLEN celery`

### HPA не масштабирует

1. Проверить метрики: `kubectl get hpa`
2. Проверить метрики сервера: `kubectl top pods -l app=fetcher`
3. Убедиться, что metrics-server установлен: `kubectl get deployment metrics-server -n kube-system`

## Production рекомендации

1. **Secrets Management**: Используйте External Secrets Operator или Vault для управления секретами
2. **Resource Quotas**: Настройте ResourceQuota для namespace
3. **Network Policies**: Настройте NetworkPolicy для ограничения сетевого доступа
4. **Pod Disruption Budget**: Создайте PDB для обеспечения доступности
5. **Monitoring**: Настройте Prometheus и Grafana для мониторинга
6. **Logging**: Настройте централизованное логирование (ELK/Loki)
---

## Навигация

[Vault](../../docs/MAIN_INDEX.md)
