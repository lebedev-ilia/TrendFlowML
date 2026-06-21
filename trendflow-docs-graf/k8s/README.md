# Kubernetes манифесты для TrendFlowML

Этот каталог содержит примеры Kubernetes конфигураций для развертывания всех компонентов системы.

## Структура

```
k8s/
├── README.md                    # Этот файл
├── infrastructure/              # Инфраструктурные компоненты
│   ├── postgres.yaml           # PostgreSQL база данных
│   ├── redis.yaml              # Redis для очередей
│   ├── minio.yaml              # MinIO (S3 storage)
│   └── triton.yaml             # Triton Inference Server
├── backend/                    # Backend API
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── fetcher/                    # Fetcher workers
│   ├── deployment.yaml
│   └── hpa.yaml                # Авто-масштабирование
├── dataprocessor/              # DataProcessor workers
│   ├── deployment.yaml
│   ├── hpa.yaml
│   └── gpu-node-selector.yaml  # Для GPU нод
└── models/                     # ML модели (если нужны отдельные сервисы)
    └── deployment.yaml
```

## Быстрый старт

### 1. Применить инфраструктуру

```bash
kubectl apply -f k8s/infrastructure/
```

### 2. Применить сервисы

```bash
kubectl apply -f k8s/backend/
kubectl apply -f k8s/fetcher/
kubectl apply -f k8s/dataprocessor/
```

### 3. Проверить статус

```bash
kubectl get pods
kubectl get services
```

## Настройка

Перед применением манифестов:

1. Создайте secrets:
```bash
kubectl create secret generic app-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=redis-password='...' \
  --from-literal=minio-access-key='...' \
  --from-literal=minio-secret-key='...'
```

2. Обновите образы в deployment.yaml (укажите ваш registry)

3. Настройте ресурсы (CPU/RAM) под ваши нужды

## Масштабирование

### Ручное масштабирование

```bash
kubectl scale deployment backend --replicas=5
kubectl scale deployment fetcher-worker --replicas=10
```

### Автоматическое масштабирование

HPA (Horizontal Pod Autoscaler) уже настроен в `hpa.yaml` файлах.

Проверить статус:
```bash
kubectl get hpa
```

## Мониторинг

После развертывания доступны:

- Grafana: `http://<ingress-ip>/grafana`
- Prometheus: `http://<ingress-ip>/prometheus`
- Jaeger: `http://<ingress-ip>/jaeger`

## GPU поддержка

Для DataProcessor workers с GPU:

1. Убедитесь, что на нодах установлен NVIDIA GPU Operator
2. Примените `dataprocessor/gpu-node-selector.yaml`
3. Проверьте, что поды запустились на GPU нодах:
```bash
kubectl get pods -o wide
```
---

## Навигация

[Vault](../docs/MAIN_INDEX.md)
