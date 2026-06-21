# Руководство по развертыванию TrendFlowML

Это практическое руководство для развертывания системы TrendFlowML от простого Docker до масштабируемого Kubernetes.

---

## 📚 Содержание

1. [Основы: что такое Docker и зачем он нужен](#1-основы-что-такое-docker)
2. [Простое развертывание: Docker Compose](#2-простое-развертывание-docker-compose)
3. [Продвинутое развертывание: Kubernetes](#3-продвинутое-развертывание-kubernetes)
4. [Масштабирование системы](#4-масштабирование-системы)
5. [Практические примеры](#5-практические-примеры)

---

## 1. Основы: что такое Docker

### Что такое контейнер?

**Контейнер** — это изолированная среда, которая содержит:
- Ваш код (Python приложение)
- Все зависимости (библиотеки, системные пакеты)
- Настройки окружения

**Аналогия**: Представьте, что контейнер — это чемодан, в котором упаковано всё необходимое для работы вашего приложения. Вы можете взять этот чемодан и запустить его на любой машине, где установлен Docker.

### Зачем нужен Docker?

1. **"У меня работает, а у тебя нет"** — больше не проблема
   - Если работает на вашей машине, будет работать везде
   
2. **Простое развертывание**
   - Один раз настроили → запускаете везде одинаково
   
3. **Изоляция**
   - Каждый сервис работает в своём контейнере
   - Не мешают друг другу

4. **Масштабирование**
   - Легко запустить несколько копий одного сервиса

### Основные команды Docker

```bash
# Собрать образ (image) из Dockerfile
docker build -t trendflow-backend:latest ./backend

# Запустить контейнер
docker run -d -p 8000:8000 trendflow-backend:latest

# Посмотреть запущенные контейнеры
docker ps

# Посмотреть логи
docker logs <container_id>

# Остановить контейнер
docker stop <container_id>
```

---

## 2. Простое развертывание: Docker Compose

### Что такое Docker Compose?

**Docker Compose** — это инструмент для запуска нескольких контейнеров одновременно и управления их взаимодействием.

**Аналогия**: Если Docker — это один контейнер, то Docker Compose — это оркестр контейнеров, которые работают вместе.

### Архитектура TrendFlowML в Docker Compose

```
┌─────────────────────────────────────────────────┐
│         Docker Compose (одна машина)            │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Backend  │  │ Fetcher  │  │ DataProc │      │
│  │  :8000   │  │  worker  │  │  worker  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │            │              │             │
│       └────────────┼──────────────┘             │
│                    │                            │
│  ┌─────────────────┼─────────────────┐         │
│  │   PostgreSQL    │   Redis         │         │
│  │   :5432         │   :6379          │         │
│  └─────────────────┴─────────────────┘         │
│                                                 │
│  ┌─────────────────────────────────────┐       │
│  │      MinIO (S3 storage) :9000       │       │
│  └─────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

### Структура docker-compose.yml

```yaml
version: '3.8'

services:
  # База данных
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trendflow
      POSTGRES_USER: trendflow
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis для очередей
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # MinIO (S3-совместимое хранилище)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    ports:
      - "9000:9000"  # API
      - "9001:9001"  # Console
    volumes:
      - minio_data:/data

  # Backend API
  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://trendflow:${DB_PASSWORD}@postgres:5432/trendflow
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

  # Fetcher worker
  fetcher-worker:
    build: ./Fetcher
    environment:
      DATABASE_URL: postgresql://trendflow:${DB_PASSWORD}@postgres:5432/trendflow
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: http://minio:9000
    depends_on:
      - postgres
      - redis
      - minio

  # DataProcessor worker
  dataprocessor-worker:
    build: ./DataProcessor
    environment:
      DATABASE_URL: postgresql://trendflow:${DB_PASSWORD}@postgres:5432/trendflow
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: http://minio:9000
      TRITON_HTTP_URL: http://triton:8000
    depends_on:
      - postgres
      - redis
      - minio

  # Triton Inference Server (для моделей)
  triton:
    image: nvcr.io/nvidia/tritonserver:latest
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./models:/models
    command: tritonserver --model-repository=/models

volumes:
  postgres_data:
  minio_data:
```

### Запуск

```bash
# Создать .env файл с паролями
cat > .env << EOF
DB_PASSWORD=your_secure_password
MINIO_PASSWORD=your_minio_password
EOF

# Запустить все сервисы
docker-compose up -d

# Посмотреть логи
docker-compose logs -f backend

# Остановить все
docker-compose down
```

### Преимущества Docker Compose

✅ **Простота**: один файл, одна команда  
✅ **Быстрый старт**: идеально для разработки и MVP  
✅ **Локальная разработка**: все сервисы на одной машине  

### Ограничения Docker Compose

❌ **Одна машина**: все контейнеры на одном сервере  
❌ **Нет авто-масштабирования**: нужно вручную добавлять worker'ы  
❌ **Нет отказоустойчивости**: если машина упала, всё упало  

---

## 3. Продвинутое развертывание: Kubernetes

### Что такое Kubernetes (K8s)?

**Kubernetes** — это система для управления контейнерами на нескольких машинах (кластере).

**Аналогия**: 
- Docker Compose = один дирижёр управляет оркестром на одной сцене
- Kubernetes = несколько дирижёров управляют оркестрами на разных сценах, и если один дирижёр устал, другой его заменяет

### Зачем нужен Kubernetes?

1. **Масштабирование**
   - Автоматически добавляет worker'ы при росте нагрузки
   - Убирает лишние при снижении

2. **Отказоустойчивость**
   - Если контейнер упал, K8s автоматически перезапустит
   - Если машина упала, контейнеры переедут на другую

3. **Управление ресурсами**
   - Контролирует использование CPU/RAM/GPU
   - Не даёт одному сервису "съесть" все ресурсы

4. **Обновления без простоя**
   - Rolling updates: обновляет по одному контейнеру
   - Если что-то пошло не так, откатывает изменения

### Основные концепции Kubernetes

#### 1. Pod (под)

**Pod** — это один или несколько контейнеров, которые работают вместе.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend-pod
spec:
  containers:
  - name: backend
    image: trendflow-backend:latest
    ports:
    - containerPort: 8000
```

#### 2. Deployment (развертывание)

**Deployment** управляет несколькими копиями Pod'ов (репликами).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-deployment
spec:
  replicas: 3  # Запустить 3 копии
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: trendflow-backend:latest
        ports:
        - containerPort: 8000
```

#### 3. Service (сервис)

**Service** — это способ доступа к Pod'ам (как единый адрес для всех реплик).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer  # Или ClusterIP для внутреннего доступа
```

#### 4. ConfigMap и Secrets

**ConfigMap** — хранит конфигурацию (не секреты).

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://..."
  log_level: "INFO"
```

**Secret** — хранит секреты (пароли, ключи).

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  password: <base64-encoded-password>
```

### Архитектура TrendFlowML в Kubernetes

```
┌─────────────────────────────────────────────────────────┐
│              Kubernetes Cluster                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Ingress (Nginx)                     │  │
│  │         (точка входа из интернета)                │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │         Backend Service (3 реплики)              │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐             │  │
│  │  │ Backend│  │ Backend│  │ Backend│             │  │
│  │  │ Pod 1  │  │ Pod 2  │  │ Pod 3  │             │  │
│  │  └────────┘  └────────┘  └────────┘             │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │      Fetcher Workers (авто-масштабирование)       │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │  │
│  │  │Worker│  │Worker│  │Worker│  │Worker│         │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘         │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │   DataProcessor Workers (GPU nodes)               │  │
│  │  ┌──────┐  ┌──────┐                              │  │
│  │  │Worker│  │Worker│  (на машинах с GPU)          │  │
│  │  └──────┘  └──────┘                              │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────┼───────────────────────────────┐  │
│  │   PostgreSQL (StatefulSet, 1 реплика)            │  │
│  │   Redis (StatefulSet, 1 реплика)                 │  │
│  │   MinIO (StatefulSet, 1 реплика)                 │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Масштабирование системы

### Горизонтальное масштабирование

**Горизонтальное масштабирование** = добавить больше машин/контейнеров.

#### Пример: масштабирование DataProcessor workers

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dataprocessor-worker
spec:
  replicas: 5  # Начать с 5 worker'ов
  # ...
```

**Автоматическое масштабирование (HPA - Horizontal Pod Autoscaler):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dataprocessor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dataprocessor-worker
  minReplicas: 2      # Минимум 2 worker'а
  maxReplicas: 20     # Максимум 20 worker'ов
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Масштабировать при 70% CPU
```

**Логика работы:**
- Если CPU > 70% → добавить worker'ы
- Если CPU < 30% → убрать лишние worker'ы
- Проверка каждые 15 секунд

### Вертикальное масштабирование

**Вертикальное масштабирование** = увеличить ресурсы одной машины (больше RAM/CPU/GPU).

```yaml
spec:
  containers:
  - name: dataprocessor-worker
    resources:
      requests:      # Минимум ресурсов
        cpu: "2"
        memory: "4Gi"
        nvidia.com/gpu: 1
      limits:        # Максимум ресурсов
        cpu: "4"
        memory: "8Gi"
        nvidia.com/gpu: 1
```

### Масштабирование по очереди

**Масштабирование на основе длины очереди Redis:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fetcher-hpa-queue
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fetcher-worker
  minReplicas: 1
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_length
      target:
        type: AverageValue
        averageValue: "10"  # 1 worker на 10 задач в очереди
```

---

## 5. Практические примеры

### Пример 1: Dockerfile для Backend

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Пример 2: Kubernetes Deployment для Backend

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  labels:
    app: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: trendflow-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Пример 3: DataProcessor Worker с GPU

```yaml
# k8s/dataprocessor-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dataprocessor-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dataprocessor-worker
  template:
    metadata:
      labels:
        app: dataprocessor-worker
    spec:
      containers:
      - name: worker
        image: trendflow-dataprocessor:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
        env:
        - name: TRITON_HTTP_URL
          value: "http://triton-service:8000"
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis-url
      nodeSelector:
        accelerator: nvidia-tesla-v100  # Только на GPU-нодах
```

### Пример 4: Полный docker-compose для разработки

См. файл `DataProcessor/docker-compose.yml` в проекте.

---

## 🚀 Пошаговый план развертывания

### Этап 1: Локальная разработка (Docker Compose)

1. Создать `docker-compose.yml`
2. Настроить `.env` с паролями
3. Запустить: `docker-compose up -d`
4. Проверить: `curl http://localhost:8000/health`

### Этап 2: Тестовый сервер (Docker Compose)

1. Арендовать VPS (например, на DigitalOcean, Hetzner)
2. Установить Docker и Docker Compose
3. Скопировать проект и `.env`
4. Запустить: `docker-compose up -d`
5. Настроить домен и SSL (Let's Encrypt)

### Этап 3: Production (Kubernetes)

1. Настроить Kubernetes кластер (GKE, EKS, или свой)
2. Создать Docker образы и загрузить в registry
3. Создать Kubernetes манифесты (deployments, services)
4. Применить: `kubectl apply -f k8s/`
5. Настроить мониторинг (Prometheus + Grafana)
6. Настроить авто-масштабирование (HPA)

---

## 📊 Мониторинг и метрики

### Что мониторить?

1. **Здоровье сервисов**
   - Health check endpoints
   - Uptime

2. **Производительность**
   - Время обработки видео
   - Использование CPU/RAM/GPU
   - Длина очереди

3. **Ошибки**
   - Количество ошибок по типам
   - Success rate

4. **Ресурсы**
   - Использование диска
   - Сетевой трафик

### Инструменты

- **Prometheus** — сбор метрик
- **Grafana** — визуализация
- **Alertmanager** — уведомления

---

## 🔒 Безопасность

1. **Secrets management**
   - Использовать Kubernetes Secrets
   - Не хранить пароли в коде

2. **Сетевая изоляция**
   - Внутренние сервисы не доступны из интернета
   - Только Backend API через Ingress

3. **TLS/SSL**
   - HTTPS для всех внешних соединений
   - mTLS для внутренних (опционально)

---

## 📝 Чеклист для production

- [ ] Все сервисы имеют health checks
- [ ] Настроен мониторинг (Prometheus + Grafana)
- [ ] Настроены алерты на критические метрики
- [ ] Настроено авто-масштабирование (HPA)
- [ ] Secrets хранятся в Kubernetes Secrets
- [ ] Настроен backup базы данных
- [ ] Настроен backup MinIO/S3
- [ ] Документированы процедуры восстановления
- [ ] Настроен CI/CD pipeline
- [ ] Проведены нагрузочные тесты

---

## 🆘 Полезные команды

### Docker Compose

```bash
# Запустить
docker-compose up -d

# Логи
docker-compose logs -f <service>

# Перезапустить сервис
docker-compose restart <service>

# Масштабировать worker
docker-compose up -d --scale dataprocessor-worker=5
```

### Kubernetes

```bash
# Применить конфигурацию
kubectl apply -f k8s/

# Посмотреть поды
kubectl get pods

# Логи
kubectl logs -f <pod-name>

# Масштабировать deployment
kubectl scale deployment backend --replicas=5

# Посмотреть метрики
kubectl top pods
```

---

## 📚 Дополнительные ресурсы

- [Docker документация](https://docs.docker.com/)
- [Kubernetes документация](https://kubernetes.io/docs/)
- [Docker Compose документация](https://docs.docker.com/compose/)
- [Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/)

---

**Вопросы?** Создайте issue в репозитории или обратитесь к команде разработки.
---

## Навигация

[Vault](MAIN_INDEX.md)
