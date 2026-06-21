# Быстрый старт: Развертывание TrendFlowML

Краткая шпаргалка для развертывания системы.

## 🚀 Вариант 1: Docker Compose (для разработки и MVP)

### Шаг 1: Подготовка

```bash
# Клонировать проект
git clone <repo-url>
cd TrendFlowML

# Создать .env файл
cat > .env << EOF
DB_PASSWORD=secure_password_here
MINIO_PASSWORD=minio_password_here
REDIS_PASSWORD=redis_password_here
EOF
```

### Шаг 2: Запуск

```bash
# Запустить все сервисы
cd DataProcessor
docker-compose up -d

# Проверить статус
docker-compose ps

# Посмотреть логи
docker-compose logs -f backend
```

### Шаг 3: Проверка

```bash
# Проверить health endpoint
curl http://localhost:8001/api/v1/health

# Масштабировать worker'ы
docker-compose up -d --scale dataprocessor-worker=3
```

---

## ☸️ Вариант 2: Kubernetes (для production)

### Предварительные требования

- Kubernetes кластер (GKE, EKS, или свой)
- `kubectl` настроен
- Docker registry для образов

### Шаг 1: Создать secrets

```bash
kubectl create secret generic app-secrets \
  --from-literal=database-url='postgresql://trendflow:password@postgres-service:5432/trendflow' \
  --from-literal=redis-url='redis://redis-service:6379/0' \
  --from-literal=postgres-password='secure_password' \
  --from-literal=minio-access-key='minioadmin' \
  --from-literal=minio-secret-key='minioadmin'
```

### Шаг 2: Применить инфраструктуру

```bash
kubectl apply -f k8s/infrastructure/
```

### Шаг 3: Применить сервисы

```bash
kubectl apply -f k8s/backend/
kubectl apply -f k8s/fetcher/
kubectl apply -f k8s/dataprocessor/
```

### Шаг 4: Проверить статус

```bash
# Посмотреть все поды
kubectl get pods

# Посмотреть сервисы
kubectl get services

# Посмотреть логи
kubectl logs -f deployment/backend
```

### Шаг 5: Масштабирование

```bash
# Ручное масштабирование
kubectl scale deployment backend --replicas=5
kubectl scale deployment fetcher-worker --replicas=10

# Проверить авто-масштабирование
kubectl get hpa
```

---

## 📊 Мониторинг

### Доступ к Grafana

```bash
# Port-forward
kubectl port-forward service/grafana 3000:3000

# Открыть в браузере
open http://localhost:3000
# Логин: admin / admin
```

### Метрики Prometheus

```bash
kubectl port-forward service/prometheus 9090:9090
open http://localhost:9090
```

---

## 🔧 Полезные команды

### Docker Compose

```bash
# Перезапустить сервис
docker-compose restart backend

# Пересобрать образ
docker-compose build --no-cache backend

# Остановить всё
docker-compose down

# Остановить и удалить volumes
docker-compose down -v
```

### Kubernetes

```bash
# Описание пода
kubectl describe pod <pod-name>

# Войти в под
kubectl exec -it <pod-name> -- /bin/sh

# Удалить deployment
kubectl delete deployment backend

# Применить изменения
kubectl apply -f k8s/backend/deployment.yaml

# Откатить deployment
kubectl rollout undo deployment/backend

# История изменений
kubectl rollout history deployment/backend
```

---

## 🐛 Troubleshooting

### Проблема: Поды не запускаются

```bash
# Проверить события
kubectl get events --sort-by='.lastTimestamp'

# Описание пода
kubectl describe pod <pod-name>

# Логи пода
kubectl logs <pod-name>
```

### Проблема: Недостаточно ресурсов

```bash
# Проверить использование ресурсов
kubectl top nodes
kubectl top pods

# Уменьшить requests/limits в deployment.yaml
```

### Проблема: База данных недоступна

```bash
# Проверить статус PostgreSQL
kubectl get pods -l app=postgres

# Проверить логи
kubectl logs -l app=postgres

# Проверить service
kubectl get service postgres-service
```

---

## 📚 Дополнительная документация

- Полное руководство: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- Kubernetes манифесты: [k8s/README.md](../k8s/README.md)
- Production архитектура: [DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md](../DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md)

---

## 🆘 Получить помощь

1. Проверьте логи: `kubectl logs <pod-name>` или `docker-compose logs`
2. Проверьте статус: `kubectl get pods` или `docker-compose ps`
3. Проверьте события: `kubectl get events`
4. Создайте issue в репозитории
---

## Навигация

[Vault](MAIN_INDEX.md)
