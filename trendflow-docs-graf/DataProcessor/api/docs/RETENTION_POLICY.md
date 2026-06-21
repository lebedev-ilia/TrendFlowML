# Retention Policy

## Обзор

Retention Policy автоматически удаляет старые данные согласно политике хранения:
- **Redis state**: Удаление старше 1 дня
- **Storage**: Удаление старше 7 дней

## Использование

### Ручной запуск через API

```bash
curl -X POST http://localhost:8000/api/v1/admin/retention/cleanup \
  -H "X-API-Key: your-api-key"
```

Ответ:
```json
{
  "redis": {
    "checked": 100,
    "deleted": 5,
    "errors": 0
  },
  "storage": {
    "checked": 50,
    "deleted": 2,
    "errors": 0
  },
  "timestamp": 1704067200.0,
  "elapsed_seconds": 12.34
}
```

### Запуск через скрипт

```bash
python -m api.retention_cleanup
```

### Автоматический запуск через Docker Compose

Сервис `dataprocessor-retention-cleanup` в `docker-compose.yml` запускается автоматически:
- Ежедневно в 2:00 UTC
- Логи сохраняются в volume `retention-cleanup-logs`

### Альтернативные варианты для production

#### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dataprocessor-retention-cleanup
spec:
  schedule: "0 2 * * *"  # Ежедневно в 2:00 UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: retention-cleanup
            image: dataprocessor-api:latest
            command: ["python", "-m", "api.retention_cleanup"]
            env:
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: dataprocessor-secrets
                  key: redis-url
            - name: STORAGE_TYPE
              value: "s3"
          restartPolicy: OnFailure
```

#### Systemd Timer

Создайте файл `/etc/systemd/system/dataprocessor-retention-cleanup.service`:
```ini
[Unit]
Description=DataProcessor Retention Cleanup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 -m api.retention_cleanup
EnvironmentFile=/etc/dataprocessor/env
```

И файл `/etc/systemd/system/dataprocessor-retention-cleanup.timer`:
```ini
[Unit]
Description=Run DataProcessor Retention Cleanup Daily

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

Запуск:
```bash
sudo systemctl enable dataprocessor-retention-cleanup.timer
sudo systemctl start dataprocessor-retention-cleanup.timer
```

## Конфигурация

Retention policy использует следующие константы (в `api/services/retention.py`):
- `REDIS_STATE_RETENTION = 24 * 3600` (1 день)
- `STORAGE_RETENTION = 7 * 24 * 3600` (7 дней)

Для изменения политики отредактируйте эти константы.

## Мониторинг

Проверьте логи retention cleanup:
```bash
# Docker Compose
docker-compose logs dataprocessor-retention-cleanup

# Kubernetes
kubectl logs -l job-name=dataprocessor-retention-cleanup
```

## Тестирование

Запустите unit тесты:
```bash
pytest api/tests/unit/test_retention.py -v
```

## Ссылки

- [Архитектура Retention Policy](../docs/DATAPROCESSOR_API_ARCHITECTURE.md#L2349)
- [Чеклист разработки](../docs/API_DEVELOPMENT_CHECKLIST.md#L878)
---

## Навигация

[README](README.md) · [Module README](../README.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
