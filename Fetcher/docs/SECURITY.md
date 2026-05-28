# Security & Privacy для Fetcher

Документ описывает настройки безопасности для Fetcher, включая TLS, proxy authentication и credential rotation.

## 1. TLS Everywhere

### PostgreSQL

Для включения TLS для PostgreSQL используйте один из способов:

**Способ 1: Через DSN (рекомендуется)**
```bash
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"
```

**Способ 2: Через отдельную переменную**
```bash
export FETCHER_POSTGRES_SSL_MODE="require"
```

Доступные значения `sslmode`:
- `disable` - SSL отключен
- `allow` - SSL опционален
- `prefer` - SSL предпочтителен, но не обязателен
- `require` - SSL обязателен (рекомендуется для production)
- `verify-ca` - SSL обязателен + проверка CA
- `verify-full` - SSL обязателен + проверка CA и hostname (максимальная безопасность)

### Redis

Для включения TLS для Redis используйте `rediss://` вместо `redis://`:

```bash
export FETCHER_REDIS_URL="rediss://user:pass@host:6380/0"
```

Или через настройки:
```bash
export FETCHER_REDIS_SSL="true"
export FETCHER_REDIS_SSL_CERT_REQS="required"
```

### S3/MinIO

Для включения TLS для S3/MinIO используйте `https://` в endpoint URL:

```bash
export FETCHER_S3_ENDPOINT_URL="https://minio.example.com:9000"
export FETCHER_S3_USE_SSL="true"
export FETCHER_S3_VERIFY_SSL="true"
```

## 2. Proxy Authentication

### Через URL (рекомендуется)

Укажите credentials прямо в proxy URL:

```bash
export FETCHER_ENABLE_PROXIES="true"
export FETCHER_PROXIES='["socks5://user:pass@proxy1.example.com:1080", "http://user2:pass2@proxy2.example.com:8080"]'
```

### Через отдельные переменные

Если credentials хранятся отдельно:

```bash
export FETCHER_PROXY_AUTH_USERNAME="proxy_user"
export FETCHER_PROXY_AUTH_PASSWORD="proxy_pass"
```

**Примечание**: Приоритет имеет URL, если credentials указаны и там, и в отдельных переменных.

## 3. Credential Rotation

### Рекомендации для Production

1. **Использование Secrets Manager**:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Kubernetes Secrets

2. **Регулярная ротация**:
   - Database passwords: каждые 90 дней
   - S3/MinIO keys: каждые 90 дней
   - Proxy credentials: каждые 30-60 дней
   - API keys: каждые 90 дней

3. **Интеграция с Fetcher**:

   **Вариант A: Через переменные окружения (Kubernetes Secrets)**
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: fetcher-secrets
   type: Opaque
   data:
     postgres-password: <base64-encoded>
     s3-secret-key: <base64-encoded>
     proxy-password: <base64-encoded>
   ```

   ```yaml
   # Deployment
   env:
     - name: FETCHER_POSTGRES_DSN
       valueFrom:
         secretKeyRef:
           name: fetcher-secrets
           key: postgres-dsn
     - name: FETCHER_S3_SECRET_KEY
       valueFrom:
         secretKeyRef:
           name: fetcher-secrets
           key: s3-secret-key
   ```

   **Вариант B: Через Vault Agent (HashiCorp Vault)**
   ```bash
   # Vault Agent автоматически обновляет secrets в файл
   # Fetcher читает из файла или переменных окружения
   export FETCHER_POSTGRES_DSN="$(cat /vault/secrets/postgres-dsn)"
   ```

4. **Graceful rotation**:
   - Fetcher поддерживает обновление credentials без перезапуска через переменные окружения
   - Для критичных изменений рекомендуется rolling restart

### Пример скрипта ротации (для reference)

```bash
#!/bin/bash
# rotate_credentials.sh

# 1. Генерируем новые credentials
NEW_PASSWORD=$(openssl rand -base64 32)

# 2. Обновляем в Secrets Manager
aws secretsmanager update-secret \
  --secret-id fetcher/postgres-password \
  --secret-string "$NEW_PASSWORD"

# 3. Обновляем в БД
psql -h $DB_HOST -U admin -c "ALTER USER fetcher WITH PASSWORD '$NEW_PASSWORD';"

# 4. Обновляем в Kubernetes Secrets
kubectl create secret generic fetcher-secrets \
  --from-literal=postgres-password="$NEW_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Rolling restart Fetcher pods
kubectl rollout restart deployment/fetcher-orchestrator
kubectl rollout restart deployment/fetcher-metadata-worker
kubectl rollout restart deployment/fetcher-download-worker
kubectl rollout restart deployment/fetcher-comments-worker
```

## 4. Best Practices

1. **Никогда не храните credentials в коде или git**
2. **Используйте минимальные привилегии** (principle of least privilege)
3. **Включайте TLS везде в production**
4. **Регулярно ротируйте credentials**
5. **Мониторьте доступ к secrets** (audit logs)
6. **Используйте separate credentials для разных окружений** (dev/staging/prod)

## 5. Проверка конфигурации

Проверьте, что TLS включен:

```bash
# Проверка PostgreSQL
psql "$FETCHER_POSTGRES_DSN" -c "SHOW ssl;" | grep "on"

# Проверка Redis
redis-cli -u "$FETCHER_REDIS_URL" --tls --insecure PING

# Проверка S3/MinIO
aws s3 ls --endpoint-url "$FETCHER_S3_ENDPOINT_URL" --no-verify-ssl
```

