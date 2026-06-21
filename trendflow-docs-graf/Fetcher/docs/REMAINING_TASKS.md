# Оставшиеся моменты разработки Fetcher

Документ содержит список задач и улучшений, которые еще не реализованы или требуют доработки.

## 🔴 Критические задачи (для production)

### 1. Завершение реализации тестов

**Статус**: Структура создана, но тесты не реализованы

- [ ] **Unit тесты на адаптеры платформ**:
  - [ ] Тесты для `YouTubeAdapter.fetch_metadata()`
  - [ ] Тесты для `YouTubeAdapter.download_video()`
  - [ ] Тесты для `YouTubeAdapter.fetch_comments()`
  - [ ] Моки для `yt-dlp` и внешних API

- [ ] **Integration тесты на полный pipeline**:
  - [ ] Тест полного pipeline с фейковым YouTube
  - [ ] Тест идемпотентности (повторный запуск не создаёт дубликаты)
  - [ ] Тест resume после сбоя
  - [ ] Тест с реальной БД и моками storage

- [ ] **Chaos тесты**:
  - [ ] Тест восстановления после падения metadata worker'а
  - [ ] Тест восстановления после падения video worker'а
  - [ ] Тест устойчивости к потере подключения к Redis
  - [ ] Тест устойчивости к потере подключения к Storage
  - [ ] Тест устойчивости к таймаутам YouTube API

### 2. Реализация DataProcessor API интеграции

**Статус**: Заглушка реализована, нужна реальная интеграция

- [ ] **Backpressure control**:
  - [ ] Реализовать реальный запрос к DataProcessor API для проверки размера очереди
  - [ ] Добавить endpoint в DataProcessor API: `/api/v1/queue/status` или использовать метрики Prometheus
  - [ ] Обработать случаи недоступности DataProcessor API

**Файл**: `fetcher/backpressure.py` (строка 64)

### 3. Определение платформы по URL

**Статус**: Только YouTube, нужно добавить поддержку других платформ

- [ ] **Улучшить `normalize_source()`**:
  - [ ] Определение платформы по URL (youtube, tiktok, instagram, etc.)
  - [ ] Поддержка разных форматов URL для каждой платформы
  - [ ] Валидация URL перед обработкой

**Файлы**:
- `fetcher/orchestrator.py` (строка 49)
- `fetcher/workers/metadata.py` (строка 48)
- `fetcher/workers/video.py` (строка 53)
- `fetcher/workers/comments.py` (строка 48)

### 4. Kubernetes deployment манифесты

**Статус**: Есть базовая структура в `k8s/fetcher/`, но не полная

- [ ] **Отдельные deployment'ы для каждого типа worker'а**:
  - [ ] `fetcher-orchestrator` (API сервис)
  - [ ] `fetcher-metadata-worker`
  - [ ] `fetcher-download-worker`
  - [ ] `fetcher-comments-worker`
  - [ ] `fetcher-finalize-worker` (artifact builder)
  - [ ] `fetcher-beat` (Celery Beat)

- [ ] **Service манифесты**:
  - [ ] Service для orchestrator API
  - [ ] Service для метрик (если нужен отдельный)

- [ ] **ConfigMap и Secrets**:
  - [ ] ConfigMap для конфигурации
  - [ ] Secrets для credentials (БД, Redis, S3, прокси)

- [ ] **HPA (Horizontal Pod Autoscaler)**:
  - [ ] Настроить авто-масштабирование для каждого типа worker'а
  - [ ] Метрики для масштабирования (CPU, память, размер очереди)

- [ ] **Resource limits**:
  - [ ] Правильные лимиты для каждого типа worker'а (согласно чеклисту)

**Текущее состояние**: Есть базовый `k8s/fetcher/deployment.yaml`, но нужно расширить

## 🟡 Важные улучшения

### 5. Kafka event streaming (Production)

**Статус**: Не реализовано, используется только Celery + Redis

- [ ] **Интеграция с Kafka**:
  - [ ] Producer для отправки событий в Kafka
  - [ ] Consumer для обработки задач из Kafka
  - [ ] Миграция с Celery на Kafka (опционально)
  - [ ] Поддержка обеих систем (Celery для MVP, Kafka для production)

**Чеклист**: Phase 4 — Queue system

### 6. Централизованное логирование

**Статус**: Логи пишутся в БД, но нет централизованного storage

- [ ] **Интеграция с ELK/Loki/Cloud**:
  - [ ] Настройка отправки логов в централизованное хранилище
  - [ ] Структурированные логи в JSON формате
  - [ ] Индексация и поиск по логам

- [ ] **Pipeline event logs**:
  - [ ] Отдельная система для событий pipeline (не только БД)
  - [ ] Интеграция с event streaming (Kafka)

**Чеклист**: Phase 6 — Logging

### 7. Snapshot schedule (configurable)

**Статус**: Только начальный snapshot, нет периодических

- [ ] **Периодические snapshots**:
  - [ ] Конфигурируемый schedule (0/7/14/21 дней или по таймстемпам)
  - [ ] Celery Beat задача для периодических snapshots
  - [ ] Обработка уже существующих видео

**Чеклист**: Phase 5 — Snapshot ingestion

### 8. Улучшение lifecycle cleanup

**Статус**: Частично реализовано

- [ ] **Очистка temp bucket**:
  - [ ] Реализовать `list_objects()` в StorageClient
  - [ ] Реализовать реальную очистку temp bucket

- [ ] **Архивация/удаление processed artifacts**:
  - [ ] Реализовать когда будет требование

**Файл**: `fetcher/lifecycle.py` (строки 211, 257)

### 9. Улучшение rate limiter

**Статус**: Базовая реализация есть, можно улучшить

- [ ] **Логирование ошибок Redis**:
  - [ ] Добавить логирование и отдельные метрики ошибок Redis
  - [ ] Обработка различных типов ошибок

- [ ] **Более гибкая схема rate limiting**:
  - [ ] Token bucket или leaky bucket вместо fixed-window
  - [ ] Поддержка разных стратегий

**Файл**: `fetcher/rate_limiter.py` (строка 68)

## 🟢 Опциональные улучшения

### 10. Поддержка других платформ

**Статус**: Только YouTube

- [ ] **TikTok adapter**:
  - [ ] Реализация `TikTokAdapter`
  - [ ] Интеграция с TikTok API или scraping

- [ ] **Instagram adapter**:
  - [ ] Реализация `InstagramAdapter`
  - [ ] Интеграция с Instagram API

- [ ] **Другие платформы**:
  - [ ] Расширяемая архитектура для добавления новых платформ

### 11. Улучшение observability

**Статус**: Базовые метрики есть

- [ ] **Дополнительные метрики**:
  - [ ] Метрики для каждого типа ошибки
  - [ ] Метрики для latency каждого этапа
  - [ ] Метрики для использования ресурсов

- [ ] **Grafana Dashboard**:
  - [ ] Создать реальный dashboard в Grafana (сейчас только описание)
  - [ ] Импортировать готовый dashboard

**Чеклист**: Phase 2 — Dashboard

### 12. Улучшение безопасности

**Статус**: Базовая безопасность есть

- [ ] **Аутентификация для admin endpoints**:
  - [ ] Добавить аутентификацию для `/admin/*` endpoints
  - [ ] Интеграция с OAuth/JWT

- [ ] **Rate limiting для API**:
  - [ ] Защита API endpoints от злоупотреблений
  - [ ] IP-based rate limiting

### 13. Улучшение документации

**Статус**: Хорошая документация, но можно дополнить

- [ ] **API документация**:
  - [ ] OpenAPI/Swagger спецификация
  - [ ] Примеры запросов и ответов

- [ ] **Deployment guide**:
  - [ ] Подробное руководство по развёртыванию в production
  - [ ] Troubleshooting guide

- [ ] **Architecture diagrams**:
  - [ ] Визуальные диаграммы архитектуры
  - [ ] Sequence diagrams для основных потоков

## 📊 Приоритизация

### Высокий приоритет (для production readiness)

1. ✅ Завершение реализации тестов (unit, integration, chaos)
2. ✅ Реализация DataProcessor API интеграции (backpressure)
3. ✅ Kubernetes deployment манифесты
4. ✅ Определение платформы по URL

### Средний приоритет (для масштабирования)

5. Kafka event streaming
6. Централизованное логирование
7. Snapshot schedule (configurable)
8. Улучшение lifecycle cleanup

### Низкий приоритет (nice to have)

9. Поддержка других платформ
10. Улучшение observability
11. Улучшение безопасности
12. Улучшение документации

## 📝 Примечания

- Большинство критических функций уже реализованы
- Основные TODO связаны с тестированием и интеграцией
- Kubernetes манифесты частично есть, но требуют доработки
- Kafka и централизованное логирование - опциональные улучшения для production
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
