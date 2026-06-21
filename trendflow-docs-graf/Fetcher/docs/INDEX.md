## Индекс документации Fetcher

Этот файл служит индексом для всей документации сервиса **Fetcher** — ingestion‑платформы для YouTube и других видеоплатформ.

Fetcher отвечает за надёжный и масштабируемый сбор видео, метаданных, временных снэпшотов и комментариев, а также подготовку артефактов и `manifest.json` для `DataProcessor`.

---

## Основные документы

- **[plan.md](./plan.md)** — целевая архитектура и high‑level план развития Fetcher
- **[checklist.md](./checklist.md)** — чеклист по фазам (Phase 0…7) для вывода Fetcher в продакшн
- **[BACKEND_CONTRACTS.md](./BACKEND_CONTRACTS.md)** — контракты между Backend, Fetcher и DataProcessor (run_id lifecycle, pipeline events, manifest, PlatformAdapter)
- **[FETCHER_OBSERVABILITY.md](./FETCHER_OBSERVABILITY.md)** — дизайн observability для Fetcher (метрики, логи, дашборды)
- **[GRAFANA_DASHBOARD.md](./GRAFANA_DASHBOARD.md)** — описание структуры Grafana dashboard для мониторинга Fetcher
- **[MONITORING_ALERTS.md](./MONITORING_ALERTS.md)** — описание monitoring alerts для Fetcher (Prometheus Alertmanager)
- **[QUEUE_ORCHESTRATION.md](./QUEUE_ORCHESTRATION.md)** — описание системы очередей и оркестрации для Fetcher (Celery + Redis)

---

## Отчёты о реализации

См. [IMPLEMENTATION/README.md](./IMPLEMENTATION/README.md)

### Phase 0 — Foundation

- [Stage 0.1: Backend contracts](./IMPLEMENTATION/2026-03-05-stage-0.1-backend-contracts.md) ⏳

---

## Быстрый поиск

### По типу документа

- **Архитектура**: `plan.md`, `BACKEND_CONTRACTS.md`
- **Чеклисты**: `checklist.md`
- **Реализация**: `IMPLEMENTATION/*.md`

### По фазе

- **Phase 0 — Foundation**: backend‑контракты, БД, storage, базовая оркестрация
- **Phase 1 — Core Ingestion Logic**: metadata/video/comments workers, artifact builder
- **Phase 2+**: observability, security, scalability, ML‑совместимость

---

## Обновление индекса

Этот индекс нужно обновлять при добавлении новых документов (особенно в `IMPLEMENTATION/`).
---

## Навигация

[Vault](../../docs/MAIN_INDEX.md)
