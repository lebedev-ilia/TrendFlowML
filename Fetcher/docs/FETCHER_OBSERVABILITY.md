## Observability Fetcher (Phase 2)

Этот документ описывает целевой дизайн наблюдаемости для Fetcher:

- Prometheus‑метрики;
- логи и событийный поток;
- требования к дашбордам Grafana.

Основан на чеклисте Phase 2 и общей архитектуре observability в проекте (см. DataProcessor/Backend).

---

## 1. Prometheus‑метрики

### 1.1. Pipeline metrics

Базовый набор:

- `fetcher_videos_downloaded_total` (Counter)
  - количество успешно скачанных видео;
  - лейблы: `platform`.
- `fetcher_videos_failed_total` (Counter)
  - количество неуспешных скачиваний/ingestion‑run’ов;
  - лейблы: `platform`, `reason`.
- `fetcher_cache_hits_total` / `fetcher_cache_miss_total` (Counter)
  - использование глобального кеша `(platform, platform_video_id)`;
  - лейблы: `platform`.
- `fetcher_download_latency_seconds` (Histogram)
  - время скачивания и upload’а видео;
  - лейблы: `platform`.
- `fetcher_metadata_latency_seconds` (Histogram)
  - время выполнения metadata worker’а;
  - лейблы: `platform`.
- `fetcher_comments_latency_seconds` (Histogram)
  - время выполнения comments worker’а;
  - лейблы: `platform`.

### 1.2. Platform error metrics

- `fetcher_youtube_429_total` (Counter)
  - количество ответов 429 / rate‑limit ошибок от YouTube;
  - лейблы: `operation` (`metadata`, `download`, `comments`).
- `fetcher_youtube_403_total` (Counter)
  - количество 403 ошибок (например, geo/age/bot);
  - лейблы: `operation`, `error_code`.
- `proxy_failure_rate` (Gauge)
  - доля неуспешных запросов по каждому прокси;
  - лейблы: `proxy_id`, `country`.
- `circuit_breaker_tripped_total` (Counter)
  - количество срабатываний circuit breaker’а;
  - лейблы: `operation`, `reason`.

### 1.3. Derived KPI

- **cache hit ratio**
  - вычисляется как `hits / (hits + misses)` по `fetcher_cache_hits_total` и `fetcher_cache_miss_total`;
- **ingestion throughput**
  - скорость обработки (videos/day, videos/minute) на основе `fetcher_videos_downloaded_total` и событий run’ов.

---

## 2. Логи и события

### 2.1. Fetcher logs

- Логи Fetcher должны:
  - быть структурированными (JSON) с полями:
    - `timestamp`, `level`, `message`;
    - `run_id`, `platform`, `platform_video_id`;
    - `stage` (`fetch_metadata`, `download_video`, `fetch_comments`, `artifact_builder`, ...).
  - писаться и в stdout (для агрегаторов логов), и в таблицу `fetch_logs` (для быстрой диагностики по run’у).

### 2.2. События pipeline

- События Fetcher описаны в `BACKEND_CONTRACTS.md` и моделях `FetcherEvent`.
- Для observability:
  - события `job.started`, `job.finished`, `job.failed` можно использовать для:
    - построения таймингов;
    - отслеживания статусов worker’ов;
    - построения алертов (частота `job.failed`).

---

## 3. Grafana‑дашборды

### 3.1. Основной дашборд Fetcher

Разделы:

- **Ingestion overview**
  - ingestion throughput (videos/minute, videos/hour);
  - доля успешных/ошибочных run’ов.
- **Queue / worker health** (после интеграции с очередью):
  - глубина очередей metadata/video/comments/finalize;
  - текущее число активных задач по типам.
- **Cache efficiency**
  - cache hit ratio;
  - динамика cache hits/misses.
- **Download performance**
  - гистограмма `fetcher_download_latency_seconds`;
  - распределение кодов ошибок (200/4xx/5xx) для download‑операций.
- **Platform health**
  - графики `fetcher_youtube_429_total`, `fetcher_youtube_403_total`;
  - `proxy_failure_rate` по прокси.

---

## 4. Связанные модули

- будущий модуль `fetcher/metrics.py`:
  - инициализация и экспорт всех Prometheus‑метрик;
  - функции‑обёртки для инкремента/тайминга в worker’ах.
- worker’ы:
  - `workers/metadata.py`, `workers/video.py`, `workers/comments.py`, `workers/artifacts.py` должны:
    - инкрементировать счётчики (успех/ошибка);
    - измерять latency основных операций;
    - логировать ключевые события в `fetch_logs`.

---

## 5. Связанные документы

- `Fetcher/docs/CORE_INGESTION.md` — логика metadata/video/comments workers и Artifact Builder.
- `Fetcher/docs/RATE_LIMITING_AND_LOCKS.md` — метрики ошибок и rate limit/circuit breaker.
- `backend/docs/EVENTS_AND_LOGGING.md` — существующая система событий и логов Backend.


