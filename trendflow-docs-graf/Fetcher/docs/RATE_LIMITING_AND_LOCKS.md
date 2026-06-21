## Rate limiting и distributed locks в Fetcher

Этот документ описывает целевой дизайн:

- Redis‑based rate limiting для запросов к платформам (в первую очередь YouTube);
- distributed locks для скачивания видео и upload’а артефактов.

Основан на `Fetcher/docs/plan.md`, раздел 5 (интеграция с YouTube, rate limits, proxy, circuit breaker) и разделе Phase 0 чеклиста.

---

## 1. Цели и ограничения

Fetcher должен:

- уважать ограничения платформ (YouTube rate limits, bot detection);
- не “стрелять себе в ногу” всплесками запросов с одного IP/прокси;
- избегать дублирующихся скачиваний/загрузок артефактов для одного и того же видео.

Решение:

- Redis‑rate‑лимитер (token bucket / leaky bucket) с ключами уровня IP/прокси и типа операции;
- Redis‑locks (SETNX/SET with NX + EX) для операций `download_video` и `upload_artifact`.

---

## 2. Rate limiting (Redis)

### 2.1. Ключи и уровни

Базовые ключи для YouTube:

- `rate:youtube:metadata:{ip_or_proxy_id}`
- `rate:youtube:download:{ip_or_proxy_id}`

Где:

- `{ip_or_proxy_id}` — идентификатор текущего исходящего IP или элемента proxy‑pool’а;
- тип операции (metadata vs download) разделён, чтобы отдельно контролировать лёгкие и тяжёлые запросы.

### 2.2. Простейшая реализация (counter + TTL)

Упрощённый алгоритм (подобие fixed window):

```python
def acquire_token(key: str, limit: int, window_sec: int) -> bool:
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window_sec)
    return count <= limit
```

Использование:

- перед вызовом `yt-dlp`/API для метаданных:
  - `acquire_token("rate:youtube:metadata:{ip}", limit=METADATA_LIMIT, window_sec=METADATA_WINDOW)`;
- перед скачиванием видео:
  - `acquire_token("rate:youtube:download:{ip}", limit=DOWNLOAD_LIMIT, window_sec=DOWNLOAD_WINDOW)`.

При `False`:

- шаг либо откладывается (retry с backoff), либо завершается с контролируемой ошибкой (`YOUTUBE_RATE_LIMITED`).

### 2.3. Конфигурация

Рекомендуемые параметры (MVP):

- метаданные:
  - `METADATA_LIMIT` ≈ 200–500 запросов / IP / час;
  - `METADATA_WINDOW` = 3600 сек.
- скачивания:
  - `DOWNLOAD_LIMIT` ≈ 50–100 запросов / IP / час;
  - `DOWNLOAD_WINDOW` = 3600 сек.

Фактические значения настраиваются через конфиг/переменные окружения Fetcher и зависят от прокси/аккаунта.

---

## 3. Distributed locks (Redis)

### 3.1. Lock для `download_video`

Цель: не скачивать одно и то же видео для одного и того же `(platform, platform_video_id)` параллельно на нескольких воркерах.

Ключ:

- `lock:video:{platform}:{platform_video_id}`

Алгоритм:

- при входе в `download_video`:
  - попытаться установить lock с TTL:

```python
def acquire_video_lock(platform: str, video_id: str, ttl_sec: int) -> bool:
    key = f"lock:video:{platform}:{video_id}"
    # Используем SET NX EX для атомарности
    return redis.set(key, "1", nx=True, ex=ttl_sec) is True
```

- если lock получен:
  - выполняем скачивание и upload;
  - по завершении:
    - при успехе — обновляем `artifacts` (`video_file` → `COMPLETED`);
    - снимаем lock (опционально, можно полагаться на TTL).
- если lock уже существует:
  - либо ждём/проверяем наличие готового артефакта (`artifacts`), периодически опрашивая БД;
  - либо сразу завершаем шаг, если артефакт уже появился.

### 3.2. Lock для upload артефактов

Цель: не допустить двойного upload’а одного и того же артефакта (особенно при retry/resume).

Ключ:

- `lock:artifact:{video_id}:{artifact_type}`

Использование:

- перед upload:
  - установить lock с небольшим TTL;
  - проверить, что в `artifacts` ещё нет записи `COMPLETED` для данного `(video_id, artifact_type)`;
  - выполнить upload и обновить запись в БД;
  - снять lock / дождаться TTL.

---

## 4. Интеграция с orchestration и адаптерами

### 4.1. Orchestrator

Orchestrator Fetcher:

- перед постановкой тяжёлых задач (например, скачивание видео) может дополнительно:
  - оценивать текущие значения rate‑лимитных ключей (для backpressure на ingestion‑уровне);
  - решать, ставить ли задачу сейчас или отложить/отклонить run.

### 4.2. Platform adapters

Адаптеры платформ (например, YouTubeAdapter) обязаны:

- перед каждым сетевым вызовом:
  - вызывать `acquire_token(...)` для соответствующего ключа;
  - корректно обрабатывать ситуацию `False` как “rate limited”.
- перед скачиванием:
  - получать lock `lock:video:{platform}:{platform_video_id}`;
  - корректно обрабатывать ситуацию, когда lock уже есть (ожидать/проверять наличие артефакта).

Таким образом, все платформенные операции централизованно проходят через слой rate limiting & locking.

---

## 5. Связанные компоненты (будущее развитие)

- **Proxy‑pool**:
  - таблица/конфиг `proxies` с `proxy_url`, `country`, `health_score`, `success_rate`, `last_used`;
  - интеграция rate‑лимитера на уровне прокси, а не только IP.
- **Circuit breaker**:
  - метрики по частоте 429/403/timeout ошибок за окно времени;
  - при превышении порога — временная блокировка операций до “остывания”.

Эти компоненты будут разрабатываться на более поздних фазах (Phase 2, Phase 7), но текущий дизайн rate‑лимитера и lock’ов не конфликтует с ними.

---

## 6. Связанные документы

- `Fetcher/docs/plan.md` — разделы про YouTube, прокси, anti‑rate‑limit.
- `Fetcher/docs/PLATFORM_ADAPTERS.md` — адаптеры платформ, которые используют rate limiting & locks.
- `Fetcher/docs/PIPELINE_ORCHESTRATION.md` — Orchestrator и state machine, куда будет интегрирован этот слой.
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
