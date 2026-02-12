# Backend Q&A → правила/контракты/архитектура (TrendFlow)

Этот документ — **интерактивное интервью** для фиксации backend‑архитектуры TrendFlow.

Формат работы:
- Я задаю вопросы (раундами).
- Ты отвечаешь прямо под вопросами.
- Я **из ответов фиксирую правила/контракты** и (если нужно) задаю следующий раунд уточнений.

Цель: за несколько раундов получить **финальные или полуфинальные** правила для backend: границы сервисов, API, контракты данных, статусы, безопасность, биллинг, кэш/ретеншен.

Источники, на которые опираемся:
- `DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md`
- `DataProcessor/docs/contracts/*` (особенно `CONTRACTS_OVERVIEW.md`, `ARTIFACTS_AND_SCHEMAS.md`, `ORCHESTRATION_AND_CACHING.md`, `ERROR_HANDLING_AND_EDGE_CASES.md`, `PRIVACY_AND_RETENTION.md`, `PRODUCT_CONTRACT.md`)
- UI ожидания из `site/SITE_SPECIFICATION.md` (создание анализа, прогресс, результаты, ЛК).

---

## Уже зафиксировано из текущих документов (не спорим, если не нужно менять)

### A) Сервисы (MVP)
- **Backend API**: auth, профили анализа, запуск run, выдача результатов, биллинг/кредиты, админка.
- **Queue + workers**: постановка задач, retry, лимиты параллелизма.
- **DataProcessor worker**: оркестратор пайплайна 1 видео → 1 run.
- **PostgreSQL**: индекс артефактов/статусы/аудит/кэш LLM‑рендера (БД — ускоритель).
- **Object Storage**: dev = filesystem, prod/MVP = **MinIO (S3‑compatible)**.
- **Triton**: отдельный сервис model serving (DataProcessor/VisualProcessor — клиенты, не “поднимают” Triton).
- **LLM gateway**: единая точка вызова LLM (ключи/лимиты/логирование prompt_version).

### B) Identity, воспроизводимость, storage
- **`run_id` генерирует backend** и передаёт в DataProcessor.
- **`config_hash` вычисляет backend** детерминированно из нормализованного profile config и передаёт в DataProcessor.
- Source-of-truth по вычислениям: **NPZ артефакты + `manifest.json`** (БД — опциональна как индекс/ускоритель).
- Storage per run:
  - `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`
  - `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/*.npz`
- LLM: **только текст**, не источник чисел/структуры (числа/графики детерминированы из NPZ).

### C) Оркестрация, кэш, ошибки
- **Queue‑based протокол** backend → DataProcessor (sync endpoint только для dev/test и отключён в проде).
- Hybrid failures: **required = fail‑fast**, **optional = best‑effort** (run ok если required ok).
- Retry только transient errors (network/timeout/503/OOM c уменьшением batch), не retry для invalid/missing dependency/auth errors.
- `frames_dir` (union sampled frames) — рабочая директория, retention **7 дней**.
- Raw OCR/comments по умолчанию не храним; включение raw только после OAuth‑верификации владельца канала; есть hard cap на retention (`hard_cap_days=60`).

---

## Раунд 1 — вопросы по backend (ответь прямо под пунктами)

### 1) Backend стек и деплой
1.1) Какой backend фреймворк выбираем для MVP: **FastAPI** / Django / NestJS / другое?

Ответ: а какой ты считаешь лучшим для оей задачи

1.2) Деплой: один сервер (docker-compose) или сразу Kubernetes?

Ответ: сразу Kubernetes

### 2) Аутентификация и аккаунты
2.1) Какие методы auth в MVP: email+password, OAuth (Google/GitHub) — что делаем первым?

Ответ: все

2.2) Роли: нужны ли `admin`/`support`/`user`? Какие операции только для админа?

Ответ: смотря для чего эти роли. решим позже но 

### 3) Биллинг/кредиты (связка с запуском run)
3.1) Единицы списания: “кредиты” (условные units) или рубли/подписки сразу? Нужен ли **billing ledger** на MVP?

Ответ: пока оставляем так что 1 рубль = 1 кредиту (unit). насчет подписок решим позже. не понимаю что такое **billing ledger**

3.2) Когда списываем: при создании run, при старте обработки, или по факту завершения? Что делаем при fail до старта?

Ответ: списываем только по факту завершения, указаных в конфиги, компонентов. Что значит fail до старта.

### 4) Профили анализа (configs) и `config_hash`
4.1) Где хранится profile config: только в БД (Postgres) или ещё как файлы/seed?

Ответ: как лучше? я думал только в бд.

4.2) Нужны ли публичные профили (“наши”), плюс пользовательские — подтверждаешь?

Ответ: да

4.3) Правило `config_hash`: нормализация JSON (сортировка ключей, исключение полей типа `updated_at`) — ок?

Ответ: ок

### 5) Запуск анализа (Create run) — контракт API
5.1) Каноничный API: `POST /api/runs` создаёт run и ставит задачу в очередь — ок?

Ответ: ок, но нужно утвердить с чет что уже есть в /home/ilya/Рабочий стол/TrendFlowML/site

5.2) Вход для run:
- YouTube URL (`platform_id="youtube"`, `video_id` извлекаем/валидация)
- Upload (видео грузим, `video_id` внутренний)

Что из этого точно делаем в MVP первым?

Ответ: Все. при YouTube URL мы все парсим сами и сами задаем ид. При Upload пользователь задает все, в том числе и ид.

5.3) Где происходит download/preprocess (yt-dlp/ffmpeg): внутри DataProcessor worker, или отдельным “ingest” сервисом?

Ответ: Будет сервис Fetcher.

### 6) Очереди и воркеры
6.1) Что выбираем для MVP: **Celery + Redis**, RQ, Dramatiq, RabbitMQ? (важно: retries, rate limits, наблюдаемость)

Ответ: **Celery + Redis**

6.2) Гранулярность задач в MVP: **1 видео = 1 job** (внутри DataProcessor DAG) — подтверждаешь?

Ответ: да

### 7) Статусы/прогресс для UI (polling/WebSocket)
Сайт ожидает прогресс (карточки “Текущие”, progress bar, этапы, логи).

7.1) Подтверди минимальный список `run.status`:
- `queued`, `running`, `succeeded`, `failed`, `cancelled`
и список стадий `stage`: `segmenter`, `audio`, `text`, `visual`, `render`

Ответ: Да, но некоторые процессоры, компоненты будут выполняться параллельно + нужно подробнее в каждой стадии.

7.2) Как отдаём прогресс: только polling (`GET /api/runs/{run_id}/status`) или добавляем WebSocket/SSE в MVP?

Ответ: WebSocket/SSE

7.3) Нужна ли пользователю “консоль логов” в UI: отдаём tail логов из backend (храним в БД) или позже?

Ответ: Нужна 

### 8) Выдача результатов (NPZ/manifest → JSON для фронта)
8.1) Backend отдаёт:
- `GET /api/runs/{run_id}/result` → JSON (агрегаты/карточки/графики) детерминированно из NPZ
- `GET /api/runs/{run_id}/manifest` → `manifest.json`
- `GET /api/runs/{run_id}/artifacts/*` → подписанные ссылки на NPZ (для админа/дебага)

Какие из этих endpoints нужны в MVP?

Ответ: все

8.2) Доступ к NPZ: обычным пользователям даём скачивание NPZ или скрываем (только JSON)?

Ответ: только json

### 9) Кэширование (heavy vs light)
9.1) Подтверди: heavy compute кэшируем по ключу (video + версии + sampling + config_hash), а light personalization/LLM рендер — отдельно.

Ответ: да

9.2) TTL кэша heavy compute настраиваемый (по умолчанию 3 дня) — ок?

Ответ: ок

### 10) Privacy/удаление/retention
10.1) Нужно ли в MVP реализовать “Delete video data” по запросу пользователя (удалить артефакты + индексы + LLM cache)?

Ответ: да

10.2) Политика raw OCR/comments: по умолчанию off; включение только после OAuth‑верификации владельца канала; hard cap 60 дней — подтверждаешь?

Ответ: да

---

## Раунд 1 — зафиксированные решения (на основе ответов)

### 1) Deployment
- **Выбор**: деплой **сразу Kubernetes** (MVP).
- **Следствие**: все сервисы должны иметь readiness/liveness, конфиги через env/secrets/configmaps, миграции БД как отдельный job.

### 2) Auth (MVP)
- **Выбор**: делаем **все** методы авторизации в MVP:
  - email+password
  - OAuth (Google, GitHub)
- **Открыто**: роли и права (`admin`/`support`) — уточнить в Раунде 2.

### 3) Billing/кредиты
- **Единица**: **1 рубль = 1 кредит (unit)** (подписки позже).
- **Списание**: **по факту завершения** компонентов, включённых в конфиг профиля.
- **Следствие**: нужен расчёт стоимости на уровне профиля/компонента и правила “что считается завершением” (ok/empty/error).
- **Термин**: *billing ledger* = таблица/журнал неизменяемых транзакций (начисления/списания/рефанды) для аудита.
- **Открыто**:
  - делаем ли ledger в MVP (я рекомендую: да, минимальный),
  - что делать при “fail до старта” (валидатор/скачивание/очередь) и при частичных ошибках (optional).

### 4) Профили анализа (`profile_config`) и `config_hash`
- **Хранение**: profile configs **в Postgres**.
- **Типы**: есть **наши публичные профили** + **пользовательские**.
- **`config_hash`**: нормализуем JSON (stable sort keys, исключаем “мета” поля типа `updated_at`) — **ок**.

### 5) Запуск анализа (Runs)
- **API**: `POST /api/runs` создаёт run и ставит задачу в очередь — **ок** (нужно совместить с UI сайта).
- **Источники**: MVP включает **оба** сценария:
  - YouTube URL: backend парсит/валидирует.
  - Upload: пользователь задаёт метаданные (включая “id” — это нужно уточнить; см. Раунд 2).
- **Fetcher**: download/preprocess выносится в **отдельный сервис Fetcher**.

### 6) Очередь и воркер‑модель
- **Queue**: **Celery + Redis** (MVP).
- **Гранулярность**: **1 видео = 1 job** (внутри DataProcessor DAG) — подтверждено.

### 7) Прогресс и логи для UI
- **Статусы run**: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
- **Стадии**: `segmenter`, `audio`, `text`, `visual`, `render` (но внутри стадий компоненты могут идти параллельно → нужна детализация).
- **Transport**: добавляем **WebSocket или SSE** в MVP (уточнить выбор и формат событий).
- **Logs**: “консоль логов” в UI **нужна** (значит нужен поток событий + storage/retention).

### 8) Результаты (JSON/manifest/artifacts)
- **Нужны endpoints**: все перечисленные:
  - `GET /api/runs/{run_id}/result` → JSON (deterministic from NPZ)
  - `GET /api/runs/{run_id}/manifest` → manifest.json
  - `GET /api/runs/{run_id}/artifacts/*` → ссылки/доступ к NPZ (для админа/дебага)
- **Политика доступа**: обычным пользователям **только JSON**, NPZ не отдаём напрямую.

### 9) Cache
- **Разделение**: heavy compute отдельно от light personalization/LLM — **да**.
- **TTL heavy**: настраиваемый, дефолт **3 дня** — **ок**.

### 10) Privacy / delete
- **Delete** по запросу пользователя в MVP: **да** (артефакты + индексы + LLM cache).
- **Raw OCR/comments**: по умолчанию off; включение только после OAuth‑верификации владельца канала; hard cap 60 дней — **да**.

---

## Раунд 2 — уточняющие вопросы (ответь прямо под пунктами)

### 1) Backend framework (final)
1.1) Я предлагаю **FastAPI** (максимально естественно к Python‑стеку DataProcessor + быстрый старт + async для SSE/интеграций). Подтверждаешь FastAPI как основной backend API?

Ответ: да

### 2) Роли и права доступа
2.1) Подтверди минимальные роли:
- `user` (по умолчанию)
- `admin` (доступ к артефактам NPZ, админ‑панель, управление профилями “наших” конфигов)
- `support` (просмотр runs пользователей и логов, но без скачивания NPZ и без изменения биллинга) — нужно?

Ответ: да

### 3) Billing ledger и списание “по факту завершения”
3.1) Подтверди: делаем **billing ledger** в MVP (журнал транзакций), чтобы баланс не был “магическим числом”.

Ответ:

3.2) Что считается “завершением компонента” для списания:
- `status=ok` списываем
- `status=empty` списываем или нет?
- `status=error` списываем или нет?

Ответ: `status=ok` списываем, `status=empty` такого не может быть так как модель так или иначе что то выдаст, `status=error` не списываем. 

3.3) “Fail до старта” — это когда run не дошёл до реальной обработки (например: валидация URL/формата не прошла, не скачалось видео, job не был принят очередью, или пользователь отменил в очереди). Подтверди: **в этих случаях списание = 0**.

Ответ: да

3.4) Недостаточно средств:
- (A) запрещаем запуск `POST /api/runs` если баланс < estimated_cost
- (B) разрешаем поставить в очередь, но не стартуем пока не пополнит
- (C) стартуем, но ограничиваем компоненты (опасно)

Выбери A/B/C.

Ответ: A. При выборе конфига там указаны желаемые алгоритмы (фичи) и сумма к каждой, если нехватает средств мы просто говрим что бы пользователь выбрал другой конфиг или пополнил баланс.

### 4) `video_id` и Upload (безопасность + дедуп)
4.1) Для YouTube: `video_id` = ID из URL (канонично) — ок.

Для upload: я **не рекомендую** позволять пользователю задавать `video_id` (коллизии, доступ к чужим данным). Подтверди правило:
- upload `video_id` генерирует backend (UUID/ULID)
- пользователь задаёт только title/desc/tags/lang/category и т.п.

Ответ: ок

4.2) Хотим ли дедуп upload по `file_hash` (если два раза загрузили один и тот же файл) или всегда считаем разными видео?

Ответ: Нужно просто единое хранилище загруженых видео, а в ЛК пользователя мы просто храним ид по которому можем найти в хранилище. Так что если 1 человек когда то загружал видео, то если 2 человек попытаеться загрузить его же мы просто сообщим что оно уже есть на сервере или без сообщения просто очень быстро отабразим в ЛК.

### 5) Fetcher service — границы ответственности
5.1) Fetcher делает только download/preprocess видео (yt-dlp/ffmpeg), или ещё и тянет мета/комменты YouTube?

Ответ: Fetcher делает все что связано со сбором данных о видео: мета, скачивание, комменты, данные канала и тд.

5.2) Как Fetcher отдаёт результат backend’у:
- (A) backend вызывает Fetcher синхронно (HTTP), Fetcher пишет `video.mp4` в MinIO и возвращает `object_key`
- (B) Fetcher сам worker (читает очередь) и пушит событие “downloaded”

Выбери A/B.

Ответ: Реши сам как правильнее, качественее

5.3) Где живёт временное сырьё (`/tmp/.../video.mp4`) и кто гарантирует cleanup/TTL?

Ответ: сначала проверка на наличие в нашем внешнем хранилище, если нет то локально в tmp качаем, после всех обработок встает на очередь на выгрузку во внешнее хранилище. cleanup/TTL - 60 дней. Но тут важно понимать что мы делаем если автор удалил видео на платформе или его заблокировано, но оно уже у нас в хранилище.

### 6) Progress transport: WebSocket vs SSE + schema событий
6.1) Выбираем одно для MVP:
- (A) SSE (проще, отлично для “стрима событий” и без stateful WS)
- (B) WebSocket (двустороннее, пригодится для “cancel”/“tail logs”, но сложнее)

Выбери A/B.

Ответ: WebSocket

6.2) Подтверди минимальный формат событий прогресса (могу зафиксировать как контракт):
- `run.stage_changed`
- `component.started`
- `component.finished` (status ok/empty/error + duration)
- `log.line` (level + message)

Ок?

Ответ: Ок

6.3) Хранение логов:
- (A) сохраняем в Postgres последние N строк (например 5k) + стримим
- (B) только стримим, без хранения (плохо для “зайти позже”)

Выбери A/B и N (если A).

Ответ: Логов будет не так много (100 - 120 сообщений а все видео), так что держим все в БД.

### 7) Детализация стадий и параллелизм
7.1) Хочешь отображать прогресс как:
- (A) только 5 стадий (segmenter/audio/text/visual/render) + общий процент
- (B) стадии + список компонентов (галочки/тайминги) как “pipeline timeline”

Выбери A/B (или “оба, но B в деталях”).

Ответ: B

### 8) Security между сервисами (K8s)
8.1) MVP в k8s: сервис‑к‑сервису:
- (A) mTLS сразу
- (B) API key/JWT между сервисами, mTLS позже

Выбери A/B.

Ответ: mTLS

---

## Раунд 2 — зафиксированные решения (на основе ответов)

### 1) Backend framework
- **Выбор**: **FastAPI** как основной backend API.

### 2) Access control / роли
- **Роли**:
  - `user`
  - `admin`
  - `support` (нужна)

### 3) Billing (уточнения по списанию)
- **Недостаточно средств**: стратегия **A** — запрещаем `POST /api/runs` если `balance < estimated_cost`, предлагаем пополнить или выбрать другой профиль.
- **Списание по результату компонента**:
  - `status=ok` → списываем
  - `status=error` → не списываем
  - `status=empty` → по твоим словам “не бывает”, но это **конфликтует с текущими контрактами** (см. Раунд 3).
- **Fail до старта** (валидация/скачивание/очередь/отмена в очереди) → **списание = 0**.

### 4) Upload `video_id` + дедуп
- **Upload `video_id`**: генерирует backend (UUID/ULID); пользователь задаёт только мета‑поля.
- **Дедуп upload**: хотим **единое хранилище** загруженных видео (dedup на уровне файла) и “привязки” в ЛК пользователя.
  - Следствие: нужен `file_hash` (или content digest) + таблица “ownership/linking”, чтобы пользователь **не мог получить доступ** к чужому upload только по совпадению хэша.

### 5) Fetcher service
- **Роль Fetcher**: отвечает **за весь сбор данных о видео**:
  - мета YouTube, данные канала
  - комментарии
  - скачивание видео (yt-dlp)
  - preprocess (ffmpeg)
- **Промежуточное сырьё**: проверяем внешнее хранилище → иначе качаем в `/tmp` → после обработок ставим на очередь выгрузку во внешнее хранилище.
- **Retention для сырья/копий видео**: **60 дней** (но нужно уточнить policy при удалении/блокировке видео на платформе; см. Раунд 3).

### 6) Progress streaming + logs
- **Transport**: **WebSocket** в MVP.
- **События**: формат событий прогресса (stage/component/log) — **ОК**.
- **Logs**: логов мало (~100–120 на видео) → храним **все** в БД (и стримим через WS).

### 7) UI прогресс
- **Отображение**: вариант **B** — pipeline timeline (стадии + список компонентов с таймингами/статусами).

### 8) Security в k8s
- **Сервис‑к‑сервису**: **mTLS сразу**.

---

## Раунд 3 — уточняющие вопросы (точечно, чтобы закрыть контракты)

### 1) Billing ledger (ответ не заполнен в Раунде 2)
1.1) Подтверди: **делаем billing ledger в MVP** (журнал транзакций), да/нет?

Ответ: Да

### 2) Конфликт: `status=empty` в контрактах
В `docs/contracts/ARTIFACTS_AND_SCHEMAS.md` и `CONTRACTS_OVERVIEW.md` явно сказано: **empty outputs валидны** (NaN + masks + `empty_reason`) и `status=empty` предусмотрен.

2.1) Подтверждаешь, что **`status=empty` остаётся в системе** (например: видео без аудио, нет лиц, нет текста, отключено политикой), даже если “модель всегда что-то выдаст”?

Ответ: `status=empty` оставляем, но в логах явно говорим что качество может ухудшиться

2.2) Если `status=empty` существует, как списываем:
- (A) списываем как `ok` (ресурсы потрачены)
- (B) не списываем (потому что “нет результата”)
- (C) зависит от причины (`empty_reason`) / от required vs optional

Выбери A/B/C.

Ответ: A

### 3) Fetcher интеграция (A/B — решение за мной, но хочу зафиксировать контракт)
С учётом Kubernetes + Celery + Redis предлагаю вариант **B**:
- backend создаёт run и кладёт job `fetch_video(run_id)` в очередь
- Fetcher worker выполняет сбор, кладёт сырьё/видео в MinIO, обновляет статус run, и ставит следующий job `process_run(run_id)` для DataProcessor
- все взаимодействия защищены mTLS

3.1) Подтверждаешь этот вариант (B) как стандарт?

Ответ: Да

### 4) Retention сырого видео и “удалено/заблокировано на платформе”
4.1) Для **YouTube URL**: мы действительно хотим хранить **копию видео** до 60 дней, или лучше хранить только derived NPZ/manifest, а `video.mp4` удалять быстро (например 24 часа) после обработки?

Ответ: Согласен с тобой

4.2) Если видео **удалено/заблокировано** на платформе, но уже есть у нас:
- (A) удаляем копию немедленно (safe-by-default)
- (B) держим до TTL (60 дней)
- (C) держим только если владелец канала верифицирован и включил опцию

Выбери A/B/C.

Ответ: Реши сам

### 5) Upload storage: права доступа при дедуп
5.1) Подтверди правило доступа:
- Даже если файл уже есть в storage (по `file_hash`), пользователь может получить к нему доступ **только** через запись “user_video link” в БД (и только к своим видео).

Ответ: Да

---

## Раунд 3 — зафиксированные решения (на основе ответов)

### 1) Billing ledger
- **Выбор**: billing ledger **делаем в MVP**.

### 2) `status=empty` (валидное состояние)
- **`status=empty` остаётся** в системе (как в существующих контрактах).
- **Коммуникация**: если required компонент = empty → это должно явно отражаться:
  - в логах/событиях
  - в UI предупреждением “качество может ухудшиться”
- **Списание**: `status=empty` оплачивается как `ok` (вариант **A**), т.к. ресурсы потрачены.

### 3) Fetcher orchestration
- **Выбор**: вариант **B** (Fetcher worker в очереди) — стандарт:
  - backend создаёт run → ставит `fetch_video(run_id)`
  - Fetcher кладёт сырьё/видео/мету в MinIO/БД → ставит `process_run(run_id)` для DataProcessor

### 4) Retention `video.mp4` + takedown policy
- **YouTube URL**: `video.mp4` **не храним 60 дней по умолчанию**. Храним только derived NPZ/manifest; `video.mp4` удаляем быстро после обработки (например, TTL 24 часа) — ты согласен с этим подходом.

- **Если видео удалено/заблокировано на платформе**:
  - **Выбор (C)**: храним копию видео **только если** владелец канала верифицирован (OAuth) **и** включил явную опцию retention для сырого видео; иначе удаляем копию (safe-by-default).
  - Derived NPZ/manifest остаются (если пользователь не запросил delete), но любой “raw” слой подчиняется privacy/retention политике.

### 5) Upload dedup ACL
- **Правило доступа подтверждено**: даже при совпадении `file_hash` доступ даётся только через запись “user_video link” в БД.

---

## Раунд 4 — финализация контрактов (DB + API + WS)

### 1) DB schema (минимальный набор таблиц)
Подтверди, что в MVP делаем минимум такие сущности (названия можно менять, смысл важен):
1) `users` (auth, роль, баланс)
2) `videos` (platform_id, video_id, source_type, мета, owner_user_id nullable)
3) `video_sources` (youtube_url / upload_object_key / hashes / durations) — можно слить в `videos`, но лучше отдельно
4) `analysis_profiles` и `profile_components` (configs)
5) `runs` (run_id, video_id, user_id, config_hash, status, stage, created/finished)
6) `run_components` (per component status/timing/schema_version/producer_version/error)
7) `artifacts` (paths/digests/sizes, component, run_id)
8) `run_logs` (run_id, ts, level, message) — раз логов мало, храним полностью
9) `billing_ledger` (immutable записи) + (опционально) `balance_snapshots`
10) `render_cache` (LLM output cache)
(да/нет + что убрать/добавить) 

Ответ: Да

### 2) Billing ledger — модель транзакций
2.1) Подтверди типы транзакций (минимум):
- `topup` (пополнение)
- `hold` (резерв под run) — опционально
- `charge` (списание за компоненты)
- `refund` (возврат/коррекция)

Ответ: Да

2.2) Нужен ли “hold” (резерв) или достаточно списывать в конце (charge) и просто запрещать запуск если нет средств (как мы решили в 3.4)?

Ответ: Да

### 3) API endpoints (MVP)
Подтверди базовый список:
- `POST /api/auth/*` (email+password + oauth callbacks)
- `GET /api/me`
- `GET /api/profiles` (наши) + `GET/POST/PUT/DELETE /api/my/profiles`
- `POST /api/videos/upload/init` + `PUT /api/videos/upload/{id}` + `POST /api/videos/upload/complete` (или проще — single endpoint; уточним)
- `POST /api/runs` (create run)
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs/{run_id}` (status + summary)
- `GET /api/runs/{run_id}/manifest`
- `GET /api/runs/{run_id}/result`
- `GET /api/runs/{run_id}/events` (WebSocket upgrade)
- `GET /api/runs` (list “текущие/завершённые”)
- `DELETE /api/videos/{video_id}` (delete request: data removal)

Ответ: (ок / правки) ок

### 4) WebSocket contract (финализируем)
4.1) Ок ли такой каноничный envelope для всех событий:
- `type` (строка)
- `run_id`
- `ts` (UTC ISO)
- `payload` (object)

Ответ: ок

4.2) Нужен ли “replay” при переподключении (клиент получил последние N событий/логов)?
Варианты:
- (A) да: `GET /api/runs/{run_id}/logs` + WS только “live”
- (B) да: WS умеет `since_ts` и отдаёт missed события
- (C) нет: только live WS (не рекомендую)

Выбери A/B/C.

Ответ: A

### 5) Cancel semantics
5.1) Если пользователь нажал “Отменить анализ”, что делаем:
- (A) мягкая отмена: прекращаем дальнейшие компоненты, сохраняем уже посчитанное (для дебага), списание = 0
- (B) жёсткая отмена: останавливаем и удаляем всё сразу, списание = 0
- (C) отмена с частичным списанием за уже выполненные компоненты

Выбери A/B/C.

Ответ: A. Списываем за прошелдшие алгоритмы (предупреждаем перед стратом)

---

## Раунд 4 — зафиксированные решения (на основе ответов)

### 1) DB schema
- **Подтверждено**: список таблиц из Раунда 4 принимаем для MVP (можно менять названия, смысл фиксируем).

### 2) Billing ledger — транзакции
- **Типы транзакций**: `topup`, `hold` (опционально), `charge`, `refund` — **ок**.
- **Открыто (нужна конкретика)**: нужен ли `hold` в MVP или достаточно `charge` в конце + запрет запуска при нехватке средств (см. Раунд 5).

### 3) API endpoints
- **Подтверждено**: список endpoints из Раунда 4 — **ок**.

### 4) WebSocket contract
- **Envelope**: `type`, `run_id`, `ts`, `payload` — **ок**.
- **Replay**: выбран вариант **A**:
  - отдельный `GET /api/runs/{run_id}/logs` (и/или events history)
  - WS = только “live”.

### 5) Cancel semantics (конфликт)
Ты выбрал (A), но дописал “списываем за прошедшие алгоритмы”, что соответствует (C).
- **Открыто**: уточняем в Раунде 5.

---

## Раунд 5 — 2 уточнения (закрываем финальные противоречия)

### 1) Нужен ли `hold` (резерв) в billing ledger?
Сейчас у нас уже есть правило “не создаём run, если balance < estimated_cost”.

1.1) Подтверди одно:
- (A) **без hold**: только `topup/charge/refund`, списание идёт по факту завершённых компонентов
- (B) **с hold**: при `POST /api/runs` создаём `hold(estimated_cost)` и потом:
  - при успехе конвертим в `charge(actual_cost)` + отпускаем остаток
  - при fail/cancel делаем `release/refund`

Выбери A/B.

Ответ: B

### 2) Cancel billing (как именно списываем при отмене)
Ты хочешь “мягкую отмену” (не делать дальнейшие компоненты), но при этом “списывать за уже выполненные алгоритмы”.

2.1) Подтверди модель отмены:
- (A) cancel = стоп пайплайна + **списание 0**
- (C) cancel = стоп пайплайна + **списываем только за компоненты, которые уже успели завершиться (`ok`/`empty`)**

Выбери A/C.

Ответ: C

2.2) Если выбрал (C): отмена разрешена на любых стадиях, или запрещаем отмену после точки “невозврата” (например, после начала heavy compute)?
- (A) разрешаем всегда
- (B) запрещаем после `segmenter` (или после начала `visual`)

Выбери A/B.

Ответ: A

## После Раунда 4 (следующий шаг)
- Превращу всё в “полуфинальные” backend контракты: DDL-черновик таблиц + OpenAPI-черновик endpoints + WS event spec.


---

## Полуфинальные backend контракты (DDL + API + WS) — Round 6 (компиляция)

Этот раздел — **конденсат** решений из Раундов 1–5 в формате, пригодном для реализации.

### 0) Канонические идентификаторы и ключи идемпотентности
- **`platform_id`**: строка, пока фиксируем `youtube`.
- **`video_id`**:
  - YouTube: каноничный ID из URL (`dQw4w9WgXcQ`).
  - Upload: **генерирует backend** (`UUID`/`ULID`), пользователь не задаёт.
- **`run_id`**: генерирует backend (UUID).
- **`config_hash`**: детерминированный хэш нормализованного profile config (stable JSON).
- **Idempotency**:
  - `POST /api/runs`: принимает заголовок `Idempotency-Key` (строка). Повтор с тем же ключом должен вернуть тот же `run_id` (если payload совпадает) или 409 (если payload отличается).
  - Upload complete: аналогично (чтобы “двойной клик” не создавал дубликаты).

---

### 1) Биллинг: единицы, hold, charge, refund (правила)
- **Единица**: `unit` (кредит), сейчас трактуем как “1 unit = 1 ₽” (пока без подписок).
- **Перед запуском**:
  - backend вычисляет `estimated_cost_units` из профиля (сумма component_cost_units).
  - если `available_balance_units < estimated_cost_units` → 409/402 (см. API ошибки) и run не создаём.
- **При создании run**:
  - создаём **hold** на `estimated_cost_units` (резерв).
- **Во время выполнения**:
  - за компонент списываем **только если он завершился** `ok` или `empty`.
  - `error` не списываем.
- **При завершении/отмене/ошибке**:
  - делаем `charge(actual_cost_units)` (фактически выполненные `ok/empty` компоненты).
  - освобождаем остаток hold (`release = estimated - actual`).
- **Cancel**:
  - пользователь может отменить **в любой момент**
  - списание = только за компоненты, которые успели завершиться (`ok/empty`).

Примечание: `empty` — валидный исход (контракты NPZ/manifest), и он **оплачивается** как `ok`.

---

### 2) Status model (run + component)
#### 2.1 Run status
- `queued` → `running` → (`succeeded` | `failed` | `cancelled`)

#### 2.2 Run stage (для UI)
- `segmenter`, `audio`, `text`, `visual`, `render`

#### 2.3 Component status
- `queued`, `running`, `ok`, `empty`, `error`, `skipped` (skipped = выключено профилем или не требуется)

---

### 3) DDL-черновик (PostgreSQL)
Это **полуфинальный** SQL-скелет (без мелких деталей), чтобы закрепить сущности/связи/индексы.

```sql
-- Users / auth
create table users (
  id uuid primary key,
  email text unique,
  password_hash text,
  role text not null default 'user', -- user|admin|support
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table user_balances (
  user_id uuid primary key references users(id) on delete cascade,
  balance_units bigint not null default 0,
  updated_at timestamptz not null default now()
);

-- Videos
create table videos (
  id uuid primary key,
  platform_id text not null, -- youtube|upload (или platform отдельно)
  video_id text not null,    -- youtube video id OR generated id for upload
  source_type text not null, -- youtube_url|upload
  title text,
  description text,
  language text,
  category text,
  created_at timestamptz not null default now(),
  unique(platform_id, video_id)
);

-- Physical files (dedup by hash), shared across users (but access is via links)
create table video_files (
  id uuid primary key,
  sha256_hex text not null unique,
  size_bytes bigint not null,
  mime_type text,
  object_key text not null, -- MinIO key for the raw mp4 (if retained)
  created_at timestamptz not null default now(),
  retention_until timestamptz -- null => not retained / deleted quickly
);

-- Link: which user has access/ownership to which upload
create table user_video_links (
  user_id uuid not null references users(id) on delete cascade,
  video_id uuid not null references videos(id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (user_id, video_id)
);

-- Optionally link video->file (youtube may be null if we don't keep mp4)
create table video_sources (
  video_id uuid primary key references videos(id) on delete cascade,
  youtube_url text,
  uploaded_file_id uuid references video_files(id),
  fetched_at timestamptz,
  duration_sec int,
  width int,
  height int
);

-- Analysis profiles
create table analysis_profiles (
  id uuid primary key,
  user_id uuid references users(id), -- null => "our" public profile
  name text not null,
  description text,
  is_public boolean not null default false,
  config_json jsonb not null,
  config_hash text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index analysis_profiles_user_hash_uq on analysis_profiles(user_id, config_hash);

create table profile_components (
  profile_id uuid not null references analysis_profiles(id) on delete cascade,
  component_name text not null,
  enabled boolean not null default true,
  required boolean not null default true,
  component_params jsonb not null default '{}'::jsonb,
  cost_units bigint not null default 0,
  primary key (profile_id, component_name)
);

-- Runs
create table runs (
  id uuid primary key,              -- run_id
  user_id uuid not null references users(id),
  video_id uuid not null references videos(id),
  profile_id uuid references analysis_profiles(id),
  config_hash text not null,
  status text not null,             -- queued|running|succeeded|failed|cancelled
  stage text,                       -- segmenter|audio|text|visual|render
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  cancel_requested_at timestamptz,
  error_code text,
  error_message text,
  estimated_cost_units bigint not null default 0,
  actual_cost_units bigint not null default 0
);
create index runs_user_created_idx on runs(user_id, created_at desc);
create index runs_video_created_idx on runs(video_id, created_at desc);

create table run_components (
  run_id uuid not null references runs(id) on delete cascade,
  component_name text not null,
  status text not null, -- queued|running|ok|empty|error|skipped
  schema_version text,
  producer_version text,
  started_at timestamptz,
  finished_at timestamptz,
  duration_ms int,
  device_used text,
  empty_reason text,
  error_code text,
  error_message text,
  cost_units bigint not null default 0,
  primary key (run_id, component_name)
);

-- Artifacts index
create table artifacts (
  id uuid primary key,
  run_id uuid not null references runs(id) on delete cascade,
  component_name text not null,
  kind text not null,        -- npz|manifest|npy|other
  object_key text not null,  -- MinIO key
  size_bytes bigint,
  sha256_hex text,
  created_at timestamptz not null default now()
);
create index artifacts_run_component_idx on artifacts(run_id, component_name);

-- Logs (store fully, ~100-120 lines per run)
create table run_logs (
  id bigserial primary key,
  run_id uuid not null references runs(id) on delete cascade,
  ts timestamptz not null default now(),
  level text not null, -- info|warning|error|debug
  message text not null
);
create index run_logs_run_ts_idx on run_logs(run_id, ts asc);

-- Billing ledger (immutable)
create table billing_ledger (
  id uuid primary key,
  user_id uuid not null references users(id),
  run_id uuid references runs(id),
  type text not null,          -- topup|hold|release|charge|refund
  amount_units bigint not null, -- positive for topup/refund/release, negative for hold/charge
  created_at timestamptz not null default now(),
  meta jsonb not null default '{}'::jsonb
);
create index billing_ledger_user_created_idx on billing_ledger(user_id, created_at desc);

-- LLM render cache
create table render_cache (
  id uuid primary key,
  video_id uuid not null references videos(id) on delete cascade,
  run_id uuid not null references runs(id) on delete cascade,
  locale text not null,
  llm_model text not null,
  prompt_hash text not null,
  text_md text not null,
  created_at timestamptz not null default now(),
  unique(video_id, run_id, locale, llm_model, prompt_hash)
);
```

---

### 4) REST API contract (полуфинал)

#### 4.1 Auth
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/oauth/{provider}/start`
- `GET /api/auth/oauth/{provider}/callback`
- `POST /api/auth/logout`
- `GET /api/me`

#### 4.2 Profiles
- `GET /api/profiles` — публичные “наши”.
- `GET /api/my/profiles`
- `POST /api/my/profiles`
- `PUT /api/my/profiles/{profile_id}`
- `DELETE /api/my/profiles/{profile_id}`

#### 4.3 Upload video
- `POST /api/videos/upload/init` → `{upload_id, video_id, presigned_put_url?}`
- `PUT /api/videos/upload/{upload_id}` (или S3 direct upload)
- `POST /api/videos/upload/complete` (Idempotency-Key)
  - server: вычисляет `sha256`, делает dedup через `video_files.sha256_hex`, создаёт `videos` + `video_sources` + `user_video_links`.

#### 4.4 Runs
- `POST /api/runs` (Idempotency-Key)
  - input: `source` = youtube_url|upload_video_id, `profile_id`
  - server:
    - проверка баланса vs `estimated_cost_units`
    - `runs` + `run_components` (план)
    - `billing_ledger: hold(-estimated_cost_units)`
    - Celery job: `fetch_video(run_id)` (Fetcher)
- `GET /api/runs` (list; фильтры: status, date range, profile, video)
- `GET /api/runs/{run_id}` (status summary, costs, stage, timestamps)
- `POST /api/runs/{run_id}/cancel`
  - cancel всегда доступен; гарантируем best-effort stop + billing по completed components.
- `GET /api/runs/{run_id}/manifest`
- `GET /api/runs/{run_id}/result` (JSON for UI)
- `GET /api/runs/{run_id}/logs` (replay/history)
- `GET /api/runs/{run_id}/events` (WebSocket)

#### 4.5 Delete
- `DELETE /api/videos/{video_id}`
  - удаляет:
    - result_store artifacts + manifest
    - render_cache
    - DB index rows (runs/components/artifacts/logs)
    - raw video copies (если есть)

#### 4.6 Ошибки (минимальный словарь)
- `401 unauthorized` / `403 forbidden`
- `404 not_found` (run/video/profile)
- `409 insufficient_funds` (нет средств под estimated_cost_units)
- `409 idempotency_conflict` (Idempotency-Key reuse with different payload)
- `422 validation_error` (плохой URL/формат)
- `503 service_unavailable` (Fetcher/DataProcessor backlog/maintenance)

---

### 5) WebSocket event spec (live-only) + replay через REST (A)

#### 5.1 WS connect
- `GET /api/runs/{run_id}/events` (upgrade to WS)
- auth: bearer token (поверх TLS; сервисы внутри k8s общаются по mTLS)
- WS даёт только **live** события; для истории:
  - `GET /api/runs/{run_id}/logs`
  - `GET /api/runs/{run_id}` (current status snapshot)

#### 5.2 Envelope (подтверждено)
```json
{
  "type": "component.finished",
  "run_id": "uuid",
  "ts": "2026-01-10T12:34:56Z",
  "payload": {}
}
```

#### 5.3 Event types (минимум)
- `run.stage_changed`:
  - payload: `{stage, progress_pct?}`
- `component.started`:
  - payload: `{component_name, kind, required}`
- `component.finished`:
  - payload: `{component_name, status, duration_ms, empty_reason?, error_code?, error_message?, cost_units}`
- `log.line`:
  - payload: `{level, message}`
- `run.status_changed`:
  - payload: `{status, error_code?, error_message?}`
- `billing.updated` (опционально, но удобно для UI):
  - payload: `{estimated_cost_units, actual_cost_units, hold_units, charged_units, released_units}`

---

### 6) Fetcher/DataProcessor orchestration (Celery) — зафиксированный поток
- `fetch_video(run_id)` (Fetcher):
  - проверяет “есть ли уже video_files для upload” / “есть ли youtube cached artifact”
  - тянет YouTube meta/channel/comments
  - скачивает mp4 во временный `/tmp`
  - кладёт нужные сырые данные (видео/мета/комменты) в MinIO (с retention policy)
  - апдейтит `runs.stage/status` + пишет логи
  - ставит задачу `process_run(run_id)`
- `process_run(run_id)` (DataProcessor):
  - читает payload/paths из DB/MinIO
  - выполняет DAG, пишет NPZ + manifest, обновляет `run_components`
  - по завершению: считает `actual_cost_units`, пишет `billing_ledger: charge(-actual)` + `release(+delta)` и фиксирует `runs.actual_cost_units`.


