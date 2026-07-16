# TrendFlow — Fetcher / dataset_collector: технический handoff-документ

> Назначение документа: полное описание системы сбора YouTube-датасета (`Fetcher/fetcher/dataset_collector/`)
> для передачи другой модели, которая будет переносить/переписывать этот пайплайн в виде автоматизации на RunPod.
> Документ собран из (а) прямого чтения исходного кода репозитория `TrendFlowML/Fetcher`, (б) форензик-анализа
> реальных прод-снапшотов состояния (`day_4`/`day_5`/`day_6` — дни 4/6 из 7-дневной недели discovery), (в) сверки
> с фактическими данными на Hugging Face. Дата анализа: 16 июля 2026. Репозиторий: `github.com/lebedev-ilia/TrendFlowML`.

---

## 1. Что делает система и зачем

TrendFlow собирает мультимодальный датасет YouTube-видео для предсказания популярности (просмотры/лайки через
14 и 21 день). Цель конкретной кампании (`dataset-20k-colab` / `100k-monthly-colab-v1`): **discover** (найти)
~100 000 видео за 1 неделю, распределённых по 18 категориям (по 5556 видео на категорию), затем в течение
следующих 4 недель повторно снимать снапшоты метрик (views/likes/comments) по расписанию **день 0, 7, 14, 21, 28**
— так получается временной ряд по каждому видео. Параллельно с discovery работают ещё два процесса: **download**
(скачивание самого видеофайла) и **enrich** (добор метаданных, которых нет в официальном YouTube Data API, через
yt-dlp). Все три потока данных (metadata-шарды, видеофайлы, enrich-json) грузятся в три отдельных датасета на
Hugging Face:

- `Ilialebedev/dataset_100k_monthly_shards` — JSON-шарды метаданных (discovery + snapshot).
- `Ilialebedev/dataset_100k_monthly_videos` — сами .mp4 файлы.
- `Ilialebedev/dataset_100k_monthly_enrich` — enrich JSON (yt-dlp: форматы, превью, субтитры).

Source-of-truth — **не база данных**, а файлы: `manifest.json` + JSONL-очереди/done-логи + JSON-шарды на диске,
периодически синхронизируемые с HF. Это принципиально важно для переноса на RunPod: вся логика "что уже сделано"
живёт в файлах состояния, не в каком-то внешнем сервисе.

Запускалось на **нескольких Google Colab-инстансах параллельно** ("main" и "worker(-ы)", условно colab-a /
colab-100k-worker-b, возможно colab-c), которые координируются друг с другом ЧЕРЕЗ HF-репозиторий (нет прямой
сети между инстансами). Colab-сессии эфемерны (~сутки жизни), поэтому весь прогресс обязан переживать рестарт
процесса — отсюда постоянный pull/push состояния в/из HF в начале и в конце каждого прохода.

---

## 2. Верхнеуровневая архитектура пайплайна

```
                    ┌────────────────────────────────────────────┐
                    │              runtime_dataset_campaign_*.json  │  ← конфиг кампании (CampaignConfig)
                    └────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼────────────────────────────────┐
        ▼                               ▼                                ▼
   [discover]                      [download]                      [enrich-metadata]
   YouTube Data API v3              pytubefix / yt-dlp                yt-dlp (metadata only)
   (search.list, videos.list,       (скачивает .mp4 файл)              (форматы, превью, субтитры —
    channels.list)                                                      то, чего нет в Data API)
        │                               │                                │
        ▼                               ▼                                ▼
   shards/metadata/**/part_*.json   downloads/videos/{cat}/{id}.mp4   shards/enrich/**/part_*.json
        │                               │                                │
        ▼                               ▼                                ▼
   upload-hf-shards                 upload-hf-videos                 upload-hf-enrich
        │                               │                                │
        ▼                               ▼                                ▼
   dataset_100k_monthly_shards   dataset_100k_monthly_videos     dataset_100k_monthly_enrich

Отдельно, по расписанию (день 0/7/14/21/28):
   [snapshot] → перечитывает views/likes/comments по уже принятым видео → snapshot-шарды → upload-hf-shards
```

Все процессы читают/пишут общий `output_dir` (в проде: `/content/dataset_runs/100k-main` и
`/content/dataset_runs/100k-monthly` — два отдельных прогона: "main" ведёт discovery + upload видео/шардов,
"worker" ведёт download + enrich + upload enrich, но оба физически могут исполнять одинаковые CLI-команды —
разделение ролей задаётся флагом `--with-discover`/`--worker-kinds` в `run-workers`, а не архитектурным барьером).

**Важный факт, подтверждённый пользователем**: discovery всегда запускается только на ОДНОМ Colab-инстансе
(main/colab-a). Download/enrich/upload воркеры могут крутиться параллельно на нескольких инстансах, разделяя
работу через `worker_shard_index`/`worker_shard_count` (например `worker_shard_count=3`, `worker_shard_index=0/1/2`)
и через coordination-claims на HF (см. §9).

---

## 3. Конфигурация кампании (`CampaignConfig`, `schemas.py` + `config.py`)

Конфиг — JSON, валидируется через Pydantic. Полный список полей, сгруппированный по смыслу:

**Категории и цели**
- `categories: List[CategoryConfig]` — каждая: `name`, `keywords: List[str]`, `target_count` (soft),
  `collect_count` (hard stop), `platform_weights` (напр. `{"youtube": 1.0}`), `filters` (мёрджится с
  `default_filters`), `youtube_relevance_languages`/`youtube_region_codes` (циклятся по индексу ключевого слова).
- В реальной кампании: 18 категорий × `target_count=collect_count=5556` = 100 008 ≈ 100k.

**Фильтры приёма видео** (`default_filters`, мёрдж с `filters` категории через `merged_filters()`)
- `duration_min_seconds`/`duration_max_seconds` (прод: 4–1500), `view_count_max` (прод: 100 000 000),
  `channel_video_cap` (прод: 100 — не больше N видео с одного канала), `outlier_policy: "reject"`.

**Распределение по возрасту видео** — `time_interval_buckets: List[{name, min_age_days, max_age_days, weight}]`.
Прод-веса сильно смещены к свежему контенту: `lt_1d=0.38`, `1d_1w=0.37`, `1w_1m=0.08`, `1m_3m=0.06`, `3m_6m=0.04`,
`6m_1y=0.03`, `1y_3y=0.02`, `gt_3y=0.02`. Это осознанный дизайн: цель — предсказывать раннюю популярность, поэтому
корпус смещён к недавно опубликованным видео.

**Snapshot-расписание**
- `snapshot_schedule_days: List[int]` — прод: `[0, 7, 14, 21, 28]`. Может быть заменено на
  `snapshot_schedule_hours`/`_minutes` (для смоук-тестов) или на `snapshot_sleep_seconds` +
  `snapshot_follow_up_count` (равномерные интервалы). Первый элемент обязан быть 0 (это и есть discovery-снапшот).

**Балансировщик** (`balancer_config_file` → `DatasetBalancer`, см. §7) — распределяет принятые видео равномерно
по полям `language, country, duration_seconds, view_count, like_count, comment_count, time_interval,
comments_available, channel`.

**Discovery**
- `discover_week_days` (прод: 7) — сколько дней длится фаза discovery, используется как gate в CLI перед стартом.
- `discover_fair_rotation: true` — round-robin по категориям (по одному keyword-батчу за раз на категорию), а не
  последовательный проход "категория за категорией".
- `discover_target_total` (100 000), `discover_fair_category_cap` (иначе вычисляется как
  `target_total / num_categories`).
- `min_videos_per_keyword` (20), `keyword_search_multiplier` (10) — сколько видео минимум нужно найти на ключевое
  слово, прежде чем двигаться дальше; `search_limit = max(remaining, min_videos_per_keyword * multiplier)`.

**HF-репозитории и лимиты аплоада** (см. §8-9 для деталей)
- `hf_repo_id`, `hf_shards_repo_id`, `hf_videos_repo_id`, `hf_enrich_repo_id` (каждый по умолчанию = `hf_repo_id`),
  `hf_coord_repo_id`, `hf_progress_repo_id` (в проде оба = shards-репо).
- `hf_token_env` — **имя** переменной окружения (не сам токен!), по умолчанию `HF_TOKEN`; есть explicit валидатор,
  который ругается, если сюда случайно вписали сам токен вместо имени переменной.
- `hf_upload_enabled`, `hf_upload_every_shards`, `hf_parallel_colab_count` (прод: 3 — сколько Colab одновременно
  шлют коммиты в один и тот же репо), `hf_repo_hourly_commit_limit` (128 — реальный лимит HF Hub на коммиты в час
  на репозиторий), `hf_commit_budget_reserve` (0.9 — резерв), `hf_commit_min_interval_seconds` (прод: 95),
  `hf_commit_hourly_limit` (100 — расчётный лимит на один Colab при параллели 3: `128 * 0.9 / 3 ≈ 38/ч`, но в
  конфиге дефолт указан 100 для одиночного Colab), `hf_shard_upload_batch_files`/`hf_video_upload_batch_files`/
  `hf_enrich_upload_batch_files` (сколько файлов коммитить за один HF commit, прод: 50/100/100).

**Downloads**
- `download_backend` ∈ `pytubefix` (default) / `yt_dlp` / `yt_dlp_first`.
- `download_pytubefix_clients` (прод: `["ANDROID_VR", "WEB"]`), `download_ytdlp_player_clients`
  (`["android","web"]`), `download_ytdlp_format` (селектор до 1080p mp4+m4a).
- `download_cookie_rotate_successes` (прод: 20 — сколько успехов подряд, прежде чем сменить cookie).
- Паузы: `download_pause_after_success_seconds` (5–10), `_after_fail_` (15), `_after_unavailable_` (15),
  `_after_bot_seconds` (120 — фиксированная пауза при обнаружении бот-детекта), `_fast_threshold_seconds` (8 —
  если скачалось подозрительно быстро, тоже пауза).

**Учётные данные и прокси**
- `cookie_files_dir` (прод: `fetcher/dataset_collector/cookies`), `cookie_file_glob` (`*.txt`).
- `youtube_keys_file` (прод: `fetcher/dataset_collector/keys/keys.txt`) — **тот самый файл с 49 Google API-ключами**.
- `proxies_file`, `use_proxies_for_discovery` (прод: `false` — для discovery проксей не нужен, только для скачивания
  видео, обход DPI/гео-блокировок).

**Очереди и ретраи**
- `queue_max_attempts` (5), `queue_retry_backoff_seconds` (900, линейный backoff: `attempt * backoff`).

**Мульти-Colab координация**
- `hf_coord_enabled`, `hf_coord_claim_ttl_seconds` (7200), `worker_id` (строка, напр. `colab-a`,
  `colab-100k-worker-b`), `worker_shard_index`/`worker_shard_count` (шардирование очередей между инстансами по
  `hash(video_id) % shard_count`).

---

## 4. Оркестрация процессов

### CLI-команды (`cli.py`)
`init-campaign`, `discover`, `snapshot`, `snapshot-poll`, `download`, `upload-hf-shards`, `upload-hf-videos`,
`upload-hf-enrich`, `enrich-metadata`, `run-workers`, `inventory-rebuild`, `export`, `status`, `validate`,
`import-seen`, `upload-hf`. Каждая — тонкая обёртка: загрузить конфиг → pull HF progress (кроме тестовых команд) →
выполнить работу → push HF progress (см. §9) → (для discover/upload-hf-shards) сразу же аплоадить накопленные шарды.

### `run-workers` (`run_workers.py`) — то, что реально крутится в проде долгоживущим процессом
- `run_all_workers()` — точка входа. Опционально берёт **lease** (файловый лок, см. §9.3), чтобы только один
  процесс на конкретной машине владел ролью воркера.
- Если передан `--with-discover` — поднимает discovery отдельным **сабпроцессом** (`subprocess.Popen`), который
  просто зацикленно дёргает CLI `discover` с паузой `interval` между проходами (легаси-путь, `_worker_loop`).
- Все остальные роли (`enrich-metadata`, `download`, `upload-hf-shards`, `upload-hf-videos`, `upload-hf-enrich`)
  — это **`DEFAULT_QUEUE_WORKERS`**, каждый крутится in-process в отдельном потоке (`_queue_worker_daemon`) без
  форка нового процесса на каждый проход: pull HF progress → выполнить проход (`run_*_queue`) → push HF progress
  (только если проход реально что-то сделал: `attempted > 0`) → пауза (idle_interval, дефолт 120с; если проход
  что-то сделал — пауза всего 1с, предполагая что очередь ещё не пуста).
- Грациозное завершение по SIGINT/SIGTERM: первый сигнал — `request_shutdown()` + SIGTERM дочерним процессам с
  grace 8с; повторный сигнал — force-kill.

### Роли двух прод-инстансов (реконструировано из форензики)
- **main** (`100k-main`, `worker_id="colab-a"`, `worker_shard_index=0/3`): discover + download + upload-hf-shards +
  upload-hf-videos. Логи: `discover_log.txt`, `logs/workers/{download,upload-hf-shards,upload-hf-videos}.log`.
- **worker** (`100k-monthly`, `worker_id="colab-100k-worker-b"`, `worker_shard_index=1/3`): download + enrich-metadata
  + upload-hf-videos + upload-hf-enrich + upload-hf-shards. Логи дополнительно содержат `enrich-metadata.log`,
  `upload-hf-enrich.log`.

Оба процесса физически смотрят на **общие** видео (через общий `video_schedule.jsonl`/очереди, синкаемые через
HF), но координируются через `worker_shard_index` + coordination-claims, чтобы не дублировать работу.

---

## 5. Файловая модель состояния (`state.py`, класс `DatasetState`)

Всё состояние кампании лежит под `output_dir`:

```
output_dir/
├── manifest.json                     ← счётчики (accepted/rejected/snapshots/downloads), category_counters,
│                                        списки shards/rejected_shards/snapshot_shards
├── shards/metadata/category=X/part_NNNNNN.json   ← принятые видео (discovery), по shard_size (100) на файл
├── shards/enrich/category=X/part_NNNNNN.json     ← enrich-записи
├── rejected/part_NNNNNN.json                     ← отклонённые видео (для аудита/дебага балансировщика)
├── downloads/videos/{category}/{video_id}.mp4    ← локальные видеофайлы (удаляются после подтверждённого HF-аплоада)
└── state/
    ├── seen_ids.jsonl                ← дедуп-ключи platform:video_id — ГЛАВНЫЙ механизм анти-дублей
    ├── video_schedule.jsonl          ← расписание снапшотов на видео (due_at по индексам 0..N)
    ├── keyword_progress.jsonl        ← прогресс по каждому (category, bucket, platform, keyword_index)
    ├── discovery_checkpoint.json / discovery_checkpoint_{category}.json  ← resume-позиция discovery
    ├── channel_counts.json           ← счётчик видео на канал (для channel_video_cap)
    ├── balancer_snapshot.json        ← счётчики балансировщика (для равномерного распределения по полям)
    ├── api_keys.json                 ← состояние пула YouTube API-ключей (см. §6 — здесь был найден критический баг)
    ├── download_done.jsonl, download queue = downloads/queue.jsonl (вне state/)
    ├── metadata_enrich_queue.jsonl / metadata_enrich_done.jsonl
    ├── hf_shard_upload_queue.jsonl / _done.jsonl
    ├── hf_video_upload_queue.jsonl / _done.jsonl
    ├── hf_enrich_upload_queue.jsonl / _done.jsonl
    ├── hf_snapshot_upload_queue.jsonl / _done.jsonl
    ├── snapshot_completion.jsonl     ← какие (video, snapshot_index) уже собраны и подтверждённо выгружены
    ├── queue_failures.jsonl          ← лог неудачных попыток (attempt, error, next_retry_at)
    ├── queue_dead_letter.jsonl       ← попытки, исчерпавшие queue_max_attempts
    ├── performance_events.jsonl
    ├── hf_commit_log.jsonl           ← для троттлинга HF-коммитов (min_interval + hourly cap)
    ├── inventory/{summary.json, shards.jsonl, videos.jsonl}   ← агрегированная статистика (см. §11)
    ├── hf_progress_cache/            ← локальный кэш последнего pull с HF (те же файлы, для сравнения при merge)
    └── coordination/{claims,done,remote,shard_cache}/  ← межколаб-координация (§9.3)
```

**Атомарность записи**: `atomic_write_json` — пишет во временный файл `{name}.{pid}.{timestamp}.tmp`, затем
`os.replace` (атомарный rename). Это защищает от полуразрушенного файла при краше **в теории**, НО: не защищает
от гонки, если два процесса одновременно читают файл для мержа, оба вычисляют новое содержимое и оба делают
`os.replace` — один из результатов будет молча перезаписан другим (last-write-wins), и именно это наблюдалось в
проде как `JSONDecodeError: Unterminated string` / `FileNotFoundError: ...tmp -> ... — skip` в логах при merge
(см. §12.3).

**Дедупликация** — единственный и главный механизм: `DatasetState.load_seen()` кэширует весь `seen_ids.jsonl` в
`Set[str]` в памяти (ключ `platform:video_id`), `is_seen()`/`mark_seen()`. Это O(N) память на весь корпус (для 100k
видео — не проблема, но при масштабировании на RunPod до нескольких сотен тысяч видео стоит иметь в виду).

**append_jsonl** — **простой append без file-locking**. Работает нормально для одного писателя, но при нескольких
параллельных процессах, пишущих в один и тот же локальный файл (что в проде не должно происходить, т.к. у каждого
Colab свой локальный `output_dir`, синк только через HF), было бы небезопасно. `file_lock()` (exclusive-create
lockfile) существует как примитив, но применяется только к `worker_leases.json`.

---

## 6. YouTube API key pool — КРИТИЧЕСКИЙ БАГ (найден и исправлен в этой сессии)

Файл: `Fetcher/fetcher/dataset_collector/discovery/youtube.py`, класс `YouTubeKeyPool`.

### Баг
`YouTubeKeyState.used_units` — счётчик потраченных quota-юнитов на ключ, **накопительный за всё время**,
инкрементируется в `record_success()` и нигде не сбрасывался по дням. `_select_key()` исключал ключ из кандидатов
навсегда, если `used_units >= daily_quota_limit` (10 000), **независимо** от того, истёк ли `disabled_until`.
Реальная YouTube Data API v3 квота сбрасывается у Google каждые 24 часа — но локальный счётчик этого не знал.

### Эффект в проде
Один discovery-процесс на 49 ключей (search.list стоит 100 units) сжёг весь пул примерно за 1-2 дня после старта
кампании (~29 июня — 2 июля). Дальше **три дня подряд** (day_4/5/6 = дни 4 и 6 из 7-дневной недели discovery,
захваченные 2, 3 и 4 июля) `api_keys.json` был **побайтово идентичен**: 17 ключей помечены Google как навсегда
`suspended` (403 "Consumer has been suspended" — по словам пользователя, это ожидаемо, старые/тестовые ключи), 31
ключ — с `used_units` в диапазоне 9900–10000 (реальная суточная квота), 1 ключ ровно на границе (`used_units:
10000`, тоже исключён `used_units < 10000` не выполняется). Итог: **0 доступных ключей**, `_select_key()` кидает
`QuotaExceededError("All YouTube API keys are exhausted or disabled")`, процесс discovery падает необработанным
исключением в первые минуты каждой Colab-сессии. `manifest.json` (`accepted=9491` из цели 100 000) не менялся все
три дня — при таком темпе неделя discovery была бы провалена почти полностью (~9.5% от цели).

### Фикс (уже применён и закоммичен, коммит `e187df9`)
Добавлено поле `quota_date` в `YouTubeKeyState`. Метод `_reset_if_new_day(state)` сбрасывает
`used_units=0, disabled_until=None, last_error=None` при смене календарной даты (UTC), вызывается из
`_select_key()` (для ВСЕХ ключей разом, не только выбранного — важно, иначе ключ, который ни разу не выбрали,
никогда бы не сбросился), `record_success()`, `record_failure()`, `quota_stats()`. Обратная совместимость: старые
`api_keys.json` без поля `quota_date` получают `None` при загрузке → трактуется как "нужен сброс" → самовосстановление
на первом же запуске после деплоя фикса. Верифицировано изолированным репро точного продового состояния (см. коммит).

### Что стоит учесть при переносе на RunPod
- Google сбрасывает квоту в **полночь по Pacific Time**, а не UTC. Текущий фикс сбрасывает по UTC-дате (сохраняя
  конвенцию, уже использовавшуюся в существующем коде для `disabled_until`) — это **консервативно безопасно**
  (в худшем случае ключ полежит чуть дольше, прежде чем реально станет доступен), но не идеально точно. Если для
  RunPod критична каждая минута квоты — стоит выровнять reset-boundary по Pacific Time.
  17 постоянно suspended-ключей будут ежедневно ретраиться заново (это ожидаемо и безопасно — они просто быстро
  словят 403 и уйдут в короткий backoff), но если Google когда-нибудь разбанит ключ — система сама подхватит это.
- **Один discover-процесс на весь пул ключей** — так и должно быть по дизайну (подтверждено пользователем). Если
  на RunPod захотите параллелить discovery на несколько подов — **обязательно** давайте каждому поду свой
  непересекающийся набор ключей (свой `youtube_keys_file`/`api_keys.json`), иначе квота будет делиться и сгорать
  ещё быстрее, чем в текущем баге.
- Рекомендуется добавить мониторинг/алерт именно на этот паттерн: `manifest.json.counters.accepted` не растёт N
  часов подряд + discovery-процесс не запущен/упал — это ровно тот сценарий, который стоил кампании 3 суток.
- `record_failure()` не различает "квота исчерпана у **этого конкретного** ключа" от "у **всех** ключей сразу" —
  соответственно нет отдельной метрики/алерта на "весь пул исчерпан", только исключение, которое роняет процесс.
  На RunPod стоит явно ловить `QuotaExceededError` в самом верхнем уровне и не давать процессу просто умереть —
  либо ждать до полуночи (Pacific) с периодической проверкой, либо алертить и требовать добавления новых ключей.

---

## 7. Discovery — логика приёма/отклонения видео (`collector.py`)

`DatasetCollector.discover_campaign()` → при `discover_fair_rotation=true` (прод) → `_discover_campaign_fair()`:
настоящий round-robin — на каждом "раунде" для каждой ещё не завершённой категории обрабатывается **один
keyword-батч** (`max_keywords=1`), затем переход к следующей категории; раунды продолжаются, пока хоть одна
категория прогрессирует. Это гарантирует равномерный прогресс по всем 18 категориям одновременно, а не
"досконально добить одну категорию, потом следующую".

`discover_category()` — основной цикл:
1. Цель категории делится на `time_interval_buckets` (веса из конфига) → **бакет** (напр. `lt_1d`).
2. Внутри бакета цель делится по `platform_weights` → **платформа** (прод — только youtube).
3. Внутри платформы — по `keywords` категории (циклический перебор), с resume по `discovery_checkpoint_{category}.json`
   (`bucket_name`, `keyword_index`).
4. На каждое найденное видео (через `adapter.discover()`, YouTube search.list + videos.list + channels.list
   батчами) — цепочка проверок:
   - **dedup**: `state.is_seen(f"{platform}:{video_id}")` → reject `duplicate_seen`.
   - **фильтры** (`VideoFilter.decide`): длительность, view_count, `channel_video_cap`, `outlier_policy` → reject
     с конкретной причиной.
   - **балансировщик** (`DatasetBalancer.decide`) — см. §7.1 → reject `balancer_rejected` или `balancer_{field}`.
   - accept → `mark_seen`, `append_schedule` (создаёт snapshot-расписание 0/7/14/21/28), `enqueue_download`,
     `buffer_accepted` → раз в `shard_size` (100) видео — flush в JSON-шард на диск.
5. На ключевое слово останавливается, набрав `min_videos_per_keyword` (20) уникальных принятых видео.
6. При `QuotaExceededError` (см. §6) — flush всего накопленного, сохранение checkpoint на текущем keyword,
   **re-raise** (процесс падает, но состояние консистентно для resume).

### 7.1 Балансировщик (`balancer.py`)
Не просто фильтр по одному правилу — старается держать распределение принятых видео равномерным по нескольким
полям (`language, country, duration_seconds, view_count, like_count, comment_count, time_interval,
comments_available, channel`). Режимы: `hard` (жёсткий порог `min_accept_score`), `soft` (вероятностный accept =
score), гибрид (default). Штраф по полю растёт, если доля значения уже выше целевой (`targets`) или выше
`max_share`. Персистентность — `state/balancer_snapshot.json`, сохраняется каждые 50 принятых видео. **Важный
нюанс для мульти-инстанс координации**: при синке через HF используется стратегия "берём более свежую версию
целиком" (`replace_if_remote_newer`), а НЕ union/сложение счётчиков — значит, если два Colab одновременно копят
локальные приросты, при push один инстанс может **затереть** прогресс другого в этом файле (не потеря видео, но
искажение статистики балансировщика, что может привести к неточному распределению на границе кампании).

### 7.2 Фактическая пропорция reject/accept в проде
На день 6/7: `accepted=9491`, `rejected=8259` — почти 1:1, то есть под капотом discovery тратит примерно вдвое
больше API-квоты, чем "чистая" находка видео (реджекты тоже стоят search/videos/channels запросов). Это усиливает
эффект бага из §6.

---

## 8. Download — скачивание видеофайлов (`downloads.py`, `cookies.py`, `proxy.py`, `download_pacing.py`)

- **Backend**: `pytubefix` (основной) с fallback на `yt-dlp`, либо чистый `yt_dlp`/`yt_dlp_first`.
- **Клиенты pytubefix**: `["ANDROID_VR", "WEB"]` (прод) — "sticky client": держится на последнем рабочем клиенте
  между видео, переключается только когда все cookie для текущего клиента исчерпаны. `WEB`-клиент требует Node.js
  для po_token (проверяется динамически, если Node недоступен — WEB исключается из ротации).
- **Cookie-ротация** (`CookieRotator`): пул `.txt` cookie-файлов (Netscape-формат), ротация после N успехов подряд
  (прод: 20) либо принудительно при bot-детекте на текущем cookie. Прод использовал минимум 3 cookie-профиля
  (видны в логах: `chan_cookie.txt`, `margo_cookie.txt`, `bon_cookie.txt`).
- **Качество**: до 1080p; если progressive-поток недоступен на нужной высоте — качается video+audio раздельно и
  мержится `ffmpeg -c copy -movflags +faststart`.
- **Классы ошибок**: `BotDetectionDownloadError` (маркеры "sign in to confirm", "po_token", "sabr maximum reload"
  и т.п.) → пауза 120с (фиксированная, НЕ экспоненциальный backoff), смена cookie/клиента; `VideoUnavailableDownloadError`
  ("is unavailable", включает приватные/удалённые видео — в проде это ~185 из ~9600 попыток на день 6, около 2%)
  → нет смысла ретраить cookies, сразу reject. Также встречается "download returned no local file" (без явной
  классификации ошибки — сетевой/таймаут сбой) и "Server error 504 Gateway Time-out" (~19 случаев/день в проде,
  стабильно).
- **Финальный fallback-путь**: pytubefix (все клиенты × все cookie) исчерпан → yt-dlp как последняя попытка.
- **Pacing**: после успеха — короткая пауза (5-10с); после bot-детекта — 120с фиксированно; после "слишком быстрого"
  скачивания (<8с, `download_fast_threshold_seconds`) — тоже пауза (подозрение на подмену/обрыв). **Важно**: паузы
  сейчас фиксированные, не экспоненциальные — при системном bot-детекте (например, если весь IP-пул RunPod
  зафлагован YouTube) это будет долбить endpoint раз в 120с бесконечно, а не отступать сильнее. Стоит учесть при
  переносе — вероятно нужен экспоненциальный backoff с потолком и общий circuit breaker.
- **После успеха**: видео НЕ считается "done" сразу после скачивания — `enqueue_hf_video_upload`, а
  `mark_download_done` происходит только после **подтверждённого HF-коммита**. Локальный `.mp4` удаляется сразу
  после успешной загрузки на HF (экономия диска — важно для Colab с ограниченным местом, для RunPod можно
  пересмотреть, если диск не проблема).
- **Прокси**: для скачивания видео (`download_only` пул, отдельно от discovery/enrich прокси) — обход
  DPI/гео-блокировок. `use_proxies_for_discovery=false` в проде — на discovery/Data-API прокси не тратятся.

---

## 9. Enrich — добор метаданных через yt-dlp (`metadata_enrichment.py`)

YouTube Data API (использованный на discovery) не отдаёт: реальные доступные форматы/качества файла, прямые ссылки
на превью нужного размера, субтитры/автосабы. `fetch_ytdlp_info()` дёргает `yt-dlp.extract_info(download=False)`
(**не скачивает видео**, только метаданные) с `subtitleslangs=["ru","en"]`. Результат — `thumbnails_ytdlp`,
`formats` (топ до 1080p), `subtitles`/`automatic_captions` (текст, не только ссылки — извлекается через
`fetch_caption_texts_from_info`). Ошибки yt-dlp (`DownloadError`) НЕ бросаются наружу — логируются и трактуются
как "failed" (запись всё равно уходит в очередь, дальше через retry-механизм §10). В проде это самый частый тип
сбоя в enrich-очереди: `"yt-dlp enrich failed"` — 475 записей, стабильное число на протяжении 3 дней (то есть эти
конкретные видео уже выработали `queue_max_attempts` и осели в retry-цикле/dead-letter, не блокируя остальные).

После получения yt-dlp info применяется **пост-enrich фильтр** (`decide_post_enrich`) — доп. проверка по данным,
недоступным на этапе discovery. При reject — запись всё равно пишется (с `rejected=True`), но видео добавляется в
`post_enrich_rejected.jsonl`, который **блокирует скачивание** этого видео в `downloads.py` (то есть discovery
может принять видео, а enrich-этап задним числом его отбраковать — тогда видеофайл вообще не будет скачан).

---

## 10. Очереди, ретраи, dead-letter (`queue_retries.py`)

Все очереди — простые append-only JSONL-файлы, без внешней БД/брокера. `queue_item_key()` формирует ключ вида
`{service}:{platform}:{category}:{video_id}` (для видео-задач) или `{service}:{shard}` (для файловых, напр.
shard-аплоад). `record_queue_failure()`: attempt+1, `next_retry_at = now + backoff*attempt` (**линейный**, не
экспоненциальный, прод `backoff=900с`); при `attempt >= queue_max_attempts` (5) — дублируется в
`queue_dead_letter.jsonl` с `dead_lettered_at`. **Нюанс, важный для переноса**: `next_retry_at` вычисляется, но
нигде явно не используется как гейт "не пытайся раньше времени" в прочитанном коде download/enrich — по факту
повторная попытка происходит просто на следующем проходе воркера, если элемент ещё не в dead-letter. Т.е.
эффективный retry-интервал = периодичность прохода воркера (`idle_interval`, обычно 120с), а не заявленный backoff
— на RunPod при проектировании стоит явно решить, нужен ли настоящий time-gated backoff (сейчас его фактически нет).

---

## 11. HF-синхронизация и координация (`hf_progress.py`, `hf_queues.py`, `hf_upload.py`)

### 11.1 Progress-файлы и merge-стратегии
Список синкаемых файлов и стратегия слияния для каждого (`_progress_specs`):
- `replace_if_remote_newer` — по `updated_at`, для `progress_meta.json`, `balancer_snapshot.json`,
  `discovery_checkpoint.json` (целиком "берём свежее", не union — риск затирания параллельного прогресса, см. §7.1).
- `jsonl_union_key` — union по ключевому полю (для `seen_ids.jsonl`, `keyword_progress.jsonl`, `download_done.jsonl`,
  `metadata_enrich_done.jsonl` и т.п.) — remote+local объединяются, **local побеждает** при конфликте по ключу.
- `jsonl_union_rows` — union по fingerprint всей строки (дедуп только полных дублей).
- `manifest_merge` — специальная логика для `manifest.json`: берёт `max()` по всем числовым counters/
  category_counters, union множеств `shards`/`snapshot_shards`/`rejected_shards`. Это самый безопасный тип мержа
  (монотонный, не теряет прогресс ни одной стороны).

Разные роли воркера (`discover`/`download`/`enrich`/`snapshot`/`workers`) синкают разные подмножества файлов
(`ROLE_PROGRESS_FILES`) — снижает объём трафика и конфликтов. **Найденный в исходнике технический долг**: словарь
`ROLE_PROGRESS_FILES` дважды объявляет ключ `"snapshot"` — второе определение перекрывает первое (добавляет
`hf_snapshot_upload_done.jsonl`, но, вероятно, теряет что-то из первого объявления) — похоже на баг дублирования
ключа, стоит перепроверить при переносе.

### 11.2 Аплоад-очереди (shards/videos/enrich)
Каждая — JSONL-очередь + JSONL `_done` (ключ). Обработка батчами (`hf_*_upload_batch_files`), один HF commit на
батч. `already_on_hf` = `key in done_keys`. **Наблюдение из прод-логов**: очереди никогда не компактируются
(старые уже обработанные строки не удаляются из `*_queue.jsonl`) — из-за этого `queue_lines` в логе с каждым
проходом включает всю историю, а не только необработанный остаток, и лог выглядит как "весь батч пропущен"
(`skipped == queue_lines`), хотя по факту это не потеря данных, а вводящий в заблуждение счётчик. **Рекомендация
для RunPod**: либо периодически компактировать очередь (переписывать файл только строками `key not in done_keys`),
либо явно логировать `pending_actionable = queue_lines - already_on_hf - dead_letter - other_category`, чтобы не
полагаться на визуальное чтение `attempted`.

### 11.3 Rate limiting HF-коммитов
`wait_for_commit_slot()` — троттлинг на два условия одновременно: `min_interval_seconds` между коммитами в один
репо (прод: 95с) и `hourly_limit` — скользящее окно коммитов за час, читается из `hf_commit_log.jsonl`. Лимит
делится между параллельными Colab через `hf_parallel_colab_count` (прод: 3 инстанса делят 128 коммитов/час на
репо → ~38/ч на инстанс с учётом резерва 0.9). **Для RunPod**: если подов будет больше 3, обязательно пересчитать
`hf_commit_hourly_limit`/`hf_parallel_colab_count` под реальное число подов, иначе HF начнёт кидать 429 на коммиты.

### 11.4 Coordination-claims (мульти-инстанс, `state/coordination/`)
`hf_coord_enabled`, `hf_coord_claim_ttl_seconds` (7200с = 2ч) — механизм "застолбить" видео/задачу за конкретным
`worker_id`, чтобы два инстанса не скачивали/не обрабатывали одно и то же одновременно. Плюс `worker_shard_index`/
`worker_shard_count` — статическое шардирование (видимо `hash(video_id) % shard_count == worker_shard_index`) как
первый уровень разделения работы, claims — как второй уровень (динамический) поверх шардирования.

### 11.5 Гонки при параллельной записи (наблюдалось в проде)
В логах регулярно: `JSONDecodeError: Unterminated string...— skip` и `FileNotFoundError: ...tmp -> ... — skip` при
merge `download_done.jsonl`, `video_schedule.jsonl`, `metadata_enrich_queue.jsonl`. Похоже на гонку при
одновременной записи several процессов в общий файл до/во время pull-merge-push цикла. `atomic_write_json` защищает
от полуразрушенного ОДНОГО файла, но не от гонки "прочитали remote → смержили → записали", если между чтением и
записью remote-версия успела измениться (нет optimistic locking/версионирования на HF-стороне). **Для RunPod**:
стоит рассмотреть либо единый "владелец записи" на каждый файл состояния (только discover-под пишет
discovery-файлы, только download-поды — download-очередь и т.п., что частично уже так по дизайну), либо
явный retry-with-backoff на merge-конфликт вместо тихого `skip`.

---

## 12. Snapshot — повторные замеры метрик по расписанию (`snapshots.py`)

`build_schedule_entry()` при discovery сразу создаёт расписание на видео: `due_at` для индексов 0 (сразу),
1 (день 7), 2 (день 14), 3 (день 21), 4 (день 28) — либо равномерно через `snapshot_sleep_seconds` +
`snapshot_follow_up_count` для тестов. `run_snapshot_poll_loop()` — бесконечный цикл: найти due
(`due_at[index] <= now` и ещё не `completed`) → собрать через `adapter.collect_snapshot()` (снова YouTube Data API
— тратит квоту из того же пула ключей!) → батчить в шарды (каждые 50 видео) → выгрузить на HF (до 3 попыток) →
только ПОСЛЕ подтверждённой HF-выгрузки пометить `mark_snapshot_completed`. Follow-up снапшоты (index>0) сжимаются
(`compact_follow_up_snapshot`) — хранят только динамические поля (views/likes/comments/subscriberCount), не
дублируют статику (title/description) из snapshot_0. Долгие ожидания между снапшотами реализованы через
`_chunked_sleep` кусками по 1 часу (не просто `sleep(7*86400)`) — специально, чтобы Colab не считал сессию
неактивной и не убивал раннтайм по таймауту; **для RunPod это ограничение неактуально** (под можно просто
останавливать и поднимать по cron/расписанию вместо непрерывного sleep-luup).

**Важно для RunPod-дизайна**: snapshot-фаза тоже расходует YouTube Data API квоту из ТОГО ЖЕ пула ключей, что и
discovery. Если discovery ещё не закончен, а уже подошли snapshot day-7 для ранних видео — они конкурируют за
квоту. В прод-снапшоте (day_6/7) `snapshot_7d=0` везде — снапшоты ещё физически не должны были начаться (неделя
discovery не кончилась), поэтому конкуренции пока не наблюдалось, но при масштабировании/ускорении на RunPod
стоит явно спроектировать приоритет между discovery-квотой и snapshot-квотой (например, отдельный под с отдельным
набором ключей для snapshot).

---

## 13. Инвентарь / отчётность (`inventory.py`)

`state/inventory/summary.json` — агрегированная статистика, пересчитывается после каждого прохода
download/enrich/upload-очередей. Ключевые метрики: `accepted`, `enriched`, `downloaded_or_on_hf`,
`uploaded_metadata_shards/enrich/video`, `lag_enrich = accepted - enriched`, `lag_download`, `lag_hf_video`,
`lag_hf_enrich`, `training_ready_snapshot0`, `training_ready_14_21` (нужны snapshot-индексы 2 и 3 = 14д и 21д),
`training_ready_7_14_21` (индексы 1,2,3). `channels.unique`/`top_count`/`top_share`/`average_videos_per_channel` —
для контроля, что один канал не доминирует в датасете. Это готовая точка для дашборда/алертинга на RunPod — эти
"lag_*" метрики напрямую показывают, где пайплайн отстаёт (в проде: `lag_download=5422` из `accepted=9611`, т.е.
скачано только 43.5%).

---

## 14. Форензика реального прогона (day_4/5/6, дни 4 и 6 из 7-дневной недели)

Коротко, для контекста "как это выглядит, когда что-то идёт не так":

- `manifest.json`, `video_schedule.jsonl`, `seen_ids.jsonl`, `keyword_progress.jsonl` были **побайтово идентичны**
  между тремя разными календарными днями (2, 3, 4 июля) — верный сигнал, что discovery-процесс не сделал ни
  одного полезного цикла (причина — баг из §6).
- `hf_shard_upload_done.jsonl` не менялся с 29 июня (за неделю до захвата) — не баг, а следствие: раз discovery
  не производит новых шардов, upload-очереди физически нечего грузить.
- `queue_failures.jsonl`/`queue_dead_letter.jsonl` росли день ото дня и у main, и у worker (worker — втрое
  интенсивнее: 990→1072→1091 против 203→338→348 у main), в основном за счёт `unavailable`/`download returned no
  local file`/`yt-dlp enrich failed`/`504 Gateway Time-out`.
- Реальные данные на HF (`dataset_100k_monthly_videos`): 4417 строк, 164 GB — примерно соответствует локальному
  счётчику `on_hf≈4189` на day_6 (расхождение объяснимо продолжающейся синхронизацией от других воркеров).
- `keys.txt` — 49 боевых Google API-ключей (17 навсегда suspended, 31 с исчерпанной суточной квотой, 1 на
  границе) — тот самый файл, что указан в `youtube_keys_file`. Хранить такой файл в общедоступном месте не стоит;
  рекомендовано пользователю ротировать при необходимости (подтверждено, что суспенды — ожидаемое старое/тестовое).

---

## 15. Чек-лист рекомендаций для переноса на RunPod

1. **Квота API-ключей**: используйте уже исправленную версию `YouTubeKeyPool` (коммит `e187df9`, файл
   `Fetcher/fetcher/dataset_collector/discovery/youtube.py`) — без этого фикса пайплайн неизбежно встанет через
   1-2 дня работы. Рассмотрите привязку reset-boundary к Pacific Time вместо UTC для точности.
2. **Один discover-под на пул ключей.** Если нужно параллелить discovery — раздайте непересекающиеся наборы
   ключей на под, не делите один `api_keys.json`/`youtube_keys_file` между несколькими discover-процессами.
3. **Персистентное хранилище обязательно.** В Colab роль "пережить рестарт" выполнял Google Drive + постоянный
   pull/push с HF в начале/конце каждого прохода. На RunPod либо используйте persistent volume под `output_dir`
   (проще, меньше HF-трафика), либо сохраните текущую HF-pull/push модель как есть — она уже реализована и
   протестирована, менять не обязательно, если не мешает.
4. **Секреты через переменные окружения**, не хардкод: `HF_TOKEN` (имя задаётся `hf_token_env`, но сам токен —
   только через env), `youtube_keys_file`/cookie-файлы — через смонтированные секреты/volume, не в образе.
5. **Компактируйте upload-очереди** (`hf_shard_upload_queue.jsonl` и т.п.) периодически — иначе логи вводят в
   заблуждение, а с ростом кампании файлы будут расти неограниченно (сейчас это не потеря данных, но плохая
   observability и лишний I/O на каждый проход).
6. **Реализуйте настоящий time-gated retry** для `queue_retries.py` — сейчас `next_retry_at` вычисляется, но
   фактически не используется как гейт; ретраи происходят на каждом проходе воркера независимо от backoff.
7. **Экспоненциальный backoff + circuit breaker для bot-детекта при скачивании** — сейчас фиксированные 120с;
   при системной блокировке IP-диапазона (вероятный риск для RunPod-датацентровых IP, YouTube агрессивно детектит
   дата-центры) стоит эскалировать паузу и/или переключаться на прокси-пул раньше.
8. **Перепроверьте дублирующийся ключ `"snapshot"` в `ROLE_PROGRESS_FILES`** (`hf_progress.py`) — похоже на баг,
   не влияющий критично, но стоящий чистки при рефакторинге.
9. **Пересчитайте `hf_commit_hourly_limit`/`hf_parallel_colab_count`** под реальное число параллельных
   RunPod-подов — иначе рискуете упереться в лимит HF Hub на коммиты в час (128/репо).
10. **Мониторинг**: используйте `state/inventory/summary.json` (`lag_download`, `lag_enrich`, `lag_hf_*`) как
    готовый источник метрик для дашборда/алертов. Отдельно — алерт "`manifest.counters.accepted` не растёт N часов"
    (это единственный надёжный ранний индикатор бага из §6, который иначе роняет процесс тихо/незаметно на
    протяжении дней).
11. **Snapshot vs discovery — конкуренция за квоту.** Если планируете ускорить кампанию на RunPod (закончить
    discovery быстрее 7 дней), явно продумайте, не начнут ли ранние snapshot'ы (день 7) конкурировать за те же
    API-ключи, пока discovery ещё не закончен для поздних категорий.
12. **Не коммитьте `keys.txt`/cookie-файлы в репозиторий или публичные артефакты.** Ключи в текущем файле уже
    частично забанены Google — значит, они точно "видны"/мониторятся; для RunPod заведите новый набор ключей
    через секрет-менеджер оркестратора (Vault/RunPod Secrets/env), не через файл в образе.

---

## 16. Карта файлов исходного кода (для быстрой навигации)

```
Fetcher/fetcher/dataset_collector/
├── cli.py                  — все CLI-команды, точка входа
├── config.py                — загрузка/дефолты CampaignConfig
├── schemas.py                — Pydantic-модели: CampaignConfig, CategoryConfig, BalancerConfig, CollectedVideo,
│                                 Snapshot, RejectedRecord, compact_follow_up_snapshot и т.д.
├── collector.py              — DatasetCollector: discover_campaign, discover_category, fair rotation
├── state.py                  — DatasetState: все пути state-файлов, atomic_write_json, append_jsonl, file_lock,
│                                 seen_ids/dedup, буферизация шардов, все enqueue/mark_done методы
├── checkpoint.py             — DiscoveryCheckpoint (resume discovery)
├── keyword_progress.py       — KeywordProgressEntry
├── balancer.py                — DatasetBalancer (равномерное распределение по полям)
├── filters.py                 — VideoFilter (duration/view_count/channel_cap/outlier, + decide_post_enrich)
├── downloads.py               — скачивание видео (pytubefix/yt-dlp), run_download_queue
├── cookies.py / proxy.py     — ротация cookie/proxy
├── download_pacing.py        — паузы между попытками скачивания
├── metadata_enrichment.py    — enrich через yt-dlp (без скачивания видео), run_metadata_enrich_queue
├── snapshots.py                — SnapshotRunner, run_snapshot_poll_loop, расписание день 0/7/14/21/28
├── queue_retries.py            — record_queue_failure, dead-letter, attempts
├── hf_upload.py                — низкоуровневые HF upload-примитивы, resolve_*_repo_id, wait_for_commit_slot
├── hf_queues.py                — run_hf_shard/video/enrich_upload_queue (батчинг, already_on_hf skip-логика)
├── hf_progress.py             — pull/push HF progress, merge-стратегии, ROLE_PROGRESS_FILES
├── hf_commit_budget.py       — resolve_hf_commit_limits (расчёт лимитов при N параллельных Colab)
├── hf_coordination.py         — claims/coordination между воркерами
├── worker_leases.py           — файловый lease-механизм (кто владеет ролью воркера)
├── worker_shutdown.py         — кооперативный shutdown-флаг (in-process)
├── run_workers.py              — оркестрация всех воркеров (run-workers команда)
├── inventory.py                — summary.json, register_shard/video, lag-метрики
├── status_report.py            — human-readable отчёт (`status` команда)
├── export.py / training_format.py — экспорт в финальный тренировочный формат
├── local_delete.py             — удаление локальных файлов после подтверждённого HF-аплоада
├── age_buckets.py               — allocate_counts по time_interval_buckets
├── legacy_import.py             — import-seen (миграция старых датасетов)
└── discovery/
    ├── base.py                 — DiscoveryAdapter интерфейс, DiscoveryCapabilities
    ├── youtube.py               — YouTubeKeyPool (§6, пофикшено), YouTubeDiscoveryAdapter
    ├── tiktok.py / instagram.py / twitch.py / rutube.py — прочие платформы (не используются в прод-кампании,
      platform_weights={"youtube":1.0})

Fetcher/fetcher/services/
├── youtube_data_client.py     — YouTubeDataClient, QuotaTracker (ВНИМАНИЕ: у этого класса ЕСТЬ корректный daily
│                                 reset через reset_at/_ensure_window — но это per-process трекер для одного
│                                 запуска, НЕ связан с персистентным YouTubeKeyState в YouTubeKeyPool, который и
│                                 был баговым; при рефакторинге стоит присмотреться, не унифицировать ли оба
│                                 механизма в один)
└── rutube_ytdlp_client.py, instagram_graph_client.py и т.д. — прочие платформы
```

---

## 17. Известные баги / технический долг (сводка)

| # | Компонент | Статус | Описание |
|---|---|---|---|
| 1 | `discovery/youtube.py` `YouTubeKeyPool` | **Исправлено** (коммит `e187df9`) | `used_units` не сбрасывался по дням → пул навсегда "умирал" через 1-2 дня |
| 2 | `hf_queues.py` upload-очереди | Не исправлено, низкий приоритет | Очереди не компактируются, логи вводят в заблуждение (`skipped == queue_lines`), реальной потери данных не обнаружено |
| 3 | `state.py` merge при параллельной записи | Не исправлено | `JSONDecodeError`/`FileNotFoundError` при гонке нескольких процессов на merge `.tmp`-файлов — тихий `skip`, потенциальная потеря приращений прогресса |
| 4 | `hf_progress.py` `ROLE_PROGRESS_FILES` | Не исправлено, найдено при чтении кода | Дублирующийся ключ `"snapshot"` в словаре — второе определение перекрывает первое |
| 5 | `queue_retries.py` backoff | Не исправлено | `next_retry_at` вычисляется, но не используется как реальный гейт — ретраи идут на каждый проход воркера, backoff декоративен |
| 6 | `download_pacing.py` bot-detection | Не исправлено | Фиксированная пауза 120с вместо экспоненциального backoff — риск при системной блокировке IP-диапазона (актуально для дата-центровых IP RunPod) |
| 7 | `balancer.py` HF-синк | Не исправлено, дизайн-решение | `replace_if_remote_newer` для `balancer_snapshot.json` — не union, риск затирания параллельного прогресса счётчиков (не видео, только статистики) |
| 8 | `keys.txt` | Операционный риск | Реальные API-ключи в файле; часть уже забанена Google — подтверждено, что ожидаемо, но всё равно не хранить в общедоступных местах |

---

*Документ подготовлен на основе форензик-анализа прод-снапшотов (day_4/5/6), прямого чтения исходного кода
`Fetcher/fetcher/dataset_collector/` (через два параллельных Task-агента, полностью прочитавших 14 файлов), и
верификации фикса ключевого бага изолированным репро. Обновлено: 16 июля 2026.*
