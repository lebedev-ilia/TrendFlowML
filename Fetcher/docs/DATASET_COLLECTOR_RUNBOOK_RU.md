# Dataset Collector Runbook

Документ фиксирует текущее состояние сборщика обучающего датасета, решения, принятые во время настройки, и план перехода от smoke-запусков к полному сбору 100k видео.

## Что Сделано

Добавлен отдельный file-first collector внутри `Fetcher`, не связанный с production API/Celery ingestion flow.

Основные возможности:

- Масштабный discovery по категориям и ключевым фразам.
- 18 категорий из первого сбора.
- Автоматическое расширение каждой категории до 300 поисковых запросов через presets.
- Сбор `snapshot_0` при discovery.
- Расписание `snapshot_1..4` через 7/14/21/28 дней.
- Глобальная дедупликация через `state/seen_ids.jsonl`.
- Shard storage вместо одного большого JSON.
- Rejected records с причинами отказа.
- Очередь скачивания видео после `snapshot_0` (`downloads/queue.jsonl`) через **pytubefix**.
- Очередь обогащения metadata через yt-dlp (`state/metadata_enrich_queue.jsonl`) — formats, thumbnails_ytdlp, subtitles.
- Очереди выгрузки в Hugging Face: shards (`hf_shard_upload_queue.jsonl`) и videos (`hf_video_upload_queue.jsonl`).
- Ротация YouTube API keys.
- Ротация proxy для YouTube Data API discovery.
- Единый список HTTP-прокси для discovery, download (pytubefix) и enrich (yt-dlp).
- Ротация cookie-файлов для yt-dlp.
- Best-effort adapters для TikTok, Twitch, Rutube.
- Export в training JSON chunks.
- Hugging Face upload hooks.

## Структура Файлов

Секреты и локальные рабочие файлы лежат здесь и закрыты в `.gitignore`:

- `Fetcher/fetcher/dataset_collector/keys/keys.txt`
- `Fetcher/fetcher/dataset_collector/proxies/proxies.txt`
- `Fetcher/fetcher/dataset_collector/cookies/*.txt`

Рабочий run создается здесь:

- `Fetcher/dataset_runs/dataset-100k/manifest.json`
- `Fetcher/dataset_runs/dataset-100k/state/seen_ids.jsonl`
- `Fetcher/dataset_runs/dataset-100k/state/video_schedule.jsonl`
- `Fetcher/dataset_runs/dataset-100k/state/api_keys.json`
- `Fetcher/dataset_runs/dataset-100k/shards/metadata/category=<category>/part_*.json`
- `Fetcher/dataset_runs/dataset-100k/shards/snapshots/snapshot=<n>/part_*.json`
- `Fetcher/dataset_runs/dataset-100k/rejected/part_*.json`
- `Fetcher/dataset_runs/dataset-100k/downloads/queue.jsonl` — скачивание mp4
- `Fetcher/dataset_runs/dataset-100k/state/metadata_enrich_queue.jsonl` — yt-dlp metadata
- `Fetcher/dataset_runs/dataset-100k/state/metadata_enrich_done.jsonl` — уже обогащённые
- `Fetcher/dataset_runs/dataset-100k/state/hf_shard_upload_queue.jsonl` — HF: metadata shards
- `Fetcher/dataset_runs/dataset-100k/state/hf_video_upload_queue.jsonl` — HF: mp4
- `Fetcher/dataset_runs/dataset-100k/downloads/videos/<category>/*.mp4` — локальные файлы

`shards/metadata/category=<cat>/part_*.json` — тот же формат, что `main_ready/data_XX.json`: объект `{video_id: {time_interval, metadata, snapshot_0, ...}}`. Поле `time_interval` — в legacy-лейблах (`1month-3month`, `less-1day`, …). Команда `export` только склеивает shards и snapshot-shards в файлы `data_XX.json`.

## Campaign Config

Основной файл:

- `Fetcher/dataset_campaign.json`

Ключевые поля:

- `categories[].name`: категория.
- `categories[].keywords`: можно оставить коротким списком, если категория есть в presets. При загрузке config collector расширит ее до 300 запросов.
- `target_count`: целевой объем после чистки.
- `collect_count`: объем с запасом под дубли/rejected.
- `snapshot_schedule_days`: сейчас `[0, 7, 14, 21, 28]`.
- `time_interval_buckets`: стратификация по возрасту видео.
- `youtube_keys_file`: путь к `youtube_keys.txt` (default: `fetcher/credentials/youtube_keys.txt`).
- `tiktok_credentials_file`, `instagram_credentials_file`, `twitch_credentials_file`: JSON в `fetcher/credentials/`.
- `credentials_dir`: каталог credentials (см. `docs/PLATFORM_CREDENTIALS.md`).
- Проверка: `python scripts/check_platform_credentials.py`.
- `proxies_file`: путь к `proxies.txt`.
- `cookie_files_dir`: папка cookie-файлов.
- `hf_repo_id`, `hf_upload_enabled`: HF upload, пока лучше держать выключенным до стабильного smoke.

## Time Interval Buckets

Видео распределяются по возрасту с упором на свежие:

- `lt_1d`: 20%
- `1d_1w`: 20%
- `1w_1m`: 12%
- `1m_3m`: 16%
- `3m_6m`: 14%
- `6m_1y`: 8%
- `1y_3y`: 6%
- `gt_3y`: 4%

Для YouTube это превращается в `publishedAfter` / `publishedBefore` в `search.list`.

На 6000 видео по категории это примерно:

- `lt_1d`: 1200
- `1d_1w`: 1200
- `1w_1m`: 720
- `1m_3m`: 960
- `3m_6m`: 840
- `6m_1y`: 480
- `1y_3y`: 360
- `gt_3y`: 240

## API Keys

Ключи лежат обычным текстом:

```text
Fetcher/fetcher/dataset_collector/keys/keys.txt
```

Формат: один ключ на строку. JSON не нужен.

Проверка ключей:

```bash
cd Fetcher
python scripts/check_youtube_keys.py fetcher/dataset_collector/keys/keys.txt \
  --proxy "http://<proxy-host>:<port>" \
  --output youtube_key_check_results.json \
  --working-output youtube_working_keys.txt
```

Результат последней проверки:

- Было проверено 124 ключа.
- `ok`: 48.
- `quota_exceeded`: 17.
- `forbidden`: 59.
- После фильтрации было оставлено 49 рабочих ключей.

## Proxy

Proxy list:

```text
Fetcher/fetcher/dataset_collector/proxies/proxies.txt
```

Формат:

```text
197.248.16.109:8080
```

Правила:

- Если схема не указана, collector считает proxy HTTP: `host:port` -> `http://host:port`.
- Один и тот же список используется для YouTube API (discovery), **pytubefix** (download) и **yt-dlp** (enrich).
- Локальные адреса (`127.0.0.1`, `localhost`) по умолчанию пропускаются (`include_local_proxies_for_discovery: false` в config).
- Для discovery желателен пул из нескольких внешних proxy с ротацией.

## Cookies

Cookie files:

```text
Fetcher/fetcher/dataset_collector/cookies/*.txt
```

Требуется Netscape cookie format, совместимый с yt-dlp. Collector и production yt-dlp adapters ротируют cookie-файлы через параметр `cookiefile`.

Cookies важны для:

- скачивания видео;
- age/session/region restrictions;
- TikTok/YouTube cases, где без cookie yt-dlp нестабилен.

## Smoke Status

Проверенные команды:

```bash
cd Fetcher
python - <<'PY'
from fetcher.dataset_collector.config import load_campaign_config
c = load_campaign_config("dataset_campaign.json")
print(len(c.categories), c.categories[0].name, len(c.categories[0].keywords))
print(c.categories[0].keywords[:5])
PY
```

Ожидаемый результат:

```text
18 Avto_i_transport 300
```

Проверка путей:

```bash
ls fetcher/dataset_collector/keys/keys.txt
ls fetcher/dataset_collector/proxies/proxies.txt
ls fetcher/dataset_collector/cookies
```

Последний успешный smoke:

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --category Sport --limit 5
```

Результат:

```text
Sport: accepted=5 rejected=0
{"accepted": 5, "rejected": 0}
```

Проверено:

- `manifest.json`: `accepted=5`, `rejected=0`, `downloads=5`.
- `seen_ids.jsonl`: 5 записей.
- `video_schedule.jsonl`: due dates на `snapshot_1..4`.
- `downloads/queue.jsonl`: 5 URL.
- `snapshot_0.comments`: комментарии начали подтягиваться.
- `snapshot_0.raw.comments_error`: появляется при сетевых/API ошибках comments fetch.

## Формат Данных

Внутренний shard формат содержит служебные поля и `raw`:

- `platform`
- `video_id`
- `url`
- `category`
- `query`
- `metadata.raw`
- `snapshot_0.raw`
- `platform_capabilities`

Это нужно для отладки и resume.

Финальный export должен быть ближе к training JSON:

- top-level key: YouTube `video_id` или `platform:video_id` для не-YouTube.
- top-level fields:
  - `platform`
  - `category`
  - `query`
  - `collected_at`
  - `time_interval`
  - `metadata`
  - `snapshot_0`
  - `snapshot_1..4`
  - `_enriched`

Служебный `raw` в финальный export не должен попадать.

Export:

```bash
python -m fetcher.dataset_collector.cli export dataset_campaign.json exported_dataset --split-count 20
```

Validate:

```bash
python -m fetcher.dataset_collector.cli validate dataset_campaign.json --required-snapshots 1
```

Для полного датасета после 4 недель:

```bash
python -m fetcher.dataset_collector.cli validate dataset_campaign.json --required-snapshots 5
```

## Download Queue

После `snapshot_0` все accepted videos попадают в:

```text
dataset_runs/dataset-100k/downloads/queue.jsonl
```

Download можно запускать отдельно:

```bash
python -m fetcher.dataset_collector.cli download dataset_campaign.json
```

Перед массовым download протестируйте прокси из `proxies.txt` на 5–20 видео (`python d.py` или `cli download --limit 5`).

## Metadata Enrich Queue (yt-dlp)

YouTube Data API не всегда отдаёт `formats`, `thumbnails_ytdlp`, `subtitles`, `automatic_captions`, `chapters`. Для этого вторая очередь:

```text
state/metadata_enrich_queue.jsonl
state/metadata_enrich_done.jsonl
```

При записи metadata shard (`part_*.json`) каждое YouTube-видео автоматически попадает в enrich-очередь. Worker обновляет shard на месте и ставит `"_enriched": {"source": "yt_dlp", ...}`.

Запуск (те же прокси из `proxies.txt` + cookies):

```bash
# Для уже собранных shards без enrich — сначала построить очередь:
python -m fetcher.dataset_collector.cli enrich-metadata dataset_campaign.json \
  --category Sport --scan-shards

# Обогатить (можно параллельно с discover в другом терминале):
python -m fetcher.dataset_collector.cli enrich-metadata dataset_campaign.json \
  --category Sport --limit 50
```

Только построить очередь без yt-dlp: `--scan-shards --scan-only`.

## Инвентарь (шарды, ID, скачивания, HF)

Обязательный учёт в `dataset_runs/dataset-100k/state/inventory/`:

| Файл | Содержимое |
|------|------------|
| `shards.jsonl` | каждый shard: путь, категория, список `video_ids`, `count` |
| `videos.jsonl` | каждое видео: `video_id`, категория, shard |
| `summary.json` | агрегаты: очереди, скачано, на HF, по категориям |

Пересборка индекса из уже существующих `part_*.json`:

```bash
python -m fetcher.dataset_collector.cli inventory-rebuild dataset_campaign.json --category Sport
```

Статус (включая `inventory`):

```bash
python -m fetcher.dataset_collector.cli status dataset_campaign.json | python -m json.tool
```

### Prometheus: очереди и HF

| Метрика | Смысл |
|---------|--------|
| `dataset_collector_download_queue_pending{category}` | в очереди на скачивание |
| `dataset_collector_videos_downloaded_local{category}` | mp4 на диске |
| `dataset_collector_videos_on_hf{category}` | видео выгружены на HF |
| `dataset_collector_hf_video_upload_queue_pending{category}` | очередь выгрузки видео |
| `dataset_collector_shards_total{category}` | шардов метаданных |
| `dataset_collector_shards_on_hf{category}` | шардов на HF |
| `dataset_collector_hf_shard_upload_queue_pending{category}` | очередь шардов |
| `dataset_collector_videos_in_shards{category}` | уникальных ID в инвентаре |
| `dataset_collector_metadata_enrich_queue_pending{category}` | очередь yt-dlp |
| `dataset_collector_videos_enriched{category}` | обогащено yt-dlp |

`category="all"` — суммарно по кампании; `category="Sport"` — по категории.

При `run-workers --metrics-port 9095` инвентарь обновляется в фоне каждые ~30 с.

## Полный тест пайплайна (пошагово)

```bash
cd Fetcher && source .fetcher_venv/bin/activate
export HF_TOKEN=hf_...   # не класть токен в dataset_campaign.json

# 0. Мониторинг (опционально)
cd monitoring && docker compose up -d && cd ..

# 1. Проверка конфига и статуса
python -m fetcher.dataset_collector.cli status dataset_campaign.json

# 2. Индекс по уже собранным shards
python -m fetcher.dataset_collector.cli inventory-rebuild dataset_campaign.json --category Sport

# 3. Backfill очередей (один раз, если данные были до очередей)
python -m fetcher.dataset_collector.cli enrich-metadata dataset_campaign.json --category Sport --scan-shards --scan-only
python -m fetcher.dataset_collector.cli upload-hf-shards dataset_campaign.json --category Sport --scan-shards --scan-only

# 4. Пилот discover (мало видео)
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --category Sport --limit 5 --metrics-port 9095

# 5. Очереди по одному (smoke, limit=1)
python -m fetcher.dataset_collector.cli enrich-metadata dataset_campaign.json --category Sport --limit 1
python -m fetcher.dataset_collector.cli download dataset_campaign.json --limit 1
python -m fetcher.dataset_collector.cli upload-hf-shards dataset_campaign.json --category Sport --limit 1
python -m fetcher.dataset_collector.cli upload-hf-videos dataset_campaign.json --category Sport --scan-downloads --limit 1

# 6. Статус + метрики
python -m fetcher.dataset_collector.cli status dataset_campaign.json
curl -s http://127.0.0.1:9095/metrics | grep dataset_collector_

# 7. Всё параллельно (боевой режим)
python -m fetcher.dataset_collector.cli run-workers dataset_campaign.json --category Sport --interval 120 --metrics-port 9095
# или: ./scripts/run_dataset_workers.sh Sport
```

## Запустить всё сразу

Один процесс-лаунчер поднимает **5 воркеров параллельно** (отдельные OS-процессы):

| Воркер | Что делает | Цикл |
|--------|------------|------|
| `discover` | сбор через YouTube API | один проход (перезапуск вручную) |
| `enrich-metadata` | yt-dlp в shards | каждые N сек |
| `download` | mp4 локально | каждые N сек |
| `upload-hf-shards` | shards → HF | каждые N сек |
| `upload-hf-videos` | mp4 → HF | каждые N сек |

```bash
cd Fetcher
source .fetcher_venv/bin/activate
export HF_TOKEN=hf_...   # имя переменной в config: hf_token_env = "HF_TOKEN"

# Prometheus/Grafana (опционально)
cd monitoring && docker compose up -d && cd ..

# Всё параллельно (логи: dataset_runs/dataset-100k/logs/workers/*.log)
python -m fetcher.dataset_collector.cli run-workers dataset_campaign.json \
  --category Sport \
  --interval 120 \
  --metrics-port 9095
```

Или скрипт:

```bash
./scripts/run_dataset_workers.sh Sport
```

Полезные флаги:

- `--no-discover` — только очереди (enrich / download / HF), если discover уже идёт в другом терминале.
- `--once` — один проход по каждой очереди и выход (удобно для теста).
- `--interval 60` — чаще опрашивать очереди (по умолчанию 120 с).

Первый раз для уже собранных shards без очередей HF/enrich:

```bash
python -m fetcher.dataset_collector.cli enrich-metadata dataset_campaign.json --category Sport --scan-shards
python -m fetcher.dataset_collector.cli upload-hf-shards dataset_campaign.json --category Sport --scan-shards
```

Остановка: `Ctrl+C` в терминале с `run-workers` (завершит дочерние процессы при выходе родителя — при необходимости `pkill -f 'dataset_collector.cli'`).

## Hugging Face Upload Queues

Две отдельные очереди для выгрузки в HF (нужен `HF_TOKEN` и `hf_repo_id` в `dataset_campaign.json`):

| Очередь | Файл | Что грузится |
|---------|------|--------------|
| **Shards** | `state/hf_shard_upload_queue.jsonl` | `shards/metadata/category=*/part_*.json` |
| **Videos** | `state/hf_video_upload_queue.jsonl` | `downloads/videos/<category>/<id>.mp4` |

Опционально в config:

```json
"hf_repo_id": "org/my-dataset",
"hf_shards_repo_id": null,
"hf_videos_repo_id": null,
"hf_shards_path_prefix": "shards/metadata",
"hf_videos_path_prefix": "videos"
```

Shards попадают в очередь при каждой записи `part_*.json`. Видео — после успешного `cli download` (локальный yt-dlp в `downloads/videos/`).

```bash
# Скачать mp4 (очередь 1)
python -m fetcher.dataset_collector.cli download dataset_campaign.json --limit 10

# Выгрузить shards в HF
python -m fetcher.dataset_collector.cli upload-hf-shards dataset_campaign.json \
  --category Sport --scan-shards --limit 5

# Выгрузить видео в HF
python -m fetcher.dataset_collector.cli upload-hf-videos dataset_campaign.json \
  --category Sport --scan-downloads --limit 5
```

## Snapshot Schedule

Сбор снапшотов:

```bash
python -m fetcher.dataset_collector.cli snapshot dataset_campaign.json --snapshot-index 1
python -m fetcher.dataset_collector.cli snapshot dataset_campaign.json --snapshot-index 2
python -m fetcher.dataset_collector.cli snapshot dataset_campaign.json --snapshot-index 3
python -m fetcher.dataset_collector.cli snapshot dataset_campaign.json --snapshot-index 4
```

Snapshot runner берет только due videos из `state/video_schedule.jsonl`.

Расписание:

- `snapshot_0`: сразу при discovery.
- `snapshot_1`: +7 дней.
- `snapshot_2`: +14 дней.
- `snapshot_3`: +21 день.
- `snapshot_4`: +28 дней.

## Тесты

Проверенная команда:

```bash
python -m pytest tests/unit/test_dataset_collector.py tests/unit/test_youtube_data_client.py
```

Последний результат:

```text
12 passed
```

Остаются warnings по Pydantic v2 deprecations. Они не блокируют работу, но позже стоит перейти с `@validator`, `.dict()`, `.parse_obj()` на Pydantic v2 API.

## План До Полного 100k

1. Не продолжать тестовые `dataset_runs/dataset-100k`; перед новым этапом удалить или переименовать папку.

2. Собрать более крупный smoke на одной категории:

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --category Sport --limit 200
```

Проверить:

- `accepted/rejected`.
- причины rejected;
- наличие comments;
- `comments_error`;
- распределение `time_interval`;
- расход ключей в `state/api_keys.json`;
- стабильность proxy.

3. Собрать pilot на одной категории:

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --category Sport --limit 1000
```

Проверить:

- сколько дублей;
- сколько long videos отсекается;
- хватает ли proxy;
- сколько comments реально подтягивается;
- размер shard-файлов;
- скорость и расход квоты.

4. Протестировать download на малой очереди:

```bash
python -m fetcher.dataset_collector.cli download dataset_campaign.json
```

Если download падает — проверить прокси в `proxies.txt` и лог `logs/workers/download.log` (pytubefix).

5. После успешного pilot запустить по 1000 на 2-3 разных категориях.

6. После этого запускать полный discovery:

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json
```

Цель:

- 18 категорий;
- `collect_count=6000` на категорию;
- около 108k raw accepted до финальной чистки;
- минимум 100k после дедупликации и фильтрации.

7. Через 7/14/21/28 дней запускать snapshot runner.

8. После `snapshot_4` сделать финальный export:

```bash
python -m fetcher.dataset_collector.cli export dataset_campaign.json exported_dataset --split-count 20
python -m fetcher.dataset_collector.cli validate dataset_campaign.json --required-snapshots 5
```

9. Только после проверки export включать Hugging Face upload:

```json
{
  "hf_repo_id": "your_username/your_dataset_repo",
  "hf_upload_enabled": true,
  "hf_upload_every_shards": 10,
  "hf_path_prefix": "raw-shards"
}
```

## Progress Logs, Resume и Checkpoint

При `discover` в stdout печатается строка прогресса:

```text
[dataset-100k] total=102_431 (baseline=100_000 run=2_431) | session=+127 keys=12/49 quota_session=18_400 | Sport 2_000/6_000 | kw [47/300] (253 left) "football highlights 2026"
```

Поля:

1. `total` — baseline + accepted текущего run.
2. `run` — accepted в `manifest` за этот campaign run.
3. `session` — accepted за текущий запуск CLI.
4. `keys` — доступные / всего YouTube API keys.
5. `quota_session` — квота, потраченная с начала сессии.
6. `kw [i/N]` — текущее ключевое слово и сколько осталось в категории.

Resume после исчерпания квоты:

- Позиция сохраняется в `state/discovery_checkpoint.json` (категория, bucket, platform, `keyword_index`, keyword).
- Отработанные ключевые слова (≥ `min_videos_per_keyword` уникальных) пишутся в `state/keyword_progress.jsonl` (`status: done`) и при следующем запуске **пропускаются**. После достижения порога collector сразу переходит к следующему keyword (не крутит поиск дальше).
- Shards пишутся инкрементально каждые `shard_size` видео (не только в конце категории).
- `seen_ids`, `video_schedule`, `downloads/queue` — как и раньше, сразу на диск.
- На следующий день запускай тот же `discover` — collector продолжит с checkpoint.
- Завершённые категории (`accepted >= collect_count`) пропускаются автоматически.

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --metrics-port 9095
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --reset-checkpoint  # начать keyword с нуля
```

## Baseline и Import Старых ID

В `dataset_campaign.json`:

```json
"baseline_accepted": 100000
```

Импорт ID из первого прогона (формат старого training JSON: ключи = video ID):

```bash
mkdir -p fetcher/dataset_collector/legacy
# положи файл, например fetcher/dataset_collector/legacy/seen_youtube_v1.json

python -m fetcher.dataset_collector.cli import-seen dataset_campaign.json \
  fetcher/dataset_collector/legacy/seen_youtube_v1.json \
  --platform youtube \
  --category legacy_v1
```

Поддерживаемые форматы: dict JSON (как `example_train_data`), list JSON, TXT (один ID на строку), CSV, JSONL.

Импорт пишет в `seen_ids.jsonl` (дедуп). `baseline_accepted` задаётся в config для отображения total в логах.

## Status

```bash
python -m fetcher.dataset_collector.cli status dataset_campaign.json
```

JSON-отчёт: total/run/session, keys, checkpoint, прогресс по категориям, распределения view/like/comment/duration из shards.

## Grafana / Prometheus

Локальный стек (Docker) — **готов к запуску**:

```bash
# 1) discover с метриками на хосте
python -m fetcher.dataset_collector.cli discover dataset_campaign.json --metrics-port 9095

# 2) Prometheus + Grafana
cd Fetcher/monitoring && docker compose up -d
```

| URL | Назначение |
|-----|------------|
| http://127.0.0.1:3001 | Grafana (admin / admin), дашборд **Dataset Collector** |
| http://127.0.0.1:9090 | Prometheus, Targets → `dataset-collector` |
| http://127.0.0.1:9095/metrics | сырые метрики процесса discover |

Подробнее: `Fetcher/monitoring/README.md`

Дашборд: heatmap view/like/comment/duration, перцентили, accept/reject rate, фильтр `$category`.

Метрики: `dataset_collector_total_with_baseline`, `dataset_collector_run_accepted`, `dataset_collector_view_count_bucket`, …

Полное распределение по всем shards (без live Prometheus):

```bash
python -m fetcher.dataset_collector.cli status dataset_campaign.json
# → JSON.distributions (p50/p90/p99)
```

## Что Еще Желательно Доделать

- Отдельный отчет по proxy health.
- Отдельный отчет по comments coverage.
- Опциональный режим `comments_required=true`.
- Миграцию Pydantic v2 API, чтобы убрать warnings.
