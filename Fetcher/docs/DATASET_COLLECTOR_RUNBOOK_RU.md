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
- Очередь скачивания видео после `snapshot_0`.
- Ротация YouTube API keys.
- Ротация proxy для YouTube Data API discovery.
- Поддержка локального/nodpi proxy как download-only.
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
- `Fetcher/dataset_runs/dataset-100k/downloads/queue.jsonl`

Важно: `shards/metadata` это внутренний raw-ish формат collector'а для resume/debug. Финальный training dataset нужно смотреть через `export`, а не напрямую по shard.

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
- `youtube_keys_file`: путь к `keys.txt`.
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
223.204.54.119:8080
169.155.50.87:1080
127.0.0.1:8084 download_only
```

Правила:

- Если схема не указана, collector считает proxy HTTP: `host:port` -> `http://host:port`.
- Локальные proxy (`127.0.0.1`, `localhost`) не используются для discovery/API parsing.
- Локальные proxy используются для download, потому что `nodpi` может помочь при скачивании видео.
- Для YouTube Data API discovery нужен большой список внешних proxy с ротацией.

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

Перед массовым download желательно отдельно протестировать `nodpi` / local proxy и cookies на 5-20 видео.

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

Если download плохо идет через внешние proxy, проверить local `nodpi` как download-only proxy.

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

## Что Еще Желательно Доделать

- Команду `status`, которая показывает компактно: accepted/rejected per category, proxy/key errors, comments coverage, due snapshots.
- Отдельный отчет по proxy health.
- Отдельный отчет по comments coverage: сколько видео с `commentCount > 0`, сколько реально имеют comments.
- Опциональный режим `comments_required=true` для категорий, где комментарии критичны.
- Миграцию Pydantic v2 API, чтобы убрать warnings.
