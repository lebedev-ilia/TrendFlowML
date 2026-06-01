# Dataset Collector: полный тест-план

## Цель

Проверить три независимых сервиса:

- `discover`: ежедневно собирает metadata по категориям до лимита/исчерпания квоты, пишет discover shards и выгружает их в HF.
- `download`: без категории читает discover metadata, скачивает уникальные видео, батчит HF upload, пишет `download_done` только после успешного upload и удаляет локальные mp4.
- `enrich`: без категории читает discover metadata, пишет отдельные `shards/enrich/category=*/part_*.json`, батчит HF upload, пишет enrich done только после успешного upload.

Все сервисы после обработки backlog должны не завершаться, а ждать новые данные и перепроверять metadata каждые 2 минуты.

## Подготовка

```bash
cd /Users/user/Desktop/TrendFlowML/Fetcher
export HF_TOKEN=...
python -m py_compile fetcher/dataset_collector/*.py fetcher/dataset_collector/discovery/*.py
python -m pytest tests/unit/test_dataset_collector.py -q
```

Порты метрик запускать разными:

- discover: `9095`
- workers: `9096`
- отдельные ручные процессы: `9097+`

## 1. Smoke Test

1. Запустить `discover` на короткий лимит.
2. Запустить `run-workers` без `--category`.
3. Убедиться, что появились:
   - `shards/metadata/category=*/part_*.json`
   - `downloads/queue.jsonl`
   - `shards/enrich/category=*/part_*.json`
   - HF queues в `state/`
4. Проверить `/metrics` на портах `9095` и `9096`.

Ожидание: сервисы не требуют category, backlog обрабатывается по всем категориям.

## 2. Restart / Interrupt Tests

### Discover

1. Запустить discover.
2. Прервать `Ctrl+C` во время keyword.
3. Запустить discover снова.

Ожидание: checkpoint продолжает с последнего keyword/bucket/platform, дубли не растут, pending records flush’ятся в shard.

### Download

1. Запустить workers.
2. Прервать во время скачивания большого видео.
3. Запустить снова.

Ожидание: `.tmp` удалены или перезаписаны, уже загруженные в HF не скачиваются повторно, локально скачанные и ожидающие HF upload не дублируются.

### Enrich

1. Прервать во время yt-dlp enrich.
2. Запустить снова.

Ожидание: уже queued/pending HF enrich не обогащаются повторно, новые ID продолжаются.

### HF Upload

1. Прервать во время HF video/enrich/shard upload.
2. Запустить снова с `--scan-*`.

Ожидание: done metadata появляется только после успешного HF commit; неуспешные элементы повторяются.

## 3. HF Commit Rate Test

1. Подготовить >100 маленьких файлов в очереди upload.
2. Запустить upload worker.
3. Проверить `state/hf_commit_log.jsonl`.

Ожидание: между commit’ами одного repo минимум `hf_commit_min_interval_seconds` (по умолчанию 37s), фактически <100 commits/hour.

## 4. Download Contract Test

Проверить порядок:

1. После локального скачивания файл есть в `downloads/videos/...`.
2. До HF upload нет строки в `download_done.jsonl`.
3. После успешного HF upload:
   - есть строка в `download_done.jsonl`
   - есть строка в `hf_video_upload_done.jsonl`
   - локальный mp4 удалён.

## 5. Enrich Payload Test

Для `shards/enrich/category=*/part_*.json` проверить каждую запись:

Разрешённые поля:

- `video_id`
- `source_shard`
- `thumbnails_ytdlp`
- `formats`
- `subtitles`
- `automatic_captions`
- `_enriched`
- `rejected`
- `rejected_reason`

Ожидание:

- `thumbnails_ytdlp`: максимум 2 лучших thumbnail.
- `formats`: максимум 2 записи: лучшее доступное разрешение и лучшее <=1080p.
- `subtitles` / `automatic_captions`: только `ru`/`en`, с `language`, `text`, `cues` при наличии таймингов.
- Нет `description`, `tags`, `duration`, `chapters`, `category`, `query`, `thumbnails` и других discover-полей.

## 6. No-Category Service Test

Запустить:

```bash
python -m fetcher.dataset_collector.cli run-workers dataset_campaign.json --interval 120 --metrics-port 9096
```

Ожидание: download/enrich/upload работают по всем категориям из metadata. `--category` использовать только для discover/debug.

## 7. Snapshot Transition Test

Сымитировать состояние 100k+ видео по 18 категориям:

1. Заполнить/подложить discover metadata shards и schedule.
2. Запустить snapshot command с нужным `snapshot_index`.
3. Проверить, что snapshot собирается по due schedule, не создаёт новых discover IDs и сохраняет timestamp `time_get`.

Ожидание: после достижения target discover не продолжает добор, а snapshot pipeline работает по schedule.

## 8. Dedup Test

1. Один и тот же `video_id` встречается в нескольких категориях/keywords.
2. Запустить discover/download/enrich.

Ожидание: discover не пишет duplicate seen; download не скачивает один и тот же key повторно; enrich не создаёт повторные done entries.

## 9. Captions Test

Видео с:

- ru manual subtitles
- en manual subtitles
- only automatic captions
- no captions
- captions returning 403

Ожидание:

- ru/en текст и cues сохраняются, если доступны.
- no captions даёт `{}` без ошибки.
- 403 по timedtext не ломает enrich и не шумит как критическая ошибка.

## 10. Metrics / Grafana Test

Проверить наличие метрик:

- discover accepted/rejected/quota/keys
- queue pending по download/enrich/HF
- local mp4 count
- videos/shards on HF
- service pass counters
- HF commit counters

Ожидание: каждый сервис имеет отдельный порт или отдельные labels (`service`, `repo_type`), Grafana panels не конфликтуют.

## 11. Disk Pressure Test

1. Скачать пачку видео.
2. Дождаться HF upload.

Ожидание: после upload локальные mp4 удаляются, размер `downloads/videos` не растёт бесконечно.

## 12. Recovery From Missing Local Files

1. Удалить локальный mp4, который стоит в HF queue.
2. Запустить upload/download.

Ожидание: HF upload помечает missing как failed, download может заново скачать video, если нет final done.

## 13. Inventory Rebuild Test

После аварийного завершения:

```bash
python -m fetcher.dataset_collector.cli inventory-rebuild dataset_campaign.json
```

Ожидание: summary соответствует фактическим metadata/enrich/download/HF done files.

## 14. Long Run Test

Запустить discover + workers на несколько часов.

Ожидание:

- нет роста дубликатов в queues
- нет превышения HF commit rate
- нет зависших child-процессов после Ctrl+C
- backlog monotonically decreases, когда discover не добавляет новые данные
