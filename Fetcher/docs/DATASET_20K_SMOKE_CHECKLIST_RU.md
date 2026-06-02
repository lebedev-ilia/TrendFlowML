# Smoke и checkpoint аудит для 20k+

## Smoke 300-500

Перед запуском в Colab обнови `yt-dlp` и поставь JS runtime:

```bash
apt-get update && apt-get install -y nodejs
python -m pip install -U yt-dlp pytubefix
```

Если уже создан `runtime_dataset_campaign_20k.json`, удали его после обновления кода, чтобы bootstrap пересоздал config с новым `pytubefix` backend, cookie rotation и `WEB` fallback:

```bash
rm -f /content/drive/MyDrive/dataset_runs/20k-test/runtime_dataset_campaign_20k.json
```

1. Запусти discover на одной категории:

```bash
python scripts/colab_20k_bootstrap.py --role discover --category Sport --limit 500
```

2. Запусти workers до почти пустых очередей:

```bash
python scripts/colab_20k_bootstrap.py --role workers --lease-name smoke-workers --metrics-port 0
```

3. Пересобери inventory и сохрани audit:

```bash
python -m fetcher.dataset_collector.cli inventory-rebuild dataset_runs/20k-test/runtime_dataset_campaign_20k.json
python scripts/audit_dataset_run.py dataset_runs/20k-test --out dataset_runs/20k-test/audit_smoke.json
```

## Что должно быть зелёным

- `queue_dead_letter = 0` или только понятные единичные ошибки.
- `lifecycle.lag_enrich` и `lifecycle.lag_download` уменьшаются после работы workers.
- `hf_commit_log.jsonl` не показывает больше `100` commits/hour на один repo.
- `channels.top_share <= 0.01` для полного запуска; на smoke одной категории может быть выше.
- `balancer_language` не доминирует настолько, что accept rate становится слишком низким.

## Checkpoint 2k-3k

После нескольких категорий:

```bash
python scripts/audit_dataset_run.py dataset_runs/20k-test --out dataset_runs/20k-test/audit_checkpoint_2k.json
```

Если `ru/en` не доминируют, проверь `youtube_relevance_languages` и keyword presets. Если `country=unknown` выше `25-30%`, ослабь country balancing только после проверки, что `channel_id` есть у большинства записей.
