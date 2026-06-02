# Google Colab Free: запуск сбора 20k+

Этот сценарий рассчитан на disposable Colab runtime: состояние лежит в Google Drive, а worker можно перезапускать после обрыва сессии.

## 1. Подготовка Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone <YOUR_REPO_URL> TrendFlowML
cd TrendFlowML/Fetcher
apt-get update && apt-get install -y nodejs
python -m pip install -e .
python -m pip install -U huggingface_hub yt-dlp pytubefix google-api-python-client
```

После `drive.mount()` удаление локальных `.mp4` через обычный `unlink` отправляет файлы в корзину Drive. Collector при путях под `/content/drive/MyDrive` удаляет видео через Drive API (без корзины). Нужны те же Google-учётные данные, что и для mount (достаточно одной авторизации в сессии).

Секреты лучше хранить в Colab Secrets или в переменных окружения:

```python
import os
os.environ["HF_TOKEN"] = "<hf token>"
os.environ["FETCHER_YOUTUBE_DATA_API_KEYS"] = "key1,key2,key3"
```

## 2. Discover по категории

```bash
python scripts/colab_20k_bootstrap.py \
  --role discover \
  --category Sport \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test \
  --limit 500
```

Для полного запуска убери `--limit` и распределяй категории между Colab-аккаунтами.

## 3. Queue workers

```bash
python scripts/colab_20k_bootstrap.py \
  --role workers \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test \
  --lease-name workers-main \
  --lease-owner colab-account-1 \
  --metrics-port 0
```

`--lease-name` защищает общий Drive state от случайного двойного запуска одного набора workers.

Для Colab профиль `dataset_campaign_20k.json` использует `pytubefix` как основной backend, но с ротацией cookies и fallback client `WEB` для генерации PO-token через Node.js. Если меняешь runtime config вручную, проверь:

```json
"download_backend": "pytubefix",
"download_cookie_rotate_successes": 20,
"download_pytubefix_clients": ["ANDROID_VR", "WEB"]
```

Если все cookies и `WEB` client поймают bot-detection, worker попробует резервный `yt-dlp` backend.

## 4. Snapshots

Для таргетов моделей нужны `14d` и `21d`; `7d` можно использовать с mask.

```bash
python scripts/colab_20k_bootstrap.py \
  --role snapshot \
  --snapshot-index 1 \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test
```

Индексы: `1=7d`, `2=14d`, `3=21d`, `4=28d`.

## 5. Проверка статуса

```bash
python scripts/colab_20k_bootstrap.py \
  --role inventory-rebuild \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test

python scripts/colab_20k_bootstrap.py \
  --role status \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test
```

Главные файлы для мониторинга:

- `state/inventory/summary.json`
- `logs/workers/*.log`
- `state/queue_failures.jsonl`
- `state/queue_dead_letter.jsonl`
- `state/hf_commit_log.jsonl`

## 6. Рекомендуемая фаза запуска

1. Smoke: `300-500` видео на одной категории.
2. Checkpoint audit: `2k-3k` accepted по нескольким категориям.
3. Полный запуск: все категории примерно по `1112` accepted.

Если после checkpoint-а `balancer_language` слишком часто отклоняет кандидатов, снизь `language.coefficient` в `dataset_balancer_20k.json` с `0.60` до `0.50-0.55`.

## 7. Место на Google Drive

Старые видео, уже попавшие в корзину, место не освобождают сами. Один раз очисти корзину вручную в Drive UI или через [Drive API `files.emptyTrash`](https://developers.google.com/drive/api/guides/delete).

Чтобы отключить безвозвратное удаление (вернуть поведение с корзиной), в campaign JSON: `"drive_permanent_delete": false`.
