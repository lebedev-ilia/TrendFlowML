# Google Colab Free: запуск сбора 20k+

Этот сценарий рассчитан на disposable Colab runtime: состояние лежит в Google Drive, а worker можно перезапускать после обрыва сессии.

**Аудит тестового прогона `20k-test-3`:** [DATASET_20K_AUDIT_20k-test-3_RU.md](./DATASET_20K_AUDIT_20k-test-3_RU.md) — полный разбор discover/download/enrich, scorecard и рекомендации перед боевым 20k.

**Colab notebooks (готовые ячейки):**

| Ноутбук | Colab | Содержимое |
|---------|-------|------------|
| [notebooks/Colab_20k_A_main.ipynb](../notebooks/Colab_20k_A_main.ipynb) | **A** | discover + все workers + HF |
| [notebooks/Colab_20k_BC_worker.ipynb](../notebooks/Colab_20k_BC_worker.ipynb) | **B** | workers без discover, shard 1/3 |
| [notebooks/Colab_20k_C_worker.ipynb](../notebooks/Colab_20k_C_worker.ipynb) | **C** | workers без discover, shard 2/3 |

Тест на **3 Colab:** A (`shard 0/3`, `20k-main`), B (`shard 1/3`, `20k-worker-b`), C (`shard 2/3`, `20k-worker-c`).

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
python -m pip install -U huggingface_hub yt-dlp "pytubefix>=9.5.0" nodejs-wheel-binaries google-api-python-client
```

После `drive.mount()` удаление локальных `.mp4` через обычный `unlink` отправляет файлы в корзину Drive. Workers запускаются отдельным subprocess — **не вызывай** `auth.authenticate_user()` внутри worker; один раз в ноутбуке:

```python
from google.colab import auth
auth.authenticate_user()
```

Затем экспортируй токен для subprocess (после `auth` в той же сессии):

```bash
python scripts/export_colab_drive_token.py \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test-2
```

Файл `.dataset_drive_token.pickle` в `output_dir` подхватится bootstrap автоматически. Без него большие `.mp4` после HF-upload уйдут в корзину; мелкие `.video.tmp`/`.audio.tmp` удаляются обычным `unlink` (в корзину, но они небольшие).

Секреты лучше хранить в Colab Secrets или в переменных окружения:

Токен HF **нельзя** класть в `hf_token_env` в JSON — там только имя переменной: `"HF_TOKEN"`.

Workers стартуют из **терминала**, не из ячейки ноутбука. Один из вариантов:

```bash
export HF_TOKEN=hf_...
export FETCHER_YOUTUBE_DATA_API_KEYS=key1,key2,key3
python scripts/colab_20k_bootstrap.py --role workers ...
```

Или Colab Secret `HF_TOKEN` — bootstrap подхватит его сам (нужен `google.colab.userdata`).

Проверка в **том же терминале**, где запускаешь workers:

```bash
echo "${HF_TOKEN:0:7}..."   # должно показать hf_...
python -c "import os; assert os.getenv('HF_TOKEN','').startswith('hf_')"
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

`--lease-name` защищает общий Drive state от случайного двойного запуска одного набора workers на **одном** output-dir. Для нескольких Colab с разными Drive-папками lease не нужен — используй HF coordination ниже.

## 3a. Несколько Colab: discover на одном, download/enrich на других

**Discover** упирается в квоту API-ключей — держи один Colab (или один процесс) с `discover` + выгрузкой метаданных на HF:

```bash
# Colab A — discover + upload metadata shards
python scripts/colab_20k_bootstrap.py --role discover --output-dir /content/drive/MyDrive/dataset_runs/20k-test ...
# в том же runtime, отдельный процесс или второй bootstrap:
python scripts/colab_20k_bootstrap.py --role workers --worker-kinds upload-hf-shards \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-test --metrics-port 0
```

**Download / enrich** можно масштабировать на Colab B, C, … — у каждого свой `output_dir` на своём Drive, общая метадата и координация через репозиторий `hf_shards_repo_id`:

| Colab | Роль | Команда |
|-------|------|---------|
| A | discover + shards → HF | `--role discover` + `upload-hf-shards` |
| B | download 0/N | `--role workers-download --worker-id colab-b --worker-shard-index 0 --worker-shard-count 3` |
| C | download 1/N | `--worker-shard-index 1` |
| D | download 2/N | `--worker-shard-index 2` |
| E | enrich 0/M | `--role workers-enrich --worker-id colab-e --worker-shard-index 0 --worker-shard-count 2` |

Пример download-воркера (shard 1 из 3):

```bash
export HF_TOKEN=hf_...
python scripts/colab_20k_bootstrap.py \
  --role workers-download \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-worker-1 \
  --worker-id colab-account-2 \
  --worker-shard-index 1 \
  --worker-shard-count 3 \
  --parallel-colab-count 3 \
  --hf-repo-prefix Ilialebedev \
  --metrics-port 0
```

Как это работает:

1. Перед каждым pass worker тянет с HF новые `shards/metadata/.../part_*.json` и файлы `state/coordination/claims/*`, `state/coordination/done/*`.
2. **Статический shard** (`worker_shard_index` / `worker_shard_count`) — детерминированное разбиение по `hash(video_key) % N`, без гонок при добавлении новых ID discover’ом.
3. **Claim на HF** — перед скачиванием/обогащением worker пишет `state/coordination/claims/{download|enrich}/{worker_id}.jsonl`; чужой активный claim (не истёкший TTL) → skip.
4. После успеха — `state/coordination/done/{service}/{worker_id}.jsonl` на HF, чтобы другие Colab не повторяли работу.

Опционально в `runtime_dataset_campaign_20k.json`:

```json
"hf_coord_enabled": true,
"worker_id": "colab-b",
"worker_shard_index": 1,
"worker_shard_count": 3
```

Важно: на discover-Colab `upload-hf-shards` должен успевать пушить метадату; download-Colab без локального discover всё равно увидит новые ID после sync с HF.

### Лимит HF commits (128/час на repo) и число Colab

Hub ограничивает **128 commit’ов в час на один dataset repo** суммарно по всем Colab. Ты задаёшь, сколько инстансов работают параллельно — от этого код считает бюджет **на этот Colab**:

```bash
# 3 Colab одновременно (A discover+shards, B и C download) — на КАЖДОМ:
python scripts/colab_20k_bootstrap.py \
  ... \
  --parallel-colab-count 3 \
  --worker-shard-count 3
```

Или в `runtime_dataset_campaign_20k.json` / env:

```json
"hf_parallel_colab_count": 3,
"hf_repo_hourly_commit_limit": 128,
"hf_commit_budget_reserve": 0.9
```

```bash
export HF_PARALLEL_COLAB_COUNT=3
```

При `hf_parallel_colab_count: 3` (и reserve 0.9): **≤38 commit’ов/час на Colab**, интервал **≥95 с** между commit’ами в один repo; claims/done на HF выгружаются **батчом в конце pass**, а не после каждого видео.

Если нужны свои числа: `"hf_commit_limits_manual": true` и явно `hf_commit_hourly_limit` / `hf_commit_min_interval_seconds`.

Для Colab профиль `dataset_campaign_20k.json` использует `pytubefix` как основной backend, но с ротацией cookies и fallback client `WEB` для генерации PO-token через Node.js. Если меняешь runtime config вручную, проверь:

```json
"download_backend": "pytubefix",
"download_cookie_rotate_successes": 20,
"download_pytubefix_clients": ["ANDROID_VR", "WEB"]
```

Если все cookies и `WEB` client поймают bot-detection, worker попробует резервный `yt-dlp` backend.

Для клиента `WEB` pytubefix вызывает Node (`botGuard.js`). Без `nodejs-wheel-binaries` в лог может сыпаться огромный stderr — worker теперь режет такой мусор; лучше установить Node-пакет выше. При bot-detection на `ANDROID_VR` worker переключится на `WEB` или yt-dlp.

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

## 8. Grafana + Prometheus через Cloudflare (Colab)

Сборщик отдаёт метрики на хосте:

| Процесс | Порт | Как запустить |
|---------|------|----------------|
| discover | **9095** | `--metrics-port 9095` |
| run-workers | **9096** | `--metrics-port 9096` (не `0`) |

| Установка | Grafana порт | cloudflared `--url` |
|-----------|--------------|---------------------|
| **Native .deb** (твой способ) | **3000** | `http://127.0.0.1:3000` |
| Docker compose | **3001** (маппинг) | `http://127.0.0.1:3001` |

### Вариант A — native (без Docker, рекомендуется в Colab)

**Не используй несколько ячеек `!nohup` подряд** — при повторном Run Colab шлёт `interrupt` и убивает Grafana (в логе `Shutdown started` + `connection refused` на :3000).

**Ячейка 1** — установка (один раз за сессию):

```python
!cd /content/TrendFlowML/Fetcher && bash scripts/colab_monitoring_native.sh install
```

**Ячейка 2** — старт (запусти один раз, не перезапускай):

```python
!cd /content/TrendFlowML/Fetcher && bash scripts/colab_monitoring_native.sh start
!curl -sf http://127.0.0.1:3000/login && echo GRAFANA_OK
!curl -sf http://127.0.0.1:9090/-/ready && echo PROMETHEUS_OK
```

Обе должны напечатать `*_OK`. Если нет — смотри `/content/grafana.log`, не запускай cloudflared.

**Ячейка 3** — только туннель (отдельно, после OK):

```python
!pkill -f cloudflared || true
!nohup cloudflared tunnel --protocol http2 --url http://127.0.0.1:3000 > /content/cloudflared_log.txt 2>&1 &
!sleep 8 && tail -20 /content/cloudflared_log.txt
```

Скопируй URL из лога, затем **ячейка 4**:

```python
import os
os.environ["GRAFANA_ROOT_URL"] = "https://ВАШ.trycloudflare.com/"
!cd /content/TrendFlowML/Fetcher && bash scripts/colab_monitoring_native.sh restart
```

Проверка:

```bash
bash scripts/colab_monitoring_native.sh status
```

Важно: **`/content/prometheus.yml` должен быть реальным файлом**, не закомментированным в ноутбуке. Скрипт копирует готовый конфиг из `monitoring/prometheus/prometheus.colab.yml` (targets `127.0.0.1:9095` и `:9096`).

Туннель (**порт 3000**):

```bash
curl -sf http://127.0.0.1:3000/login && echo OK
cloudflared tunnel --protocol http2 --url http://127.0.0.1:3000
```

После появления URL (например `https://pool-shut-noble-trim.trycloudflare.com`):

```bash
export GRAFANA_ROOT_URL="https://pool-shut-noble-trim.trycloudflare.com/"
bash scripts/colab_monitoring_native.sh restart
```

Логин Grafana: `admin` / `admin`. Дашборд **Dataset Collector** — в папке provisioning.

Остановка:

```bash
pkill -f cloudflared
bash scripts/colab_monitoring_native.sh stop
```

### Вариант B — Docker

```bash
cd /content/TrendFlowML/Fetcher/monitoring
docker compose -f docker-compose.yml -f docker-compose.colab.yml up -d
curl -sf http://127.0.0.1:3001/login && echo OK
GRAFANA_PORT=3001 bash start_colab_tunnel.sh
```

Для Docker после URL: `GRAFANA_ROOT_URL=...` + `docker compose ... -f docker-compose.colab-tunnel.yml up -d --force-recreate grafana`.

### «Нет данных» в дашборде

1. discover `--metrics-port 9095`, workers `--metrics-port 9096`;
2. `curl -s http://127.0.0.1:9095/metrics | head` и `:9096/metrics`;
3. http://127.0.0.1:9090/targets — оба job **UP**;
4. На дашборде категория не **All**, интервал Last 30 minutes.

### 8b. Дашборд Coord Sync (multi-Colab)

Отдельный дашборд **`Dataset Collector — Coord Sync`** (`uid=dataset-collector-coord-sync`) в папке *Dataset Collector*.

Показывает при `hf_coord_enabled=true`:

- `dataset_collector_coord_enabled`, worker_id / shard_index / shard_count
- последний HF sync: metadata shards pulled, claim/done files, active claims, global done
- skip rate: `coord_shard`, `coord_claimed`, `coord_done`, `coord_claim_busy`
- lifecycle lag и throughput download/enrich

Проверка метрик на worker-Colab:

```bash
curl -s http://127.0.0.1:9096/metrics | grep dataset_collector_coord
```

На **каждом** download/enrich Colab свой `:9096` — в Grafana виден один instance за сессию; для сравнения нескольких Colab нужен отдельный Prometheus на каждом или federation.

## 9. Полная цепочка на нескольких Colab (команды)

Общие переменные (подставь свои пути и namespace HF):

```bash
export REPO=/content/TrendFlowML
export FETCHER=$REPO/Fetcher
export HF_PREFIX=Ilialebedev          # HF namespace
export HF_TOKEN=hf_...              # не в JSON!
export YT_KEYS=/content/drive/MyDrive/keys/keys.txt
export COOKIES_DIR=$FETCHER/fetcher/dataset_collector/cookies
export PROXIES_FILE=$FETCHER/fetcher/dataset_collector/proxies/proxies.txt
```

### Шаг 0 — на каждом Colab (один раз за сессию)

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
git clone <YOUR_REPO_URL> TrendFlowML
cd TrendFlowML/Fetcher
apt-get update -qq && apt-get install -y -qq nodejs
python -m pip install -q -e .
python -m pip install -q -U huggingface_hub yt-dlp "pytubefix>=9.5.0" nodejs-wheel-binaries google-api-python-client
```

Drive token для subprocess (ноутбук, один раз):

```python
from google.colab import auth
auth.authenticate_user()
```

```bash
python scripts/export_colab_drive_token.py \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main
```

---

### Colab A — discover + metadata → HF

`output-dir` — **главная** папка кампании (discover пишет shards сюда).

```bash
cd $FETCHER
export HF_TOKEN=hf_...
export FETCHER_YOUTUBE_DATA_API_KEYS="$(paste -sd, $YT_KEYS)"

# Терминал 1: discover (без --limit для полного прогона)
python scripts/colab_20k_bootstrap.py \
  --role discover \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main \
  --hf-repo-prefix $HF_PREFIX \
  --youtube-keys-file $YT_KEYS \
  --metrics-port 9095
```

```bash
# Терминал 2: выгрузка metadata shards на HF
python scripts/colab_20k_bootstrap.py \
  --role workers \
  --worker-kinds upload-hf-shards \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main \
  --hf-repo-prefix $HF_PREFIX \
  --metrics-port 9096
```

Опционально — мониторинг (native):

```bash
bash scripts/colab_monitoring_native.sh install
bash scripts/colab_monitoring_native.sh start
# discover :9095, workers :9096 — см. §8
```

Проверка:

```bash
python scripts/colab_20k_bootstrap.py \
  --role inventory-rebuild \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main
python scripts/colab_20k_bootstrap.py \
  --role status \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main
```

---

### Colab B, C, D — download (3 шарда, N=3)

У каждого Colab **свой** `output-dir` на Drive (локальные mp4 + state), общая метадата на `hf_shards_repo_id`.

**Colab B** — shard `0/3`:

```bash
cd $FETCHER
export HF_TOKEN=hf_...

python scripts/colab_20k_bootstrap.py \
  --role workers-download \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-dl-0 \
  --hf-repo-prefix $HF_PREFIX \
  --worker-id colab-dl-0 \
  --worker-shard-index 0 \
  --worker-shard-count 3 \
  --cookie-files-dir $COOKIES_DIR \
  --proxies-file $PROXIES_FILE \
  --metrics-port 9096
```

**Colab C** — shard `1/3`:

```bash
python scripts/colab_20k_bootstrap.py \
  --role workers-download \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-dl-1 \
  --hf-repo-prefix $HF_PREFIX \
  --worker-id colab-dl-1 \
  --worker-shard-index 1 \
  --worker-shard-count 3 \
  --cookie-files-dir $COOKIES_DIR \
  --proxies-file $PROXIES_FILE \
  --metrics-port 9096
```

**Colab D** — shard `2/3`:

```bash
python scripts/colab_20k_bootstrap.py \
  --role workers-download \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-dl-2 \
  --hf-repo-prefix $HF_PREFIX \
  --worker-id colab-dl-2 \
  --worker-shard-index 2 \
  --worker-shard-count 3 \
  --cookie-files-dir $COOKIES_DIR \
  --proxies-file $PROXIES_FILE \
  --metrics-port 9096
```

В `runtime_dataset_campaign_20k.json` на download-Colab при необходимости:

```json
"download_backend": "yt_dlp_first",
"hf_coord_enabled": true
```

(`--role workers-download` включает `hf_coord` автоматически.)

---

### Colab E, F — enrich (2 шарда, M=2)

**Colab E** — shard `0/2`:

```bash
cd $FETCHER
export HF_TOKEN=hf_...

python scripts/colab_20k_bootstrap.py \
  --role workers-enrich \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-en-0 \
  --hf-repo-prefix $HF_PREFIX \
  --worker-id colab-en-0 \
  --worker-shard-index 0 \
  --worker-shard-count 2 \
  --cookie-files-dir $COOKIES_DIR \
  --metrics-port 9096
```

**Colab F** — shard `1/2`:

```bash
python scripts/colab_20k_bootstrap.py \
  --role workers-enrich \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-en-1 \
  --hf-repo-prefix $HF_PREFIX \
  --worker-id colab-en-1 \
  --worker-shard-index 1 \
  --worker-shard-count 2 \
  --cookie-files-dir $COOKIES_DIR \
  --metrics-port 9096
```

---

### Сводка ролей

| Colab | Роль bootstrap | Порты metrics | output-dir |
|-------|----------------|---------------|------------|
| A | `discover` + `workers --worker-kinds upload-hf-shards` | 9095 + 9096 | `20k-main` |
| B–D | `workers-download` shard 0..2/3 | 9096 | `20k-dl-0..2` |
| E–F | `workers-enrich` shard 0..1/2 | 9096 | `20k-en-0..1` |

HF repos (при `--hf-repo-prefix Ilialebedev`):

- `Ilialebedev/dataset_20k_colab_shards` — metadata + `state/coordination/*`
- `Ilialebedev/dataset_20k_colab_videos` — mp4
- `Ilialebedev/dataset_20k_colab_enrich` — enrich shards

---

### Resume после обрыва Colab

```bash
# тот же output-dir и worker-id / shard-index, что при первом запуске
python scripts/colab_20k_bootstrap.py --role workers-download ... --metrics-port 9096
```

Очереди и coord claims подхватываются из локального state + HF sync.

### Snapshots (позже, с Colab A или отдельного)

```bash
python scripts/colab_20k_bootstrap.py \
  --role snapshot \
  --snapshot-index 1 \
  --output-dir /content/drive/MyDrive/dataset_runs/20k-main
```

Индексы: `1=7d`, `2=14d`, `3=21d`, `4=28d`.

### Типичные ошибки

- `prometheus.yml` в ноутбуке только в комментариях — Prometheus стартует без scrape targets.
- Туннель на **3001** при native Grafana — нужен **3000**.
- `Ctrl+C` / перезапуск runtime — Grafana/Prometheus из `nohup` умирают; снова `colab_monitoring_native.sh start`.
