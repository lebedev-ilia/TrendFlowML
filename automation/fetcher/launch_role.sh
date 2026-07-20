#!/usr/bin/env bash
# Запуск роли Fetcher dataset collector на ОДНОМ поде. Копируется на под и выполняется там
# (см. automation/fetcher/provision.py, которое кладёт секреты и вызывает этот скрипт по SSH).
#
# Использование на поде:
#   HF_TOKEN=hf_... WORKER_ID=fetcher-main WORKER_SHARD_INDEX=0 WORKER_SHARD_COUNT=3 ROLE=main \
#     bash launch_role.sh
#
# ROLE=main    -> discover (в фоне, с автовосстановлением после исчерпания квоты — уже в коде CLI) +
#                 workers (свой шард)
# ROLE=worker  -> только workers (свой шард)
set -euo pipefail

REPO=/workspace/TrendFlowML
FETCHER="$REPO/Fetcher"
OUTPUT_DIR=/workspace/dataset_runs/100k-monthly
LOGDIR=/workspace/logs
VENV=/workspace/venv
mkdir -p "$LOGDIR" "$OUTPUT_DIR"

# ffmpeg — системный пакет (НЕ на Network Volume, теряется при пересоздании контейнера пода),
# нужен yt-dlp/pytubefix для склейки видео+аудио дорожек при скачивании. Идемпотентная проверка
# на каждом запуске (баг найден 2026-07-16: без ffmpeg все download-воркеры падают в цикле).
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[launch_role] ffmpeg не найден, ставлю..." >> "$LOGDIR/setup.log"
  apt-get update -qq >> "$LOGDIR/setup.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >> "$LOGDIR/setup.log" 2>&1
  echo "[launch_role] ffmpeg готов" >> "$LOGDIR/setup.log"
fi

# Node.js — тоже системный пакет, теряется при пересоздании контейнера. yt-dlp использует его как
# JS-рантайм для решения YouTube signature-challenge при выборе форматов; без него download-воркер
# массово проваливает попытки ("Signature solving failed"). Найдено вотчдогом 2026-07-17 на
# fetcher-worker-b: dead_letter вырос до 1452/9500 (почти всё) без Node.js, упал до 1 после установки.
if ! command -v node >/dev/null 2>&1; then
  echo "[launch_role] node не найден, ставлю..." >> "$LOGDIR/setup.log"
  apt-get update -qq >> "$LOGDIR/setup.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nodejs >> "$LOGDIR/setup.log" 2>&1
  echo "[launch_role] node готов" >> "$LOGDIR/setup.log"
fi

# ВАЖНО: venv на Network Volume (/workspace), а НЕ системный pip3 — контейнер пода эфемерен
# (пережил один раз необъяснимое пересоздание, см. deploy.py/README.md), а /workspace персистентен.
# Системные пакеты (pip3 install без venv) теряются при пересоздании пода, venv на volume — нет.
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[launch_role] venv не найден, создаю на Network Volume ($VENV)..." >> "$LOGDIR/setup.log"
  python3 -m venv "$VENV" --system-site-packages
  source "$VENV/bin/activate"
  pip install -q -U pip
  pip install -q -r "$FETCHER/requirements.txt" >> "$LOGDIR/setup.log" 2>&1
  pip install -q -U huggingface_hub yt-dlp pytubefix >> "$LOGDIR/setup.log" 2>&1
  echo "[launch_role] venv готов" >> "$LOGDIR/setup.log"
else
  source "$VENV/bin/activate"
fi
PY="$VENV/bin/python3"

: "${WORKER_ID:?нужен WORKER_ID}"
: "${WORKER_SHARD_INDEX:?нужен WORKER_SHARD_INDEX}"
: "${WORKER_SHARD_COUNT:?нужен WORKER_SHARD_COUNT}"
: "${ROLE:?нужен ROLE=main|worker}"
: "${HF_TOKEN:?нужен HF_TOKEN}"

export HF_TOKEN
echo "$HF_TOKEN" > "$OUTPUT_DIR/.hf_token"

cd "$FETCHER"

CONFIG_OVERRIDES="$OUTPUT_DIR/config_overrides_${WORKER_ID}.json"
"$PY" - "$CONFIG_OVERRIDES" <<PYEOF
import json, sys
overrides = {
    "hf_parallel_colab_count": 5,
    "hf_commit_min_interval_seconds": 95,
    "hf_shard_upload_batch_files": 50,
    "hf_video_upload_batch_files": 100,
    "hf_enrich_upload_batch_files": 100,
    "download_pause_after_success_seconds": 10,
    "download_pause_after_fail_seconds": 15,
    "download_pause_after_bot_seconds": 120,
    "download_pause_after_fast_seconds": 15,
    # ИСТОРИЯ (не удалять — объясняет, почему тут было "yt_dlp", а не наоборот):
    # 2026-07-19 утро: pytubefix был 100% забанен (14 подряд bot_detection на одном видео,
    # ~28 мин впустую до yt-dlp) -> поставили "yt_dlp" (без фолбэка вообще).
    # 2026-07-19 вечер (эта правка): "yt_dlp" сам сломался — 100% FAIL на всех 3 подах
    # ("Requested format is not available" / "Only images are available", проверено живым SSH
    # в download.log). При этом ТЕ ЖЕ логи показывают, что ПЕРЕД переключением на "yt_dlp"
    # pytubefix:ANDROID_VR давал 396 OK / 49 FAIL (~89% успех) — т.е. состояние изменилось,
    # именно ANDROID_VR клиент сейчас жив. Раз оба бэкенда время от времени то живы, то мертвы —
    # используем "yt_dlp_first": пробуем yt-dlp (быстро, если жив), при ЛЮБОЙ неудаче падаем на
    # pytubefix (полная куки x клиент матрица, сейчас рабочая через ANDROID_VR). Худший случай —
    # +15с (download_pause_after_fail_seconds) на видео, если оба бэкенда сейчас мертвы; не 28 мин.
    "download_backend": "yt_dlp_first",
    "worker_id": "$WORKER_ID",
    "worker_shard_index": int("$WORKER_SHARD_INDEX"),
    "worker_shard_count": int("$WORKER_SHARD_COUNT"),
}
with open(sys.argv[1], "w") as f:
    json.dump(overrides, f, indent=2)
PYEOF

# --- workers (download+enrich+upload, свой шард) — крутится всегда (main=шард 0/5 на RunPod,
# 4 остальных шарда — на Colab через Colab_100k_monthly_worker.ipynb) ---
nohup "$PY" scripts/colab_20k_bootstrap.py \
  --campaign-profile 100k-monthly \
  --role workers \
  --output-dir "$OUTPUT_DIR" \
  --hf-repo-prefix Ilialebedev \
  --interval 120 \
  --metrics-port 0 \
  --worker-id "$WORKER_ID" \
  --worker-shard-index "$WORKER_SHARD_INDEX" \
  --worker-shard-count "$WORKER_SHARD_COUNT" \
  --parallel-colab-count 5 \
  --config-overrides-json "$CONFIG_OVERRIDES" \
  --hf-coord \
  > "$LOGDIR/workers_${WORKER_ID}.log" 2>&1 < /dev/null &
disown
echo "workers ($WORKER_ID) pid=$!"

if [ "$ROLE" = "main" ]; then
  # --- discover (только на main) — CLI сам восстанавливается после исчерпания квоты API-ключей
  # (патч 2026-07-16, cli.py::command_discover), процесс НЕ нужно перезапускать снаружи по крашу
  # квоты. Внешний цикл — только на случай, если процесс всё же завершится штатно (весь список
  # категорий добит для сегодняшнего дня) или упадёт по РЕАЛЬНОЙ ошибке (не квота).
  nohup bash -c '
    cd '"$FETCHER"'
    while true; do
      '"$PY"' scripts/colab_20k_bootstrap.py \
        --campaign-profile 100k-monthly \
        --role discover \
        --output-dir '"$OUTPUT_DIR"' \
        --hf-repo-prefix Ilialebedev \
        --metrics-port 0 \
        --config-overrides-json '"$CONFIG_OVERRIDES"' \
        >> '"$LOGDIR"'/discover.log 2>&1
      echo "[launch_role] discover процесс завершился (rc=$?), пауза 60с и перезапуск" >> '"$LOGDIR"'/discover.log
      sleep 60
    done
  ' > "$LOGDIR/discover_wrapper.log" 2>&1 < /dev/null &
  disown
  echo "discover ($WORKER_ID) pid=$!"
fi

echo "Запущено. Логи: $LOGDIR/"
