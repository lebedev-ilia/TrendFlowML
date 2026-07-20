#!/usr/bin/env bash
# Поднимает локальный PO Token provider (bgutil-ytdlp-pot-provider) для yt-dlp — HTTP-сервер на
# 127.0.0.1:4416, который yt-dlp подхватывает автоматически (plugin auto-detect, без флагов).
#
# Почему нужен: YouTube с некоторого момента (обнаружено 2026-07-19) требует Proof-of-Origin
# token почти на всех клиентах — без него yt-dlp массово падает ("Requested format is not
# available" / bot detection), даже со свежим IP и рабочими куки (подтверждено живым тестом на
# новом RunPod-поде в другом датацентре). bgutil генерирует токен локально через BotGuard
# (LuanRT/BgUtils), yt-dlp находит его сам через PO Token Provider Framework.
#
# Используется на RunPod (launch_role.sh) и в Colab (секция ноутбука) — принимает BASE_DIR первым
# аргументом (где клонировать репозиторий провайдера): /workspace на RunPod (персистентно), /content
# в Colab (эфемерно, но сервер поднимается заново на каждый Colab-сеанс — это нормально, старт ~10с).
#
# Идемпотентно: если сервер уже слушает порт 4416 — ничего не делает.
set -euo pipefail

BASE_DIR="${1:-/workspace}"
PROVIDER_DIR="$BASE_DIR/bgutil-ytdlp-pot-provider"
PROVIDER_VERSION="1.3.1"
PORT="${POT_PROVIDER_PORT:-4416}"
LOG_DIR="${POT_PROVIDER_LOG_DIR:-$BASE_DIR/logs}"
mkdir -p "$LOG_DIR"

log() { echo "[setup_pot_provider] $*"; }

# --- уже поднят? ---
if curl -s -o /dev/null -m 3 "http://127.0.0.1:${PORT}/ping"; then
  log "уже запущен на 127.0.0.1:${PORT} — пропускаю"
  exit 0
fi

# --- Deno (>=2.0.0) ---
if ! command -v deno >/dev/null 2>&1 && [ ! -x "$HOME/.deno/bin/deno" ]; then
  log "Deno не найден, ставлю..."
  curl -fsSL https://deno.land/install.sh | sh -s -- >> "$LOG_DIR/setup_pot_provider.log" 2>&1
fi
export PATH="$HOME/.deno/bin:$PATH"
DENO_BIN="$(command -v deno || echo "$HOME/.deno/bin/deno")"
log "deno: $("$DENO_BIN" --version 2>&1 | head -1)"

# --- git clone / update провайдера ---
if [ ! -d "$PROVIDER_DIR/.git" ]; then
  log "клонирую bgutil-ytdlp-pot-provider $PROVIDER_VERSION..."
  git clone --single-branch --branch "$PROVIDER_VERSION" \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$PROVIDER_DIR" \
    >> "$LOG_DIR/setup_pot_provider.log" 2>&1
fi

cd "$PROVIDER_DIR/server"
if [ ! -d node_modules ]; then
  log "deno install (сборка server)..."
  "$DENO_BIN" install --allow-scripts=npm:canvas --frozen >> "$LOG_DIR/setup_pot_provider.log" 2>&1
fi

# --- pip plugin (yt-dlp сам находит провайдер через него) ---
PY="${POT_PROVIDER_PYTHON:-python3}"
"$PY" -m pip install -q -U bgutil-ytdlp-pot-provider >> "$LOG_DIR/setup_pot_provider.log" 2>&1 || \
  log "WARN: pip install bgutil-ytdlp-pot-provider не удался (см. $LOG_DIR/setup_pot_provider.log)"

# --- запуск сервера в фоне ---
log "запускаю сервер на порту $PORT..."
cd "$PROVIDER_DIR/server/node_modules"
nohup "$DENO_BIN" run --allow-env --allow-net --allow-ffi=. --allow-read=. \
  ../src/main.ts --port "$PORT" \
  >> "$LOG_DIR/pot_provider_server.log" 2>&1 < /dev/null &
disown
SERVER_PID=$!
log "сервер pid=$SERVER_PID, лог: $LOG_DIR/pot_provider_server.log"

# --- health check (до 15с) ---
for i in $(seq 1 15); do
  sleep 1
  if curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/ping"; then
    log "OK: провайдер отвечает на 127.0.0.1:${PORT}"
    exit 0
  fi
done
log "WARN: провайдер не ответил за 15с — проверь $LOG_DIR/pot_provider_server.log"
exit 0
